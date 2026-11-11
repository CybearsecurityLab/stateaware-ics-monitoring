import pandas as pd
import numpy as np
import pytest
import wntr

from simulation import RunSpec, run_nominal
from simulation.cyber import _validate_replay, run_cyber, write_cyber_bundle
from tests.simulation.test_runner import make_network


def controlled(path):
    wn = wntr.network.WaterNetworkModel()
    wn.add_reservoir('R', base_head=10)
    wn.add_tank('T', elevation=0, init_level=10, min_level=0, max_level=30, diameter=20)
    wn.add_curve('c', 'HEAD', [(0, 40), (0.02, 20), (0.04, 0)])
    wn.add_pump('P', 'R', 'T', pump_type='HEAD', pump_parameter='c', initial_status='CLOSED')
    for name, relation, threshold, status in [('low', '<', 5, 'OPEN'), ('high', '>', 15, 'CLOSED')]:
        cond = wntr.network.controls.ValueCondition(wn.get_node('T'), 'level', relation, threshold)
        action = wntr.network.controls.ControlAction(wn.get_link('P'), 'status', wntr.network.LinkStatus[status.title()])
        wn.add_control(name, wntr.network.controls.Control(cond, action))
    wn.options.time.duration = 1200
    wn.options.time.hydraulic_timestep = 300
    wn.options.time.report_timestep = 300
    wntr.network.write_inpfile(wn, str(path))
    return path


def event(kind, target, value, **kw):
    return {**dict(kind=kind, target=target, value=value, start=300, end=900), **kw}


def test_nominal_segmented(tmp_path):
    spec = RunSpec(make_network(tmp_path / 'n.inp'), 'n')
    pd.testing.assert_frame_equal(run_cyber(spec).plant_truth, run_nominal(spec).plant_truth, check_exact=False, rtol=1e-8, atol=1e-8)


def test_feedback_scada_labels_and_boundaries(tmp_path):
    spec = RunSpec(controlled(tmp_path / 'n.inp'), 'n')
    base = run_cyber(spec)
    attack = event('sensor', 'T', 0, mode='constant', scope='both')
    b = run_cyber(spec, [attack])
    assert b.labels.attack_active.tolist() == [False, True, True, False, False]
    assert b.controller_views['PLC']['T'].iloc[1:3].tolist() == [0, 0]
    assert b.plant_truth.s_P.iloc[2] == 1
    assert b.labels.physical_impact.iloc[2:].any()
    scada = run_cyber(spec, [{**attack, 'scope': 'scada'}])
    pd.testing.assert_frame_equal(scada.plant_truth, base.plant_truth)
    assert scada.labels.delivery_or_command_modified.iloc[1]
    assert not scada.labels.physical_impact.any()


def test_actuator_threshold_and_priority(tmp_path):
    spec = RunSpec(controlled(tmp_path / 'n.inp'), 'n')
    b = run_cyber(spec, [event('actuator', 'P', 'OPEN'), event('actuator', 'P', 'CLOSED', start=600, end=900, priority=1)])
    assert b.plant_truth.s_P.iloc[2] == 1
    assert b.plant_truth.s_P.iloc[3] == 0
    c = run_cyber(spec, [event('threshold', 'control 1', 20)])
    assert c.plant_truth.s_P.iloc[2] == 1
    assert c.manifest['controls'][0]['threshold'] == pytest.approx(5)
    release = run_cyber(spec, [event('actuator', 'P', 'OPEN')])
    assert release.plant_truth.s_P.tolist() == [0, 0, 1, 1, 0]
    assert not release.labels.attack_active.iloc[3]
    assert release.labels.physical_impact.iloc[3]
    restored = run_cyber(spec, [event('sensor', 'T', 1, mode='constant', scope='controller', start=0, end=1200),
                                event('threshold', 'control 1', 0, start=0, end=900)])
    assert restored.plant_truth.s_P.tolist() == [0, 0, 0, 0, 1]


def test_sensor_modes_and_ownership(tmp_path):
    spec = RunSpec(controlled(tmp_path / 'n.inp'), 'n')
    plcs = tmp_path / 'plcs.yaml'
    plcs.write_text('- name: A\n  sensors: [T]\n  actuators: [P]\n')
    freeze = run_cyber(spec, [event('sensor', 'T', None, mode='freeze', scope='scada')], plcs_path=plcs)
    assert freeze.observed['T'].iloc[1] == freeze.observed['T'].iloc[2]
    assert freeze.observed['T'].iloc[3] == freeze.plant_truth['T'].iloc[3]
    offset = run_cyber(spec, [event('sensor', 'T', 2, mode='offset', scope='controller')], plcs_path=plcs)
    assert offset.controller_views['A']['T'].iloc[1] == pytest.approx(offset.plant_truth['T'].iloc[1] + 2)
    series = run_cyber(spec, [event('sensor', 'T', {'300': 1, '600': 2}, mode='series', scope='scada')], plcs_path=plcs)
    assert series.observed['T'].iloc[1:3].tolist() == [1, 2]
    plcs.write_text('- name: A\n  sensors: [T]\n  actuators: [P]\n- name: B\n  sensors: [T]\n  actuators: [P]\n')
    with pytest.raises(ValueError, match='ownership'):
        run_cyber(spec, plcs_path=plcs)


def test_sensor_priority_and_execution_log(tmp_path):
    spec = RunSpec(controlled(tmp_path / 'n.inp'), 'n', seed=19)
    attacks = [event('sensor', 'T', 1, mode='constant', scope='scada'),
               event('sensor', 'T', 2, mode='constant', scope='scada')]
    b = run_cyber(spec, attacks)
    assert b.observed['T'].iloc[1] == 2
    assert b.events[1]['seed'] == 19
    assert b.events[1]['applied'] == [{'recipient': 'scada', 'value': 2.0}]
    attacks[0]['priority'] = 10
    assert run_cyber(spec, attacks).observed['T'].iloc[1] == 1


def test_time_controls_and_incomplete_runs_rejected(tmp_path, monkeypatch):
    path = make_network(tmp_path / 'n.inp')
    wn = wntr.network.WaterNetworkModel(str(path))
    condition = wntr.network.controls.SimTimeCondition(wn, '=', 300)
    action = wntr.network.controls.ControlAction(wn.get_link('in'), 'status', wntr.network.LinkStatus.Closed)
    wn.add_control('time', wntr.network.controls.Control(condition, action))
    wntr.network.write_inpfile(wn, str(path))
    with pytest.raises(ValueError, match='unsupported'):
        run_cyber(RunSpec(path, 'n'))
    path = make_network(path)
    original = wntr.sim.WNTRSimulator.run_sim
    def incomplete(self, **kwargs):
        result = original(self, **kwargs)
        result.time = []
        return result
    monkeypatch.setattr(wntr.sim.WNTRSimulator, 'run_sim', incomplete)
    with pytest.raises(RuntimeError, match='incomplete'):
        run_cyber(RunSpec(path, 'n'))


def test_serialization_and_speed_rejection(tmp_path):
    path = controlled(tmp_path / 'n.inp')
    b = run_cyber(RunSpec(path, 'n'), [event('sensor', 'T', 0, mode='constant', scope='scada')])
    write_cyber_bundle(b, tmp_path / 'a')
    write_cyber_bundle(b, tmp_path / 'b')
    for p in (tmp_path / 'a').iterdir():
        assert p.read_bytes() == (tmp_path / 'b' / p.name).read_bytes()
    wn = wntr.network.WaterNetworkModel(str(path))
    wn.get_link('P').base_speed = 0.8
    wntr.network.write_inpfile(wn, str(path))
    with pytest.raises(ValueError, match='speed'):
        run_cyber(RunSpec(path, 'n'))


def test_initial_command_drives_first_interval(tmp_path):
    path = controlled(tmp_path / 'n.inp')
    wn = wntr.network.WaterNetworkModel(str(path))
    wn.get_node('T').init_level = 4
    wntr.network.write_inpfile(wn, str(path))
    b = run_cyber(RunSpec(path, 'initial'))
    initial = b.hydraulic_truth.iloc[0]
    assert b.plant_truth.s_P.iloc[0] == 0
    assert initial.s_P == 1
    assert initial.f_P > 0
    volume_change = (b.plant_truth['T'].iloc[1] - b.plant_truth['T'].iloc[0]) * np.pi * 100
    assert volume_change == pytest.approx(initial.f_P * 300, abs=1e-5)


def test_tank_limit_intermediate_solves_and_scan_comparison(tmp_path):
    path = controlled(tmp_path / 'n.inp')
    wn = wntr.network.WaterNetworkModel(str(path))
    wn.get_node('T').init_level = 29.99
    wn.get_link('P').initial_status = wntr.network.LinkStatus.Open
    for name, _ in list(wn.controls()):
        wn.remove_control(name)
    wntr.network.write_inpfile(wn, str(path))
    spec = RunSpec(path, 'limit')
    base = run_cyber(spec)
    assert any(0 < t < 300 for t in base.hydraulic_truth.time_seconds)
    assert base.hydraulic_truth.s_P.iloc[1] == 0
    attack = run_cyber(spec, [event('actuator', 'P', 'CLOSED', start=0, end=300)])
    assert attack.plant_truth.time_seconds.tolist() == base.plant_truth.time_seconds.tolist()
    assert attack.hydraulic_truth.time_seconds.tolist() != base.hydraulic_truth.time_seconds.tolist()
    assert attack.labels.physical_impact.any()


def test_remote_sensors_and_recipient_specific_spoof(tmp_path):
    path = controlled(tmp_path / 'n.inp')
    wn = wntr.network.WaterNetworkModel(str(path))
    wn.add_pump('Q', 'R', 'T', pump_type='HEAD', pump_parameter='c', initial_status='CLOSED')
    cond = wntr.network.controls.ValueCondition(wn.get_node('T'), 'level', '<', 5)
    act = wntr.network.controls.ControlAction(wn.get_link('Q'), 'status', wntr.network.LinkStatus.Open)
    wn.add_control('other', wntr.network.controls.Control(cond, act))
    wntr.network.write_inpfile(wn, str(path))
    plcs = tmp_path / 'plcs.yaml'
    plcs.write_text('- name: A\n  actuators: [P]\n- name: B\n  sensors: [T]\n  actuators: [Q]\n')
    b = run_cyber(RunSpec(path, 'remote'), [event('sensor', 'T', 0, mode='constant', scope='controller', controller='A')], plcs_path=plcs)
    assert b.manifest['sensor_dependencies']['A'] == {'T': 'B'}
    assert b.manifest['actuator_ownership'] == {'P': 'A', 'Q': 'B'}
    assert b.controller_views['A']['T'].iloc[1] == 0
    assert b.controller_views['B']['T'].iloc[1] == b.plant_truth['T'].iloc[1]
    assert b.plant_truth.s_P.iloc[2] == 1 and b.plant_truth.s_Q.iloc[2] == 0


def test_override_label_persists_and_validation(tmp_path):
    spec = RunSpec(controlled(tmp_path / 'n.inp'), 'labels')
    b = run_cyber(spec, [event('actuator', 'P', 'OPEN')])
    assert b.labels.delivery_or_command_modified.tolist() == [False, True, True, False, False]
    b.controller_views.clear()
    with pytest.raises(ValueError, match='controller'):
        b.validate()
    b = run_cyber(spec)
    b.labels['attack_active'] = b.labels['attack_active'].astype(int)
    with pytest.raises(ValueError, match='label types'):
        b.validate()


def test_exact_thresholds_hold_initial_state(tmp_path):
    spec = RunSpec(controlled(tmp_path / 'n.inp'), 'threshold')
    low, high = [c['threshold'] for c in run_cyber(spec).manifest['controls']]
    attacks = [event('sensor', 'T', {'0': low, '300': 4, '600': high, '900': 16}, mode='series', scope='controller', start=0, end=1200)]
    b = run_cyber(spec, attacks)
    assert b.hydraulic_truth.s_P.tolist() == [0, 1, 1, 0, 0]


def test_patterned_demand_and_reservoir_head(tmp_path):
    path = make_network(tmp_path / 'n.inp')
    wn = wntr.network.WaterNetworkModel(str(path))
    wn.add_pattern('demand_pattern', [1, 2, 0.5, 1])
    wn.add_pattern('head_pattern', [1, 1.1, 0.9, 1])
    wn.get_node('J').demand_timeseries_list[0].pattern_name = 'demand_pattern'
    wn.get_node('R').head_timeseries.pattern_name = 'head_pattern'
    wn.options.time.pattern_timestep = 300
    wntr.network.write_inpfile(wn, str(path))
    spec = RunSpec(path, 'patterns')
    cyber = run_cyber(spec)
    nominal = run_nominal(spec)
    pd.testing.assert_frame_equal(cyber.hydraulic_truth, nominal.plant_truth, check_exact=False, rtol=1e-6, atol=1e-7)
    assert cyber.plant_truth.demand_J.iloc[:3].tolist() == pytest.approx([0.002, 0.004, 0.001])
    assert cyber.plant_truth.head_R.iloc[:3].tolist() == pytest.approx([30, 33, 27])


def test_override_release_restores_active_valve_command(tmp_path, monkeypatch):
    wn = wntr.network.WaterNetworkModel()
    wn.add_reservoir('R', base_head=20)
    wn.add_junction('A', elevation=0)
    wn.add_junction('J', base_demand=0.001, elevation=0)
    wn.add_pipe('pipe', 'R', 'A', length=100, diameter=0.3, roughness=100)
    wn.add_valve('V', 'A', 'J', diameter=0.2, valve_type='PRV', initial_setting=30, initial_status='ACTIVE')
    wn.options.time.duration = 1200
    wn.options.time.hydraulic_timestep = 300
    wn.options.time.report_timestep = 300
    path = tmp_path / 'valve.inp'
    wntr.network.write_inpfile(wn, str(path))
    spec = RunSpec(path, 'valve')
    actions = []
    original = wntr.network.controls.ControlAction.run_control_action
    def record(action):
        target, attr = action.target()
        if target.name == 'V' and attr == 'status':
            actions.append(int(action._value))
        return original(action)
    monkeypatch.setattr(wntr.network.controls.ControlAction, 'run_control_action', record)
    b = run_cyber(spec, [event('actuator', 'V', 'CLOSED')])
    assert b.plant_truth.s_V.iloc[0] == 1
    assert b.labels.delivery_or_command_modified.iloc[1:3].all()
    assert actions == [0, 0, 2]
    baseline = run_cyber(spec)
    pd.testing.assert_frame_equal(b.hydraulic_truth.iloc[3:].reset_index(drop=True), baseline.hydraulic_truth.iloc[3:].reset_index(drop=True), check_exact=False, rtol=1e-8, atol=1e-8)


def replayable(path):
    path = controlled(path)
    wn = wntr.network.WaterNetworkModel(str(path))
    wn.get_node('T').init_level = 4
    wntr.network.write_inpfile(wn, str(path))
    return path


def replay_event(**kw):
    return {**dict(kind='sensor', target='T', mode='replay', scope='controller', period=1, start=600, end=900, seed=5), **kw}


def test_replay_delivers_pre_attack_history_in_order(tmp_path):
    spec = RunSpec(replayable(tmp_path / 'n.inp'), 'replay')
    base = run_cyber(spec)
    attack = replay_event(period=2, start=600, end=1200)
    b = run_cyber(spec, [attack])
    view = b.controller_views['PLC']['T']
    assert view.iloc[2] == pytest.approx(base.plant_truth['T'].iloc[0])
    assert view.iloc[3] == pytest.approx(base.plant_truth['T'].iloc[1])
    assert view.iloc[2] != pytest.approx(b.plant_truth['T'].iloc[2])
    applied = [entry for record in b.events for entry in record['applied']]
    assert [entry['source_time_seconds'] for entry in applied] == [0.0, 300.0]
    assert [entry['source_value'] for entry in applied] == pytest.approx(base.plant_truth['T'].iloc[:2].tolist())
    assert all(entry['noise'] == 0.0 and entry['clamp'] == 'none' for entry in applied)
    assert all(entry['recipient'] == 'PLC' for entry in applied)


def test_replay_is_recipient_specific(tmp_path):
    spec = RunSpec(replayable(tmp_path / 'n.inp'), 'recipient')
    base = run_cyber(spec)
    constant = {'kind': 'sensor', 'target': 'T', 'value': 2.0, 'mode': 'constant', 'scope': 'scada', 'start': 0, 'end': 600}
    b = run_cyber(spec, [constant, replay_event(scope='both', period=1, start=600, end=900)])
    assert b.observed['T'].iloc[2] == pytest.approx(2.0)
    assert b.controller_views['PLC']['T'].iloc[2] == pytest.approx(base.plant_truth['T'].iloc[1])
    recipients = {entry['recipient']: entry for record in b.events if record['time_seconds'] == 600.0 for entry in record['applied']}
    assert recipients['scada']['source_value'] == pytest.approx(2.0)
    assert recipients['PLC']['source_value'] == pytest.approx(base.plant_truth['T'].iloc[1])


def test_replay_requires_pre_attack_history(tmp_path):
    spec = RunSpec(replayable(tmp_path / 'n.inp'), 'history')
    with pytest.raises(ValueError, match='pre-attack history'):
        run_cyber(spec, [replay_event(period=1, start=0, end=300)])
    with pytest.raises(ValueError, match='pre-attack history'):
        run_cyber(spec, [replay_event(period=4, start=600, end=1200)])


def test_replay_noise_is_reproducible_and_clamped(tmp_path):
    spec = RunSpec(replayable(tmp_path / 'n.inp'), 'noise')
    noisy = replay_event(period=1, noise=0.5, seed=11)
    first = run_cyber(spec, [noisy])
    second = run_cyber(spec, [noisy])
    assert first.controller_views['PLC']['T'].tolist() == second.controller_views['PLC']['T'].tolist()
    entries = [entry for record in first.events for entry in record['applied']]
    assert len(entries) == 1
    assert abs(entries[0]['noise']) <= 0.5
    assert entries[0]['value'] == pytest.approx(entries[0]['source_value'] + entries[0]['noise'])
    other = run_cyber(spec, [replay_event(period=1, noise=0.5, seed=12)])
    other_entries = [entry for record in other.events for entry in record['applied']]
    assert other_entries[0]['noise'] != entries[0]['noise']
    clamped = run_cyber(spec, [replay_event(period=1, max=3.5)])
    entry = [e for record in clamped.events for e in record['applied']][0]
    assert entry['value'] == pytest.approx(3.5) and entry['clamp'] == 'max'
    clamped = run_cyber(spec, [replay_event(period=1, min=4.5)])
    entry = [e for record in clamped.events for e in record['applied']][0]
    assert entry['value'] == pytest.approx(4.5) and entry['clamp'] == 'min'
    both = run_cyber(spec, [replay_event(period=1, scope='both', noise=0.5, seed=11)])
    entries = [entry for record in both.events for entry in record['applied']]
    assert {entry['recipient'] for entry in entries} == {'scada', 'PLC'}
    assert entries[0]['noise'] != entries[1]['noise']


def test_freeze_dos_holds_value_and_resumes(tmp_path):
    spec = RunSpec(replayable(tmp_path / 'n.inp'), 'dos')
    base = run_cyber(spec)
    b = run_cyber(spec, [event('sensor', 'T', None, mode='freeze', scope='controller', start=300, end=900)])
    view = b.controller_views['PLC']['T']
    assert view.iloc[1] == pytest.approx(view.iloc[2])
    assert view.iloc[1] == pytest.approx(base.plant_truth['T'].iloc[1])
    assert view.iloc[2] != pytest.approx(b.plant_truth['T'].iloc[2])
    assert view.iloc[3] == pytest.approx(b.plant_truth['T'].iloc[3])
    assert b.labels.attack_active.tolist() == [False, True, True, False, False]
    assert b.labels.delivery_or_command_modified.tolist() == [False, False, True, False, False]


def test_replay_boundaries_and_validation(tmp_path):
    spec = RunSpec(replayable(tmp_path / 'n.inp'), 'bounds')
    b = run_cyber(spec, [replay_event(period=1, start=900, end=1200)])
    assert b.labels.attack_active.tolist() == [False, False, False, True, False]
    assert b.controller_views['PLC']['T'].iloc[4] == pytest.approx(b.plant_truth['T'].iloc[4])
    with pytest.raises(ValueError, match='positive integer period'):
        run_cyber(spec, [replay_event(period=0)])
    with pytest.raises(ValueError, match='positive integer period'):
        run_cyber(spec, [replay_event(period=1.5)])
    with pytest.raises(ValueError, match='noise amplitude'):
        run_cyber(spec, [replay_event(noise=-1.0)])
    with pytest.raises(ValueError, match='clamp bounds'):
        run_cyber(spec, [replay_event(max='high')])
    with pytest.raises(ValueError, match='clamp bounds'):
        run_cyber(spec, [replay_event(min=5.0, max=1.0)])
    with pytest.raises(ValueError, match='clamp bounds'):
        _validate_replay({'start': 0, 'end': 300, 'period': 1, 'max': float('nan')}, 300, 1200)
    with pytest.raises(ValueError, match='noise amplitude'):
        _validate_replay({'start': 0, 'end': 300, 'period': 1, 'noise': float('inf')}, 300, 1200)
    with pytest.raises(ValueError, match='invalid sensor attack'):
        run_cyber(spec, [replay_event(target='missing')])
    with pytest.raises(ValueError, match='only targets scada'):
        run_cyber(spec, [replay_event(scope='scada', controller='PLC')])
    with pytest.raises(ValueError, match='no controller recipient'):
        run_cyber(spec, [replay_event(scope='controller', controller='OTHER')])


def test_replay_delivery_is_separate_from_physical_impact(tmp_path):
    spec = RunSpec(replayable(tmp_path / 'n.inp'), 'labels')
    b = run_cyber(spec, [replay_event(period=1, scope='scada', start=600, end=900)])
    assert b.labels.attack_active.iloc[2]
    assert b.labels.delivery_or_command_modified.iloc[2]
    assert not b.labels.physical_impact.any()
    pd.testing.assert_frame_equal(b.plant_truth, run_cyber(spec).plant_truth)
