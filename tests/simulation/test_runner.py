import hashlib

import numpy as np
import pandas as pd
import pytest
import wntr

from simulation import RunSpec, load_bundle, run_nominal, write_bundle
from stl.residuals import (junction_continuity_residual, pipe_energy_residual,
                           pump_dispatch, tank_volume_residual, valve_dispatch)


def make_network(path, demand_model="DDA", reservoir_head=30):
    wn = wntr.network.WaterNetworkModel()
    wn.add_reservoir("R", base_head=reservoir_head)
    wn.add_junction("J", base_demand=0.002, elevation=0)
    wn.add_tank("T", elevation=0, init_level=10, min_level=0, max_level=20, diameter=20)
    wn.add_pipe("in", "R", "J", length=100, diameter=0.3, roughness=100)
    wn.add_pipe("tank", "J", "T", length=100, diameter=0.3, roughness=100)
    wn.options.time.duration = 1200
    wn.options.time.hydraulic_timestep = 300
    wn.options.time.report_timestep = 600
    wn.options.hydraulic.demand_model = demand_model
    wn.options.hydraulic.minimum_pressure = 0
    wn.options.hydraulic.required_pressure = 40
    wntr.network.write_inpfile(wn, str(path))
    return path


def test_full_signals_continuity_tank_and_provenance(tmp_path):
    path = make_network(tmp_path / "network.inp")
    bundle = run_nominal(RunSpec(path, "run", 12, "train"))
    frame = bundle.plant_truth
    assert frame.time_seconds.tolist() == [0, 300, 600, 900, 1200]
    assert {"f_in", "f_tank", "s_in", "s_tank", "J", "T", "demand_J", "leak_J", "emitter_J", "head_R"} <= set(frame)
    assert np.allclose(frame.f_in - frame.f_tank, frame.demand_J + frame.leak_J, atol=1e-7)
    assert (frame.f_tank > 0).all()
    assert (frame["T"].diff().dropna() > 0).all()
    assert np.allclose(frame["T"].diff().iloc[1:] * np.pi * 100, frame.f_tank.iloc[:-1] * 300, atol=1e-5)
    assert (frame.s_in == 1).all()
    assert np.allclose(frame.demand_J, 0.002)
    assert (frame.emitter_J == 0).all()
    assert bundle.manifest["inp_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert bundle.manifest["wntr_version"] == wntr.__version__
    assert bundle.manifest["timesteps"]["report_timestep"] == 600
    assert bundle.events == []
    bundle.observed.loc[0, "J"] = 999
    assert bundle.controller_view.loc[0, "J"] != 999
    assert bundle.plant_truth.loc[0, "J"] != 999


def test_pdd_delivered_demand_and_isolation(tmp_path):
    path = make_network(tmp_path / "network.inp", "PDD", 5)
    a = run_nominal(RunSpec(path, "a", split="train"))
    b = run_nominal(RunSpec(path, "b", split="test"))
    assert a.manifest["demand_model"] in ("PDD", "PDA")
    assert (a.plant_truth.demand_J < 0.002).all()
    assert a.manifest["split"] != b.manifest["split"]
    pd.testing.assert_frame_equal(a.plant_truth, b.plant_truth)


def test_write_load_deterministic_and_validate(tmp_path):
    bundle = run_nominal(RunSpec(make_network(tmp_path / "n.inp"), "a"))
    write_bundle(bundle, tmp_path / "a")
    write_bundle(bundle, tmp_path / "b")
    for filename in ("observed.csv", "plant_truth.csv", "controller_view.csv", "events.json", "manifest.json"):
        assert (tmp_path / "a" / filename).read_bytes() == (tmp_path / "b" / filename).read_bytes()
    loaded = load_bundle(tmp_path / "a")
    pd.testing.assert_frame_equal(bundle.plant_truth, loaded.plant_truth, check_dtype=False)
    loaded.observed.drop(columns="J", inplace=True)
    with pytest.raises(ValueError):
        loaded.validate()


def test_pump_and_valve_results(tmp_path):
    wn = wntr.network.WaterNetworkModel()
    wn.add_reservoir("R", base_head=10)
    wn.add_junction("A", elevation=0)
    wn.add_junction("B", base_demand=0.001, elevation=0)
    wn.add_curve("curve", "HEAD", [(0, 40), (0.02, 20), (0.04, 0)])
    wn.add_pump("P", "R", "A", pump_type="HEAD", pump_parameter="curve", speed=1.0)
    wn.add_valve("V", "A", "B", diameter=0.2, valve_type="TCV", initial_setting=2)
    wn.options.time.duration = 600
    wn.options.time.hydraulic_timestep = 300
    wn.options.time.report_timestep = 300
    path = tmp_path / "pump.inp"
    wntr.network.write_inpfile(wn, str(path))
    bundle = run_nominal(RunSpec(path, "pump"))
    frame = bundle.plant_truth
    assert (frame.f_P > 0).all()
    assert (frame.s_P == 1).all()
    assert (frame.s_V == 2).all()
    assert np.allclose(frame.setting_V, 2)
    assert (frame.speed_P == 1).all()
    assert "speed_P" not in bundle.manifest["unavailable_channels"]


def test_unsupported_emitters_and_speeds(tmp_path):
    path = make_network(tmp_path / "n.inp")
    wn = wntr.network.WaterNetworkModel(str(path))
    wn.get_node("J").emitter_coefficient = 0.01
    wntr.network.write_inpfile(wn, str(path))
    with pytest.raises(ValueError, match="nonzero emitter"):
        run_nominal(RunSpec(path, "emitter"))

    wn = wntr.network.WaterNetworkModel()
    wn.add_reservoir("R", base_head=10)
    wn.add_junction("J", base_demand=0.001)
    wn.add_curve("c", "HEAD", [(0, 40), (0.02, 20), (0.04, 0)])
    wn.add_pump("P", "R", "J", pump_type="HEAD", pump_parameter="c", speed=0.8)
    wntr.network.write_inpfile(wn, str(path))
    with pytest.raises(ValueError, match="non-unit or patterned"):
        run_nominal(RunSpec(path, "speed"))
    wn.get_link("P").base_speed = 1
    wn.add_pattern("pat", [1, 1])
    wn.get_link("P").speed_timeseries.pattern_name = "pat"
    wntr.network.write_inpfile(wn, str(path))
    with pytest.raises(ValueError, match="non-unit or patterned"):
        run_nominal(RunSpec(path, "pattern"))
    wn.get_link("P").speed_timeseries.pattern_name = None
    condition = wntr.network.controls.SimTimeCondition(wn, "=", 300)
    action = wntr.network.controls.ControlAction(wn.get_link("P"), "base_speed", 0.8)
    wn.add_control("speed", wntr.network.controls.Control(condition, action))
    wntr.network.write_inpfile(wn, str(path))
    with pytest.raises(ValueError, match="controlled speed"):
        run_nominal(RunSpec(path, "controlled"))


def test_timing_policy(tmp_path):
    path = make_network(tmp_path / "n.inp")
    wn = wntr.network.WaterNetworkModel(str(path))
    wn.options.time.hydraulic_timestep = 600
    wn.options.time.report_timestep = 300
    wn.options.time.report_start = 300
    wntr.network.write_inpfile(wn, str(path))
    b = run_nominal(RunSpec(path, "timing"))
    assert b.plant_truth.time_seconds.tolist() == [0, 300, 600, 900, 1200]
    assert b.manifest["timesteps"]["hydraulic_timestep"] == 600
    assert b.manifest["effective_timing"]["hydraulic_timestep"] == 300
    wn.options.time.duration = 1000
    wntr.network.write_inpfile(wn, str(path))
    with pytest.raises(ValueError, match="duration must lie"):
        run_nominal(RunSpec(path, "duration"))
    wn.options.time.duration = 1200
    wn.options.time.report_start = 100
    wntr.network.write_inpfile(wn, str(path))
    with pytest.raises(ValueError, match="report start must lie"):
        run_nominal(RunSpec(path, "report-start"))


@pytest.mark.parametrize("mutation", ["truncate", "gap", "columns", "nan"])
def test_incomplete_bundles_are_rejected(tmp_path, mutation):
    b = run_nominal(RunSpec(make_network(tmp_path / "n.inp"), "run"))
    if mutation in ("truncate", "gap"):
        for frame in (b.observed, b.plant_truth, b.controller_view):
            frame.drop(index=frame.index[-1 if mutation == "truncate" else 1], inplace=True)
    elif mutation == "columns":
        for frame in (b.observed, b.plant_truth, b.controller_view):
            frame.drop(columns="J", inplace=True)
        b.manifest["column_units"].pop("J")
        b.manifest["expected_columns"].remove("J")
        b.manifest["required_channels"].remove("J")
    else:
        b.plant_truth["J"] = np.nan
    with pytest.raises(ValueError):
        b.validate()


def test_independent_runs_are_deterministic(tmp_path):
    spec = RunSpec(make_network(tmp_path / "n.inp"), "same", 123)
    a, b = run_nominal(spec), run_nominal(spec)
    assert a.manifest == b.manifest
    for name in ("observed", "plant_truth", "controller_view"):
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name))
    write_bundle(a, tmp_path / "a")
    write_bundle(b, tmp_path / "b")
    for name in ("observed.csv", "plant_truth.csv", "controller_view.csv", "events.json", "manifest.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_five_templates_and_valve_transitions(tmp_path):
    wn = wntr.network.WaterNetworkModel()
    wn.add_reservoir("R", base_head=20)
    wn.add_junction("A", base_demand=0.001)
    wn.add_junction("B", base_demand=0.001)
    wn.add_junction("C", base_demand=0.001)
    wn.add_tank("T", elevation=3, init_level=20, min_level=0, max_level=50, diameter=20)
    wn.add_curve("curve", "HEAD", [(0, 40), (0.02, 20), (0.04, 0)])
    wn.add_pump("P", "R", "A", pump_type="HEAD", pump_parameter="curve")
    wn.add_pipe("pipe", "A", "B", length=100, diameter=0.3, roughness=100)
    wn.add_valve("V", "B", "C", diameter=0.2, valve_type="TCV", initial_setting=2)
    wn.add_pipe("out", "C", "T", length=100, diameter=0.3, roughness=100)
    wn.add_pipe("bypass", "B", "C", length=300, diameter=0.15, roughness=100)
    valve = wn.get_link("V")
    for name, sec, attr, value in (("close", 300, "status", wntr.network.LinkStatus.Closed),
                                    ("open", 600, "status", wntr.network.LinkStatus.Open),
                                    ("active", 900, "setting", 2.0)):
        condition = wntr.network.controls.SimTimeCondition(wn, "=", sec)
        action = wntr.network.controls.ControlAction(valve, attr, value)
        wn.add_control(name, wntr.network.controls.Control(condition, action))
    wn.options.time.duration = 1200
    wn.options.time.hydraulic_timestep = 300
    wn.options.time.report_timestep = 300
    path = tmp_path / "all.inp"
    wntr.network.write_inpfile(wn, str(path))
    bundle = run_nominal(RunSpec(path, "all"))
    f = bundle.plant_truth
    assert {x["family"] for x in bundle.manifest["template_coverage"] if x["status"] == "compiled"} == {"junction_continuity", "pipe_energy", "tank_volume", "pump_head", "valve_dispatch"}
    assert f.set_index("time_seconds").loc[[0, 300, 600, 900, 1200], "s_V"].tolist() == [2, 0, 1, 2, 2]
    for i, row in f.iterrows():
        assert abs(junction_continuity_residual(inflows=[row.f_P], outflows=[row.f_pipe], delivered_demand=row.demand_A, emitter_withdrawals=[row.leak_A, row.emitter_A])) < 1e-7
        assert abs(pipe_energy_residual(flow=row.f_pipe, length=100, diameter=0.3, roughness=100, head_start=row.A, head_end=row.B)) < 1e-3
        pump = pump_dispatch(flow=row.f_P, status="OPEN", speed=row.speed_P, curve_points=[(0, 40), (0.02, 20), (0.04, 0)], head_start=row.head_R, head_end=row.A)
        assert abs(pump.residual) < 1e-3
        dispatch = valve_dispatch(valve_type="TCV", flow=row.f_V, status={0: "CLOSED", 1: "OPEN", 2: "ACTIVE"}[row.s_V], setting=row.setting_V, diameter=0.2, head_start=row.B, head_end=row.C)
        assert dispatch.residual is not None and abs(dispatch.residual) < 1e-4
        if i:
            prev = f.iloc[i - 1]
            residual = tank_volume_residual(level_start=prev["T"], level_end=row["T"], net_inflow=prev.f_out - prev.leak_T, dt_s=row.time_seconds - prev.time_seconds, diameter=20)
            assert abs(residual) < 1e-4
