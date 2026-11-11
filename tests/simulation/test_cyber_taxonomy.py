"""Focused tests for the attack-class taxonomy and observation-only feedback."""

import pandas as pd
import pytest

from simulation import RunSpec
from simulation.cyber import (
    ATTACK_CLASS_TAXONOMY,
    derive_attack_class,
    run_cyber,
)
from tests.simulation.test_cyber import controlled, event


def test_legacy_events_derive_explicit_classes(tmp_path):
    spec = RunSpec(controlled(tmp_path / 'n.inp'), 'taxonomy')
    b = run_cyber(spec, [event('sensor', 'T', 1, mode='offset', scope='scada')])
    assert b.labels.attack_class.tolist() == [
        'normal', 'sensor_offset', 'sensor_offset', 'normal', 'normal']
    assert b.manifest['attack_class_taxonomy'] == ATTACK_CLASS_TAXONOMY
    assert derive_attack_class({'kind': 'sensor', 'mode': 'freeze'}) == 'sensor_freeze'
    assert derive_attack_class({'kind': 'sensor', 'mode': 'replay'}) == 'sensor_replay'
    assert derive_attack_class({'kind': 'actuator'}) == 'actuator_dos'
    assert derive_attack_class({'kind': 'actuator_feedback'}) == 'actuator_feedback_spoof'
    assert derive_attack_class({'kind': 'threshold'}) == 'control_guard_violation'


def test_mixed_and_explicit_composite_classes(tmp_path):
    spec = RunSpec(controlled(tmp_path / 'n.inp'), 'composite')
    attacks = [
        {**event('sensor', 'T', 1, mode='offset', scope='scada'), 'attack_class': 'adaptive'},
        event('actuator', 'P', 'OPEN'),
    ]
    b = run_cyber(spec, attacks)
    assert b.events[0]['attack_class'] == 'adaptive'
    assert b.events[1]['attack_class'] == 'actuator_dos'
    assert b.labels.attack_class.iloc[1:3].tolist() == ['coordinated', 'coordinated']

    tagged = [{**event('sensor', 'T', 1, mode='offset', scope='scada'),
               'attack_class': 'coordinated'},
              {**event('actuator', 'P', 'OPEN'), 'attack_class': 'coordinated'}]
    assert run_cyber(spec, tagged).labels.attack_class.iloc[1:3].tolist() == [
        'coordinated', 'coordinated']


def test_invalid_or_normal_event_classes_are_rejected(tmp_path):
    spec = RunSpec(controlled(tmp_path / 'n.inp'), 'invalid')
    with pytest.raises(ValueError, match='attack_class'):
        run_cyber(spec, [{**event('sensor', 'T', 1, mode='offset', scope='scada'),
                          'attack_class': 'made_up'}])
    with pytest.raises(ValueError, match='attack_class'):
        run_cyber(spec, [{**event('sensor', 'T', 1, mode='offset', scope='scada'),
                          'attack_class': 'normal'}])


def test_observation_only_feedback_never_changes_plant_truth(tmp_path):
    spec = RunSpec(controlled(tmp_path / 'n.inp'), 'feedback')
    base = run_cyber(spec)
    feedback = {
        'kind': 'actuator_feedback', 'target': 'P', 'value': 'OPEN',
        'scope': 'scada', 'start': 300, 'end': 900,
    }
    b = run_cyber(spec, [feedback])
    pd.testing.assert_frame_equal(b.plant_truth, base.plant_truth)
    pd.testing.assert_frame_equal(b.hydraulic_truth, base.hydraulic_truth)
    assert b.observed['s_P'].iloc[1:3].tolist() == [1, 1]
    assert b.labels.attack_class.iloc[1:3].tolist() == [
        'actuator_feedback_spoof', 'actuator_feedback_spoof']
    assert b.labels.delivery_or_command_modified.iloc[1:3].all()
    assert not b.labels.physical_impact.any()
    assert b.events[0]['applied'][0] == {
        'recipient': 'scada', 'channel': 'status', 'column': 's_P', 'value': 1.0}


def test_physical_actuator_command_differs_from_feedback_spoof(tmp_path):
    spec = RunSpec(controlled(tmp_path / 'n.inp'), 'contrast')
    base = run_cyber(spec)
    physical = run_cyber(spec, [event('actuator', 'P', 'OPEN')])
    feedback = run_cyber(spec, [{
        'kind': 'actuator_feedback', 'target': 'P', 'value': 'OPEN',
        'scope': 'scada', 'start': 300, 'end': 900,
    }])
    assert physical.labels.physical_impact.any()
    assert physical.labels.attack_class.iloc[1] == 'actuator_dos'
    assert not feedback.labels.physical_impact.any()
    assert feedback.labels.attack_class.iloc[1] == 'actuator_feedback_spoof'
    assert feedback.plant_truth['s_P'].iloc[2] == base.plant_truth['s_P'].iloc[2] == 0


def test_feedback_controller_spoof_is_rejected_to_preserve_plant_truth(tmp_path):
    spec = RunSpec(controlled(tmp_path / 'n.inp'), 'feedback-controller')
    feedback = {
        'kind': 'actuator_feedback', 'target': 'P', 'value': 'OPEN',
        'scope': 'controller', 'controller': 'A', 'start': 300, 'end': 900,
    }
    with pytest.raises(ValueError, match='scope=scada'):
        run_cyber(spec, [feedback])
    with pytest.raises(ValueError, match='scope=scada'):
        run_cyber(spec, [{**feedback, 'scope': 'both'}])


def test_observable_label_covers_actuator_visibility_after_injection(tmp_path):
    spec = RunSpec(controlled(tmp_path / 'n.inp'), 'observable')
    b = run_cyber(spec, [event('actuator', 'P', 'OPEN', end=600)])
    assert b.labels.attack_active.tolist() == [False, True, False, False, False]
    assert not b.labels.observable_attack.iloc[:2].any()
    assert b.labels.observable_attack.iloc[2]
    evaluation = b.labels.attack_active | b.labels.observable_attack
    assert evaluation.iloc[:3].tolist() == [False, True, True]


def test_old_labels_without_attack_class_still_validate(tmp_path):
    spec = RunSpec(controlled(tmp_path / 'n.inp'), 'legacy-labels')
    b = run_cyber(spec)
    b.labels = b.labels.drop(columns='attack_class')
    b.validate()
