"""Focused tests for threshold-based logic-attack planning and metadata."""

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_logic_attacks.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_logic_attacks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


logic = load_script()


def make_frame(values, step=300):
    return pd.DataFrame({
        "time_seconds": list(range(0, len(values) * step, step)),
        "T1": values,
        "T2": [6.0] * len(values),
    })


def make_manifest():
    return {
        "controls": [
            {"id": "control 1", "link": "PU1", "sensor": "T1", "relation": "gt",
             "threshold": 8.0, "value": 0, "owner": "PLC"},
            {"id": "control 2", "link": "PU1", "sensor": "T1", "relation": "lt",
             "threshold": 5.0, "value": 1, "owner": "PLC"},
            {"id": "control 3", "link": "V2", "sensor": "T2", "relation": "lt",
             "threshold": 0.5, "value": 1, "owner": "PLC"},
        ],
        "expected_columns": ["time_seconds", "T1", "T2", "speed_PU1"],
    }


def spec():
    return logic.TopologySpec(
        "anytown", 1800, 300, "reuse", Path("/nonexistent"))


def plan(frame=None, manifest=None, **kwargs):
    options = {"period_scans": 1, "sensor_window_scans": 2, "dos_window_scans": 2}
    options.update(kwargs)
    return logic.build_plan(
        "anytown", frame if frame is not None else make_frame([6, 7, 8.5, 7, 4.5, 3, 6]),
        manifest if manifest is not None else make_manifest(), spec(), **options)


def test_windows_use_only_pump_guard_transitions():
    manifest = make_manifest()
    windows = logic.control_windows(
        make_frame([6, 7, 8.5, 7, 4.5, 3, 6]), manifest["controls"],
        step=300, duration=1800, pumps=logic.pump_links(manifest))
    assert [(w["link"], w["direction"], w["time_seconds"], w["forced_status"])
            for w in windows] == [
        ("PU1", "above_high", 600, "OPEN"),
        ("PU1", "below_low", 1200, "CLOSED"),
    ]
    assert all(w["link"] != "V2" for w in windows)


def test_initial_true_guard_is_eligible_before_later_crossings():
    manifest = make_manifest()
    windows = logic.control_windows(
        make_frame([9, 8.5, 8.2, 6, 4.0, 3, 6]), manifest["controls"],
        step=300, duration=1800, pumps=logic.pump_links(manifest))
    assert [(w["direction"], w["time_seconds"]) for w in windows] == [
        ("above_high", 0), ("below_low", 1200)]


def test_plan_derives_classes_and_magnitudes_from_thresholds():
    item = plan()
    by_name = {scenario["name"]: scenario for scenario in item["scenarios"]}

    guard_open = by_name["guard_open_PU1"]
    assert guard_open["attack_class"] == "control_guard_violation"
    assert guard_open["events"] == [{
        "kind": "actuator", "target": "PU1", "value": "OPEN",
        "start": 600, "end": 900, "attack_class": "control_guard_violation"}]
    assert guard_open["selection"]["window"]["threshold"] == 8.0

    guard_closed = by_name["guard_closed_PU1"]
    assert guard_closed["events"][0]["value"] == "CLOSED"
    assert guard_closed["events"][0]["start"] == 1200

    assert by_name["sensor_offset_T1"]["events"][0]["value"] == pytest.approx(1.5)
    assert by_name["adaptive_T1"]["events"][0]["value"] == pytest.approx(0.03)
    assert by_name["sensor_replay_T1"]["events"][0]["period"] == 1
    assert by_name["actuator_feedback_spoof_PU1"]["events"][0]["kind"] == "actuator_feedback"
    assert by_name["actuator_feedback_spoof_PU1"]["events"][0]["scope"] == "scada"
    assert by_name["coordinated_PU1"]["attack_class"] == "coordinated"
    assert [event["attack_class"] for event in by_name["coordinated_PU1"]["events"]] == [
        "coordinated", "coordinated"]

    assert item["selection_policy"]["detector_results_consulted"] is False
    assert item["inherited_leak_labels"] is False
    assert item["attack_taxonomy"]["normal"]
    assert set(item["pump_links"]) == {"PU1"}


def test_replay_falls_back_to_a_feasible_period():
    item = plan(period_scans=100)
    by_name = {scenario["name"]: scenario for scenario in item["scenarios"]}
    event = by_name["sensor_replay_T1"]["events"][0]
    assert event["start"] == 1200
    assert event["period"] == 3
    assert event["start"] - event["period"] * 300 >= 0
    assert "period 3 scans" in by_name["sensor_replay_T1"]["selection"]["magnitude"]


def test_plan_without_crossing_or_pump_fails_explicitly():
    flat = make_frame([6, 6, 6, 6, 6, 6, 6])
    with pytest.raises(logic.LogicAttackError, match="no true pump guard state"):
        plan(frame=flat)
    manifest = make_manifest()
    manifest["controls"] = [manifest["controls"][2]]
    with pytest.raises(logic.LogicAttackError, match="no true pump guard state"):
        plan(manifest=manifest)


def test_write_plan_records_nominal_and_no_inherited_labels(tmp_path):
    item = plan()
    logic.write_plan(tmp_path, "anytown", item, {
        "kind": "reuse", "path": "/nominal", "duration": 1800, "scan_step": 300,
        "plant_truth_sha256": "a", "manifest_sha256": "b"})
    payload = json.loads((tmp_path / "anytown" / "plan.json").read_text())
    assert payload["nominal"]["scan_step"] == 300
    assert payload["selection_policy"]["detector_results_consulted"] is False
    assert payload["inherited_leak_labels"] is False


def test_execute_scenarios_writes_attack_class_and_no_leak_labels(tmp_path):
    item = plan()
    outputs = logic.execute_scenarios(
        "anytown", spec(), item, out_root=tmp_path, inp_path=tmp_path / "topology.inp",
        names=["guard_open_PU1"],
        runner=lambda run_spec, events: object(),
        writer=lambda bundle, path: None,
    )
    assert len(outputs) == 1
    payload = json.loads((outputs[0] / "scenario.json").read_text())
    assert payload["attack_class"] == "control_guard_violation"
    assert payload["detector_results_consulted"] is False
    assert payload["labels"]["battledim_leak_labels_used"] is False
    assert payload["labels"]["inherited_leak_labels"] is False
    assert payload["events"][0]["attack_class"] == "control_guard_violation"
