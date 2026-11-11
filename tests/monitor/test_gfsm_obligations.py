import json
from pathlib import Path

import pandas as pd

from monitor.gfsm_detector import GfsmAnomalyDetector
from monitor.gfsm_obligations import GfsmObligationDetector


def write_model(tmp_path: Path, *, conflicting=False):
    gfsm_dir = tmp_path / "gfsm"
    gfsm_dir.mkdir(exist_ok=True)
    transitions = [
        {"id": "open", "from_state": "0", "to_state": "1", "condition": "T1 < 2.4"},
        {"id": "close", "from_state": "1", "to_state": "0", "condition": "T1 > 3.9"},
    ]
    if conflicting:
        transitions.append({"id": "other", "from_state": "0", "to_state": "0", "condition": "T1 < 2.4"})
    (gfsm_dir / "test.gfsm.json").write_text(json.dumps({
        "states": {"PLC1.S:0": {}, "PLC1.S:1": {}},
        "transitions": [
            {"from": "PLC1.S:0", "to": "PLC1.S:1", "condition": "T1 < 2.4"},
            {"from": "PLC1.S:1", "to": "PLC1.S:0", "condition": "T1 > 3.9"},
        ],
    }))
    (gfsm_dir / "test_gfsm_manifest.json").write_text(json.dumps({
        "plcs": [{"name": "PLC1", "artifacts": {"fsm_json": "plc1.fsm.json"}}]
    }))
    (gfsm_dir / "plc1.fsm.json").write_text(json.dumps({
        "function_blocks": [{"name": "S", "case_variable": "S", "transitions": transitions}]
    }))
    return gfsm_dir


def detector(tmp_path, *, obligations=True, conflicting=False):
    return GfsmObligationDetector(
        gfsm_dir=write_model(tmp_path, conflicting=conflicting),
        topology="test",
        fb_to_col={("PLC1", "S"): "p1"},
        check_obligations=obligations,
    ).fit([])


def test_timing_uses_previous_input_not_destination_input(tmp_path):
    frame = pd.DataFrame({"p1": [0, 1], "l_t1": [1.0, 99.0]})
    write_model(tmp_path)
    original = GfsmAnomalyDetector(
        gfsm_dir=tmp_path / "gfsm", topology="test", fb_to_col={("PLC1", "S"): "p1"}
    ).fit([])
    assert original.predict(frame).flags.tolist() == [0, 1]
    assert detector(tmp_path, obligations=False).predict(frame).flags.tolist() == [0, 0]


def test_sustained_unrealized_action_is_flagged_with_explanation(tmp_path):
    result = detector(tmp_path).predict_detailed(
        pd.DataFrame({"p1": [0, 0], "l_t1": [1.0, 1.0]})
    )
    assert result.flags.tolist() == [0, 1]
    assert result.reasons == [{
        "type": "action_realization",
        "plc": "PLC1",
        "actuator": "S",
        "case_var": "S",
        "source_state": "0",
        "actual_state": "0",
        "input_row": 0,
        "observation_row": 1,
        "expected_state": "1",
        "guards": ["T1 < 2.4"],
        "reason": "enabled action was not realized in the next observation",
    }]


def test_realized_open_and_close_actions_do_not_flag(tmp_path):
    frame = pd.DataFrame({"p1": [0, 1, 0], "l_t1": [1.0, 9.0, 9.0]})
    assert detector(tmp_path).predict(frame).flags.tolist() == [0, 0, 0]


def test_conflicting_enabled_actions_are_not_silently_resolved(tmp_path):
    result = detector(tmp_path, conflicting=True).predict_detailed(
        pd.DataFrame({"p1": [0, 0], "l_t1": [1.0, 1.0]})
    )
    assert result.flags.tolist() == [0, 1]
    assert result.reasons[0]["reason"] == "conflicting source obligations"
    assert result.reasons[0]["expected_states"] == ["0", "1"]
