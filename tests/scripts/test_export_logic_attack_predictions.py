"""Focused, mocked tests for the logic-attack prediction exporter."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import export_logic_attack_predictions as logic_export
from scripts import sweep_topology_tolerances as sweep


def _write_scenario(directory, values, *, attack_active=None, attack_class=None):
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "time_seconds": list(range(0, 300 * len(values), 300)),
        "value": list(values),
    }).to_csv(directory / "observed.csv", index=False)
    labels = {"attack_active": attack_active or [0] * len(values)}
    if attack_class is not None:
        labels["attack_class"] = attack_class
    pd.DataFrame(labels).to_csv(directory / "labels.csv", index=False)


def _hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(root).rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def fake_detectors(monkeypatch):
    monkeypatch.setattr(sweep.export, "load_network", lambda path: None)
    monkeypatch.setattr(sweep.export, "_gfsm_components", lambda data: [])
    monkeypatch.setattr(sweep.export, "_resolve_gfsm_columns", lambda **kw: ({}, {}))

    class Detector:
        def __init__(self, network, config):
            self.config = config
            self.unknown_policy = config["unknown_policy"]

        def fit(self, frames):
            return self

        def to_artifact(self):
            return {"tolerances": {"m": 1.0}, "formulas": {}, "coverage": {}}

    def stl_predict(detector, frame):
        values = frame["value"].to_numpy()
        return (
            (values >= 5).astype(int),
            (values >= 5).astype(float) * 2,
            np.zeros(len(frame), dtype=int),
        )

    class Gfsm:
        def __init__(self, **kwargs):
            pass

        def fit(self, frames):
            return self

        def predict(self, frame):
            flags = (frame["value"].to_numpy() >= 20).astype(int)
            return type("Result", (), {"flags": flags})()

    monkeypatch.setattr(sweep.export, "TopologyStlDetector", Detector)
    monkeypatch.setattr(sweep.export, "_stl_predict", stl_predict)
    monkeypatch.setattr(sweep.export, "GfsmAnomalyDetector", Gfsm)


def test_ltown_external_nominal_labels_immutability(tmp_path, monkeypatch, fake_detectors):
    logic_root = tmp_path / "logic"
    topology_dir = logic_root / "ltown"
    topology_dir.mkdir(parents=True)
    pdd_root = tmp_path / "pdd" / "ltown"
    pdd_root.mkdir(parents=True)
    nominal_dir = pdd_root / "nominal_two_day"
    inp = pdd_root / "topology.inp"
    inp.write_text("NETWORK")

    nominal_values = [0] * 8 + [10, 10]
    _write_scenario(nominal_dir, nominal_values)
    _write_scenario(
        topology_dir / "guard_open_PUMP_1", [0, 0, 0, 30, 30, 0],
        attack_active=[0, 0, 0, 1, 1, 0],
        attack_class=["normal", "normal", "normal",
                      "control_guard_violation", "control_guard_violation", "normal"],
    )
    (topology_dir / ".guard_open_PUMP_1.staging").mkdir()
    write_partial = topology_dir / "half_written"
    write_partial.mkdir()
    pd.DataFrame({"time_seconds": [0]}).to_csv(
        write_partial / "observed.csv", index=False)
    plan_path = topology_dir / "plan.json"
    plan_path.write_text(json.dumps({
        "nominal": {"path": str(nominal_dir), "kind": "existing_real_pdd_nominal"},
        "scenarios": [{"name": "guard_open_PUMP_1"}],
    }))

    def fake_surrogate(*, out_root, inp_path):
        assert Path(inp_path) == inp
        gfsm_dir = Path(out_root) / "ltown" / "generated" / "ltown" / "gfsm"
        gfsm_dir.mkdir(parents=True, exist_ok=True)
        (gfsm_dir / "ltown.gfsm.json").write_text("{}")
        return gfsm_dir, gfsm_dir.parent.parent

    monkeypatch.setattr(sweep.export, "_build_ltown_surrogate", fake_surrogate)

    before = {**_hashes(logic_root), **_hashes(pdd_root)}
    result = logic_export.export_topology(
        topology="ltown",
        logic_root=logic_root,
        data_root=tmp_path / "data",
        out_root=tmp_path / "out",
        ltown_nominal_dir=nominal_dir,
        ltown_inp=inp,
    )
    after = {**_hashes(logic_root), **_hashes(pdd_root)}
    assert before == after

    predictions = pd.read_csv(
        tmp_path / "out" / "ltown" / "detector_predictions.csv")
    assert list(predictions.columns) == logic_export.PREDICTION_COLUMNS
    assert set(predictions["scenario"]) == {"guard_open_PUMP_1", "nominal_two_day"}
    assert "half_written" not in set(predictions["scenario"])
    assert all(not name.startswith(".") for name in predictions["scenario"])

    nominal_rows = predictions[predictions["scenario"] == "nominal_two_day"]
    assert len(nominal_rows) == 2
    assert set(nominal_rows["attack_class"]) == {"normal"}
    attack_rows = predictions[predictions["scenario"] == "guard_open_PUMP_1"]
    assert set(attack_rows.loc[attack_rows.y_true == 1, "attack_class"]) == {
        "control_guard_violation"}

    expected = np.logical_or(
        predictions["y_pred_gfsm"].to_numpy(),
        predictions["y_pred_stl"].to_numpy()).astype(int)
    assert np.array_equal(predictions["y_pred"].to_numpy(), expected)
    assert np.allclose(
        predictions["combined_score"].to_numpy(),
        np.maximum(predictions["stl_score"].to_numpy(),
                   predictions["y_pred_gfsm"].to_numpy(float)))

    metrics = json.loads(
        (tmp_path / "out" / "ltown" / "metrics.json").read_text())
    assert metrics["config"]["percentile"] == 99.0
    assert metrics["config"]["additive_scale"] == 1.0
    assert metrics["config"]["tuned_on_attacks"] is False
    assert metrics["stl"]["tolerances"] == {"m": 1.0}
    assert metrics["nominal_false_alarms"]["held_out"] is True
    assert metrics["nominal_false_alarms"]["rows"] == 2
    assert metrics["nominal_false_alarms"]["gfsm"] == 0
    assert metrics["nominal_false_alarms"]["stl"] == 2
    assert metrics["plan"]["nominal_path_matches"] is True
    assert str(inp) in metrics["inputs"]
    assert str(plan_path) in metrics["inputs"]
    assert str(nominal_dir / "observed.csv") in metrics["inputs"]
    assert {item["name"] for item in metrics["excluded"]} == {
        ".guard_open_PUMP_1.staging", "half_written"}
    assert result["arms"]["gfsm"]["mcc"] != result["arms"]["stl"]["mcc"]


def test_arm_metrics_keep_gfsm_and_stl_scores_distinct():
    frame = pd.DataFrame({
        "scenario": ["nominal", "nominal", "attack", "attack"],
        "y_true": [0, 0, 1, 1],
        "y_pred_gfsm": [0, 0, 0, 1],
        "y_pred_phi": [0, 0, 0, 0],
        "phi_score": [0.0, 0.0, 0.0, 0.0],
        "y_pred_stl": [0, 1, 0, 1],
        "y_pred": [0, 1, 0, 1],
        "stl_score": [0.0, 3.0, 0.0, 2.0],
        "combined_score": [0.0, 3.0, 0.0, 2.0],
        "uncalibrated_count": [0, 0, 1, 0],
    })
    result = logic_export.metrics(frame, frame["scenario"] == "nominal")
    assert result["rows"] == 4 and result["attack_rows"] == 2
    assert result["unknown_rows"] == 1 and result["unknown_residuals"] == 1
    assert result["gfsm"]["mcc"] != result["stl"]["mcc"]
    assert result["gfsm"]["auprc"] != result["stl"]["auprc"]
    assert result["gfsm"]["nominal_false_alarms"] == 0
    assert result["stl"]["nominal_false_alarms"] == 1
    assert result["combined"]["nominal_false_alarms"] == 1
    assert result["combined"]["mcc"] == pytest.approx(
        sweep.export.detection_metrics(
            frame["y_true"], frame["y_pred"], frame["combined_score"])["mcc"])


def test_observation_aware_metrics_preserve_injection_labels():
    nominal = pd.DataFrame({"time_seconds": [0, 300, 600], "status": [1, 1, 1]})
    attacked = nominal.copy()
    attacked.loc[1, "status"] = 0
    visible = logic_export.observable_attack_mask(attacked, nominal)
    frame = pd.DataFrame({
        "y_true": [1, 0, 0],
        "y_true_observation_aware": np.logical_or([1, 0, 0], visible).astype(int),
        "y_pred_gfsm": [0, 1, 0], "y_pred_stl": [0, 0, 0],
        "y_pred_phi": [0, 0, 0], "phi_score": [0., 0., 0.],
        "y_pred": [0, 1, 0], "stl_score": [0., 0., 0.],
        "combined_score": [0., 1., 0.], "uncalibrated_count": [0, 0, 0],
    })
    mask = pd.Series([False, False, False])
    primary = logic_export.metrics(frame, mask)
    secondary = logic_export.metrics(frame, mask, "y_true_observation_aware")
    assert frame.y_true.tolist() == [1, 0, 0]
    assert primary["gfsm"]["tp"] == 0 and primary["gfsm"]["fp"] == 1
    assert secondary["gfsm"]["tp"] == 1 and secondary["gfsm"]["fp"] == 0


def test_observation_comparison_rejects_unmatched_times():
    nominal = pd.DataFrame({"time_seconds": [0, 300], "status": [1, 1]})
    attacked = pd.DataFrame({"time_seconds": [600], "status": [0]})
    with pytest.raises(logic_export.LogicAttackExportError, match="times"):
        logic_export.observable_attack_mask(attacked, nominal)
