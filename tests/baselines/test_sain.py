"""Tests for the SAIN-style local (per-CASE-state) invariant baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baselines import sain
from baselines.sain import (
    SAIN_ADAPTATIONS,
    SAIN_CODE_COMMIT,
    SAIN_CODE_URL,
    SAIN_PAPER_URL,
    ComparisonScenario,
    SainLocalDetector,
    compare_local_vs_global,
    local_state_ids,
    mine_local_invariants,
    run_sain_local_comparison,
)
from invariants.driver import mine_topology
from invariants.model import Atom, MinedRule
from monitor.invariants_detector import InvariantsAnomalyDetector
from monitor.protocol import StepFlags
from stl.metrics import detection_metrics

COMPONENTS = [("PLC1", "S"), ("PLC2", "T")]
FB_TO_COL = {("PLC1", "S"): "p1", ("PLC2", "T"): "p2"}


# ------------------------------------------------------------------ helpers


def _bound_miner(recorded: list[pd.DataFrame]):
    """Deterministic stand-in for NiaARM: bound each feature by its range."""

    def _mine(slice_df, feature_cols, cfg):
        recorded.append(slice_df.copy())
        rules = []
        for i, col in enumerate(feature_cols):
            if col not in slice_df.columns:
                continue
            vals = slice_df[col].to_numpy(dtype="float64")
            if len(vals) == 0:
                continue
            rules.append(MinedRule(
                id=f"r{i}", antecedent=[],
                consequent=[Atom(col=col, op="in",
                                val=[float(vals.min()), float(vals.max())])],
                support=1.0, confidence=1.0,
            ))
        return rules
    return _mine


@pytest.fixture()
def miner(monkeypatch):
    recorded: list[pd.DataFrame] = []
    fake = _bound_miner(recorded)
    monkeypatch.setattr("baselines.sain.mine_state", fake)
    monkeypatch.setattr("invariants.driver.mine_state", fake)
    return recorded


def _write_global_inputs(tmp_path: Path, cal: pd.DataFrame,
                         states: dict[str, list[str]]) -> Path:
    gfsm_dir = tmp_path / "gfsm"
    gfsm_dir.mkdir(exist_ok=True)
    (gfsm_dir / "synth.gfsm.json").write_text(json.dumps({
        "initial": next(iter(states)),
        "states": states,
        "transitions": [],
        "metadata": {"source_file": "x", "extraction_date": "",
                      "total_states": len(states), "total_transitions": 0},
        "max_states": 100,
    }))
    (gfsm_dir / "synth_gfsm_manifest.json").write_text(json.dumps({"x": 1}))
    ds_dir = tmp_path / "synth" / "dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "dataset_manifest.yaml").write_text(
        "files:\n  calibration:\n    - cal.csv\n  evaluation: []\n"
        "column_map: {}\nattack_windows: []\n"
    )
    cal.to_csv(ds_dir / "cal.csv", index=False)
    return gfsm_dir


def _three_combo_train(n: int = 20) -> pd.DataFrame:
    """Combos (0,0), (0,1), (1,0) — PLC1.S:0 spans two composite states."""
    a = pd.DataFrame({"p1": [0] * n, "p2": [0] * n, "t1": [5.0] * n})
    b = pd.DataFrame({"p1": [0] * n, "p2": [1] * n, "t1": [6.0] * n})
    c = pd.DataFrame({"p1": [1] * n, "p2": [0] * n, "t1": [6.0] * n})
    return pd.concat([a, b, c], ignore_index=True)


def _local_model(frames_train, frames_val, feature_cols=("t1",), **kw):
    return mine_local_invariants(
        topology="synth", components=COMPONENTS, fb_to_col=FB_TO_COL,
        training_frames=frames_train, validation_frames=frames_val,
        feature_cols=list(feature_cols), min_observations=5, max_evals=50,
        seed=42, fp_budget=0.2, **kw,
    )


# ------------------------------------------------------------------- tests


def test_parallel_component_mining_matches_serial():
    train = _three_combo_train(20)
    val = _three_combo_train(5)
    serial = _local_model([train], [val], n_jobs=1)
    parallel = _local_model([train], [val], n_jobs=2)
    assert parallel["components"] == serial["components"]
    assert parallel["calibration"] == serial["calibration"]
    assert parallel["niaarm"]["n_jobs"] == 2
    assert serial["niaarm"]["n_jobs"] == 1


def test_local_and_global_partitions_differ(tmp_path: Path, miner):
    """The local per-CASE partition is coarser than the composite one.

    Row R (p1=0, p2=0, t1=5.5) is inside the *local* PLC1.S:0 / PLC2.T:0
    ranges (both span t1 in [5, 6]) but outside the *global* composite state
    PLC1.S:0|PLC2.T:0 range (t1 in [5, 5]).
    """
    train = _three_combo_train(20)
    val = _three_combo_train(5)
    states = {
        "PLC1.S:0|PLC2.T:0": ["0", "0"],
        "PLC1.S:0|PLC2.T:1": ["0", "1"],
        "PLC1.S:1|PLC2.T:0": ["1", "0"],
    }
    gfsm_dir = _write_global_inputs(tmp_path, train, states)
    inv_dir = tmp_path / "invariants"
    mine_topology(
        topology="synth", data_root=tmp_path, gfsm_dir=gfsm_dir,
        out_dir=inv_dir, fb_to_col=FB_TO_COL, min_observations=5,
        max_evals=50, seed=42, feature_cols=["t1"], fp_budget=0.2,
        training_frames=[train], validation_frames=[val],
    )
    phi = json.loads((inv_dir / "synth_phi.json").read_text())
    global_det = InvariantsAnomalyDetector(
        invariants_dir=inv_dir, gfsm_dir=gfsm_dir, data_root=tmp_path,
        topology="synth", components=COMPONENTS, fb_to_col=FB_TO_COL,
    ).fit([])

    model = _local_model([train], [val])
    local_det = SainLocalDetector(model)

    # structural difference: 3 composite states vs 2x2 local states
    assert len(phi["states"]) == 3
    assert set(model["components"]) == {"PLC1.S", "PLC2.T"}
    assert set(model["components"]["PLC1.S"]["local_states"]) == {"0", "1"}
    assert set(model["components"]["PLC2.T"]["local_states"]) == {"0", "1"}

    row = pd.DataFrame({"p1": [0], "p2": [0], "t1": [5.5]})
    assert global_det.predict(row).flags[0] == 1
    assert local_det.predict(row).flags[0] == 0


def test_unknown_local_state_is_flagged(miner):
    train = _three_combo_train(20)
    model = _local_model([train], [_three_combo_train(5)])
    det = SainLocalDetector(model)
    row = pd.DataFrame({"p1": [7], "p2": [0], "t1": [5.0]})
    out = det.predict(row)
    assert out.flags[0] == 1
    # only PLC1.S knows about state 7 -> exactly one local detector fired
    assert out.scores[0] == 1.0


def test_per_component_fusion_or_and_score_alignment(miner):
    """Three scenario groups, so the two components have different bounds.

    G1 = (p1=0, p2=0) t1=1.0, G2 = (p1=0, p2=1) t1=2.0, G3 = (p1=1, p2=0)
    t1=3.0.  PLC1.S:0 spans G1+G2 -> t1 in [1, 2]; PLC2.T:0 spans G1+G3 ->
    t1 in [1, 3]; PLC2.T:1 is G2 -> t1 in [2, 2].
    """
    frame = pd.concat([
        pd.DataFrame({"p1": [0] * 20, "p2": [0] * 20, "t1": [1.0] * 20}),
        pd.DataFrame({"p1": [0] * 20, "p2": [1] * 20, "t1": [2.0] * 20}),
        pd.DataFrame({"p1": [1] * 20, "p2": [0] * 20, "t1": [3.0] * 20}),
    ], ignore_index=True)
    model = _local_model([frame], [frame], feature_cols=("t1",))
    det = SainLocalDetector(model)

    eval_frame = pd.DataFrame({
        "p1": [0, 0, 0],
        "p2": [0, 0, 1],
        "t1": [1.0, 2.5, 2.5],
    })
    out = det.predict(eval_frame)
    # OR across the local detectors: flags stay binary
    assert list(out.flags) == [0, 1, 1]
    # scores count the flagging components, aligned row-by-row with flags
    assert list(out.scores) == [0.0, 1.0, 2.0]
    assert np.array_equal(out.flags > 0, out.scores > 0)


def test_local_threshold_is_selected_on_validation(miner):
    """The per-component threshold comes from the validation rows only."""
    train = pd.DataFrame({
        "p1": [0] * 30, "p2": [0] * 30,
        "t1": [1.0] * 30, "t2": [1.0] * 30,
    })
    val = pd.DataFrame({
        "p1": [0] * 10, "p2": [0] * 10,
        "t1": [5.0] * 10, "t2": [1.0] * 10,
    })
    model = _local_model([train], [val], feature_cols=("t1", "t2"))
    comp = model["components"]["PLC1.S"]
    # every validation row violates the mined bound -> the validation-selected
    # threshold must be > 1 (a training-selected one would be 1)
    assert comp["violation_threshold"] == 2
    assert model["calibration"]["threshold_source"] == "validation"
    assert model["calibration"]["fit_rows"] == 30
    assert model["calibration"]["threshold_rows"] == 10


def test_validation_rows_never_reach_the_local_miner(miner):
    train = pd.DataFrame({
        "p1": [0] * 30, "p2": [0] * 30,
        "t1": [1.0] * 30, "t2": [1.0] * 30,
    })
    val = pd.DataFrame({
        "p1": [0] * 10, "p2": [0] * 10,
        "t1": [7.0] * 10, "t2": [1.0] * 10,
    })
    _local_model([train], [val], feature_cols=("t1", "t2"))
    seen = pd.concat(miner, ignore_index=True)
    assert set(seen["t1"].unique()) == {1.0}
    # two components, one local state each, 30 training rows per slice
    assert len(seen) == 60


def test_artifacts_are_deterministic(miner):
    train = _three_combo_train(20)
    val = _three_combo_train(5)
    a = _local_model([train], [val])
    b = _local_model([train], [val])
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(
        b, sort_keys=True, default=str)
    # the artifact is JSON-serializable and round-trips
    payload = json.loads(json.dumps(a, sort_keys=True, default=str))
    det = SainLocalDetector(payload)
    row = pd.DataFrame({"p1": [0], "p2": [0], "t1": [5.5]})
    assert det.predict(row).flags[0] == 0


def test_upstream_reference_is_documented():
    assert SAIN_CODE_URL.startswith("https://github.com/")
    assert "SAIN" in SAIN_CODE_URL
    assert len(SAIN_CODE_COMMIT) == 40
    assert "usenix.org" in SAIN_PAPER_URL
    assert len(SAIN_ADAPTATIONS) >= 3
    doc = sain.__doc__ or ""
    assert "not" in doc and "byte-exact" in doc
    assert SAIN_CODE_COMMIT in doc


def test_feature_parity_between_local_and_global(tmp_path: Path, miner):
    train = _three_combo_train(20)
    val = _three_combo_train(5)
    states = {
        "PLC1.S:0|PLC2.T:0": ["0", "0"],
        "PLC1.S:0|PLC2.T:1": ["0", "1"],
        "PLC1.S:1|PLC2.T:0": ["1", "0"],
    }
    gfsm_dir = _write_global_inputs(tmp_path, train, states)
    inv_dir = tmp_path / "invariants"
    manifest = mine_topology(
        topology="synth", data_root=tmp_path, gfsm_dir=gfsm_dir,
        out_dir=inv_dir, fb_to_col=FB_TO_COL, min_observations=5,
        max_evals=50, seed=42, feature_cols=["t1", "t2"], fp_budget=0.2,
        training_frames=[train], validation_frames=[val],
    )
    model = _local_model([train], [val], feature_cols=("t1", "t2"))
    assert model["feature_cols"] == manifest["calibration"]["feature_cols"]


def test_comparison_entry_point_metrics_schema(miner):
    frame = pd.DataFrame({
        "p1": [0] * 40, "p2": [0] * 40,
        "t1": [0.5] * 40, "t2": [0.5] * 40,
    })
    model = _local_model([frame], [frame], feature_cols=("t1", "t2"))
    local_det = SainLocalDetector(model)

    class _Global:
        name = "invariants"

        def predict(self, f: pd.DataFrame) -> StepFlags:
            flags = np.zeros(len(f), dtype=int)
            flags[1] = 1
            return StepFlags(flags=flags, scores=flags.astype(float))

    eval_frame = pd.DataFrame({
        "p1": [0] * 3, "p2": [0] * 3,
        "t1": [0.5, 9.0, 0.5], "t2": [0.5, 0.5, 0.5],
    })
    labels = np.array([0, 1, 0])
    out = compare_local_vs_global(
        scenarios=[ComparisonScenario("s1", eval_frame, labels)],
        local_detector=local_det, global_detector=_Global(),
        feature_cols=["t1", "t2"],
    )
    expected = set(detection_metrics(labels, labels * 0, labels * 0.0))
    assert set(out["scenarios"][0]["local"]["metrics"]) == expected
    assert set(out["scenarios"][0]["global"]["metrics"]) == expected
    assert set(out["summary"]["local"]) == expected
    assert out["feature_cols"] == ["t1", "t2"]


def test_local_state_ids_match_the_gfsm_encoding():
    frame = pd.DataFrame({"p1": [0, 1, 2], "p2": [0, 1, 0]})
    assert list(local_state_ids(frame, "p1")) == ["0", "1", "2"]
    assert list(local_state_ids(frame, "p2")) == ["0", "1", "0"]


def test_run_sain_local_comparison_end_to_end(tmp_path: Path, miner):
    """The reusable entry point mines both sides and scores them alike."""
    data_root = tmp_path / "data"
    ds_dir = data_root / "anytown" / "dataset"
    ds_dir.mkdir(parents=True)
    cal0 = pd.DataFrame({
        "p1": [0] * 20, "p2": [0] * 20, "t1": [1.0] * 20, "t2": [0.5] * 20,
    })
    cal1 = pd.DataFrame({
        "p1": [0] * 20, "p2": [1] * 20, "t1": [2.0] * 20, "t2": [0.5] * 20,
    })
    cal0.to_csv(ds_dir / "cal0.csv", index=False)
    cal1.to_csv(ds_dir / "cal1.csv", index=False)
    ev = pd.DataFrame({
        "p1": [0, 0, 0], "p2": [0, 0, 1],
        "t1": [1.0, 5.0, 2.0], "t2": [0.5, 0.5, 0.5],
        "label": [0, 1, 0],
    })
    ev.to_csv(ds_dir / "eval.csv", index=False)
    (ds_dir / "dataset_manifest.yaml").write_text(
        "files:\n"
        "  calibration:\n    - cal0.csv\n    - cal1.csv\n"
        "  evaluation:\n    - eval.csv\n"
        "column_map:\n  S: p1\n  T: p2\n"
        "attack_windows: []\n"
    )

    gfsm_dir = data_root / "generated" / "anytown" / "gfsm"
    gfsm_dir.mkdir(parents=True)
    (gfsm_dir / "anytown.gfsm.json").write_text(json.dumps({
        "initial": "PLC1.S:0|PLC2.T:0",
        "states": {
            "PLC1.S:0|PLC2.T:0": ["0", "0"],
            "PLC1.S:0|PLC2.T:1": ["0", "1"],
        },
        "transitions": [],
        "metadata": {"source_file": "x", "extraction_date": "",
                      "total_states": 2, "total_transitions": 0},
        "max_states": 100,
    }))
    (gfsm_dir / "anytown_gfsm_manifest.json").write_text(json.dumps({"x": 1}))

    out = run_sain_local_comparison(
        topology="anytown", data_root=data_root, min_observations=5,
        max_evals=50, fp_budget=0.2, out_dir=tmp_path / "out",
    )
    assert out["schema"] == "baselines/compare/v1"
    assert len(out["scenarios"]) == 1
    expected = set(detection_metrics([0, 1, 0], [0, 0, 0], [0.0, 0.0, 0.0]))
    assert set(out["scenarios"][0]["local"]["metrics"]) == expected
    assert set(out["scenarios"][0]["global"]["metrics"]) == expected
    assert (tmp_path / "out" / "sain_local_vs_global.json").exists()
    # both sides were built on the same feature columns
    assert out["feature_cols"] == ["t1", "t2"]
