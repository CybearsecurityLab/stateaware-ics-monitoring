"""Leak-free calibration of the Φ violation threshold (train/validation split)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from invariants.driver import mine_topology
from invariants.model import Atom, InvariantsError, MinedRule


def _dataset(tmp_path: Path, cal: pd.DataFrame, name: str = "synth") -> Path:
    gfsm_dir = tmp_path / "gfsm"
    gfsm_dir.mkdir(exist_ok=True)
    (gfsm_dir / f"{name}.gfsm.json").write_text(json.dumps({
        "initial": "PLC1.S:0",
        "states": {"PLC1.S:0": ["0"], "PLC1.S:1": ["1"]},
        "transitions": [], "metadata": {"source_file": "x",
            "extraction_date": "", "total_states": 2, "total_transitions": 0},
        "max_states": 100,
    }))
    (gfsm_dir / f"{name}_gfsm_manifest.json").write_text(json.dumps({"x": 1}))
    ds_dir = tmp_path / name / "dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "dataset_manifest.yaml").write_text(
        "files:\n  calibration:\n    - cal.csv\n  evaluation: []\n"
        "column_map: {}\nattack_windows: []\n"
    )
    cal.to_csv(ds_dir / "cal.csv", index=False)
    return gfsm_dir


def _cal_frame(n_clean: int = 90, n_bad: int = 10) -> pd.DataFrame:
    """90 clean rows (t1=1.0) followed by 10 violating rows (t1=5.0)."""
    return pd.DataFrame({
        "p1": [0] * (n_clean + n_bad),
        "t1": [1.0] * n_clean + [5.0] * n_bad,
        "f1": [0.5] * (n_clean + n_bad),
    })


def _fake_miner(recorded: list[pd.DataFrame]):
    """Deterministic miner: t1 must stay in [0, 1.5]."""

    def _mine(slice_df, feature_cols, cfg):
        recorded.append(slice_df.copy())
        if "t1" not in feature_cols:
            return []
        return [MinedRule(
            id="r0", antecedent=[],
            consequent=[Atom(col="t1", op="in", val=[0.0, 1.5])],
            support=1.0, confidence=1.0,
        )]
    return _mine


@pytest.fixture()
def mined(monkeypatch):
    recorded: list[pd.DataFrame] = []
    monkeypatch.setattr("invariants.driver.mine_state", _fake_miner(recorded))
    return recorded


def test_threshold_is_selected_on_validation_not_training(tmp_path: Path,
                                                           mined):
    cal = _cal_frame()
    gfsm_dir = _dataset(tmp_path, cal)
    fb_to_col = {("PLC1", "S"): "p1"}

    train = cal.iloc[:90].reset_index(drop=True)
    val = cal.iloc[90:].reset_index(drop=True)

    out = tmp_path / "invariants"
    manifest = mine_topology(
        topology="synth", data_root=tmp_path, gfsm_dir=gfsm_dir, out_dir=out,
        fb_to_col=fb_to_col, min_observations=10, max_evals=50, seed=42,
        feature_cols=["t1"], fp_budget=0.2,
        training_frames=[train], validation_frames=[val],
    )
    phi = json.loads((out / "synth_phi.json").read_text())

    # The validation rows all violate the rule -> a validation-selected
    # threshold must be > 1.  A threshold calibrated on the training rows
    # (which are all clean) would have been 1.
    assert phi["violation_threshold"] == 2
    cal_prov = manifest["calibration"]
    assert cal_prov["mode"] == "train_validation_split"
    assert cal_prov["threshold_source"] == "validation"
    assert cal_prov["fit_rows"] == 90
    assert cal_prov["threshold_rows"] == 10


def test_legacy_mode_keeps_threshold_on_training(tmp_path: Path, mined):
    """Backwards compatibility: without a split, behaviour is unchanged."""
    cal = _cal_frame()
    gfsm_dir = _dataset(tmp_path, cal)
    out = tmp_path / "invariants"
    manifest = mine_topology(
        topology="synth", data_root=tmp_path, gfsm_dir=gfsm_dir, out_dir=out,
        fb_to_col={("PLC1", "S"): "p1"}, min_observations=10, max_evals=50,
        seed=42, feature_cols=["t1"], fp_budget=0.2,
    )
    phi = json.loads((out / "synth_phi.json").read_text())
    # legacy: fit and threshold on the same 100 rows -> 10% violations <=
    # budget 0.2 -> K = 1 (the historical, leaky behaviour).
    assert phi["violation_threshold"] == 1
    cal_prov = manifest["calibration"]
    assert cal_prov["mode"] == "legacy_full_calibration"
    assert cal_prov["threshold_source"] == "calibration"
    assert cal_prov["fit_rows"] == 100
    assert cal_prov["threshold_rows"] == 100
    assert "split" not in cal_prov


def test_validation_rows_never_reach_the_miner(tmp_path: Path, mined):
    cal = _cal_frame()
    gfsm_dir = _dataset(tmp_path, cal)
    train = cal.iloc[:90].reset_index(drop=True)
    val = cal.iloc[90:].reset_index(drop=True)
    mine_topology(
        topology="synth", data_root=tmp_path, gfsm_dir=gfsm_dir,
        out_dir=tmp_path / "invariants", fb_to_col={("PLC1", "S"): "p1"},
        min_observations=10, max_evals=50, seed=42, feature_cols=["t1"],
        fp_budget=0.2, training_frames=[train], validation_frames=[val],
    )
    assert mined, "the miner was never called"
    seen = pd.concat(mined, ignore_index=True)
    assert set(seen["t1"].unique()) == {1.0}
    assert len(seen) == 90


def test_split_manifest_is_embedded_verbatim(tmp_path: Path, mined):
    from dataio.splits import split_calibration_frames

    cal = _cal_frame()
    gfsm_dir = _dataset(tmp_path, cal)
    frames = [("sc0", cal.iloc[:50]), ("sc1", cal.iloc[50:])]
    split = split_calibration_frames("anytown", frames, seed=42)

    out = tmp_path / "invariants"
    manifest = mine_topology(
        topology="synth", data_root=tmp_path, gfsm_dir=gfsm_dir, out_dir=out,
        fb_to_col={("PLC1", "S"): "p1"}, min_observations=10, max_evals=50,
        seed=42, feature_cols=["t1"], fp_budget=0.2,
        training_frames=split.train_frames,
        validation_frames=split.validation_frames,
        split_manifest=split.manifest,
    )
    assert manifest["calibration"]["split"]["strategy"] == "scenario_split"
    assert manifest["calibration"]["split"] == split.manifest
    phi = json.loads((out / "synth_phi.json").read_text())
    assert phi["calibration"]["split"] == split.manifest


def test_training_and_validation_must_be_given_together(tmp_path: Path,
                                                        mined):
    cal = _cal_frame()
    gfsm_dir = _dataset(tmp_path, cal)
    with pytest.raises(InvariantsError, match="must be provided together"):
        mine_topology(
            topology="synth", data_root=tmp_path, gfsm_dir=gfsm_dir,
            out_dir=tmp_path / "invariants", fb_to_col={("PLC1", "S"): "p1"},
            training_frames=[cal.iloc[:90]],
        )
    with pytest.raises(InvariantsError, match="must be provided together"):
        mine_topology(
            topology="synth", data_root=tmp_path, gfsm_dir=gfsm_dir,
            out_dir=tmp_path / "invariants", fb_to_col={("PLC1", "S"): "p1"},
            validation_frames=[cal.iloc[90:]],
        )


def test_feature_columns_are_derived_from_fit_rows_only(tmp_path: Path,
                                                         mined):
    cal = _cal_frame()
    gfsm_dir = _dataset(tmp_path, cal)
    train = cal.iloc[:90].reset_index(drop=True)
    # the validation frame carries an extra numeric column that must not
    # influence the feature selection
    val = cal.iloc[90:].reset_index(drop=True)
    val["leak_only_in_validation"] = 1.0

    out = tmp_path / "invariants"
    manifest = mine_topology(
        topology="synth", data_root=tmp_path, gfsm_dir=gfsm_dir, out_dir=out,
        fb_to_col={("PLC1", "S"): "p1"}, min_observations=10, max_evals=50,
        seed=42, fp_budget=0.2,
        training_frames=[train], validation_frames=[val],
    )
    feats = manifest["calibration"]["feature_cols"]
    assert "leak_only_in_validation" not in feats
    assert "t1" in feats and "f1" in feats
    # the state column is not a feature by default
    assert "p1" not in feats


def test_explicit_feature_columns_are_used_verbatim(tmp_path: Path, mined):
    """Actuator + sensor features are honoured when supplied identically."""
    cal = _cal_frame()
    gfsm_dir = _dataset(tmp_path, cal)
    train = cal.iloc[:90].reset_index(drop=True)
    val = cal.iloc[90:].reset_index(drop=True)
    out = tmp_path / "invariants"
    manifest = mine_topology(
        topology="synth", data_root=tmp_path, gfsm_dir=gfsm_dir, out_dir=out,
        fb_to_col={("PLC1", "S"): "p1"}, min_observations=10, max_evals=50,
        seed=42, feature_cols=["p1", "t1", "f1"], fp_budget=0.2,
        training_frames=[train], validation_frames=[val],
    )
    assert manifest["calibration"]["feature_cols"] == ["p1", "t1", "f1"]
    phi = json.loads((out / "synth_phi.json").read_text())
    assert phi["calibration"]["feature_cols"] == ["p1", "t1", "f1"]
