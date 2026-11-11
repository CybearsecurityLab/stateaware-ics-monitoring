"""Split isolation / determinism tests for dataio.splits."""

from __future__ import annotations

import pandas as pd
import pytest

from dataio.model import DataIoError
from dataio.splits import (
    CalibrationSplit,
    split_calibration_frames,
    split_dataset,
)


def _frame(n: int, offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({
        "p1": [0] * n,
        "t1": [1.0 + offset + i * 0.01 for i in range(n)],
    })


def _scenarios(n: int) -> list[tuple[str, pd.DataFrame]]:
    return [(f"sc{i:02d}", _frame(10, offset=i)) for i in range(n)]


# ---------------------------------------------------------------- anytown


def test_anytown_split_is_whole_scenario():
    frames = _scenarios(5)
    split = split_calibration_frames("anytown", frames, seed=42, fraction=0.8)
    assert isinstance(split, CalibrationSplit)
    assert split.strategy == "scenario_split"
    train_names = split.manifest["train"]["scenarios"]
    val_names = split.manifest["validation"]["scenarios"]
    assert len(train_names) == 4 and len(val_names) == 1
    # no scenario is split across the two sides
    assert not set(train_names) & set(val_names)
    assert set(train_names) | set(val_names) == {f"sc{i:02d}" for i in range(5)}
    # every scenario lands wholly on one side (row counts add up)
    assert (split.train_rows + split.validation_rows) == 50
    assert split.manifest["isolation"]["disjoint"] is True
    assert split.manifest["isolation"]["shared_scenarios"] == []


def test_anytown_split_is_deterministic_for_a_seed():
    frames = _scenarios(10)
    a = split_calibration_frames("anytown", frames, seed=42)
    b = split_calibration_frames("anytown", frames, seed=42)
    assert a.manifest["train"]["scenarios"] == b.manifest["train"]["scenarios"]
    assert (a.manifest["validation"]["scenarios"]
            == b.manifest["validation"]["scenarios"])
    assert a.manifest["scenarios"] == b.manifest["scenarios"]


def test_anytown_split_ignores_input_order():
    frames = _scenarios(10)
    a = split_calibration_frames("anytown", frames, seed=42)
    b = split_calibration_frames("anytown", list(reversed(frames)), seed=42)
    assert set(a.manifest["train"]["scenarios"]) == set(
        b.manifest["train"]["scenarios"])
    assert set(a.manifest["validation"]["scenarios"]) == set(
        b.manifest["validation"]["scenarios"])


def test_anytown_seed_changes_the_partition():
    frames = _scenarios(10)
    partitions = {
        tuple(split_calibration_frames(
            "anytown", frames, seed=s).manifest["validation"]["scenarios"])
        for s in range(1, 21)
    }
    assert len(partitions) > 1


def test_anytown_manifest_records_exact_split():
    frames = _scenarios(4)
    split = split_calibration_frames("anytown", frames, seed=42, fraction=0.75)
    man = split.manifest
    assert man["schema"] == "dataio/splits/v1"
    assert man["topology"] == "anytown"
    assert man["strategy"] == "scenario_split"
    assert man["seed"] == 42 and man["fraction"] == 0.75
    assert len(man["scenarios"]) == 4
    for entry in man["scenarios"]:
        assert set(entry) == {"scenario", "rows", "digest", "partition"}
        assert entry["partition"] in ("train", "validation")
        assert len(entry["digest"]) == 64
    # digests are content-derived: the same frame gives the same digest
    assert man["scenarios"][0]["digest"] == man["scenarios"][0]["digest"]


def test_anytown_single_scenario_is_rejected():
    with pytest.raises(DataIoError, match="empty side"):
        split_calibration_frames("anytown", [("only", _frame(10))])


# ------------------------------------------------------------------ ctown


def test_ctown_split_is_chronological():
    df = _frame(100)
    split = split_calibration_frames("ctown", [("cal", df)], fraction=0.8)
    assert split.strategy == "chronological_split"
    assert split.train_rows == 80
    assert split.validation_rows == 20
    pd.testing.assert_frame_equal(split.train_frames[0], df.iloc[:80])
    pd.testing.assert_frame_equal(split.validation_frames[0], df.iloc[80:])
    chrono = split.manifest["chronological"]
    assert chrono["total_rows"] == 100
    assert chrono["cut_row"] == 80


def test_ctown_split_cuts_across_frame_boundaries():
    df1, df2 = _frame(30), _frame(70, offset=1.0)
    split = split_calibration_frames(
        "ctown", [("a", df1), ("b", df2)], fraction=0.8)
    assert split.train_rows == 80
    assert split.validation_rows == 20
    # no row is duplicated or dropped
    rejoined = pd.concat(split.train_frames + split.validation_frames,
                         ignore_index=True)
    pd.testing.assert_frame_equal(rejoined, pd.concat([df1, df2],
                                                      ignore_index=True))


def test_ctown_split_manifest_exposes_cut():
    df = _frame(50)
    split = split_calibration_frames("ctown", [("cal", df)], fraction=0.6)
    chrono = split.manifest["chronological"]
    assert chrono["cut_row"] == 30
    assert chrono["frames"][0]["train_rows"] == 30
    assert chrono["frames"][0]["validation_rows"] == 20


# ------------------------------------------------------- evaluation safety


def test_evaluation_scenarios_are_untouched():
    frames = _scenarios(4)
    before = [(name, df.copy()) for name, df in frames]
    split = split_calibration_frames(
        "anytown", frames, seed=42, evaluation_scenarios=["eval_a", "eval_b"])
    ev = split.manifest["evaluation"]
    assert ev["scenarios"] == ["eval_a", "eval_b"]
    assert ev["count"] == 2
    assert ev["untouched"] is True
    assert ev["included_in_split"] is False
    # the calibration frames handed in are not mutated by the split
    for (name, df), (name2, before_df) in zip(frames, before):
        assert name == name2
        pd.testing.assert_frame_equal(df, before_df)


def test_evaluation_scenario_inside_calibration_is_rejected():
    frames = _scenarios(3)
    with pytest.raises(DataIoError, match="also present in the calibration"):
        split_calibration_frames(
            "anytown", frames, evaluation_scenarios=["sc00"])


def test_duplicate_scenario_identity_is_rejected():
    frames = [("dup", _frame(5)), ("dup", _frame(5))]
    with pytest.raises(DataIoError, match="duplicate calibration scenario"):
        split_calibration_frames("anytown", frames)


def test_unsupported_topology_is_rejected():
    with pytest.raises(DataIoError, match="unsupported topology"):
        split_calibration_frames("wadi", _scenarios(3))


def test_bad_fraction_is_rejected():
    with pytest.raises(DataIoError, match="split fraction"):
        split_calibration_frames("anytown", _scenarios(3), fraction=1.0)


def test_split_dataset_uses_dataset_calibration_frames():
    class _Scen:
        def __init__(self, name):
            self.name = name

    class _Ds:
        topology = "anytown"
        calibration_frames = _scenarios(4)
        eval_scenarios = [_Scen("eval_x")]

    split = split_dataset(_Ds(), seed=42)
    assert split.manifest["evaluation"]["scenarios"] == ["eval_x"]
    assert split.train_rows + split.validation_rows == 40
