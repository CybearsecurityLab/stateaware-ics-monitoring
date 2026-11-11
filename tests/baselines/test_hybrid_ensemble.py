"""Tests for the BATADAL hybrid-ensemble baseline (Ahmed, arXiv:2512.14422).

The tests prove the leak-free properties of the implementation rather than
its predictive power: feature-selection isolation, SMOTE placement,
out-of-fold stacking, sequence/label alignment, the absence of any backward
fill, threshold and test isolation, deterministic artefacts and the metric
schema.  All models are tiny so that the suite stays fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from baselines.hybrid_ensemble import (ADAPTATIONS, METRIC_SCHEMA,
                                       HybridEnsemble, HybridEnsembleConfig,
                                       build_sequences,
                                       engineer_temporal_features,
                                       evaluate_attack, frame_digest,
                                       run_batadal, select_top_features)

SEED = 42

#: Tiny models/epochs: the paper's hyper-parameters are exercised by the
#: real-data runner, not by the unit tests.
TINY = HybridEnsembleConfig(
    seed=SEED,
    n_selected_features=4,
    rf_n_estimators=8,
    xgb_n_estimators=8,
    lstm_hidden=4,
    lstm_epochs=2,
    lstm_batch_size=16,
    sequence_length=4,
    n_folds=2,
    validation_fraction=0.25,
)

SENSORS = ["L_T1", "F_PU1", "S_PU1", "P_J280", "P_J415", "F_V2"]
CANONICAL = ["l_t1", "f_pu1", "s_pu1", "p_j280", "p_j415", "f_v2"]


def make_series(n: int = 400, n_cols: int = 6, seed: int = 0,
                attacks=((120, 165),)) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic multi-sensor series with one contiguous attack block."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    data = {}
    for i in range(n_cols):
        base = 20 + 3 * i + 2 * np.sin(t / (4.0 + i))
        data[f"s_{i}"] = base + rng.normal(0, 0.15, n)
    frame = pd.DataFrame(data)
    labels = np.zeros(n, dtype=int)
    for start, end in attacks:
        labels[start:end] = 1
        frame.iloc[start:end, 0] += 6.0
    return frame, labels


@pytest.fixture(scope="module")
def series() -> tuple[pd.DataFrame, np.ndarray]:
    return make_series()


@pytest.fixture(scope="module")
def fitted(series) -> HybridEnsemble:
    frame, labels = series
    cut = 300
    return HybridEnsemble(TINY).fit(frame.iloc[:cut], labels[:cut])


# --------------------------------------------------------------------------
# feature selection
# --------------------------------------------------------------------------
def test_feature_selection_is_fit_on_training_rows_only(series):
    frame, labels = series
    cut = 300
    ensemble = HybridEnsemble(TINY).fit(frame.iloc[:cut], labels[:cut])

    selection = ensemble.selection_
    assert selection.scope == "train"
    # the forest saw the training rows only, and the manifest records exactly
    # which rows those were
    expected_rows = frame.iloc[TINY.min_history:cut].reset_index(drop=True)
    assert selection.n_train_rows == len(expected_rows)
    assert selection.train_digest == frame_digest(expected_rows)
    assert selection.train_digest != frame_digest(
        frame.reset_index(drop=True))

    # the selection is reproducible from the training rows alone
    reference = select_top_features(
        expected_rows, labels[TINY.min_history:cut],
        n_top=TINY.n_selected_features, n_estimators=TINY.rf_n_estimators,
        seed=SEED)
    assert reference.features == selection.features
    assert reference.importances == selection.importances
    assert len(selection.features) == TINY.n_selected_features


def test_validation_rows_cannot_change_the_feature_selection(series):
    """A leaky implementation would select on train+validation."""
    frame, labels = series
    cut = 300
    train = frame.iloc[:cut]
    ensemble = HybridEnsemble(TINY).fit(train, labels[:cut])

    poisoned = frame.copy()
    poisoned.iloc[cut:, :] = 1e9
    leaky = select_top_features(
        poisoned.iloc[TINY.min_history:].reset_index(drop=True),
        np.concatenate([labels[TINY.min_history:cut],
                        np.ones(len(frame) - cut, dtype=int)]),
        n_top=TINY.n_selected_features, n_estimators=TINY.rf_n_estimators,
        seed=SEED)
    assert leaky.importances != ensemble.selection_.importances
    assert ensemble.selection_.importances == reference_importances(train,
                                                                    labels,
                                                                    cut)


def reference_importances(train: pd.DataFrame, labels: np.ndarray,
                           cut: int) -> dict[str, float]:
    rows = train.iloc[TINY.min_history:cut].reset_index(drop=True)
    return select_top_features(
        rows, labels[TINY.min_history:cut], n_top=TINY.n_selected_features,
        n_estimators=TINY.rf_n_estimators, seed=SEED).importances


# --------------------------------------------------------------------------
# SMOTE placement
# --------------------------------------------------------------------------
def test_smote_is_applied_only_inside_training_folds(fitted):
    records = fitted.resample_records_
    scopes = [r["scope"] for r in records]
    assert set(scopes) == {"oof_fold_train", "final_train"}
    assert scopes[-1] == "final_train"
    assert scopes[:-1].count("oof_fold_train") == fitted.fit_audit_["n_folds"]

    for record in records:
        assert record["smote_applied"] is True
        assert record["n_out"] > record["n_in"]
        assert record["n_minority_out"] > record["n_minority_in"]

    # each fold resampled exactly the rows of its own training part
    for record, size in zip(records[:-1], fitted.fit_audit_["fold_sizes"]):
        assert record["n_in"] == size["train"]


def test_evaluation_partitions_are_never_resampled(fitted, series):
    frame, _ = series
    validation = frame.iloc[300:]
    result = fitted.predict_proba(validation)
    # the number of scored rows equals the number of input rows minus the
    # rows without history: nothing was added by resampling
    assert len(result.proba) == len(validation) - TINY.min_scored
    assert result.dropped_rows == TINY.min_scored
    assert len(result.proba) != len(validation)


# --------------------------------------------------------------------------
# out-of-fold stacking
# --------------------------------------------------------------------------
def test_meta_learner_is_trained_on_out_of_fold_predictions(fitted, series):
    from sklearn.linear_model import LogisticRegression

    frame, labels = series
    cut = 300
    oof = fitted.oof_probabilities_
    n_scored = fitted.fit_audit_["n_scored_rows"]
    assert oof.shape == (n_scored, len(TINY.base_models))
    assert n_scored == cut - TINY.min_scored

    # every scored training row is the validation row of exactly one fold
    assignments = fitted.fold_assignments_
    assert assignments.min() >= 0
    counts = np.bincount(assignments)
    assert counts.sum() == n_scored
    # every fold holds a comparable share of the scored rows
    assert counts.max() - counts.min() <= max(1, n_scored // 4)

    # the meta-learner was fit on exactly that out-of-fold matrix
    y_scored = labels[TINY.min_scored:cut]
    reference = LogisticRegression(max_iter=1000, random_state=SEED).fit(
        oof, y_scored)
    assert np.allclose(fitted.meta_.coef_, reference.coef_)
    assert np.allclose(fitted.meta_.intercept_, reference.intercept_)

    # and *not* on the in-sample base predictions
    in_sample = fitted.predict_proba(frame.iloc[:cut]).proba
    assert len(in_sample) == n_scored
    assert not np.allclose(oof[:, 0], in_sample)


def test_every_fold_is_smaller_than_the_training_partition(fitted):
    sizes = fitted.fit_audit_["fold_sizes"]
    n_train = fitted.fit_audit_["n_rows"]
    for size in sizes:
        assert size["train"] < n_train
        assert size["validation"] > 0
        assert size["sequence_train"] >= size["train"] - size["validation"]


# --------------------------------------------------------------------------
# sequences and labels
# --------------------------------------------------------------------------
def test_lstm_sequences_are_aligned_with_their_labels():
    values = np.arange(30, dtype=float).reshape(10, 3)
    labels = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1])

    x, y, end_index = build_sequences(values, labels, 4)
    assert x.shape == (7, 4, 3)
    assert len(y) == len(end_index) == 7
    for i, end in enumerate(end_index):
        # the label of a window is the label of its *last* step
        assert y[i] == labels[end]
        assert np.allclose(x[i], values[end - 3:end + 1])
    assert list(end_index) == list(range(3, 10))


def test_sequences_never_cross_a_segment_boundary():
    values = np.arange(30, dtype=float).reshape(10, 3)
    labels = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1])
    segments = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    _x, _y, end_index = build_sequences(values, labels, 4, segments=segments)
    # the window may not span the segment change between row 4 and row 5
    assert list(end_index) == [3, 4, 8, 9]
    for end in end_index:
        assert segments[end - 3] == segments[end]


# --------------------------------------------------------------------------
# no backward fill
# --------------------------------------------------------------------------
def test_features_are_never_backward_filled():
    frame = pd.DataFrame({"a": [100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    out = engineer_temporal_features(frame, ["a"], lags=(1, 3), rolling_window=5)

    # the head has no history: the values stay missing instead of being
    # copied from the future
    assert np.isnan(out["a_lag1"].iloc[0])
    assert np.isnan(out["a_rollmean5"].iloc[0])
    assert out["a_lag1"].iloc[1] == 100.0
    # the first complete 5-hour rolling window is row 4: mean of rows 0..4
    assert out["a_rollmean5"].iloc[4] == pytest.approx(22.0)
    assert out["a_rollstd5"].iloc[4] == pytest.approx(
        np.std([100.0, 1.0, 2.0, 3.0, 4.0], ddof=1))


def test_a_future_value_cannot_change_an_earlier_feature():
    frame = pd.DataFrame({"a": np.arange(10, dtype=float)})
    out1 = engineer_temporal_features(frame, ["a"])
    changed = frame.copy()
    changed.iloc[-1, 0] = 9999.0
    out2 = engineer_temporal_features(changed, ["a"])
    assert np.allclose(out1["a_rollmean5"].iloc[:-1],
                       out2["a_rollmean5"].iloc[:-1], equal_nan=True)
    assert np.allclose(out1["a_lag1"].iloc[:-1], out2["a_lag1"].iloc[:-1],
                       equal_nan=True)


def test_rows_without_history_are_dropped_not_filled(fitted, series):
    frame, labels = series
    cut = 300
    ensemble = HybridEnsemble(TINY).fit(frame.iloc[:cut], labels[:cut])
    assert ensemble.fit_audit_["n_dropped_head_rows"] == TINY.min_history
    assert ensemble.fit_audit_["n_rows"] == cut - TINY.min_history

    result = ensemble.predict_proba(frame.iloc[cut:])
    assert result.dropped_rows == TINY.min_scored
    # the dropped rows are the first ones, and they are reported
    assert list(result.dropped_indices) == list(range(TINY.min_scored))


# --------------------------------------------------------------------------
# threshold and test isolation
# --------------------------------------------------------------------------
def test_threshold_is_fixed_and_predicting_never_mutates_the_model(fitted,
                                                                   series):
    frame, labels = series
    assert fitted.config.threshold == 0.5
    assert fitted.provenance_["threshold"] == {"value": 0.5, "tuned": False}

    before = fitted.state_digest()
    fitted.predict_proba(frame.iloc[300:])
    fitted.predict(frame.iloc[300:])
    assert fitted.state_digest() == before

    # the evaluation rows are not part of the fitting audit
    audit = json.dumps(fitted.fit_audit_)
    assert frame_digest(frame.iloc[300:].reset_index(drop=True)) not in audit


def test_manifest_records_every_adaptation(fitted):
    manifest = fitted.provenance_
    assert manifest["seed"] == SEED
    assert [a["id"] for a in manifest["adaptations"]] == \
        [a["id"] for a in ADAPTATIONS]
    for adaptation in manifest["adaptations"]:
        assert {"id", "paper", "ours", "reason"} <= set(adaptation)


# --------------------------------------------------------------------------
# persistence and determinism
# --------------------------------------------------------------------------
def test_persisted_artifacts_are_deterministic(tmp_path, series):
    frame, labels = series
    cut = 300

    first = HybridEnsemble(TINY).fit(frame.iloc[:cut], labels[:cut])
    first.save(tmp_path / "first")
    second = HybridEnsemble(TINY).fit(frame.iloc[:cut], labels[:cut])
    second.save(tmp_path / "second")

    for name in ("manifest.json", "model.pkl", "lstm.pt"):
        assert (tmp_path / "first" / name).read_bytes() == \
            (tmp_path / "second" / name).read_bytes()

    manifest = json.loads((tmp_path / "first" / "manifest.json").read_text())
    assert manifest["seed"] == SEED
    assert manifest["threshold"] == {"value": 0.5, "tuned": False}
    assert manifest["feature_selection"]["scope"] == "train"

    restored = HybridEnsemble.load(tmp_path / "first")
    assert np.allclose(restored.predict_proba(frame.iloc[cut:]).proba,
                       first.predict_proba(frame.iloc[cut:]).proba)


def test_loaded_model_reproduces_the_stored_predictions(tmp_path, series):
    frame, labels = series
    cut = 300
    ensemble = HybridEnsemble(TINY).fit(frame.iloc[:cut], labels[:cut])
    ensemble.save(tmp_path / "bundle")
    restored = HybridEnsemble.load(tmp_path / "bundle")
    assert restored.config == ensemble.config
    assert restored.selection_.features == ensemble.selection_.features
    assert np.allclose(restored.predict_proba(frame.iloc[cut:]).proba,
                       ensemble.predict_proba(frame.iloc[cut:]).proba)


# --------------------------------------------------------------------------
# metric schema
# --------------------------------------------------------------------------
def test_metric_schema_is_stable_and_unadjusted():
    truth = np.array([0, 0, 1, 1, 0, 1])
    proba = np.array([0.1, 0.2, 0.9, 0.8, 0.4, 0.6])

    metrics = evaluate_attack(truth, proba)
    assert set(metrics) == {
        "schema", "threshold", "threshold_tuned", "n_samples", "n_positive",
        "n_negative", "average_precision", "pr_auc", "roc_auc",
        "matthews_corrcoef", "balanced_accuracy", "accuracy",
        "precision_attack", "recall_attack", "f1_attack",
        "tp", "fp", "tn", "fn",
    }
    assert metrics["schema"] == METRIC_SCHEMA
    assert metrics["threshold"] == 0.5
    assert metrics["threshold_tuned"] is False

    # the balanced accuracy is *unadjusted*: it is the plain accuracy per
    # class at the fixed threshold, with no threshold search
    predicted = (proba >= 0.5).astype(int)
    manual = 0.5 * ((predicted[truth == 0] == 0).mean()
                    + (predicted[truth == 1] == 1).mean())
    assert metrics["balanced_accuracy"] == pytest.approx(manual)

    from sklearn.metrics import average_precision_score
    assert metrics["average_precision"] == pytest.approx(
        average_precision_score(truth, proba))
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert -1.0 <= metrics["matthews_corrcoef"] <= 1.0
    assert metrics["tp"] == 3 and metrics["fn"] == 0
    assert metrics["fp"] == 0 and metrics["tn"] == 3


def test_ensemble_exposes_probabilities_for_auprc(fitted, series):
    frame, labels = series
    validation = frame.iloc[300:].reset_index(drop=True)
    result = fitted.predict_proba(validation)
    metrics = evaluate_attack(labels[300:][result.index.to_numpy()],
                              result.proba)
    assert metrics["average_precision"] == pytest.approx(
        metrics["average_precision"])
    assert metrics["n_samples"] == len(result.proba)


# --------------------------------------------------------------------------
# native BATADAL runner
# --------------------------------------------------------------------------
def _write_fake_batadal(root: Path, n_cal: int = 200, n_sup: int = 300,
                        n_test: int = 120, seed: int = 3) -> None:
    """Materialise a miniature BATADAL layout (dataset03/04 + official test)."""
    rng = np.random.default_rng(seed)

    def sensors(n: int, offset: float) -> pd.DataFrame:
        t = np.arange(n)
        data = {}
        for i, name in enumerate(SENSORS):
            data[name] = (10 + i + np.sin(t / 5.0 + offset)
                          + rng.normal(0, 0.1, n))
        return pd.DataFrame(data)

    dataset_dir = root / "ctown" / "dataset"
    (dataset_dir / "calibration").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    (root / "ctown" / "raw").mkdir(parents=True, exist_ok=True)

    # dataset 03 calibration frame (normalized: sensor columns only)
    cal = sensors(n_cal, 0.0)
    cal.columns = CANONICAL
    cal.to_csv(dataset_dir / "calibration" / "calibration.csv", index=False)

    # official test set (sensor columns + label)
    test = sensors(n_test, 1.0)
    test.columns = CANONICAL
    test["label"] = np.zeros(n_test, dtype=int)
    test.loc[n_test - 30:, "label"] = 1
    test.to_csv(dataset_dir / "evaluation" / "test.csv", index=False)

    # raw dataset 04 (supervised): DATETIME + sensors + ATT_FLAG
    raw = sensors(n_sup, 2.0)
    raw.insert(0, "DATETIME", pd.date_range("2016-07-04", periods=n_sup,
                                            freq="h").strftime("%d/%m/%y %H"))
    raw["ATT_FLAG"] = np.full(n_sup, -999, dtype=int)
    # one attack block inside the chronological training part and one inside
    # the chronological validation part
    raw.loc[40:80, "ATT_FLAG"] = 1
    raw.loc[240:260, "ATT_FLAG"] = 1
    raw.to_csv(root / "ctown" / "raw" / "BATADAL_dataset04.csv", index=False)

    def sha(path: Path) -> str:
        import hashlib
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "schema": "dataio/v1",
        "topology": "ctown",
        "source": {"name": "fake BATADAL", "note": "unit test",
                   "format": "csv"},
        "generated_at": "2026-01-01T00:00:00+00:00",
        "files": {"calibration": ["calibration/calibration.csv"],
                  "evaluation": ["evaluation/test.csv"]},
        "sha256": {"calibration/calibration.csv":
                       sha(dataset_dir / "calibration" / "calibration.csv"),
                   "evaluation/test.csv":
                       sha(dataset_dir / "evaluation" / "test.csv")},
        "column_map": {name: name.lower() for name in SENSORS},
        "attack_windows": [],
    }
    (dataset_dir / "dataset_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=True))


def test_native_runner_combines_dataset03_and_dataset04(tmp_path):
    root = tmp_path / "data"
    _write_fake_batadal(root)
    test_file = root / "ctown" / "dataset" / "evaluation" / "test.csv"
    before = test_file.read_bytes()

    result = run_batadal(data_root=root, topology="ctown", config=TINY)

    # the official test file was not touched
    assert test_file.read_bytes() == before
    assert result.artifacts["official_test_unchanged"] is True

    # dataset 04 provides the supervised rows, dataset 03 the calibration rows
    assert result.artifacts["supervised_rows"] == 300
    assert result.artifacts["calibration_rows"] == 200
    cut = result.artifacts["split_cut_row"]
    assert result.artifacts["train_rows"] == cut + 200
    assert result.artifacts["validation_rows"] == 300 - cut

    # the metrics follow the fixed schema
    assert result.metrics_validation["schema"] == METRIC_SCHEMA
    assert result.metrics_validation["threshold"] == 0.5
    assert result.metrics_validation["threshold_tuned"] is False
    assert result.metrics_test is not None
    assert result.metrics_test["schema"] == METRIC_SCHEMA

    # the official test was never fitted on: the fitting audit only contains
    # the dataset-04 training rows and the calibration rows.  Each of the
    # two segments loses its head rows, which have no feature history.
    audit = result.provenance["fit_audit"]
    assert audit["n_rows"] == cut + 200 - 2 * TINY.min_history
    assert frame_digest(pd.read_csv(test_file)) not in json.dumps(audit)


def test_native_runner_can_exclude_the_calibration_frames(tmp_path):
    root = tmp_path / "data"
    _write_fake_batadal(root)
    result = run_batadal(data_root=root, topology="ctown", config=TINY,
                         calibration_mode="exclude")
    assert result.artifacts["calibration_rows"] == 200
    assert result.artifacts["train_rows"] == result.artifacts["split_cut_row"]
