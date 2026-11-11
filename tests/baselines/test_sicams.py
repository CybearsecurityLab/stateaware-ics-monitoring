from __future__ import annotations

import inspect
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.linalg import solve_triangular

from baselines.sicams import (
    METRIC_KEYS,
    SCHEME_HOUR_OF_DAY,
    SCHEME_LTOWN_AREA_A,
    SCHEME_LTOWN_AREA_B,
    SCHEME_LTOWN_AREA_C,
    FeatureDimensionError,
    InsufficientClusterSamplesError,
    NotFittedError,
    SicamsBaseline,
    SicamsConfig,
    SicamsTuningGrid,
    SingularCovarianceError,
    UnknownClusterError,
    artifact_json,
    build_artifact,
    causal_moving_average,
    hysteresis_state,
    pointwise_metrics,
    select_config_on_validation,
    temporal_cluster_labels,
    write_artifact,
)


def _normal_train(seed: int = 7) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 3.0],
            [4.0, 2.0],
            [5.0, 5.0],
        ]
    )


def test_hand_computed_whitening_t2_and_f_rescaling():
    train = _normal_train()
    config = SicamsConfig(
        significance=0.05, moving_average_window=1, hysteresis=False
    )
    model = SicamsBaseline(config).fit(train, cluster_ids=np.zeros(len(train)))
    assert model.n_features_ == 2

    mean = train.mean(axis=0)
    covariance = np.cov(train.T, ddof=1)
    np.testing.assert_allclose(model.clusters_[0].mean, mean)
    np.testing.assert_allclose(model.clusters_[0].covariance, covariance)
    np.testing.assert_allclose(
        model.clusters_[0].cholesky, np.linalg.cholesky(covariance)
    )

    x = np.array([[2.5, 1.75]])
    residual = x[0] - mean
    expected_t2 = float(residual @ np.linalg.inv(covariance) @ residual)
    whitened = solve_triangular(
        np.linalg.cholesky(covariance), residual, lower=True
    )
    np.testing.assert_allclose(float(whitened @ whitened), expected_t2)

    result = model.predict(x, cluster_ids=[0])
    np.testing.assert_allclose(result.t2, [expected_t2])
    n, s = train.shape
    expected_tf2 = (n - s) / (s * (n - 1)) * expected_t2
    np.testing.assert_allclose(result.tf2, [expected_tf2])
    np.testing.assert_allclose(
        result.upper_threshold, [stats.f.ppf(0.95, s, n - s)]
    )
    np.testing.assert_allclose(
        result.lower_threshold, [(n - s) / (n - s - 2)]
    )
    np.testing.assert_allclose(result.moving_average, [expected_tf2])


def test_cluster_isolation_uses_each_cluster_moments_and_thresholds():
    rng = np.random.default_rng(3)
    c0 = rng.normal(loc=[0.0, 0.0], scale=[1.0, 1.5], size=(8, 2))
    c1 = rng.normal(loc=[20.0, -8.0], scale=[3.0, 0.7], size=(10, 2))
    train = np.vstack([c0, c1])
    ids = np.array([0] * len(c0) + [1] * len(c1))
    config = SicamsConfig(
        significance=0.05, moving_average_window=1, hysteresis=False
    )
    model = SicamsBaseline(config).fit(train, cluster_ids=ids)

    assert set(model.clusters_) == {0, 1}
    np.testing.assert_allclose(model.clusters_[0].mean, c0.mean(axis=0))
    np.testing.assert_allclose(model.clusters_[1].mean, c1.mean(axis=0))
    assert (
        model.clusters_[0].upper_threshold
        != model.clusters_[1].upper_threshold
    )
    assert (
        model.clusters_[0].lower_threshold
        != model.clusters_[1].lower_threshold
    )

    offsets = np.array([[0.5, -0.25], [1.25, 0.75]])
    for key in (0, 1):
        stat = model.clusters_[key]
        expected = []
        for delta in offsets:
            residual = stat.mean + delta - stat.mean
            expected.append(
                float(residual @ np.linalg.inv(stat.covariance) @ residual)
            )
        result = model.predict(
            stat.mean + offsets, cluster_ids=[key] * len(offsets)
        )
        np.testing.assert_allclose(result.t2, expected)

    with pytest.raises(UnknownClusterError):
        model.predict(offsets, cluster_ids=[2, 2])


def test_cluster_specific_insufficient_samples_are_explicit():
    train = np.vstack([_normal_train(), _normal_train()[:4]])
    ids = np.array([0] * 6 + [1] * 4)
    with pytest.raises(InsufficientClusterSamplesError) as excinfo:
        SicamsBaseline().fit(train, cluster_ids=ids)
    assert excinfo.value.cluster_id == 1
    assert excinfo.value.n_k == 4
    assert excinfo.value.s == 2


def test_singular_covariance_is_reported_not_hidden():
    rng = np.random.default_rng(11)
    x = rng.normal(size=20)
    train = np.column_stack([x, 2.0 * x])
    with pytest.raises(SingularCovarianceError) as excinfo:
        SicamsBaseline().fit(train)
    err = excinfo.value
    assert err.cluster_id == 0
    assert err.s == 2
    assert err.rank == 1
    assert "rank" in str(err)


def test_feature_count_and_fit_state_are_explicit():
    model = SicamsBaseline()
    with pytest.raises(NotFittedError):
        model.predict(_normal_train())
    model.fit(_normal_train())
    with pytest.raises(FeatureDimensionError):
        model.predict(np.zeros((3, 3)))


def test_causal_moving_average_uses_only_past_and_current():
    values = np.arange(1.0, 8.0)
    np.testing.assert_allclose(
        causal_moving_average(values, 3), [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
    )

    rng = np.random.default_rng(21)
    train = rng.normal(size=(20, 2))
    test = rng.normal(size=(12, 2))
    config = SicamsConfig(
        significance=0.1, moving_average_window=4, hysteresis=True
    )
    model = SicamsBaseline(config).fit(train)
    full = model.predict(test)
    prefix = model.predict(test[:6])
    np.testing.assert_allclose(full.moving_average[:6], prefix.moving_average)
    np.testing.assert_array_equal(full.flags[:6], prefix.flags)


def test_hysteresis_uses_strict_upper_entry_and_lower_reset():
    scores = np.array([0.0, 2.0, 1.5, 0.5, 2.0])
    upper = np.ones(5)
    lower = np.full(5, 0.75)
    assert hysteresis_state(scores, upper, lower).tolist() == [0, 1, 1, 0, 1]

    strict = np.array([1.0, 1.1, 0.75, 0.74])
    assert hysteresis_state(strict, np.ones(4), np.full(4, 0.75)).tolist() == [
        0,
        1,
        1,
        0,
    ]


def test_temporal_cluster_mappings_native_and_adapted():
    timestamps = pd.date_range("2019-01-01", periods=120, freq="h")
    hourly = temporal_cluster_labels(timestamps, SCHEME_HOUR_OF_DAY)
    native_a = temporal_cluster_labels(timestamps, SCHEME_LTOWN_AREA_A)
    native_c = temporal_cluster_labels(timestamps, SCHEME_LTOWN_AREA_C)
    native_b = temporal_cluster_labels(timestamps, SCHEME_LTOWN_AREA_B)

    np.testing.assert_array_equal(hourly, native_a)
    assert hourly[:3].tolist() == [0, 1, 2]
    assert sorted(set(hourly)) == list(range(24))
    assert native_c[:5].tolist() == [0, 0, 1, 1, 2]
    assert sorted(set(native_c)) == list(range(12))
    assert set(native_b) == {0}

    rng = np.random.default_rng(5)
    frame = rng.normal(size=(120, 2))
    adapted = SicamsBaseline(
        SicamsConfig(temporal_scheme=SCHEME_HOUR_OF_DAY)
    ).fit(frame, timestamps=timestamps)
    adapted_result = adapted.predict(frame, timestamps=timestamps)
    assert adapted_result.provenance["native_vs_adaptation"][
        "scheme_is_native_ltown"
    ] is False

    native = SicamsBaseline(
        SicamsConfig(temporal_scheme=SCHEME_LTOWN_AREA_C)
    ).fit(frame, timestamps=timestamps)
    native_result = native.predict(frame, timestamps=timestamps)
    assert len(native.clusters_) == 12
    assert native_result.provenance["native_vs_adaptation"][
        "scheme_is_native_ltown"
    ] is True
    assert native_result.provenance["temporal_clusters"][
        "declared_scheme"
    ] == SCHEME_LTOWN_AREA_C


def test_pointwise_metric_schema_hand_computed():
    labels = np.array([0, 0, 1, 1])
    flags = np.array([0, 1, 1, 1])
    scores = np.array([0.1, 0.4, 0.8, 0.9])
    metrics = pointwise_metrics(labels, flags, scores)

    assert set(metrics) == set(METRIC_KEYS)
    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 0
    assert metrics["positive_class"] == "attack"
    assert metrics["balanced_accuracy_adjusted"] is False
    assert metrics["scope"] == "pointwise_binary_detection_only"
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["mcc"] == pytest.approx(2.0 / math.sqrt(12.0))
    assert metrics["attack_average_precision"] == pytest.approx(1.0)
    for forbidden in ("localization", "volume", "pelt", "event"):
        assert all(forbidden not in key.lower() for key in metrics)


def test_validation_selection_uses_validation_only_and_no_test_refit():
    rng = np.random.default_rng(42)
    train = np.vstack(
        [rng.normal(0.0, 1.0, size=(12, 2)), rng.normal(8.0, 1.0, size=(12, 2))]
    )
    train_ids = np.array([0] * 12 + [1] * 12)
    validation = np.vstack(
        [rng.normal(0.0, 1.0, size=(30, 2)), rng.normal(8.0, 1.0, size=(30, 2))]
    )
    validation_ids = np.array([0] * 30 + [1] * 30)
    labels = np.zeros(60, dtype=int)
    for base in (0, 30):
        labels[base + np.array([2, 9, 17, 24])] = 1
        validation[base + np.array([2, 9, 17, 24])] += 5.0

    grid = SicamsTuningGrid(
        significances=(0.01, 0.05, 0.2),
        moving_average_windows=(1, 4),
        selection_metric="mcc",
    )
    selection = select_config_on_validation(
        train,
        validation,
        labels,
        train_clusters=train_ids,
        validation_clusters=validation_ids,
        grid=grid,
    )

    assert len(selection.candidates) == 6
    assert selection.provenance["selection_data"] == "validation_only"
    assert selection.provenance["test_data_used"] is False
    assert selection.provenance["labels_used_for_fit"] is False
    assert selection.config.significance in grid.significances
    assert selection.config.moving_average_window in grid.moving_average_windows

    scores = [c["validation_score"] for c in selection.candidates]
    finite = [s for s in scores if s is not None and not math.isnan(s)]
    best = max(finite)
    chosen = next(c for c in selection.candidates if c["validation_score"] == best)
    assert selection.config.significance == chosen["significance"]
    assert selection.config.moving_average_window == chosen[
        "moving_average_window"
    ]

    assert "labels" not in inspect.signature(SicamsBaseline.fit).parameters
    assert "labels" not in inspect.signature(SicamsBaseline.predict).parameters
    assert not any(
        "test" in name
        for name in inspect.signature(select_config_on_validation).parameters
    )

    model = SicamsBaseline(selection.config).fit(train, cluster_ids=train_ids)
    before = [
        (c.mean.copy(), c.covariance.copy(), c.upper_threshold)
        for c in model.clusters_.values()
    ]
    model.predict(validation, cluster_ids=validation_ids)
    after = [
        (c.mean.copy(), c.covariance.copy(), c.upper_threshold)
        for c in model.clusters_.values()
    ]
    for old, new in zip(before, after):
        np.testing.assert_array_equal(old[0], new[0])
        np.testing.assert_array_equal(old[1], new[1])
        assert old[2] == new[2]

    for key, stat in model.clusters_.items():
        expected_upper = stats.f.ppf(
            1.0 - selection.config.significance, 2, stat.n_k - 2
        )
        assert stat.upper_threshold == pytest.approx(expected_upper)
        assert stat.lower_threshold == pytest.approx(
            (stat.n_k - 2) / (stat.n_k - 4)
        )


def test_deterministic_artifact_predictions_provenance_and_metrics(tmp_path):
    timestamps = pd.date_range("2019-06-01", periods=168, freq="h")
    rng = np.random.default_rng(99)
    train = rng.normal(size=(168, 3))
    test = rng.normal(size=(48, 3))
    labels = np.zeros(48, dtype=int)
    labels[[5, 11, 23, 37]] = 1
    test[[5, 11, 23, 37]] += 4.0

    config = SicamsConfig(
        significance=0.05,
        moving_average_window=6,
        temporal_scheme=SCHEME_LTOWN_AREA_A,
        hysteresis=True,
    )
    model = SicamsBaseline(config).fit(train, timestamps=timestamps)
    result = model.predict(test, timestamps=timestamps[:48])
    other = SicamsBaseline(config).fit(train, timestamps=timestamps).predict(
        test, timestamps=timestamps[:48]
    )

    np.testing.assert_array_equal(result.t2, other.t2)
    np.testing.assert_array_equal(result.tf2, other.tf2)
    np.testing.assert_array_equal(result.flags, other.flags)

    text_a = artifact_json(
        result, labels=labels, dataset={"name": "synthetic", "split": "test"}
    )
    text_b = artifact_json(
        other, labels=labels, dataset={"name": "synthetic", "split": "test"}
    )
    assert text_a == text_b
    payload = json.loads(text_a)
    assert payload["schema"] == "baselines/sicams/v1"
    assert set(payload) == {"schema", "config", "provenance", "predictions", "metrics", "dataset"}
    assert payload["config"] == config.to_dict()
    assert set(payload["predictions"]) == {
        "t2",
        "tf2",
        "moving_average",
        "flags",
        "upper_threshold",
        "lower_threshold",
        "cluster_ids",
    }
    assert set(payload["metrics"]) == set(METRIC_KEYS)
    provenance = payload["provenance"]
    assert provenance["native_vs_adaptation"]["scheme_is_native_ltown"] is True
    assert provenance["missing_original_settings"]
    assert provenance["omitted_native_components"]
    assert provenance["determinism"]["refit_on_evaluation"] is False
    assert provenance["temporal_clusters"]["n_clusters"] == 24

    target = write_artifact(
        tmp_path / "sicams.json", result, labels=labels, dataset={"name": "synthetic"}
    )
    assert target.read_text() == artifact_json(
        result, labels=labels, dataset={"name": "synthetic"}
    ) + "\n"

    built = build_artifact(result, labels=labels)
    assert "dataset" not in built
    assert built["metrics"]["n_samples"] == 48
