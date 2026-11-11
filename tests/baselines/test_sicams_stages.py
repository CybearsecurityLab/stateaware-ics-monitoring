import itertools

import numpy as np
import pandas as pd
import pytest

from baselines.sicams import SicamsConfig
from baselines.sicams_stages import (
    MissingSuppliedInputError,
    SicamsFullPipeline,
    _pelt,
    build_spatial_partition,
    detect_change_points,
    discriminate_anomaly,
    fit_leak_volume_regression,
    laplacian_prelocalization,
)


@pytest.mark.parametrize("min_segment", [1, 2, 3])
def test_pelt_objective_matches_exhaustive_segmentations(min_segment):
    values = np.array([0.1, -0.1, 2.1, 2.0, 1.9, -1.0, -0.9, -1.1])
    penalty = 0.7
    cuts, _, _, actual = _pelt(values, penalty=penalty, min_segment=min_segment)
    objectives = []
    for size in range(len(values)):
        for candidate in itertools.combinations(range(1, len(values)), size):
            boundaries = [0, *candidate, len(values)]
            if any(b - a < min_segment for a, b in zip(boundaries, boundaries[1:])):
                continue
            cost = penalty * len(candidate)
            for a, b in zip(boundaries, boundaries[1:]):
                segment = values[a:b]
                cost += np.sum((segment - segment.mean()) ** 2)
            objectives.append(cost)
    assert actual == pytest.approx(min(objectives))
    assert all(b - a >= min_segment for a, b in zip([0, *cuts], [*cuts, len(values)]))


def test_change_points_classify_up_and_down_and_report_offline_latency():
    values = np.r_[np.zeros(20), np.full(20, 5.0), np.zeros(20)]
    result = detect_change_points(values, penalty=1.0, min_segment=4)
    assert result.raw_change_points == (20, 40)
    assert [point.label for point in result.classified] == ["anomaly_start", "anomaly_end"]
    assert result.online is False
    assert result.latency == "retrospective_full_series"


def test_supplied_topology_partition_and_laplacian_boundary_conditions():
    partition = build_spatial_partition(["a", "b"], supplied={"left": ["a"], "right": ["b"]})
    assert partition.names("left") == ("a",)
    graph = {"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]}
    result = laplacian_prelocalization(graph, {"a": 0., "d": 3.})
    assert result.field == pytest.approx({"a": 0., "b": 1., "c": 2., "d": 3.})


def test_fault_discrimination_requires_all_supplied_ranges():
    missing = discriminate_anomaly([8., 8.], [5., 5.], ["a", "b"])
    assert missing.status == "unavailable"
    ranges = {"a": (-1., 1.), "b": (-1., 1.)}
    fault = discriminate_anomaly([8., 0.], [5., 0.], ["a", "b"], operational_ranges=ranges)
    hydraulic = discriminate_anomaly([8., 8.], [5., 5.], ["a", "b"], operational_ranges=ranges)
    assert fault.status == "sensor_fault"
    assert hydraulic.status == "hydraulic_anomaly"


def test_volume_regression_uses_supplied_training_targets_only():
    feature = np.linspace(1., 5., 20)
    model = fit_leak_volume_regression(feature, 3. + 2. * feature, window=1)
    before = (model.intercept, model.slope)
    assert before == pytest.approx((3., 2.))
    assert model.predict([7., 8.]) == pytest.approx([17., 19.])
    assert (model.intercept, model.slope) == before
    assert model.to_dict()["published_reference_applied"] is False


def test_concentration_rule_is_independent_and_ranges_must_be_complete():
    direct = discriminate_anomaly(
        [0., 0.], [3., 1.], ["a", "b"], concentration_threshold=0.8)
    assert direct.status == "sensor_fault"
    incomplete = discriminate_anomaly(
        [2., 2.], [1., 1.], ["a", "b"], operational_ranges={"a": (-1., 1.)})
    assert incomplete.status == "unavailable"


def test_volume_regression_requires_independent_training_series():
    values = np.linspace(1., 6., 10)
    pipeline = SicamsFullPipeline(SicamsConfig(moving_average_window=1))
    with pytest.raises(MissingSuppliedInputError, match="volume_training_X"):
        pipeline.fit(values, leak_volume_targets=values)
    pipeline = SicamsFullPipeline(SicamsConfig(moving_average_window=1))
    pipeline.fit(values, volume_training_X=values,
                 leak_volume_targets=3. + 2. * values,
                 volume_training_cluster_ids=np.zeros(10, dtype=int),
                 volume_window=1)
    assert pipeline.volume_model_.target_source == "independent_regression_training_pairs"
    tf2 = pipeline.detect(values, cluster_ids=np.zeros(10, dtype=int)).tf2
    slope, intercept = np.polyfit(tf2, 3. + 2. * values, 1)
    assert pipeline.volume_model_.slope == pytest.approx(slope)
    assert pipeline.volume_model_.intercept == pytest.approx(intercept)


def test_pipeline_training_schema_projection_and_causal_prefix():
    rng = np.random.default_rng(4)
    train = pd.DataFrame(rng.normal(size=(60, 2)), columns=["a", "b"])
    train["time_seconds"] = np.arange(60) * 300
    train["leak_truth"] = 0.
    pipeline = SicamsFullPipeline(SicamsConfig(moving_average_window=3))
    pipeline.fit(train)
    assert pipeline.feature_names_ == ["a", "b"]
    test = train[["b", "a"]].copy()
    test["scenario_extra"] = 999.
    actual = pipeline.detect(test)
    expected = pipeline.detect(train)
    assert actual.tf2 == pytest.approx(expected.tf2)
    assert pipeline.verify_online_causality(
        test, prefix=20, cluster_ids=np.zeros(len(test), dtype=int))["online_causal"] is True
    full = pipeline.run_full(test)
    assert full.leak_volume["status"] == "unavailable"
    assert full.prelocalization["status"] == "unavailable"


def test_calibration_interval_does_not_fit_outside_rows():
    rng = np.random.default_rng(9)
    values = rng.normal(size=(60, 2))
    times = pd.date_range("2020-01-01", periods=60, freq="h")
    interval = (times[10], times[49])
    config = SicamsConfig(moving_average_window=1)
    whole = SicamsFullPipeline(config).fit(
        values, timestamps=times, cluster_ids=np.zeros(60, dtype=int),
        calibration_interval=interval)
    subset = SicamsFullPipeline(config).fit(values[10:50], cluster_ids=np.zeros(40, dtype=int))
    test = rng.normal(size=(10, 2))
    assert whole.detect(test, cluster_ids=np.zeros(10, dtype=int)).t2 == pytest.approx(
        subset.detect(test, cluster_ids=np.zeros(10, dtype=int)).t2)
