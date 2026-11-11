import numpy as np
import pytest

from monitor.fusion import compute_fused_score, fuse_intersection, fuse_or


def test_fuse_or_truth_table():
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 0, 1])
    assert fuse_or([a, b]).tolist() == [0, 1, 1, 1]


def test_fuse_or_single():
    a = np.array([0, 1, 0])
    assert fuse_or([a]).tolist() == [0, 1, 0]


def test_alarm_union_truth_table():
    logical_alarm = np.array([0, 0, 1, 1])
    physical_alarm = np.array([0, 1, 0, 1])
    assert fuse_or([logical_alarm, physical_alarm]).tolist() == [0, 1, 1, 1]


def test_alarm_union_three_detectors_truth_table():
    gfsm = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    inv = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    stl = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    expected = [0, 1, 1, 1, 1, 1, 1, 1]
    assert fuse_or([gfsm, inv, stl]).tolist() == expected


def test_compute_fused_score_ranking_preserved():
    fused = np.array([0, 1, 1, 1])
    stl_scores = np.array([0.0, 1.2, 3.5, 2.1])
    scores = compute_fused_score(fused, stl_scores)
    assert scores.tolist() == [0.0, 2.2, 4.5, 3.1]
    assert scores[2] > scores[3] > scores[1] > scores[0]


def test_compute_fused_score_alarm_guaranteed_positive():
    fused = np.array([0, 1, 0])
    stl_scores = np.array([0.0, 0.0, 0.0])
    scores = compute_fused_score(fused, stl_scores)
    assert scores.tolist() == [0.0, 1.0, 0.0]
    assert (scores[fused == 1] > 0).all()


def test_compute_fused_score_none_stl_scores():
    fused = np.array([0, 1, 0])
    scores = compute_fused_score(fused, None)
    assert scores.tolist() == [0.0, 1.0, 0.0]


def test_compute_fused_score_empty():
    empty = np.array([], dtype=int)
    scores = compute_fused_score(empty, np.array([]))
    assert len(scores) == 0


def test_compute_fused_score_threshold_and_smoothed_non_alarm():
    flags = np.array([0, 1, 0, 1, 0, 1])
    stl = np.array([5.0, 0.0, 0.25, 2.0, 0.75, 0.5])
    scores = compute_fused_score(flags, stl)
    np.testing.assert_array_equal(scores >= 1.0, flags)
    assert scores[0] == np.nextafter(1.0, 0.0)
    assert scores[0] < scores[1]
    assert scores[2] < scores[4] < scores[0]
    assert scores[1] < scores[5] < scores[3]
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("flags", [1, [[0, 1]], [0, 2], [-1, 1], [0, np.nan], [np.inf]])
def test_compute_fused_score_rejects_invalid_flags(flags):
    with pytest.raises(ValueError):
        compute_fused_score(flags)


@pytest.mark.parametrize("scores", [0.0, [[0.0, 1.0]], [0.0], [0.0, 1.0, 2.0],
                                    [0.0, -1.0], [0.0, np.nan],
                                    [0.0, np.inf], [0.0, -np.inf]])
def test_compute_fused_score_rejects_invalid_stl_scores(scores):
    with pytest.raises(ValueError):
        compute_fused_score([0, 1], scores)


def test_score_consistency_with_auprc():
    pytest.importorskip("sklearn")
    from stl.metrics import detection_metrics

    y_true = np.array([0, 1, 1, 0])
    fused = np.array([0, 1, 1, 0])
    stl_scores = np.array([0.0, 0.0, 2.0, 0.0])
    fused_scores = compute_fused_score(fused, stl_scores)
    metrics_fused = detection_metrics(y_true, fused, fused_scores)
    metrics_stl_only = detection_metrics(y_true, fused, stl_scores)
    assert metrics_fused["aucpr"] == 1.0
    assert metrics_stl_only["aucpr"] < 1.0


def test_run_topology_default_fusion_and_score_persistence(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    from types import SimpleNamespace
    import pandas as pd
    from monitor.driver import run_topology
    from monitor.protocol import StepFlags

    scenario = SimpleNamespace(
        name="test_scen",
        frame=pd.DataFrame({"x": [1, 2, 3]}),
        labels=np.array([0, 1, 1]),
    )
    dataset = SimpleNamespace(
        calibration_frames=[],
        eval_scenarios=[scenario],
    )

    class DummyStl:
        def fit(self, _frames):
            return self

        def predict(self, _frame):
            return StepFlags(
                flags=np.array([0, 0, 1]),
                scores=np.array([0.0, 0.0, 2.5]),
            )

    class DummyGfsm:
        def fit(self, _frames):
            return self

        def predict(self, _frame):
            return StepFlags(flags=np.array([0, 1, 0]))

    class DummyInvariants:
        def fit(self, _frames):
            return self

        def predict(self, _frame):
            return StepFlags(flags=np.array([0, 0, 0]))

    monkeypatch.setattr("monitor.driver.load_topology", lambda *a, **k: dataset)
    monkeypatch.setattr(
        "monitor.driver.resolve_fb_to_col_from_paths",
        lambda *a, **k: ({}, [], {}),
    )
    monkeypatch.setattr("monitor.driver.StlAnomalyDetector", lambda *a, **k: DummyStl())
    monkeypatch.setattr("monitor.driver.GfsmAnomalyDetector", lambda *a, **k: DummyGfsm())
    monkeypatch.setattr(
        "monitor.driver.InvariantsAnomalyDetector",
        lambda *a, **k: DummyInvariants(),
    )

    out_default = tmp_path / "out_default"
    manifest_default = run_topology(
        topology="ctown",
        data_root=tmp_path,
        out_dir=out_default,
    )
    assert manifest_default["fusion"] == "or"
    pred_default = pd.read_csv(out_default / "predictions.csv")
    assert "anomaly_score" in pred_default.columns
    assert pred_default["anomaly_score"].tolist() == [0.0, 1.0, 3.5]
    assert pred_default["y_pred"].tolist() == [0, 1, 1]

    out_ablation = tmp_path / "out_ablation"
    manifest_ablation = run_topology(
        topology="ctown",
        data_root=tmp_path,
        out_dir=out_ablation,
        fusion="intersection",
    )
    assert manifest_ablation["fusion"] == "intersection"
    pred_ablation = pd.read_csv(out_ablation / "predictions.csv")
    assert "anomaly_score" in pred_ablation.columns
    assert pred_ablation["y_pred"].tolist() == [0, 0, 0]
