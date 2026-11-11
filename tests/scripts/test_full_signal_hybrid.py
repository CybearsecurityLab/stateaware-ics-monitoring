import json

import numpy as np
import pandas as pd

from baselines.hybrid_ensemble import HybridEnsemble, HybridEnsembleConfig
from scripts.run_full_signal_hybrid import observable_frame, run_topology, score


def test_observable_frame_excludes_simulator_truth_and_labels():
    frame = pd.DataFrame({
        "time_seconds": [0, 300],
        "f_sensor": [1., 2.],
        "leak_truth": [0., 1.],
        "demand_truth": [0., 1.],
        "emitter_truth": [0., 0.],
        "attack_active": [0, 1],
        "attack_class": ["normal", "sensor_offset"],
    })
    assert list(observable_frame(frame).columns) == ["f_sensor"]


def test_training_warmup_preserves_every_evaluation_row():
    from types import SimpleNamespace

    class DummyModel:
        config = SimpleNamespace(min_scored=2, threshold=0.5)

        def __init__(self):
            self.frame = None

        def predict_proba(self, frame, segments):
            self.frame = frame.copy()
            return SimpleNamespace(
                index=np.arange(2, len(frame)),
                proba=np.array([0.75, 0.25]),
                dropped_rows=2,
            )

    training = pd.DataFrame({
        "time_seconds": [0, 300, 600],
        "f_sensor": [1., 2., 3.],
    })
    run = SimpleNamespace(
        name="attack",
        frame=pd.DataFrame({"time_seconds": [900, 1200], "f_sensor": [4., 5.]}),
        y_true=np.array([1, 0]),
        attack_class=np.array(["control_guard_violation", "normal"]),
    )
    model = DummyModel()
    predictions, metrics = score(model, run, ("f_sensor",), training)
    assert model.frame["f_sensor"].tolist() == [2., 3., 4., 5.]
    assert predictions.row.tolist() == [0, 1]
    assert predictions.y_true.tolist() == [1, 0]
    assert predictions.time_seconds.tolist() == [900, 1200]
    assert predictions.score.tolist() == [0.75, 0.25]
    assert metrics["nominal_warmup_rows"] == 2
    assert metrics["dropped_rows"] == 0
    assert metrics["row_population"] == {
        "evaluation_rows": 2,
        "scored_evaluation_rows": 2,
        "dropped_evaluation_rows": 0,
        "warmup_rows_prepended": 2,
        "model_input_rows": 4,
    }


def test_reused_model_warms_nominal_test_from_training_rows_only(
        tmp_path, monkeypatch):
    from types import SimpleNamespace

    class DummyModel:
        config = SimpleNamespace(
            min_scored=2,
            threshold=0.5,
            to_dict=lambda: {"min_scored": 2, "threshold": 0.5},
        )
        raw_columns_ = ["f_sensor"]
        provenance_ = {}

        def __init__(self):
            self.frame = None

        def predict_proba(self, frame, segments):
            self.frame = frame.copy()
            return SimpleNamespace(
                index=np.arange(2, len(frame)),
                proba=np.zeros(len(frame) - 2),
                dropped_rows=2,
            )

    training = SimpleNamespace(
        name="train",
        frame=pd.DataFrame({
            "time_seconds": [0, 300, 600, 900],
            "f_sensor": [1., 2., 3., 4.],
        }),
        digest="train-sha256",
    )
    nominal_test = SimpleNamespace(
        name="nominal_test",
        frame=pd.DataFrame({
            "time_seconds": [12000, 12300, 12600],
            "f_sensor": [100., 101., 102.],
        }),
        y_true=np.zeros(3, dtype=int),
        attack_class=np.array(["normal"] * 3),
    )
    nominal_split = SimpleNamespace(
        train=training,
        test=nominal_test,
        manifest={"fractions": {"train": 0.64}},
    )
    discovery = SimpleNamespace(
        nominal=SimpleNamespace(frame=pd.concat([
            training.frame, nominal_test.frame,
        ], ignore_index=True)),
        attacks=[],
    )
    model = DummyModel()
    out_dir = tmp_path / "anytown"
    out_dir.mkdir(parents=True)
    (out_dir / "validation_predictions.csv").write_text("row,score,y_true\n")
    (out_dir / "metrics.json").write_text(json.dumps({
        "observable_policy": {"selection_provenance": {"scope": "train"}},
        "episodes": {"manifest_sha256": "episode-manifest"},
        "metrics": {"validation": {"mcc": 0.0}},
        "model_provenance": {"raw_columns": ["f_sensor"]},
    }))
    monkeypatch.setattr(
        "scripts.run_full_signal_hybrid.discover_full_signal_runs",
        lambda _topology: discovery,
    )
    monkeypatch.setattr(
        "scripts.run_full_signal_hybrid.split_nominal_run",
        lambda _nominal: nominal_split,
    )
    monkeypatch.setattr(
        "scripts.run_full_signal_hybrid.HybridEnsemble.load",
        lambda _path: model,
    )

    metadata = run_topology("anytown", tmp_path, reuse_model=True)
    assert model.frame["f_sensor"].tolist() == [3., 4., 100., 101., 102.]
    assert model.frame["f_sensor"].iloc[:2].tolist() == [3., 4.]
    assert not model.frame["f_sensor"].iloc[:2].isin([100., 101., 102.]).any()
    assert len(pd.read_csv(out_dir / "predictions.csv")) == len(nominal_test.frame)
    assert metadata["warmup"]["source"] == "nominal_split.train.frame"
    assert metadata["warmup"]["training_split_sha256"] == "train-sha256"
    assert metadata["prediction_row_population"]["dropped_evaluation_rows"] == 0
    assert metadata["model_reused_without_retraining"] is True
    assert metadata["validation_predictions"]["preserved_from_existing_run"] is True


def test_equal_mean_ensemble_mode(tmp_path):
    rng = np.random.default_rng(5)
    x = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"])
    y = np.r_[np.zeros(50, dtype=int), np.ones(50, dtype=int)]
    model = HybridEnsemble(HybridEnsembleConfig(
        ensemble_mode="mean", base_models=("rf",), n_folds=2,
        lstm_epochs=1, rolling_window=2, sequence_length=2,
        rf_n_estimators=5,
    ))
    model.fit(x, y)
    result = model.predict_proba(x)
    assert model.meta_ is None
    assert 0 < len(result.proba) <= len(x)
    assert np.all((result.proba >= 0) & (result.proba <= 1))
    model.save(tmp_path / "model")
