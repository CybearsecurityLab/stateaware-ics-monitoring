import numpy as np
import pandas as pd
import pytest

from scripts import sweep_topology_tolerances as sweep


def test_grid_and_scaled_calibration_isolation(monkeypatch):
    fits = []
    configs = []

    class Detector:
        def __init__(self, network, config):
            configs.append(config)

        def fit(self, frames):
            fits.append(frames[0].copy())
            frames[0].iloc[0, 1] = -100
            return self

        def to_artifact(self):
            return {}

    monkeypatch.setattr(sweep.export, "TopologyStlDetector", Detector)
    monkeypatch.setattr(sweep.export, "_stl_predict", lambda detector, frame: (
        np.zeros(len(frame), dtype=int), np.zeros(len(frame)), np.zeros(len(frame), dtype=int)))
    class Phi:
        model = {}

        def predict(self, frame):
            return type("Result", (), {
                "flags": np.zeros(len(frame), dtype=int),
                "scores": np.zeros(len(frame)),
            })()

    calibration = pd.DataFrame({"time_seconds": [0, 1], "value": [3, 4]})
    attack = pd.DataFrame({"time_seconds": [0, 1], "value": [999, 999]})
    prepared = {
        "network": None,
        "calibration": calibration,
        "phi": Phi(),
        "timing_evaluated": {"attack": np.array([0, 0])},
        "obligation_evaluated": {"attack": np.array([0, 0])},
        "obligation_reasons": {"attack": []},
        "scenarios": {"attack": (attack, np.array([0, 1]), np.array([0, 1]))},
    }
    original_cal = calibration.copy(deep=True)
    original_attack = attack.copy(deep=True)
    assert len(sweep.grid()) == 9
    for percentile, scale in sweep.grid():
        sweep.evaluate(prepared, percentile, scale)
    for fitted in fits:
        pd.testing.assert_frame_equal(fitted, original_cal)
    pd.testing.assert_frame_equal(calibration, original_cal)
    pd.testing.assert_frame_equal(attack, original_attack)
    for config, (percentile, scale) in zip(configs, sweep.grid()):
        assert config["percentile"] == percentile
        assert config["closed_flow_epsilon"] == 1e-6
        assert config["additive"] == {k: v * scale for k, v in sweep.export.DEFAULT_ADDITIVE.items()}


def test_evaluate_combines_gfsm_phi_and_stl(monkeypatch):
    class Detector:
        def __init__(self, network, config): pass
        def fit(self, frames): return self
        def to_artifact(self): return {}

    class Phi:
        model = {}
        def predict(self, frame):
            return type("Result", (), {
                "flags": np.array([0, 1]), "scores": np.array([0., 1.])
            })()

    monkeypatch.setattr(sweep.export, "TopologyStlDetector", Detector)
    monkeypatch.setattr(
        sweep.export, "_stl_predict",
        lambda detector, frame: (
            np.array([1, 0]), np.array([2., 0.]), np.array([0, 0])
        ),
    )
    frame = pd.DataFrame({"value": [1, 2]})
    prepared = {
        "network": None,
        "calibration": frame.copy(),
        "phi": Phi(),
        "timing_evaluated": {"attack": np.array([0, 0])},
        "obligation_evaluated": {"attack": np.array([0, 0])},
        "obligation_reasons": {"attack": []},
        "scenarios": {"attack": (frame, np.array([0, 1]), np.array([0, 0]))},
    }
    evaluated, _ = sweep.evaluate(prepared, 99, 1)
    assert evaluated["combined"].tolist() == [1, 1]
    assert evaluated["combined_score"].tolist() == [2., 1.]
    assert evaluated["phi"].tolist() == [0, 1]


def test_metrics_use_arm_specific_scores_and_nominal_false_alarms():
    frame = pd.DataFrame({"y_true": [0, 0, 1, 1], "gfsm": [0, 0, 0, 1],
                          "gfsm_timing": [0, 0, 0, 0],
                          "gfsm_obligations": [0, 0, 0, 0],
                          "gfsm_obligation_score": [0., 0., 0., 0.],
                          "combined_obligations": [1, 1, 0, 1],
                          "combined_obligations_score": [1., 1., 0., 2.],
                          "phi": [0, 0, 0, 0], "phi_score": [0., 0., 0., 0.],
                          "stl": [0, 1, 0, 1],
                          "stl_score": [0., 1., 0., 2.], "combined": [1, 1, 0, 1],
                          "combined_score": [1., 1., 0., 2.], "unknown": [0, 2, 1, 0]})
    result = sweep.metrics(frame, pd.Series([True, True, False, False]))
    score_columns = {
        "gfsm": "gfsm", "gfsm_timing": "gfsm_timing",
        "gfsm_obligations": "gfsm_obligation_score", "phi": "phi_score",
        "stl": "stl_score", "combined": "combined_score",
        "combined_obligations": "combined_obligations_score",
    }
    for arm, score_column in score_columns.items():
        expected = sweep.export.detection_metrics(
            frame.y_true, frame[arm], frame[score_column]
        )
        assert result[arm]["mcc"] == pytest.approx(expected["mcc"])
        assert result[arm]["auprc"] == pytest.approx(expected["aucpr"])
    assert result["stl"]["nominal_false_alarms"] == 1
    assert result["combined"]["nominal_false_alarms"] == 2
    assert result["unknown_rows"] == 2 and result["unknown_residuals"] == 3


def test_load_scenario_aligns_labels_with_chronological_observations(tmp_path):
    pd.DataFrame({
        "time_seconds": [300, 0], "value": [10., 0.]
    }).to_csv(tmp_path / "observed.csv", index=False)
    pd.DataFrame({
        "time_seconds": [300, 0],
        "attack_active": [1, 0],
        "attack_class": ["sensor_offset", "normal"],
    }).to_csv(tmp_path / "labels.csv", index=False)
    frame, truth, _, labels = sweep._load_scenario(tmp_path)
    assert frame.time_seconds.tolist() == [0, 300]
    assert truth.tolist() == [0, 1]
    assert labels.attack_class.tolist() == ["normal", "sensor_offset"]


def test_phi_training_boundary_matches_shared_64_fraction():
    assert int(577 * sweep.PHI_TRAIN_FRACTION) == 369


def test_prepare_split_and_independent_runs(tmp_path, monkeypatch):
    root = tmp_path / "full"
    nominal = root / "anytown" / "nominal"
    attack = root / "anytown" / "attack"
    nominal.mkdir(parents=True)
    attack.mkdir()
    (root / "anytown" / "topology.inp").write_text("network")
    for directory, values in [(nominal, range(10)), (attack, range(100, 105))]:
        pd.DataFrame({"time_seconds": range(len(values)), "value": values}).to_csv(directory / "observed.csv", index=False)
        pd.DataFrame({"attack_active": [0] * len(values)}).to_csv(directory / "labels.csv", index=False)
    gfsm_dir = tmp_path / "data/generated/anytown/gfsm"
    gfsm_dir.mkdir(parents=True)
    (gfsm_dir / "anytown.gfsm.json").write_text("{}")
    monkeypatch.setattr(sweep.export, "_gfsm_components", lambda data: [])
    monkeypatch.setattr(sweep.export, "_resolve_gfsm_columns", lambda **kw: ({}, {}))
    monkeypatch.setattr(sweep.export, "load_network", lambda path: None)
    calls = []

    class Gfsm:
        def __init__(self, **kwargs): pass
        def fit(self, frames): return self
        def predict(self, frame):
            calls.append(frame.value.tolist())
            return type("Result", (), {"flags": np.zeros(len(frame), dtype=int)})()

    monkeypatch.setattr(sweep.export, "GfsmAnomalyDetector", Gfsm)

    class Phi:
        model = {}

        @classmethod
        def fit(cls, **kwargs):
            assert kwargs["training_frame"].value.tolist() == list(range(6))
            assert kwargs["validation_frame"].value.tolist() == [6, 7]
            return cls()

    monkeypatch.setattr(sweep, "FullSignalPhiDetector", Phi)

    class Obligations:
        def __init__(self, **kwargs): pass
        def fit(self, frames): return self
        def predict(self, frame):
            return type("Result", (), {"flags": np.zeros(len(frame), dtype=int)})()
        def predict_detailed(self, frame):
            flags = np.zeros(len(frame), dtype=int)
            return type("Result", (), {
                "flags": flags, "scores": flags.astype(float), "reasons": [],
            })()

    monkeypatch.setattr(sweep, "GfsmObligationDetector", Obligations)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    prepared = sweep.prepare("anytown", root, tmp_path / "data", tmp_path / "out")
    assert prepared["calibration"].value.tolist() == list(range(8))
    assert calls == [list(range(100, 105)), [8, 9]]
    assert before == {p: p.read_bytes() for p in before}
    assert len(prepared["inputs"]) == 6
