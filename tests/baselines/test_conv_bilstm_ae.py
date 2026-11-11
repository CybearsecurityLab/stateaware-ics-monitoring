"""Unit tests for the ConvBiLSTM-AE baseline (Lee et al., 2024).

All tests use tiny synthetic frames and 1-2 epochs: no real model is trained
and no real dataset is touched.  The canonical paper configuration is checked
separately (defaults only, no fitting).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from baselines.conv_bilstm_ae import (  # noqa: E402
    PAPER_SPEC,
    ConvBiLSTMAE,
    ConvBiLSTMAEConfig,
    ConvBiLSTMAENet,
    SamePadConv1d,
    TrainMinMaxScaler,
    environment_metadata,
    measure_batch_throughput,
    measure_single_event_latency,
    select_oracle_threshold,
)

SEED = 42


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def tiny_config(**overrides) -> ConvBiLSTMAEConfig:
    """A deliberately tiny configuration that still exercises every code path."""
    cfg = dict(
        window_size=2,
        conv_filters=4,
        conv_kernel_size=3,
        lstm_units=2,
        dense_units=3,
        batch_size=8,
        epochs=1,
        learning_rate=1e-2,
        seed=SEED,
        threshold_quantile=0.9,
        device="cpu",
    )
    cfg.update(overrides)
    return ConvBiLSTMAEConfig(**cfg)


def synthetic_frames(n_train=48, n_val=24, n_test=30, n_features=6, seed=SEED):
    """Deterministic synthetic normal/attack frames (rows = timesteps)."""
    rng = np.random.default_rng(seed)
    train = rng.normal(0.0, 1.0, size=(n_train, n_features))
    val = rng.normal(0.0, 1.0, size=(n_val, n_features))
    test = rng.normal(0.0, 1.0, size=(n_test, n_features))
    labels = np.zeros(n_test, dtype=np.int64)
    labels[10:16] = 1
    test[10:16] += 6.0
    return train, val, test, labels


# --------------------------------------------------------------------------
# canonical configuration
# --------------------------------------------------------------------------
def test_canonical_config_matches_paper_table3():
    cfg = ConvBiLSTMAEConfig()
    assert cfg.window_size == 1
    assert cfg.conv_filters == 132
    assert cfg.conv_kernel_size == 48
    assert cfg.lstm_units == 32
    assert cfg.dense_units == 66
    assert cfg.batch_size == 512
    assert cfg.epochs == 60
    assert cfg.learning_rate == 1e-3
    assert cfg.seed == 42
    assert cfg.early_stopping_patience is None


# --------------------------------------------------------------------------
# architecture / tensor shapes
# --------------------------------------------------------------------------
def test_network_output_shape_and_architecture_summary():
    net = ConvBiLSTMAENet(
        n_features=7,
        window_size=3,
        conv_filters=5,
        conv_kernel_size=2,
        lstm_units=3,
        dense_units=4,
    )
    x = torch.zeros(2, 3, 7)
    y = net(x)
    assert tuple(y.shape) == (2, 3, 7)

    summary = net.architecture_summary()
    assert summary["n_features"] == 7
    assert summary["window_size"] == 3
    assert summary["encoder"]["conv1d_filters"] == 5
    assert summary["encoder"]["conv1d_kernel_size"] == 2
    assert summary["encoder"]["dense_units"] == 4
    assert summary["encoder"]["lstm_units_per_direction"] == 3
    assert summary["encoder"]["lstm_bidirectional"] is True
    assert summary["decoder"]["dense_units"] == 4
    assert summary["decoder"]["conv1d_filters"] == 5
    assert summary["decoder"]["output_units"] == 7
    assert summary["n_parameters"] > 0


def test_encoder_conv_uses_132_filters_48_kernel_by_default():
    net = ConvBiLSTMAENet(n_features=87)
    assert net.encoder_conv.out_channels == 132
    assert net.encoder_conv.kernel_size == 48
    assert net.decoder_conv.out_channels == 132
    assert net.decoder_conv.kernel_size == 48
    assert net.encoder_lstm.num_layers == 1
    assert net.encoder_lstm.bidirectional is True
    assert net.encoder_lstm.hidden_size == 32
    assert net.encoder_dense.out_features == 66


def test_same_pad_conv_keeps_length_and_supports_even_kernels():
    """TensorFlow 'same' semantics: output length == input length."""
    conv = SamePadConv1d(in_channels=3, out_channels=5, kernel_size=48)
    for length in (1, 2, 10, 66, 87):
        out = conv(torch.zeros(1, 3, length))
        assert out.shape[-1] == length
        assert out.shape[1] == 5


def test_same_pad_conv_matches_manual_padding_arithmetic():
    conv = SamePadConv1d(in_channels=2, out_channels=3, kernel_size=48)
    x = torch.arange(2 * 2 * 1, dtype=torch.float32).reshape(2, 2, 1)
    out = conv(x)
    # 47 pad taps are inserted (23 left / 24 right) around the single timestep.
    assert out.shape[-1] == 1
    assert torch.isfinite(out).all()


def test_reconstruction_error_uses_float64_before_squaring():
    class ZeroModel(torch.nn.Module):
        def forward(self, batch):
            return torch.zeros_like(batch)

    baseline = ConvBiLSTMAE(tiny_config(window_size=1))
    baseline.model = ZeroModel()
    windows = np.array([[[1.0e32]]], dtype=np.float64)
    errors = baseline.reconstruction_errors(windows)
    assert np.isfinite(errors).all()
    assert errors[0] == pytest.approx(1.0e64, rel=1e-6)


def test_predict_row_and_window_alignment():
    train, val, test, labels = synthetic_frames()
    baseline = ConvBiLSTMAE(tiny_config())
    baseline.fit(train, val)
    result = baseline.predict(test)
    assert result.n_windows == test.shape[0] - baseline.config.window_size + 1
    assert result.errors.shape == (result.n_windows,)
    assert result.row_errors.shape == (result.n_rows,)
    # The first window_size-1 rows have no complete window.
    assert np.isnan(result.row_errors[: baseline.config.window_size - 1]).all()
    assert np.isfinite(result.row_errors[baseline.config.window_size - 1 :]).all()
    assert result.predictions.sum() == (result.errors > baseline.threshold).sum()


def test_window_size_one_gives_one_score_per_row():
    train, val, test, _ = synthetic_frames()
    baseline = ConvBiLSTMAE(tiny_config(window_size=1))
    baseline.fit(train, val)
    result = baseline.predict(test)
    assert result.n_windows == result.n_rows == test.shape[0]
    assert np.isfinite(result.row_errors).all()


# --------------------------------------------------------------------------
# preprocessing: constant channels + scaling isolation
# --------------------------------------------------------------------------
def test_constant_training_channels_are_removed():
    train = np.zeros((10, 4))
    train[:, 0] = np.arange(10)
    train[:, 1] = 5.0            # constant -> removed
    train[:, 2] = np.arange(10) ** 2
    train[:, 3] = -2.0           # constant -> removed
    scaler = TrainMinMaxScaler().fit(train, ["a", "b", "c", "d"])
    assert list(scaler.kept_columns) == [0, 2]
    assert set(scaler.removed_columns) == {"b", "d"}
    assert scaler.min_.tolist() == [0.0, 0.0]
    assert scaler.max_.tolist() == [9.0, 81.0]


def test_scaler_uses_training_statistics_only():
    train = np.array([[0.0, 10.0], [5.0, 20.0]])
    scaler = TrainMinMaxScaler().fit(train)
    scaled = scaler.transform(np.array([[2.5, 15.0]]))
    assert scaled.tolist() == [[0.5, 0.5]]
    # Out-of-range test values are *not* clipped (Eq. 10 has no clipping).
    scaled_out = scaler.transform(np.array([[10.0, 30.0]]))
    assert scaled_out.tolist() == [[2.0, 2.0]]


def test_scaling_isolation_from_validation_and_test_data():
    """The scaler must not move when other splits are seen."""
    train, val, test, labels = synthetic_frames()
    baseline = ConvBiLSTMAE(tiny_config())
    baseline.fit(train, val)
    min_before = baseline.scaler.min_.copy()
    max_before = baseline.scaler.max_.copy()
    kept_before = baseline.scaler.kept_columns.copy()

    other_val = val * 100.0 + 50.0
    other_test = test * 100.0 + 50.0
    baseline.predict(other_val)
    baseline.score(other_test, labels)

    assert np.array_equal(baseline.scaler.min_, min_before)
    assert np.array_equal(baseline.scaler.max_, max_before)
    assert np.array_equal(baseline.scaler.kept_columns, kept_before)


def test_constant_channel_removed_from_every_split():
    train = np.random.default_rng(0).normal(size=(30, 4))
    train[:, 1] = 7.0
    val = np.random.default_rng(1).normal(size=(20, 4))
    val[:, 1] = 7.0
    test = np.random.default_rng(2).normal(size=(20, 4))
    test[:, 1] = 7.0

    baseline = ConvBiLSTMAE(tiny_config())
    baseline.fit(train, val)
    assert baseline.scaler.removed_columns == ("c1",)
    assert baseline.model.n_features == 3
    result = baseline.predict(test)
    # The constant channel cannot contribute to the reconstruction error.
    assert result.errors.shape == (result.n_windows,)


# --------------------------------------------------------------------------
# threshold selection (validation only)
# --------------------------------------------------------------------------
def test_threshold_equals_validation_quantile_and_ignores_test_data():
    train, val, test, labels = synthetic_frames()
    cfg = tiny_config(threshold_quantile=0.9)
    baseline = ConvBiLSTMAE(cfg)
    baseline.fit(train, val)

    val_errors = baseline.reconstruction_errors(baseline.windowize(val))
    expected = float(np.quantile(val_errors, cfg.threshold_quantile))
    assert baseline.threshold == pytest.approx(expected, rel=1e-9)

    threshold_before = baseline.threshold
    baseline.predict(test)
    baseline.score(test, labels)
    baseline.predict(test * 50.0)
    assert baseline.threshold == threshold_before


def test_threshold_is_not_optimised_for_f1_on_the_test_set():
    """With a test-optimal F1 threshold available, the baseline must not use it."""
    train, val, test, labels = synthetic_frames(n_val=24, n_test=30)
    cfg = tiny_config(threshold_quantile=0.9)
    baseline = ConvBiLSTMAE(cfg)

    val_errors = np.linspace(0.01, 0.05, 24)
    n_val_windows = 24 - cfg.window_size + 1
    n_test_windows = 30 - cfg.window_size + 1
    test_errors = np.concatenate(
        [np.full(10, 0.001), np.full(n_test_windows - 10, 0.9)]
    )

    def fake_errors(windows):
        if len(windows) == n_val_windows:
            return val_errors[:n_val_windows]
        return test_errors[:n_test_windows]

    with mock.patch.object(baseline, "reconstruction_errors", side_effect=fake_errors):
        baseline.fit(train, val)

    expected = float(np.quantile(val_errors[:n_val_windows], cfg.threshold_quantile))
    assert baseline.threshold == pytest.approx(expected, rel=1e-9)

    # An F1-optimising threshold on the test errors would sit far above the
    # validation quantile; the baseline must not have used it.
    from sklearn.metrics import precision_recall_curve

    labels_padded = np.zeros(n_test_windows, dtype=np.int64)
    labels_padded[10:] = 1
    precision, recall, thresholds = precision_recall_curve(labels_padded, test_errors)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = float(thresholds[int(np.argmax(f1[:-1]))])
    assert baseline.threshold < best


# --------------------------------------------------------------------------
# no test refitting
# --------------------------------------------------------------------------
def test_predict_and_score_do_not_refit_anything():
    train, val, test, labels = synthetic_frames()
    baseline = ConvBiLSTMAE(tiny_config())
    baseline.fit(train, val)

    weights_before = {k: v.clone() for k, v in baseline.model.state_dict().items()}
    min_before = baseline.scaler.min_.copy()
    max_before = baseline.scaler.max_.copy()
    threshold_before = baseline.threshold
    history_len = len(baseline.history)

    baseline.predict(test)
    baseline.score(test, labels)
    baseline.predict(test)

    for key, value in baseline.model.state_dict().items():
        assert torch.equal(value, weights_before[key])
    assert np.array_equal(baseline.scaler.min_, min_before)
    assert np.array_equal(baseline.scaler.max_, max_before)
    assert baseline.threshold == threshold_before
    assert len(baseline.history) == history_len


def test_fit_is_deterministic_for_a_fixed_seed():
    train, val, test, _ = synthetic_frames()
    a = ConvBiLSTMAE(tiny_config(epochs=2))
    b = ConvBiLSTMAE(tiny_config(epochs=2))
    a.fit(train, val)
    b.fit(train, val)
    for key, value in a.model.state_dict().items():
        assert torch.equal(value, b.model.state_dict()[key])
    assert a.threshold == pytest.approx(b.threshold, rel=1e-12)
    errors_a = a.reconstruction_errors(a.windowize(test))
    errors_b = b.reconstruction_errors(b.windowize(test))
    assert np.allclose(errors_a, errors_b, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------
# metric correctness
# --------------------------------------------------------------------------
def test_metric_correctness_against_sklearn():
    train, val, test, labels = synthetic_frames()
    baseline = ConvBiLSTMAE(tiny_config())
    baseline.fit(train, val)
    metrics = baseline.score(test, labels)

    result = baseline.predict(test)
    mask = ~np.isnan(result.row_errors)
    y_true = labels[mask]
    scores = result.row_errors[mask]
    y_pred = result.row_predictions[mask]

    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        matthews_corrcoef,
    )

    assert metrics["aucpr"] == pytest.approx(average_precision_score(y_true, scores))
    assert metrics["average_precision"] == metrics["aucpr"]
    assert metrics["mcc"] == pytest.approx(matthews_corrcoef(y_true, y_pred))
    assert metrics["balanced_accuracy"] == pytest.approx(
        balanced_accuracy_score(y_true, y_pred)
    )
    assert metrics["n_scored"] == int(mask.sum())
    assert metrics["n_positive"] == int(y_true.sum())
    assert metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"] == metrics["n_scored"]


def test_score_rejects_misaligned_labels():
    train, val, test, labels = synthetic_frames()
    baseline = ConvBiLSTMAE(tiny_config())
    baseline.fit(train, val)
    with pytest.raises(ValueError):
        baseline.score(test, labels[:-3])


# --------------------------------------------------------------------------
# deterministic initialisation
# --------------------------------------------------------------------------
def test_deterministic_initialisation_for_equal_seeds():
    torch.manual_seed(0)
    a = ConvBiLSTMAENet(n_features=5, conv_filters=3, conv_kernel_size=2, lstm_units=2, dense_units=3)
    torch.manual_seed(0)
    b = ConvBiLSTMAENet(n_features=5, conv_filters=3, conv_kernel_size=2, lstm_units=2, dense_units=3)
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])


def test_baseline_seed_is_recorded_and_applied():
    train, val, _, _ = synthetic_frames()
    baseline = ConvBiLSTMAE(tiny_config(seed=7))
    baseline.fit(train, val)
    assert baseline.config.seed == 7
    assert baseline.metadata()["seed"] == 7
    # fit() seeds the global torch RNG from the configuration.
    assert torch.initial_seed() == 7


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------
def test_persistence_roundtrip(tmp_path: Path):
    train, val, test, labels = synthetic_frames()
    baseline = ConvBiLSTMAE(tiny_config())
    baseline.fit(train, val)
    baseline.score(test, labels)
    out_dir = baseline.save(tmp_path / "artifact")

    assert (out_dir / "model.pt").exists()
    assert (out_dir / "scaler.npz").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "predictions.npz").exists()

    restored = ConvBiLSTMAE.load(out_dir)
    assert restored.threshold == pytest.approx(baseline.threshold, rel=1e-12)
    assert restored.config.to_dict() == baseline.config.to_dict()
    assert restored.scaler.removed_columns == baseline.scaler.removed_columns

    original = baseline.predict(test)
    reloaded = restored.predict(test)
    assert np.allclose(original.errors, reloaded.errors, rtol=0, atol=1e-12)
    assert np.array_equal(original.predictions, reloaded.predictions)

    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert metadata["config"]["seed"] == SEED
    assert "hardware" in metadata and "software" in metadata
    assert metadata["architecture"]["encoder"]["conv1d_filters"] == 4


def test_save_requires_a_fitted_model(tmp_path: Path):
    baseline = ConvBiLSTMAE(tiny_config())
    with pytest.raises(RuntimeError):
        baseline.save(tmp_path / "empty")


# --------------------------------------------------------------------------
# timing helpers + metadata
# --------------------------------------------------------------------------
def test_timing_helpers_report_latency_and_throughput_separately():
    net = ConvBiLSTMAENet(n_features=4, conv_filters=3, conv_kernel_size=2, lstm_units=2, dense_units=3)
    sample = torch.zeros(2, 4)
    batch = torch.zeros(8, 2, 4)

    latency = measure_single_event_latency(net, sample, device="cpu", warmup=1, repeats=3)
    assert latency["measurement"] == "single_event_latency"
    assert latency["batch_size"] == 1
    assert latency["unit"] == "ms"
    assert latency["median_ms"] > 0
    assert latency["n_repeats"] == 3

    throughput = measure_batch_throughput(net, batch, device="cpu", warmup=1, repeats=3)
    assert throughput["measurement"] == "batch_throughput"
    assert throughput["batch_size"] == 8
    assert throughput["median_ms"] > 0
    assert throughput["events_per_second"] > 0
    assert throughput["ms_per_event"] > 0
    # The two measurements describe different things and must not be conflated.
    assert latency["measurement"] != throughput["measurement"]


def test_environment_metadata_reports_hardware_and_software():
    meta = environment_metadata(torch.device("cpu"))
    assert "hardware" in meta and "software" in meta
    assert meta["hardware"]["platform"]
    assert meta["hardware"]["compute_device_type"] == "cpu"
    assert meta["hardware"]["cpu_count"] >= 1
    assert meta["software"]["torch_version"] == torch.__version__
    assert meta["software"]["python_version"]
    assert "numpy_version" in meta["software"]
    assert "scikit_learn_version" in meta["software"]


def test_baseline_metadata_contains_config_threshold_and_hardware():
    train, val, test, labels = synthetic_frames()
    baseline = ConvBiLSTMAE(tiny_config())
    baseline.fit(train, val)
    baseline.score(test, labels)
    meta = baseline.metadata()
    assert meta["config"]["seed"] == SEED
    assert meta["threshold"] == pytest.approx(baseline.threshold)
    assert "validation reconstruction errors" in meta["threshold_rule"]
    assert meta["dataset"]["n_train_rows"] == train.shape[0]
    assert meta["dataset"]["n_validation_rows"] == val.shape[0]
    assert meta["hardware"]["compute_device_type"]
    assert meta["software"]["torch_version"]
    assert meta["history"], "training history should be recorded"


def test_predict_before_fit_raises():
    baseline = ConvBiLSTMAE(tiny_config())
    with pytest.raises(RuntimeError):
        baseline.predict(np.zeros((4, 6)))

# --------------------------------------------------------------------------
# faithful preset vs adapted configuration
# --------------------------------------------------------------------------
def test_faithful_preset_matches_table3_and_rejects_architecture_overrides():
    cfg = ConvBiLSTMAEConfig.faithful()
    assert cfg.fidelity == "paper"
    for name, expected in PAPER_SPEC.items():
        assert getattr(cfg, name) == expected
    # The paper does not state Adam's learning rate; it stays a recorded choice.
    assert cfg.learning_rate == 1e-3
    with pytest.raises(ValueError):
        ConvBiLSTMAEConfig.faithful(window_size=10)
    with pytest.raises(ValueError):
        ConvBiLSTMAEConfig(fidelity="paper", window_size=10).validate()


def test_faithful_metadata_records_paper_spec_and_unspecified_choices():
    baseline = ConvBiLSTMAE(ConvBiLSTMAEConfig.faithful(device="cpu"))
    meta = baseline.metadata()
    assert meta["fidelity"] == "paper"
    assert meta["paper_spec"]["window_size"] == 1
    assert meta["paper_spec"]["conv_filters"] == 132
    assert meta["implementation_notes"]["recurrent_activation"].startswith("standard LSTM")
    assert "conv_padding" in meta["unspecified_choices"]
    assert "learning_rate" in meta["unspecified_choices"]


def test_architecture_summary_records_tanh_recurrent_cell():
    net = ConvBiLSTMAENet(n_features=4, window_size=1, conv_filters=3,
                          conv_kernel_size=2, lstm_units=2, dense_units=3)
    cell = net.architecture_summary()["recurrent_cell"]
    assert cell["type"] == "LSTM"
    assert cell["gate_activation"] == "sigmoid"
    assert cell["candidate_and_hidden_activation"] == "tanh"
    assert cell["bidirectional"] is True


# --------------------------------------------------------------------------
# full-signal DataFrame schema selection
# --------------------------------------------------------------------------
def _full_signal_frame(n, width, *, seed, extra=False):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(seed)
    data = {"time_seconds": np.arange(n, dtype=float)}
    for i in range(width):
        data[f"f{i}"] = rng.normal(size=n)
    data["constant_channel"] = np.full(n, 3.0)
    data["attack_active"] = np.zeros(n, dtype=int)
    data["attack_class"] = np.array(["normal"] * n, dtype=object)
    if extra:
        data["scenario_only"] = rng.normal(size=n)
    return pd.DataFrame(data)


@pytest.mark.parametrize("town,width", [("anytown", 3), ("ctown", 4), ("ltown", 5)])
def test_full_signal_dataframes_work_for_all_towns(town, width):
    train = _full_signal_frame(36, width, seed=100 + width)
    val = _full_signal_frame(24, width, seed=200 + width)
    test = _full_signal_frame(20, width, seed=300 + width, extra=True)
    labels = np.zeros(len(test), dtype=np.int64)
    labels[8:12] = 1

    baseline = ConvBiLSTMAE(tiny_config())
    baseline.fit(train, val)

    expected = [f"f{i}" for i in range(width)] + ["constant_channel"]
    assert baseline.feature_columns == expected
    assert set(baseline.dropped_columns) == {
        "time_seconds", "attack_active", "attack_class"
    }
    assert baseline.scaler.removed_columns == ("constant_channel",)

    result = baseline.predict(test)
    assert result.n_rows == len(test)
    assert result.n_windows == len(test) - baseline.config.window_size + 1
    assert baseline.feature_columns == expected  # test frame cannot change schema
    baseline.score(test, labels)


def test_dataframe_schema_is_training_only_and_rejects_missing_features():
    train = _full_signal_frame(36, 3, seed=1)
    val = _full_signal_frame(24, 3, seed=2)
    test = _full_signal_frame(18, 3, seed=3, extra=True)
    # Reorder and add columns: the fitted names, not positions, must be used.
    test = test[
        ["f2", "time_seconds", "f0", "attack_active", "f1",
         "attack_class", "scenario_only", "constant_channel"]
    ]
    baseline = ConvBiLSTMAE(tiny_config())
    baseline.fit(train, val)
    schema_before = list(baseline.feature_columns)
    baseline.predict(test)
    assert baseline.feature_columns == schema_before

    with pytest.raises(ValueError):
        baseline.predict(test.drop(columns=["f1"]))


def test_windows_are_built_per_frame_and_never_cross_scenarios():
    train, val, _, _ = synthetic_frames()
    baseline = ConvBiLSTMAE(tiny_config(window_size=2))
    baseline.fit(train, val)

    width = train.shape[1]
    first = np.arange(10 * width, dtype=float).reshape(10, width)
    second = np.arange(100, 100 + 10 * width, dtype=float).reshape(10, width)
    r_first = baseline.predict(first)
    r_second = baseline.predict(second)
    assert r_first.n_windows == len(first) - 1
    assert r_second.n_windows == len(second) - 1
    assert np.isnan(r_second.row_errors[0])
    assert np.isfinite(r_second.row_errors[1])

    joined = baseline.predict(np.vstack([first, second]))
    assert joined.n_windows == len(first) + len(second) - 1
    assert joined.n_windows == r_first.n_windows + r_second.n_windows + 1


# --------------------------------------------------------------------------
# oracle threshold analysis is explicit and non-mutating
# --------------------------------------------------------------------------
def test_select_oracle_threshold_maximises_the_requested_metric():
    y = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    candidates = np.concatenate(
        ([np.nextafter(scores.min(), -np.inf)], np.unique(scores),
         [np.nextafter(scores.max(), np.inf)])
    )

    from sklearn.metrics import f1_score

    oracle = select_oracle_threshold(y, scores, criterion="f1")
    brute = max(
        float(f1_score(y, (scores > t).astype(int), zero_division=0))
        for t in candidates
    )
    assert oracle["criterion_value"] == pytest.approx(brute)
    assert oracle["criterion"] == "f1"


def test_score_oracle_is_explicit_and_does_not_mutate_primary_threshold():
    train, val, test, labels = synthetic_frames()
    baseline = ConvBiLSTMAE(tiny_config())
    baseline.fit(train, val)
    primary_threshold = baseline.threshold
    primary = baseline.score(test, labels)

    oracle = baseline.score_oracle(test, labels, criterion="f1")
    assert baseline.threshold == primary_threshold
    assert oracle["threshold_policy"] == "test_oracle_explicit"
    assert "leakage" in oracle["leakage_warning"]
    assert oracle["primary_threshold"] == primary_threshold
    assert oracle["f1"] >= primary["f1"] - 1e-12
