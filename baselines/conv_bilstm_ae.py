"""ConvBiLSTM-AE baseline (Lee et al., 2024).

Paper-grounded implementation of the hybrid convolutional + bidirectional
LSTM autoencoder in:

    B. Lee, S. Kim, M. Maqsood, J. Moon, S. Rho,
    "Advancing Autoencoder Architectures for Enhanced Anomaly Detection in
    Multivariate Industrial Time Series", Computers, Materials & Continua,
    vol. 81, no. 1, pp. 1275-1298, 2024. doi:10.32604/cmc.2024.054826

Primary source (page references are the CMC printed pages)
-----------------------------------------------------------
* Table 2, p. 1281-1282: HAI 23.05 normal-only train/validation, two labelled
  test scenarios; 87 variables including a timestamp.
* Algorithm 3, p. 1285-1286: ConvBiLSTM-AE trains an encoder/decoder with
  Conv1D -> BiLSTM and reconstructs the input with an MSE loss.
* Fig. 3 / Eqs. (1)-(6), p. 1283: standard LSTM cell.  Gates use sigmoid
  activations; the cell candidate and hidden state use tanh.  This is the
  recurrent activation implemented by ``torch.nn.LSTM``.
* Eqs. (7)-(9), p. 1283-1284: bidirectional LSTM concatenates the forward and
  backward hidden states.
* Sec. 2.3, p. 1284 and Sec. 2.4, p. 1285: the 1D CNN is applied on the time
  axis and the deep autoencoder is symmetric.
* Sec. 3.1, p. 1287: timestamp excluded, min-max scaling with training
  statistics (Eq. 10); implementation is TensorFlow/Keras.
* Sec. 3.2, p. 1287-1288: pointwise reconstruction error Eq. (11) and
  threshold rule Eq. (12).  The paper picks the threshold that gives the best
  test performance; that is oracle analysis, not a deployable rule.
* Table 3, p. 1289: Conv1D filters 132, kernel 48, LSTM units 32, Dense 66,
  window 1, symmetric decoder; batch 512, Adam, MSE, 60 epochs (p. 1289).

Paper-specified configuration
-----------------------------
* unsupervised normal-only training;
* Conv1D(132, kernel 48) -> Dense(66) -> BiLSTM(32) encoder;
* symmetric decoder -> Dense(66) -> Conv1D(132, kernel 48) -> Dense(input
  width);
* sequence window 1 (Table 3);
* standard tanh/sigmoid LSTM cell (Eqs. 1-6);
* pointwise MSE reconstruction loss (Eq. 11) and MSE training loss;
* min-max scaling from training rows only (Eq. 10), no clipping;
* batch 512, Adam, 60 epochs;
* test-optimal threshold (Eq. 12), exposed here only through
  :meth:`ConvBiLSTMAE.score_oracle`.

Leak-free primary protocol (this repository, explicitly not the paper)
-----------------------------------------------------------------------
The paper selects the threshold on the labelled test set.  Reusing that rule
to report scores leaks test labels.  The primary :meth:`ConvBiLSTMAE.fit`
therefore stores the ``threshold_quantile`` quantile of the *normal validation*
reconstruction errors.  Test labels are never read by ``fit``, ``predict`` or
``score``.  :meth:`ConvBiLSTMAE.score_oracle` is a separate, explicitly named
analysis that recomputes the paper's test-optimal threshold and never changes
the fitted threshold.

Unspecified choices (recorded, not invented as paper facts)
-----------------------------------------------------------
The paper does not state the following.  Defaults are chosen for a runnable
TensorFlow-era baseline and are listed in ``UNSPECIFIED_CHOICES``:
Conv1D padding (``same`` is required to keep a length-1 window alive with
kernel 48), Dense activation (ReLU), Conv1D/output activation (linear),
learning rate (1e-3), Adam beta/epsilon and weight initialisation (framework
defaults), batch shuffling (shuffled each epoch), early stopping (none, full
60 epochs), and whether Table 3's LSTM 32 is per direction or total (read as
per direction, i.e. 64 concatenated features, matching Keras
``Bidirectional(LSTM(32))``).  Static training channels are removed from
training statistics only; the paper says static variables were excluded but
does not define the detection split.

No author code artifact is cited by the paper.  The only repository it cites is
the HAI dataset (``github.com/icsdataset/hai``), so Table 3 and the text are
the authority here.  Earlier drafts of this module mentioned an unlinked
"HAI-Anomaly-Detection" repository; that source is not used.

The module requires ``torch`` (the ``[baselines]`` extra); ``pandas`` is only
needed when the caller passes a DataFrame.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = [
    "ConvBiLSTMAE",
    "ConvBiLSTMAEConfig",
    "ConvBiLSTMAENet",
    "PAPER_SPEC",
    "PredictionResult",
    "SamePadConv1d",
    "TrainMinMaxScaler",
    "UNSPECIFIED_CHOICES",
    "environment_metadata",
    "measure_batch_throughput",
    "measure_single_event_latency",
    "run_baseline",
    "select_oracle_threshold",
]


# Table 3 (p. 1289) plus the training settings on p. 1289.  Only these values
# are enforced by the ``fidelity="paper"`` preset; everything else is either a
# documented adaptation or an unspecified choice.
PAPER_SPEC: dict[str, Any] = {
    "window_size": 1,
    "conv_filters": 132,
    "conv_kernel_size": 48,
    "lstm_units": 32,
    "dense_units": 66,
    "batch_size": 512,
    "epochs": 60,
    "early_stopping_patience": None,
}

# Columns that the full-signal adapter emits for provenance/labels rather than
# model features.  They are dropped only when fitting on a DataFrame; the
# resulting list is stored and reused for every later scenario.
DEFAULT_NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "time_seconds",
    "timestamp",
    "time",
    "datetime",
    "date",
    "index",
    "attack_active",
    "attack_class",
    "label",
    "labels",
    "y_true",
)

ORACLE_CRITERIA = ("f1", "accuracy", "balanced_accuracy", "mcc")

IMPLEMENTATION_NOTES = {
    # paper-backed
    "sequence_window": "1 (Table 3, p. 1289); not a shortcut to remove",
    "recurrent_activation": (
        "standard LSTM cell, sigmoid gates and tanh candidate/hidden "
        "(Eqs. 1-6, p. 1283); torch.nn.LSTM matches"
    ),
    "reconstruction_loss": "pointwise MSE (Eq. 11, p. 1287; training loss p. 1289)",
    "scaling": "min-max from training rows only (Eq. 10, p. 1287), no clipping",
    "training": "batch 512, Adam, MSE, 60 epochs (p. 1289)",
    "threshold_primary": (
        "validation-quantile of normal reconstruction errors; leak-free "
        "repository protocol, not the paper's test-optimal rule"
    ),
    "threshold_oracle": (
        "paper rule (Sec. 3.2, p. 1287) exposed only by score_oracle"
    ),
}

UNSPECIFIED_CHOICES = {
    "conv_padding": (
        "not stated; TensorFlow-'same' keeps a length-1 window alive with "
        "kernel 48 and is the only runnable reading of Table 3"
    ),
    "dense_activation": "not stated; ReLU used",
    "conv_and_output_activation": "not stated; linear used",
    "learning_rate": "not stated; 1e-3 used",
    "gradient_clip_norm": "not stated; norm 1.0 used to prevent non-finite training",
    "adam_betas_and_epsilon": "not stated; PyTorch defaults used",
    "weight_initialisation": "not stated; PyTorch defaults used",
    "batch_order": "not stated; training rows shuffled each epoch",
    "early_stopping": "not stated; full 60 epochs, no early stopping",
    "lstm_units_reading": (
        "Table 3 says 32; read as 32 per direction (64 concatenated), "
        "matching Keras Bidirectional(LSTM(32))"
    ),
    "static_channel_split": (
        "paper excludes static variables; detected on training rows only here"
    ),
    "threshold_quantile": (
        "the paper gives per-test thresholds for reporting, not a deployable "
        "quantile; 0.995 normal-validation coverage chosen here"
    ),
}


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass
class ConvBiLSTMAEConfig:
    """ConvBiLSTM-AE hyper-parameters.

    ``fidelity="adapted"`` (the default) allows the tiny test configurations
    and any deliberate deviation.  ``fidelity="paper"`` is the strict preset
    created by :meth:`faithful`; it rejects architecture/training overrides
    that are not in Table 3.  In both modes the primary threshold is the
    leak-free validation quantile; the paper's test-optimal threshold is
    available only through :meth:`ConvBiLSTMAE.score_oracle`.

    All unspecified values are listed in ``UNSPECIFIED_CHOICES`` and recorded
    in the run metadata rather than presented as paper facts.
    """

    # --- architecture (paper Table 3, p. 1289) ---
    window_size: int = 1
    conv_filters: int = 132
    conv_kernel_size: int = 48
    lstm_units: int = 32
    dense_units: int = 66

    # --- optimisation (paper p. 1289) ---
    batch_size: int = 512
    epochs: int = 60
    learning_rate: float = 1e-3
    gradient_clip_norm: float | None = 1.0
    early_stopping_patience: int | None = None

    # --- protocol ---
    fidelity: str = "adapted"  # "paper" | "adapted"
    seed: int = 42
    threshold_quantile: float = 0.995
    oracle_criterion: str = "f1"
    non_feature_columns: tuple[str, ...] = DEFAULT_NON_FEATURE_COLUMNS
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"

    @classmethod
    def faithful(cls, **overrides: Any) -> "ConvBiLSTMAEConfig":
        """Paper Table 3 preset; non-paper overrides are rejected.

        Allowed overrides are protocol choices not fixed by the paper:
        ``seed``, ``threshold_quantile``, ``oracle_criterion``,
        ``non_feature_columns``, ``learning_rate`` and ``device``.  Any
        architecture/training override raises during :meth:`validate`.
        """
        allowed = {
            "seed",
            "threshold_quantile",
            "oracle_criterion",
            "non_feature_columns",
            "learning_rate",
            "gradient_clip_norm",
            "device",
        }
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(
                "faithful() only accepts protocol overrides; "
                f"refusing {sorted(unknown)}"
            )
        config = cls(fidelity="paper", **PAPER_SPEC, **overrides)
        config.validate()
        return config

    def validate(self) -> None:
        if self.fidelity not in {"paper", "adapted"}:
            raise ValueError(
                f"fidelity must be 'paper' or 'adapted', got {self.fidelity!r}"
            )
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        if self.conv_filters < 1 or self.conv_kernel_size < 1:
            raise ValueError("conv_filters and conv_kernel_size must be >= 1")
        if self.lstm_units < 1 or self.dense_units < 1:
            raise ValueError("lstm_units and dense_units must be >= 1")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be >= 1")
        if not 0.0 < self.threshold_quantile < 1.0:
            raise ValueError(
                f"threshold_quantile must be in (0, 1), got {self.threshold_quantile}"
            )
        if self.oracle_criterion not in ORACLE_CRITERIA:
            raise ValueError(
                f"oracle_criterion must be one of {ORACLE_CRITERIA}, "
                f"got {self.oracle_criterion!r}"
            )
        if self.fidelity == "paper":
            mismatched = {
                name: (getattr(self, name), expected)
                for name, expected in PAPER_SPEC.items()
                if getattr(self, name) != expected
            }
            if mismatched:
                details = ", ".join(
                    f"{name}={actual!r} (paper {expected!r})"
                    for name, (actual, expected) in sorted(mismatched.items())
                )
                raise ValueError(
                    "fidelity='paper' rejects non-Table-3 values: " + details
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------
@dataclass
class TrainMinMaxScaler:
    """Min-max scaler fitted on training rows only.

    Training-constant channels (max == min over the training rows) are removed
    outright -- the paper excludes such "static variables" because they carry
    no reconstruction signal (Sec. 2.1).  Min/max are the training minima and
    maxima (Eq. 10) and are *not* clipped, so out-of-range attack values keep
    their (large) reconstruction error.
    """

    min_: np.ndarray | None = None
    max_: np.ndarray | None = None
    kept_columns: np.ndarray | None = None
    removed_columns: tuple[str, ...] = ()
    n_input_columns: int = 0

    def fit(self, X: np.ndarray, column_names: Sequence[str] | None = None) -> "TrainMinMaxScaler":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D frame, got shape {X.shape}")
        if X.shape[0] == 0:
            raise ValueError("cannot fit a scaler on an empty frame")
        if column_names is None:
            column_names = [f"c{i}" for i in range(X.shape[1])]
        if len(column_names) != X.shape[1]:
            raise ValueError("column_names length does not match the frame width")

        self.n_input_columns = int(X.shape[1])
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        kept = np.flatnonzero(maxs > mins)
        removed_idx = np.flatnonzero(maxs <= mins)
        if kept.size == 0:
            raise ValueError("every channel is constant in the training frame")
        self.min_ = mins[kept].copy()
        self.max_ = maxs[kept].copy()
        self.kept_columns = kept.copy()
        self.removed_columns = tuple(str(column_names[i]) for i in removed_idx)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None or self.kept_columns is None:
            raise RuntimeError("scaler is not fitted")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D frame, got shape {X.shape}")
        if X.shape[1] != self.n_input_columns:
            raise ValueError(
                f"frame width {X.shape[1]} does not match the fitted width "
                f"{self.n_input_columns}"
            )
        subset = X[:, self.kept_columns]
        scale = self.max_ - self.min_
        return (subset - self.min_) / scale

    def state_dict(self) -> dict[str, Any]:
        return {
            "min_": self.min_,
            "max_": self.max_,
            "kept_columns": self.kept_columns,
            "removed_columns": list(self.removed_columns),
            "n_input_columns": self.n_input_columns,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "TrainMinMaxScaler":
        scaler = cls()
        scaler.min_ = np.asarray(state["min_"], dtype=np.float64)
        scaler.max_ = np.asarray(state["max_"], dtype=np.float64)
        scaler.kept_columns = np.asarray(state["kept_columns"], dtype=np.int64)
        scaler.removed_columns = tuple(state.get("removed_columns", ()))
        scaler.n_input_columns = int(state["n_input_columns"])
        return scaler


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class SamePadConv1d(nn.Module):
    """``nn.Conv1d`` with TensorFlow ``"same"`` padding semantics.

    PyTorch rejects even kernel sizes for ``padding="same"``; TensorFlow and
    Keras (the framework named in Sec. 3.1, p. 1287) accept them.  The paper
    does not state a padding mode, so ``same`` is a recorded choice needed to
    keep Table 3's length-1 window alive with kernel 48.  The padding is
    therefore applied explicitly::

        out_len = ceil(L / stride)
        pad_total = max((out_len - 1) * stride + kernel - L, 0)
        pad_left = pad_total // 2

    The convolution is executed as an equivalent 2-D convolution over a
    trailing singleton axis.  Mathematically this is identical to
    ``nn.Conv1d``, but the MPS backend aborts (``Weights tensor and ndArray
    input channel mismatch``) when a ``conv1d`` receives a non-contiguous
    gradient -- which is exactly what the (channels-last) encoder/decoder
    produce.  The 4-D formulation is portable across CPU, CUDA and MPS.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.conv = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolution backends require a contiguous input; the encoder feeds a
        # transposed (window, features) tensor, so materialise it here.
        x = x.contiguous()
        length = x.shape[-1]
        out_len = math.ceil(length / self.stride)
        pad_total = max((out_len - 1) * self.stride + self.kernel_size - length, 0)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x4 = x.unsqueeze(-1)  # (B, C, L, 1)
        if pad_total:
            x4 = F.pad(x4, (0, 0, pad_left, pad_right))
        weight = self.conv.weight.unsqueeze(-1)  # (out, in, k, 1)
        y = F.conv2d(x4, weight, bias=self.conv.bias, stride=(self.stride, 1))
        return y.squeeze(-1)

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding=same"
        )


class ConvBiLSTMAENet(nn.Module):
    """ConvBiLSTM-AE network (paper Table 3, symmetric decoder).

    Tensor layout: ``(batch, window, features)``.

    Encoder: ``Conv1D(conv_filters, conv_kernel_size)`` over the time axis ->
    ``Dense(dense_units)`` -> ``BiLSTM(lstm_units)`` (last timestep).
    Decoder: ``RepeatVector`` -> ``BiLSTM(lstm_units)`` -> ``Dense(dense_units)``
    -> ``Conv1D(conv_filters, conv_kernel_size)`` -> ``Dense(n_features)``.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int = 1,
        conv_filters: int = 132,
        conv_kernel_size: int = 48,
        lstm_units: int = 32,
        dense_units: int = 66,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        self.conv_filters = int(conv_filters)
        self.conv_kernel_size = int(conv_kernel_size)
        self.lstm_units = int(lstm_units)
        self.dense_units = int(dense_units)

        self.encoder_conv = SamePadConv1d(self.n_features, self.conv_filters, self.conv_kernel_size)
        self.encoder_dense = nn.Linear(self.conv_filters, self.dense_units)
        self.encoder_lstm = nn.LSTM(
            self.dense_units, self.lstm_units, batch_first=True, bidirectional=True
        )
        self.decoder_lstm = nn.LSTM(
            self.lstm_units * 2, self.lstm_units, batch_first=True, bidirectional=True
        )
        self.decoder_dense = nn.Linear(self.lstm_units * 2, self.dense_units)
        self.decoder_conv = SamePadConv1d(self.dense_units, self.conv_filters, self.conv_kernel_size)
        self.output_dense = nn.Linear(self.conv_filters, self.n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (batch, window, features), got {tuple(x.shape)}")
        if x.shape[-1] != self.n_features:
            raise ValueError(
                f"expected {self.n_features} features, got {x.shape[-1]}"
            )
        h = self.encoder_conv(x.transpose(1, 2)).transpose(1, 2)   # (B, W, filters)
        h = F.relu(self.encoder_dense(h))                          # (B, W, dense)
        encoded, _ = self.encoder_lstm(h)                          # (B, W, 2*units)
        latent = encoded[:, -1, :]                                 # (B, 2*units)
        repeated = latent.unsqueeze(1).expand(-1, self.window_size, -1)
        decoded, _ = self.decoder_lstm(repeated)                   # (B, W, 2*units)
        decoded = F.relu(self.decoder_dense(decoded))              # (B, W, dense)
        decoded = self.decoder_conv(decoded.transpose(1, 2)).transpose(1, 2)
        return self.output_dense(decoded)                          # (B, W, features)

    def architecture_summary(self) -> dict[str, Any]:
        """Machine-readable description of the built architecture."""
        return {
            "n_features": self.n_features,
            "window_size": self.window_size,
            "recurrent_cell": {
                "type": "LSTM",
                "gate_activation": "sigmoid",
                "candidate_and_hidden_activation": "tanh",
                "bidirectional": True,
                "units_per_direction": self.lstm_units,
            },
            "encoder": {
                "order": ["Conv1D", "Dense", "BiLSTM"],
                "conv1d_filters": self.encoder_conv.out_channels,
                "conv1d_kernel_size": self.encoder_conv.kernel_size,
                "conv1d_padding": "same",
                "dense_units": self.encoder_dense.out_features,
                "dense_activation": "relu",
                "lstm_units_per_direction": self.lstm_units,
                "lstm_bidirectional": True,
            },
            "decoder": {
                "order": ["RepeatVector", "BiLSTM", "Dense", "Conv1D", "Dense"],
                "repeat_vector": self.window_size,
                "lstm_units_per_direction": self.lstm_units,
                "lstm_bidirectional": True,
                "dense_units": self.decoder_dense.out_features,
                "dense_activation": "relu",
                "conv1d_filters": self.decoder_conv.out_channels,
                "conv1d_kernel_size": self.decoder_conv.kernel_size,
                "conv1d_padding": "same",
                "output_units": self.n_features,
                "output_activation": "linear",
            },
            "n_parameters": sum(p.numel() for p in self.parameters()),
        }


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------
@dataclass
class PredictionResult:
    """Scores and binary decisions produced by :meth:`ConvBiLSTMAE.predict`."""

    errors: np.ndarray            # reconstruction error per window
    predictions: np.ndarray       # 1 = anomaly, per window
    row_errors: np.ndarray        # reconstruction error per input row (NaN padded)
    row_predictions: np.ndarray   # 1 = anomaly, per input row
    threshold: float
    n_rows: int
    n_windows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "errors": self.errors,
            "predictions": self.predictions,
            "row_errors": self.row_errors,
            "row_predictions": self.row_predictions,
            "threshold": self.threshold,
            "n_rows": self.n_rows,
            "n_windows": self.n_windows,
        }


# --------------------------------------------------------------------------
# hardware / software metadata + timing helpers
# --------------------------------------------------------------------------
def _device_label(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Metal (MPS)"
    return platform.processor() or platform.machine() or "CPU"


def environment_metadata(device: torch.device | str | None = None) -> dict[str, Any]:
    """Explicit hardware / software metadata for every produced artifact."""
    if isinstance(device, str):
        device = torch.device(device)
    if device is None:
        device = torch.device("cpu")
    try:
        import sklearn  # noqa: PLC0415 — local import keeps the metadata helper self-contained

        sklearn_version = sklearn.__version__
    except Exception:  # pragma: no cover - sklearn is a hard dependency
        sklearn_version = "unavailable"

    hardware: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "compute_device_type": device.type,
        "compute_device_name": _device_label(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index or 0
        hardware["cuda_device_count"] = torch.cuda.device_count()
        hardware["cuda_device_capability"] = list(torch.cuda.get_device_capability(idx))
        hardware["cuda_total_memory_bytes"] = int(
            torch.cuda.get_device_properties(idx).total_memory
        )
    software: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn_version,
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }
    return {"hardware": hardware, "software": software}


def synchronize_device(device: torch.device | str) -> None:
    """Block until every queued CUDA/MPS kernel has completed.

    Timing helpers must call this before and after a measured region, otherwise
    asynchronous GPU/MPS execution makes the measurement meaningless.
    """
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def measure_single_event_latency(
    model: nn.Module,
    sample: torch.Tensor,
    *,
    device: torch.device | str = "cpu",
    warmup: int = 5,
    repeats: int = 50,
) -> dict[str, Any]:
    """Latency of scoring **one** event (a single window) end to end.

    ``sample`` must have shape ``(window, features)`` (no batch axis).  The
    returned milliseconds are measured on the host clock with the accelerator
    synchronised around every timed region, so they include kernel launch and
    synchronisation overhead -- i.e. they are the latency a single event would
    experience in a monitoring loop, not a throughput number.
    """
    if sample.dim() == 3 and sample.shape[0] == 1:
        sample = sample[0]
    if sample.dim() != 2:
        raise ValueError(f"expected a (window, features) sample, got {tuple(sample.shape)}")
    device = torch.device(device)
    batch = sample.unsqueeze(0).to(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(max(int(warmup), 0)):
            model(batch)
        synchronize_device(device)
        timings_ms: list[float] = []
        for _ in range(max(int(repeats), 1)):
            synchronize_device(device)
            start = time.perf_counter()
            model(batch)
            synchronize_device(device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    arr = np.asarray(timings_ms, dtype=np.float64)
    return {
        "measurement": "single_event_latency",
        "unit": "ms",
        "batch_size": 1,
        "n_repeats": int(arr.size),
        "warmup": int(warmup),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "std_ms": float(arr.std(ddof=0)),
        "device": str(device),
    }


def measure_batch_throughput(
    model: nn.Module,
    batch: torch.Tensor,
    *,
    device: torch.device | str = "cpu",
    warmup: int = 2,
    repeats: int = 10,
) -> dict[str, Any]:
    """Throughput of scoring a **batch** of events (windows) at once.

    ``batch`` must have shape ``(batch, window, features)``.  This is the
    amortised cost per event when events are processed in batches, which is
    different from -- and typically much better than -- the single-event
    latency measured by :func:`measure_single_event_latency`.
    """
    if batch.dim() != 3:
        raise ValueError(f"expected a (batch, window, features) batch, got {tuple(batch.shape)}")
    device = torch.device(device)
    batch = batch.to(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(max(int(warmup), 0)):
            model(batch)
        synchronize_device(device)
        timings_ms: list[float] = []
        for _ in range(max(int(repeats), 1)):
            synchronize_device(device)
            start = time.perf_counter()
            model(batch)
            synchronize_device(device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    arr = np.asarray(timings_ms, dtype=np.float64)
    n_events = int(batch.shape[0])
    median_ms = float(np.median(arr))
    return {
        "measurement": "batch_throughput",
        "unit": "ms",
        "batch_size": n_events,
        "n_repeats": int(arr.size),
        "warmup": int(warmup),
        "median_ms": median_ms,
        "mean_ms": float(arr.mean()),
        "p95_ms": float(np.percentile(arr, 95)),
        "events_per_second": float(n_events / (median_ms / 1000.0)) if median_ms > 0 else float("inf"),
        "ms_per_event": float(median_ms / n_events) if n_events else float("nan"),
        "device": str(device),
    }


# --------------------------------------------------------------------------
# the baseline
# --------------------------------------------------------------------------
def _is_pandas_frame(X: Any) -> bool:
    return hasattr(X, "columns") and hasattr(X, "loc") and hasattr(X, "select_dtypes")


def _normalise_column(name: Any) -> str:
    return str(name).strip().casefold()


def _select_dataframe_features(
    frame: Any,
    non_feature_columns: Sequence[str],
    feature_columns: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Select numeric feature columns from a DataFrame by training schema.

    With ``feature_columns=None`` the schema is inferred: numeric, non-boolean
    columns whose names are not in ``non_feature_columns``.  With an explicit
    schema the same names must exist and be numeric; extra columns are
    ignored.  In both cases the returned ``dropped`` list records every input
    column that was not used.
    """
    import pandas as pd  # local import: arrays do not require pandas

    excluded = {_normalise_column(c) for c in non_feature_columns}
    if feature_columns is None:
        selected: list[str] = []
        for column in frame.columns:
            if _normalise_column(column) in excluded:
                continue
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                selected.append(str(column))
        if not selected:
            raise ValueError("no numeric feature columns remain after schema selection")
    else:
        selected = [str(c) for c in feature_columns]
        missing = [c for c in selected if c not in frame.columns]
        if missing:
            raise ValueError(
                "frame is missing fitted feature columns: " + ", ".join(missing)
            )
        non_numeric = [
            c
            for c in selected
            if not pd.api.types.is_numeric_dtype(frame[c])
            or pd.api.types.is_bool_dtype(frame[c])
        ]
        if non_numeric:
            raise ValueError(
                "fitted feature columns are not numeric: " + ", ".join(non_numeric)
            )

    values = frame.loc[:, selected].to_numpy(dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D frame, got shape {values.shape}")
    used = set(selected)
    dropped = [str(c) for c in frame.columns if str(c) not in used]
    return values, selected, dropped


def _windowize(rows: np.ndarray, window_size: int) -> np.ndarray:
    """Sliding windows with stride 1: ``(n - w + 1, w, features)``."""
    if window_size == 1:
        return rows[:, None, :]
    n_windows = rows.shape[0] - window_size + 1
    if n_windows < 1:
        raise ValueError(
            f"not enough rows ({rows.shape[0]}) for window_size={window_size}"
        )
    strides = rows.strides
    shape = (n_windows, window_size, rows.shape[1])
    return np.lib.stride_tricks.as_strided(rows, shape=shape, strides=(strides[0], strides[0], strides[1]))


def select_oracle_threshold(
    y_true: Any,
    scores: Any,
    criterion: str = "f1",
) -> dict[str, Any]:
    """Select the paper's test-optimal threshold **using test labels**.

    This is an explicitly leaky oracle analysis, not a deployable rule.  The
    prediction convention matches :meth:`ConvBiLSTMAE.predict`: a window is
    anomalous iff ``score > threshold``.  Candidate thresholds are every
    distinct score plus one step below the minimum and above the maximum, so
    the all-positive and all-normal decisions are both reachable.
    """
    if criterion not in ORACLE_CRITERIA:
        raise ValueError(
            f"criterion must be one of {ORACLE_CRITERIA}, got {criterion!r}"
        )
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size != s.size:
        raise ValueError(f"labels length {y.size} does not match scores length {s.size}")
    if y.size == 0:
        raise ValueError("cannot select a threshold on empty scores")
    if not np.all(np.isfinite(s)):
        raise ValueError("oracle threshold requires finite scores")

    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    total_pos = int(y_sorted.sum())
    cumulative_pos = np.concatenate(([0], np.cumsum(y_sorted)))
    unique = np.unique(s_sorted)
    candidates = np.concatenate(
        ([np.nextafter(unique[0], -np.inf)], unique, [np.nextafter(unique[-1], np.inf)])
    )

    # Number of scores <= candidate: everything to the right is predicted 1.
    idx = np.searchsorted(s_sorted, candidates, side="right")
    predicted_pos = s.size - idx
    tp = total_pos - cumulative_pos[idx]
    fp = predicted_pos - tp
    fn = total_pos - tp
    tn = s.size - predicted_pos - fn

    if criterion == "f1":
        denom = 2 * tp + fp + fn
        metric = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    elif criterion == "accuracy":
        metric = (tp + tn) / s.size
    elif criterion == "balanced_accuracy":
        tpr = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
        tnr = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) > 0)
        metric = 0.5 * (tpr + tnr)
    else:  # mcc
        denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        metric = np.divide(
            tp * tn - fp * fn,
            np.sqrt(denom, dtype=float),
            out=np.zeros_like(tp, dtype=float),
            where=denom > 0,
        )

    best = int(np.argmax(metric))
    return {
        "threshold": float(candidates[best]),
        "criterion": criterion,
        "criterion_value": float(metric[best]),
        "n_scored": int(s.size),
        "n_positive": total_pos,
        "tp": int(tp[best]),
        "fp": int(fp[best]),
        "tn": int(tn[best]),
        "fn": int(fn[best]),
    }


def _classification_metrics(
    y_true: np.ndarray, scores: np.ndarray, y_pred: np.ndarray
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "aucpr": float(average_precision_score(y_true, scores)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    out["average_precision"] = out["aucpr"]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)})
    out["accuracy"] = float((tp + tn) / max(y_true.size, 1))
    if len(np.unique(y_true)) > 1:
        out["aucroc"] = float(roc_auc_score(y_true, scores))
    else:
        out["aucroc"] = float("nan")
    return out


def _resolve_device(request: str) -> torch.device:
    if request == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device='mps' requested but MPS is not available")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - older torch builds
        pass


class ConvBiLSTMAE:
    """ConvBiLSTM-AE anomaly detector.

    Lifecycle: :meth:`fit` (train frames + validation frames, both normal
    only) -> :meth:`predict` / :meth:`score` (test frames) -> :meth:`save` /
    :meth:`load`.

    The scaler, the network weights and the decision threshold are estimated
    **only** from the training and validation frames; ``predict``/``score``
    never re-fit anything.
    """

    def __init__(self, config: ConvBiLSTMAEConfig | None = None):
        self.config = config or ConvBiLSTMAEConfig()
        self.config.validate()
        self.device = _resolve_device(self.config.device)
        self.scaler = TrainMinMaxScaler()
        self.model: ConvBiLSTMAENet | None = None
        self.threshold: float | None = None
        self.history: list[dict[str, float]] = []
        self.train_rows: int = 0
        self.validation_rows: int = 0
        self.n_features_in: int = 0
        self.column_names: list[str] = []
        self.feature_columns: list[str] = []
        self.dropped_columns: tuple[str, ...] = ()
        self._feature_schema_from_dataframe: bool = False
        self._last_result: PredictionResult | None = None
        self._last_predictions: dict[str, np.ndarray] | None = None

    # -- properties ------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self.model is not None and self.threshold is not None

    # -- input schema ----------------------------------------------------
    def _prepare_features(
        self,
        X: Any,
        *,
        fitting: bool = False,
        feature_columns: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Return numeric feature values and the column schema.

        DataFrames are the normal path for the full-signal adapter: on
        ``fit`` the schema is inferred from the training frame only (numeric
        columns minus ``non_feature_columns``), and every later frame is
        projected onto those stored names.  Numpy arrays are positional and
        must already contain only feature columns.  No test/scenario frame
        ever changes the schema.
        """
        if _is_pandas_frame(X):
            if fitting:
                values, columns, dropped = _select_dataframe_features(
                    X, self.config.non_feature_columns, feature_columns
                )
                self.feature_columns = list(columns)
                self.dropped_columns = tuple(dropped)
                self._feature_schema_from_dataframe = True
                return values, list(columns)
            if self._feature_schema_from_dataframe:
                values, columns, _ = _select_dataframe_features(
                    X, self.config.non_feature_columns, self.feature_columns
                )
                return values, list(columns)
            values, columns, _ = _select_dataframe_features(
                X, self.config.non_feature_columns, None
            )
            if len(columns) != len(self.feature_columns):
                raise ValueError(
                    f"frame has {len(columns)} usable numeric columns but the "
                    f"fitted schema has {len(self.feature_columns)}"
                )
            return values, list(self.feature_columns)

        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"expected a 2-D frame, got shape {values.shape}")
        if fitting:
            if feature_columns is None:
                columns = [f"c{i}" for i in range(values.shape[1])]
            else:
                columns = [str(c) for c in feature_columns]
                if len(columns) != values.shape[1]:
                    raise ValueError(
                        "feature_columns length does not match the array width"
                    )
            self.feature_columns = list(columns)
            self.dropped_columns = ()
            self._feature_schema_from_dataframe = False
            return values, list(columns)
        if values.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"frame width {values.shape[1]} does not match the fitted schema "
                f"width {len(self.feature_columns)}"
            )
        return values, list(self.feature_columns)

    # -- fitting ---------------------------------------------------------
    def fit(
        self,
        train: Any,
        validation: Any,
        *,
        feature_columns: Sequence[str] | None = None,
    ) -> "ConvBiLSTMAE":
        """Train the autoencoder on normal frames and pick the threshold.

        ``train`` and ``validation`` are 2-D normal-only frames (rows =
        timesteps, columns = channels).  They may be numpy arrays or DataFrames
        from :mod:`dataio.full_signal`; with DataFrames the feature schema is
        selected from ``train`` only.  The scaler is fitted on ``train`` only;
        the threshold is the ``threshold_quantile`` quantile of the validation
        reconstruction errors.
        """
        cfg = self.config
        cfg.validate()
        _set_seed(cfg.seed)

        train_values, train_columns = self._prepare_features(
            train, fitting=True, feature_columns=feature_columns
        )
        val_values, _ = self._prepare_features(validation)
        if train_values.shape[1] != val_values.shape[1]:
            raise ValueError(
                "train and validation frames must have the same number of channels "
                f"({train_values.shape[1]} vs {val_values.shape[1]})"
            )
        self.column_names = train_columns
        self.n_features_in = int(train_values.shape[1])
        self.train_rows = int(train_values.shape[0])
        self.validation_rows = int(val_values.shape[0])

        self.scaler = TrainMinMaxScaler().fit(train_values, train_columns)
        n_features = int(self.scaler.kept_columns.size)

        train_windows = _windowize(self.scaler.transform(train_values), cfg.window_size)
        val_windows = _windowize(self.scaler.transform(val_values), cfg.window_size)

        self.model = ConvBiLSTMAENet(
            n_features=n_features,
            window_size=cfg.window_size,
            conv_filters=cfg.conv_filters,
            conv_kernel_size=cfg.conv_kernel_size,
            lstm_units=cfg.lstm_units,
            dense_units=cfg.dense_units,
        ).to(self.device)

        train_tensor = torch.from_numpy(np.ascontiguousarray(train_windows)).float().to(self.device)
        val_tensor = torch.from_numpy(np.ascontiguousarray(val_windows)).float().to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        rng = np.random.default_rng(cfg.seed)
        best_val = math.inf
        best_state: dict[str, torch.Tensor] | None = None
        patience_left = cfg.early_stopping_patience

        self.history = []
        for epoch in range(cfg.epochs):
            self.model.train()
            order = rng.permutation(train_tensor.shape[0])
            epoch_loss = 0.0
            n_seen = 0
            for start in range(0, len(order), cfg.batch_size):
                idx = order[start : start + cfg.batch_size]
                batch = train_tensor[idx]
                optimizer.zero_grad(set_to_none=True)
                reconstructed = self.model(batch)
                loss = F.mse_loss(reconstructed, batch)
                loss.backward()
                if self.config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(idx)
                n_seen += len(idx)
            train_loss = epoch_loss / max(n_seen, 1)
            val_loss = float(self._mean_loss(val_tensor))
            self.history.append(
                {"epoch": float(epoch + 1), "train_loss": train_loss, "val_loss": val_loss}
            )
            if val_loss < best_val - 1e-12:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                patience_left = cfg.early_stopping_patience
            elif patience_left is not None:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if cfg.early_stopping_patience is not None and best_state is not None:
            self.model.load_state_dict(best_state)

        # Threshold: validation-only, declared normal-coverage quantile.
        val_errors = self.reconstruction_errors(val_windows)
        self.threshold = float(np.quantile(val_errors, cfg.threshold_quantile))
        self._last_result = None
        self._last_predictions = None
        return self

    def _mean_loss(self, tensor: torch.Tensor) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, tensor.shape[0], self.config.batch_size):
                batch = tensor[start : start + self.config.batch_size]
                loss = F.mse_loss(self.model(batch), batch)
                total += float(loss) * len(batch)
                count += len(batch)
        self.model.train()
        return total / max(count, 1)

    # -- inference -------------------------------------------------------
    def windowize(self, X: Any) -> np.ndarray:
        """Scale a frame with the fitted scaler and build sliding windows.

        Returns ``(n_windows, window_size, n_kept_channels)``.  The scaler is
        *applied*, never re-fitted, and windows never cross frame/scenario
        boundaries because each call windows one frame independently.
        """
        if self.scaler.min_ is None:
            raise RuntimeError("the scaler is not fitted")
        values, _ = self._prepare_features(X)
        return _windowize(self.scaler.transform(values), self.config.window_size)

    def reconstruction_errors(self, windows: np.ndarray) -> np.ndarray:
        """Pointwise reconstruction MSE (paper Eq. 11) for each window."""
        if self.model is None:
            raise RuntimeError("model is not fitted")
        tensor = torch.from_numpy(np.ascontiguousarray(windows)).float().to(self.device)
        self.model.eval()
        errors: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, tensor.shape[0], self.config.batch_size):
                batch = tensor[start : start + self.config.batch_size]
                reconstructed = self.model(batch)
                residual = reconstructed.double() - batch.double()
                per_sample = (residual ** 2).mean(dim=(1, 2))
                errors.append(per_sample.detach().cpu().numpy().astype(np.float64))
        return np.concatenate(errors) if errors else np.zeros(0, dtype=np.float64)

    def predict(self, X: Any) -> PredictionResult:
        """Score frames; no re-fitting happens here."""
        if not self.is_fitted:
            raise RuntimeError("call fit() before predict()")
        values, _ = self._prepare_features(X)
        windows = _windowize(self.scaler.transform(values), self.config.window_size)
        errors = self.reconstruction_errors(windows)
        predictions = (errors > float(self.threshold)).astype(np.int64)

        n_rows = values.shape[0]
        w = self.config.window_size
        row_errors = np.full(n_rows, np.nan, dtype=np.float64)
        row_predictions = np.zeros(n_rows, dtype=np.int64)
        if w == 1:
            row_errors[:] = errors
            row_predictions[:] = predictions
        else:
            row_errors[w - 1 :] = errors
            row_predictions[w - 1 :] = predictions

        result = PredictionResult(
            errors=errors,
            predictions=predictions,
            row_errors=row_errors,
            row_predictions=row_predictions,
            threshold=float(self.threshold),
            n_rows=n_rows,
            n_windows=int(errors.size),
        )
        self._last_result = result
        self._last_predictions = {
            "errors": errors,
            "predictions": predictions,
            "row_errors": row_errors,
            "row_predictions": row_predictions,
        }
        return result

    def _scored_rows(
        self, X: Any, y: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, PredictionResult]:
        labels = np.asarray(y).astype(np.int64).reshape(-1)
        result = self.predict(X)
        if labels.size != result.n_rows:
            raise ValueError(
                f"labels length {labels.size} does not match frame rows {result.n_rows}"
            )
        mask = ~np.isnan(result.row_errors)
        return labels[mask], result.row_errors[mask], result.row_predictions[mask], result

    def score(self, X: Any, y: Any) -> dict[str, Any]:
        """Attack-positive metrics at the **leak-free** primary threshold.

        ``y`` holds the per-row attack labels (1 = attack).  Leading rows that
        have no complete window (``window_size > 1``) are excluded.  The
        threshold comes from normal validation errors; test labels are used
        only to compute the reported metrics, never to choose the threshold.
        """
        y_true, scores, y_pred, _ = self._scored_rows(X, y)
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError(
                "non-finite reconstruction error; refusing to compute detection metrics"
            )
        out: dict[str, Any] = {
            "n_scored": int(y_true.size),
            "n_positive": int(y_true.sum()),
            "threshold": float(self.threshold),
            "threshold_quantile": float(self.config.threshold_quantile),
            "threshold_policy": "validation_quantile",
            "prevalence": float(y_true.mean()) if y_true.size else float("nan"),
        }
        if y_true.size == 0:
            return out
        out.update(_classification_metrics(y_true, scores, y_pred))
        return out

    def score_oracle(
        self,
        X: Any,
        y: Any,
        criterion: str | None = None,
    ) -> dict[str, Any]:
        """Explicit oracle analysis using the paper's test-optimal threshold.

        This *does* read test labels to choose the threshold, exactly as the
        paper's Sec. 3.2 reporting does.  It never changes ``self.threshold``
        and is not used by :meth:`fit`, :meth:`predict` or :meth:`score`; use
        it only to compare against the paper's reported operating points.
        """
        criterion = criterion or self.config.oracle_criterion
        y_true, scores, _, _ = self._scored_rows(X, y)
        out: dict[str, Any] = {
            "n_scored": int(y_true.size),
            "n_positive": int(y_true.sum()),
            "threshold_policy": "test_oracle_explicit",
            "leakage_warning": (
                "oracle leakage: threshold selected on labelled test scores; "
                "not deployable and not used by fit/predict/score"
            ),
            "primary_threshold": float(self.threshold),
            "threshold_quantile": float(self.config.threshold_quantile),
            "oracle_criterion": criterion,
            "prevalence": float(y_true.mean()) if y_true.size else float("nan"),
        }
        if y_true.size == 0:
            return out
        oracle = select_oracle_threshold(y_true, scores, criterion=criterion)
        y_pred = (scores > oracle["threshold"]).astype(np.int64)
        out["threshold"] = float(oracle["threshold"])
        out["oracle_criterion_value"] = float(oracle["criterion_value"])
        out["oracle"] = oracle
        out.update(_classification_metrics(y_true, scores, y_pred))
        return out

    # -- metadata --------------------------------------------------------
    def metadata(self) -> dict[str, Any]:
        """Hardware/software metadata plus the full run configuration."""
        meta = environment_metadata(self.device)
        meta["config"] = self.config.to_dict()
        meta["fidelity"] = self.config.fidelity
        meta["paper_spec"] = dict(PAPER_SPEC)
        meta["implementation_notes"] = dict(IMPLEMENTATION_NOTES)
        meta["unspecified_choices"] = dict(UNSPECIFIED_CHOICES)
        meta["seed"] = int(self.config.seed)
        meta["dataset"] = {
            "n_input_channels": self.n_features_in,
            "n_kept_channels": int(self.scaler.kept_columns.size)
            if self.scaler.kept_columns is not None
            else 0,
            "feature_columns": list(self.feature_columns),
            "feature_schema_source": (
                "dataframe" if self._feature_schema_from_dataframe else "array"
            ),
            "dropped_non_feature_columns": list(self.dropped_columns),
            "removed_constant_channels": list(self.scaler.removed_columns),
            "n_train_rows": self.train_rows,
            "n_validation_rows": self.validation_rows,
            "window_size": self.config.window_size,
        }
        meta["architecture"] = self.model.architecture_summary() if self.model else None
        meta["threshold"] = self.threshold
        meta["reconstruction_error_precision"] = (
            "float32 model inference; float64 residual subtraction, squaring, "
            "and mean reduction"
        )
        meta["threshold_rule"] = (
            f"quantile({self.config.threshold_quantile}) of validation reconstruction errors"
        )
        meta["threshold_oracle_rule"] = (
            "test-label-optimal threshold available only via score_oracle"
        )
        meta["history"] = self.history
        return meta

    # -- persistence -----------------------------------------------------
    def save(self, directory: str | os.PathLike[str]) -> Path:
        """Persist the fitted baseline (weights, scaler, config, metadata)."""
        if not self.is_fitted:
            raise RuntimeError("nothing to save: the baseline is not fitted")
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                "config": self.config.to_dict(),
                "n_features": int(self.scaler.kept_columns.size),
            },
            out_dir / "model.pt",
        )
        np.savez(
            out_dir / "scaler.npz",
            min_=self.scaler.min_,
            max_=self.scaler.max_,
            kept_columns=self.scaler.kept_columns,
            removed_columns=np.asarray(self.scaler.removed_columns, dtype=object),
            n_input_columns=np.asarray([self.scaler.n_input_columns]),
        )
        (out_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        metadata = self.metadata()
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
        )
        if self._last_predictions is not None:
            np.savez(out_dir / "predictions.npz", **self._last_predictions)
        digest = _sha256(out_dir / "model.pt")
        (out_dir / "checksums.json").write_text(
            json.dumps({"model.pt": digest}, indent=2, sort_keys=True), encoding="utf-8"
        )
        return out_dir

    @classmethod
    def load(cls, directory: str | os.PathLike[str]) -> "ConvBiLSTMAE":
        """Restore a baseline saved with :meth:`save`."""
        in_dir = Path(directory)
        blob = torch.load(in_dir / "model.pt", map_location="cpu", weights_only=True)
        config = ConvBiLSTMAEConfig(**blob["config"])
        obj = cls(config)
        obj.device = _resolve_device(config.device)
        scaler_state = np.load(in_dir / "scaler.npz", allow_pickle=True)
        obj.scaler = TrainMinMaxScaler.from_state_dict(
            {
                "min_": scaler_state["min_"],
                "max_": scaler_state["max_"],
                "kept_columns": scaler_state["kept_columns"],
                "removed_columns": [str(c) for c in scaler_state["removed_columns"].tolist()],
                "n_input_columns": int(scaler_state["n_input_columns"][0]),
            }
        )
        obj.n_features_in = obj.scaler.n_input_columns
        obj.model = ConvBiLSTMAENet(
            n_features=int(blob["n_features"]),
            window_size=config.window_size,
            conv_filters=config.conv_filters,
            conv_kernel_size=config.conv_kernel_size,
            lstm_units=config.lstm_units,
            dense_units=config.dense_units,
        ).to(obj.device)
        obj.model.load_state_dict(blob["state_dict"])
        metadata_path = in_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            obj.threshold = float(metadata["threshold"])
            obj.history = metadata.get("history", [])
            dataset = metadata.get("dataset", {})
            obj.train_rows = int(dataset.get("n_train_rows", 0))
            obj.validation_rows = int(dataset.get("n_validation_rows", 0))
            stored_columns = dataset.get("feature_columns")
            if stored_columns:
                obj.feature_columns = [str(c) for c in stored_columns]
            else:
                width = int(dataset.get("n_input_channels", obj.scaler.n_input_columns))
                obj.feature_columns = [f"c{i}" for i in range(width)]
            obj.n_features_in = int(
                dataset.get("n_input_channels", len(obj.feature_columns))
            )
            obj.dropped_columns = tuple(
                str(c) for c in dataset.get("dropped_non_feature_columns", ())
            )
            obj._feature_schema_from_dataframe = (
                dataset.get("feature_schema_source") == "dataframe"
            )
        else:
            obj.feature_columns = [f"c{i}" for i in range(obj.scaler.n_input_columns)]
            obj.n_features_in = obj.scaler.n_input_columns
            obj._feature_schema_from_dataframe = False
        predictions_path = in_dir / "predictions.npz"
        if predictions_path.exists():
            data = np.load(predictions_path)
            obj._last_predictions = {k: data[k] for k in data.files}
        return obj


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# convenience entry point
# --------------------------------------------------------------------------
def run_baseline(
    train: Any,
    validation: Any,
    test: Any,
    y_test: Any,
    config: ConvBiLSTMAEConfig | None = None,
    *,
    artifact_dir: str | os.PathLike[str] | None = None,
    latency_repeats: int = 50,
    throughput_repeats: int = 10,
) -> dict[str, Any]:
    """Fit, score, time and (optionally) persist the ConvBiLSTM-AE baseline.

    Returns a dictionary with ``metrics``, ``timing`` and ``metadata`` keys.
    """
    baseline = ConvBiLSTMAE(config)
    baseline.fit(train, validation)
    metrics = baseline.score(test, y_test)

    windows = baseline.windowize(test)
    tensor = torch.from_numpy(np.ascontiguousarray(windows)).float()
    timing = {
        "single_event_latency": measure_single_event_latency(
            baseline.model,
            tensor[0],
            device=baseline.device,
            repeats=latency_repeats,
        ),
        "batch_throughput": measure_batch_throughput(
            baseline.model,
            tensor,
            device=baseline.device,
            repeats=throughput_repeats,
        ),
    }
    metadata = baseline.metadata()
    metadata["timing"] = timing
    if artifact_dir is not None:
        baseline.save(artifact_dir)
    return {"metrics": metrics, "timing": timing, "metadata": metadata, "baseline": baseline}
