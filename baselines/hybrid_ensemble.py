"""BATADAL hybrid ensemble baseline (Ahmed, arXiv:2512.14422).

This module re-implements the hybrid stacked ensemble described in the full
paper *"Hybrid Ensemble Method for Detecting Cyber-Attacks in Water
Distribution Systems Using the BATADAL Dataset"* (W. Ahmed,
arXiv:2512.14422v1, 16 Dec 2025).  It was written from the paper text
(Sec. III.B--III.G and Sec. IV), **not** from summary tables quoted
elsewhere, and it does not reuse any third-party re-implementation.

Paper protocol
--------------
* a preliminary Random Forest is trained on the *raw* numeric sensors and
  the top-10 features by importance are kept;
* for every selected feature the paper engineers ``lag1``, ``lag3``, the
  first difference ``d1``, a 5-hour rolling mean and a 5-hour rolling
  standard deviation (43 sensors -> 93 columns);
* 80/20 stratified train/validation split;
* SMOTE is applied to the *training* split only;
* Random Forest (300 trees) and XGBoost (500 estimators, ``max_depth=6``,
  ``learning_rate=0.05``) on the standardised matrix;
* a single 64-unit LSTM layer with 20% dropout and a sigmoid output, trained
  on sequences of 10 consecutive steps with class weights inversely
  proportional to the class frequencies, on the min-max scaled matrix;
* the three attack-class probabilities feed a logistic-regression
  meta-learner (stacking).

Leak-free protocol used here (every deviation is listed in
:data:`ADAPTATIONS` and copied into the persisted manifest)
---------------------------------------------------------------------
* the outer split is **chronological** (no shuffling);
* feature selection is fit on the **training rows only**;
* engineered series are **never backward filled**: only a forward fill is
  used and the rows that still lack history are *dropped*;
* the meta-learner is trained on **out-of-fold** base predictions and SMOTE
  is applied **inside the training folds only** (never to a validation or
  test partition);
* the decision threshold is **fixed at 0.5** -- never tuned, and no model
  is ever refitted on evaluation data;
* every artefact records the seed (42 by default), the input digests and the
  library versions.

Public API
----------
``HybridEnsembleConfig``
    Hyper-parameters (paper values by default).
``HybridEnsemble``
    ``fit`` / ``predict_proba`` / ``predict`` / ``save`` / ``load``.
``evaluate_attack``
    Attack-positive metric bundle (average precision / AUPRC, Matthews
    correlation, *unadjusted* balanced accuracy at the fixed threshold,
    ROC-AUC, ...).
``run_batadal``
    Native runner: raw ``BATADAL_dataset04.csv`` (supervised) + the
    normalized dataset-03 calibration frames, leaving the official BATADAL
    test set untouched.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "ADAPTATIONS",
    "BatadalRunResult",
    "HybridEnsemble",
    "HybridEnsembleConfig",
    "HybridEnsembleError",
    "MissingDependencyError",
    "PredictionResult",
    "build_sequences",
    "engineer_temporal_features",
    "evaluate_attack",
    "run_batadal",
    "select_top_features",
]

#: Version of the persisted bundle layout.
BUNDLE_SCHEMA = "baselines/hybrid_ensemble/bundle/v1"
#: Version of the metric dictionary returned by :func:`evaluate_attack`.
METRIC_SCHEMA = "baselines/hybrid_ensemble/metrics/v1"
#: Default random seed used everywhere (the paper does not state one).
DEFAULT_SEED = 42
#: Fixed decision threshold.  Never tuned.
FIXED_THRESHOLD = 0.5


class HybridEnsembleError(RuntimeError):
    """Unrecoverable error in the hybrid-ensemble baseline."""


class MissingDependencyError(HybridEnsembleError):
    """An optional dependency of the ``[baselines]`` extra is missing."""


#: Every deviation from the paper protocol, with the reason.  The list is
#: embedded verbatim in every persisted manifest.
ADAPTATIONS: tuple[dict[str, str], ...] = (
    {
        "id": "outer_split_chronological",
        "paper": "80/20 stratified train/validation split.",
        "ours": "80/20 chronological split of the labelled series.",
        "reason": "A stratified split lets later (attack) samples inform "
                  "earlier training rows; chronological order is the only "
                  "split that cannot look backwards.",
    },
    {
        "id": "feature_selection_train_only",
        "paper": "Preliminary RF feature selection (top-10) on the data.",
        "ours": "The preliminary RF is fit on training rows only; the "
                "selected feature list is frozen before any validation or "
                "test row is touched.",
        "reason": "Selecting features on the full data leaks validation and "
                  "test information into the model.",
    },
    {
        "id": "no_backward_fill",
        "paper": "Lag/rolling NaNs are forward- and backward-filled.",
        "ours": "Only a forward fill is used; rows that still lack history "
                "are dropped (the first max(rolling_window-1, "
                "sequence_length-1) rows of every partition).",
        "reason": "Backward fill copies future values into the past and is a "
                  "direct look-ahead leak.",
    },
    {
        "id": "out_of_fold_stacking",
        "paper": "The meta-learner is trained on base predictions obtained on "
                 "the (resampled) training data.",
        "ours": "The meta-learner is trained on out-of-fold base predictions "
                "(K folds inside the training partition); the deployed base "
                "models are refitted on the whole training partition.",
        "reason": "In-sample base predictions are over-confident, so the "
                  "meta-learner would learn weights that do not transfer.",
    },
    {
        "id": "smote_inside_training_folds_only",
        "paper": "SMOTE is applied to the training split only.",
        "ours": "SMOTE is applied to the training part of every out-of-fold "
                "split and to the final training partition; no "
                "validation/test partition is ever resampled.",
        "reason": "Resampling an evaluation partition inflates every metric.",
    },
    {
        "id": "smote_not_applied_to_lstm",
        "paper": "SMOTE is applied to the training data that feeds every "
                 "model.",
        "ours": "SMOTE feeds the tabular branch (RF/XGB) only; the LSTM is "
                "trained on the original temporal sequences with the "
                "class-weight balancing described in the paper.",
        "reason": "Synthesising time steps destroys the temporal dependence "
                  "the LSTM is supposed to model.",
    },
    {
        "id": "fixed_threshold",
        "paper": "The threshold is implicitly the default 0.5.",
        "ours": "The threshold is fixed at 0.5 and recorded as untuned.",
        "reason": "Tuning a threshold on an evaluation partition is "
                  "optimistic; the balanced accuracy is reported unadjusted.",
    },
    {
        "id": "metric_set",
        "paper": "ROC-AUC, accuracy, attack precision/recall/F1.",
        "ours": "Attack-positive average precision (AUPRC), Matthews "
                "correlation, unadjusted balanced accuracy at the fixed "
                "threshold, plus ROC-AUC and the attack-class "
                "precision/recall/F1 for comparability with the paper.",
        "reason": "The manuscript under revision reports MCC, AUPRC and "
                  "balanced accuracy on the attack class.",
    },
    {
        "id": "segment_aware_features",
        "paper": "Lag/rolling features are computed on one contiguous series.",
        "ours": "Lag/rolling/difference features and the LSTM sequences are "
                "computed inside segments, so no feature crosses a segment "
                "boundary.",
        "reason": "The native runner concatenates two different BATADAL "
                  "simulations; a lag across the junction would mix "
                  "unrelated trajectories.",
    },
    {
        "id": "dataset_substitution",
        "paper": "BATADAL training datasets 1 and 2 are concatenated.",
        "ours": "The native runner trains on the raw BATADAL dataset 04 "
                "(labelled, ATT_FLAG) and adds the normalized dataset-03 "
                "calibration frames as extra normal-operation rows; the "
                "official BATADAL test set is never fitted on.",
        "reason": "Datasets 01/02 are not distributed with this repository; "
                  "dataset 04 is the labelled supervised source and "
                  "dataset 03 is the clean calibration source.",
    },
)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _import_optional(module: str):
    """Import an optional ``[baselines]`` dependency with a clear message."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the env
        raise MissingDependencyError(
            f"the hybrid-ensemble baseline needs the optional dependency "
            f"{module!r}; install it with `uv sync --extra baselines` ({exc})"
        ) from exc


def _load_backends() -> None:
    """Import the optional backends in a safe order.

    xgboost and torch each ship their own OpenMP runtime.  When torch is
    imported first, the next xgboost call segfaults on some platforms
    (macOS/libgomp).  Importing xgboost first is safe and costs nothing.
    """
    try:
        importlib.import_module("xgboost")
    except ImportError:  # pragma: no cover - depends on the env
        pass
    importlib.import_module("torch")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    _load_backends()
    torch = _import_optional("torch")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # a single torch thread: xgboost and torch ship different OpenMP
    # runtimes and multi-threaded torch can deadlock after xgboost ran
    torch.set_num_threads(1)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def frame_digest(frame: pd.DataFrame) -> str:
    """Stable content digest of a frame (column order + values + position)."""
    payload = frame.to_csv(index=True, lineterminator="\n").encode("utf-8")
    return _sha256_bytes(payload)


def array_digest(values: np.ndarray) -> str:
    """Stable content digest of a label/probability vector."""
    return _sha256_bytes(np.ascontiguousarray(values).tobytes())


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "sklearn", "xgboost", "torch", "imblearn"):
        try:
            mod = importlib.import_module(name)
        except ImportError:  # pragma: no cover - depends on the env
            versions[name] = "missing"
        else:
            versions[name] = str(getattr(mod, "__version__", "unknown"))
    return versions


def _canonical_columns(columns: Sequence[str]) -> dict[str, str]:
    """snake_case canonicalisation of column names (same rule as dataio)."""
    import re

    pattern = re.compile(r"[^0-9a-z]+")
    out: dict[str, str] = {}
    for raw in columns:
        name = pattern.sub("_", str(raw).strip().lower()).strip("_")
        out[str(raw)] = name
    return out


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class HybridEnsembleConfig:
    """Hyper-parameters of the hybrid ensemble (paper values by default)."""

    seed: int = DEFAULT_SEED
    #: number of raw features kept by the preliminary RF (paper: 10)
    n_selected_features: int = 10
    #: lag horizons in hours (paper: 1 and 3)
    lags: tuple[int, ...] = (1, 3)
    #: rolling window in hours (paper: 5)
    rolling_window: int = 5
    #: Random Forest size (paper: 300)
    rf_n_estimators: int = 300
    #: XGBoost size (paper: 500)
    xgb_n_estimators: int = 500
    #: XGBoost maximum depth (paper: 6)
    xgb_max_depth: int = 6
    #: XGBoost learning rate (paper: 0.05)
    xgb_learning_rate: float = 0.05
    #: LSTM hidden units (paper: 64)
    lstm_hidden: int = 64
    #: LSTM dropout (paper: 0.2)
    lstm_dropout: float = 0.2
    #: sequence length in steps (paper: 10)
    sequence_length: int = 10
    #: LSTM epochs (not specified in the paper)
    lstm_epochs: int = 50
    #: LSTM batch size (not specified in the paper)
    lstm_batch_size: int = 64
    #: LSTM learning rate (not specified in the paper)
    lstm_learning_rate: float = 1e-3
    #: number of out-of-fold splits inside the training partition
    n_folds: int = 5
    #: ``blocked`` | ``stratified`` | ``chronological``
    fold_strategy: str = "blocked"
    #: SMOTE neighbourhood size (not specified in the paper)
    smote_k_neighbors: int = 5
    #: enable SMOTE for the tabular branch
    use_smote: bool = True
    #: fixed decision threshold (never tuned)
    threshold: float = FIXED_THRESHOLD
    #: fraction of the labelled series held out chronologically
    validation_fraction: float = 0.2
    #: base learners fed to the meta-learner
    base_models: tuple[str, ...] = ("rf", "xgb", "lstm")
    #: logistic-regression stacking or an equal-probability average
    ensemble_mode: str = "stacked"

    def __post_init__(self) -> None:
        if self.fold_strategy not in ("blocked", "stratified", "chronological"):
            raise HybridEnsembleError(
                f"unknown fold strategy {self.fold_strategy!r}; expected one "
                "of 'blocked', 'stratified', 'chronological'")
        if not 0.0 < self.validation_fraction < 1.0:
            raise HybridEnsembleError(
                "validation_fraction must be in (0, 1), got "
                f"{self.validation_fraction!r}")
        if self.n_folds < 2:
            raise HybridEnsembleError("n_folds must be >= 2")
        if not self.base_models:
            raise HybridEnsembleError("base_models must not be empty")
        unknown = set(self.base_models) - {"rf", "xgb", "lstm"}
        if unknown:
            raise HybridEnsembleError(
                f"unknown base model(s) {sorted(unknown)}")
        if self.ensemble_mode not in {"stacked", "mean"}:
            raise HybridEnsembleError("ensemble_mode must be stacked or mean")

    @property
    def min_history(self) -> int:
        """Rows that must be dropped so that no engineered feature is
        undefined.

        the largest lag and the rolling window both need complete history.
        The rows are dropped, never back-filled.
        """
        return max(self.rolling_window - 1, max(self.lags, default=0))

    @property
    def min_scored(self) -> int:
        """Rows of a partition that can actually be scored.

        The engineered matrix starts at ``min_history``; a sequence window
        additionally needs ``sequence_length - 1`` *complete* rows of that
        matrix, so the first scored row of a partition is
        ``min_history + sequence_length - 1``.
        """
        return self.min_history + self.sequence_length - 1

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lags"] = list(self.lags)
        data["base_models"] = list(self.base_models)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HybridEnsembleConfig":
        payload = dict(data)
        payload["lags"] = tuple(payload.get("lags", (1, 3)))
        payload["base_models"] = tuple(payload.get("base_models",
                                                  ("rf", "xgb", "lstm")))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


# --------------------------------------------------------------------------
# feature selection
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureSelection:
    """Outcome of the preliminary Random Forest feature selection."""

    features: tuple[str, ...]
    importances: dict[str, float]
    n_train_rows: int
    train_digest: str
    scope: str = "train"

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "importances": {k: float(v) for k, v in self.importances.items()},
            "n_train_rows": int(self.n_train_rows),
            "train_digest": self.train_digest,
            "scope": self.scope,
        }


def select_top_features(
    frame: pd.DataFrame,
    y: Sequence[int],
    *,
    n_top: int = 10,
    n_estimators: int = 300,
    seed: int = DEFAULT_SEED,
) -> FeatureSelection:
    """Preliminary Random Forest feature selection (paper Sec. III.B.3).

    The Random Forest is fit on the **raw** numeric columns of ``frame``,
    which must contain training rows only.  The returned
    :class:`FeatureSelection` records the digest of the data the forest was
    fit on, so a reviewer can prove the selection never saw a validation or
    test row.
    """
    from sklearn.ensemble import RandomForestClassifier

    if frame.shape[1] == 0:
        raise HybridEnsembleError("no feature columns to select from")
    labels = np.asarray(y).astype(int).ravel()
    if len(labels) != len(frame):
        raise HybridEnsembleError(
            f"feature selection got {len(frame)} rows and {len(labels)} labels")
    if len(np.unique(labels)) < 2:
        raise HybridEnsembleError(
            "feature selection needs both classes in the training partition")

    forest = RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=1)
    forest.fit(frame.to_numpy(dtype=float), labels)
    importances = forest.feature_importances_
    order = np.argsort(-importances, kind="stable")
    chosen = tuple(str(frame.columns[i]) for i in order[:n_top])
    return FeatureSelection(
        features=chosen,
        importances={str(c): float(v)
                     for c, v in zip(frame.columns, importances)},
        n_train_rows=int(len(frame)),
        train_digest=frame_digest(frame),
        scope="train",
    )


# --------------------------------------------------------------------------
# temporal feature engineering
# --------------------------------------------------------------------------
def engineer_temporal_features(
    frame: pd.DataFrame,
    selected: Sequence[str],
    *,
    lags: Sequence[int] = (1, 3),
    rolling_window: int = 5,
    segments: Sequence[int] | None = None,
    forward_fill: bool = True,
) -> pd.DataFrame:
    """Engineer the temporal features described in the paper (Sec. III.C).

    For every column in ``selected`` the paper adds ``lag1``, ``lag3``, the
    first difference ``d1``, a ``rolling_window``-hour rolling mean and a
    ``rolling_window``-hour rolling standard deviation.  All raw columns are
    kept, so the frame holds ``n_raw + 5 * n_selected`` columns (43 + 50 =
    93 for the paper's top-10 selection).

    Leak control
    ------------
    * every feature is strictly backward looking (lags and trailing rolling
      windows), never centered;
    * **no backward fill** is performed -- only a forward fill, applied
      inside a segment, so a missing value can only be replaced by a *past*
      observation;
    * when ``segments`` is given, lags/rolling/differences and the forward
      fill never cross a segment boundary.
    """
    work = frame.reset_index(drop=True)
    missing = [c for c in selected if c not in work.columns]
    if missing:
        raise HybridEnsembleError(
            f"selected feature(s) not present in the frame: {missing}")

    if segments is None:
        seg = pd.Series(np.zeros(len(work), dtype=int), index=work.index)
    else:
        seg = pd.Series(np.asarray(segments, dtype=int).ravel(),
                        index=work.index)
        if len(seg) != len(work):
            raise HybridEnsembleError(
                f"segments has {len(seg)} entries for {len(work)} rows")

    grouped = work.groupby(seg, sort=False)
    out = work.copy()
    for lag in lags:
        for col in selected:
            out[f"{col}_lag{lag}"] = grouped[col].shift(lag).to_numpy()
    for col in selected:
        out[f"{col}_d1"] = (work[col] - out[f"{col}_lag1"]).to_numpy()
        roll_mean = (grouped[col]
                     .rolling(window=rolling_window,
                              min_periods=rolling_window)
                     .mean()
                     .reset_index(level=0, drop=True))
        roll_std = (grouped[col]
                    .rolling(window=rolling_window,
                             min_periods=rolling_window)
                    .std()
                    .reset_index(level=0, drop=True))
        out[f"{col}_rollmean{rolling_window}"] = roll_mean.to_numpy()
        out[f"{col}_rollstd{rolling_window}"] = roll_std.to_numpy()

    if forward_fill:
        numeric = list(out.select_dtypes(include=[np.number]).columns)
        out[numeric] = out[numeric].groupby(seg, sort=False).ffill()
    return out


def build_sequences(
    values: np.ndarray,
    labels: Sequence[int],
    sequence_length: int,
    *,
    segments: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build LSTM sequences and their aligned labels.

    A sample is the ``sequence_length`` rows that *end* at row ``i``; its
    label is the label of row ``i`` (the last step of the window), so the
    label always describes the most recent observation of the window.

    Returns
    -------
    ``(X, y, end_index)`` with ``X`` of shape
    ``(n, sequence_length, n_feat)``, ``y`` of shape ``(n,)`` and
    ``end_index`` giving, for every sample, the row index of its last step.
    """
    data = np.asarray(values, dtype=float)
    if data.ndim != 2:
        raise HybridEnsembleError(
            f"build_sequences expects a 2-D matrix, got {data.ndim}-D")
    y = np.asarray(labels).astype(int).ravel()
    if len(y) != len(data):
        raise HybridEnsembleError(
            f"build_sequences got {len(data)} rows and {len(y)} labels")
    if sequence_length < 1:
        raise HybridEnsembleError("sequence_length must be >= 1")
    if segments is None:
        seg = np.zeros(len(data), dtype=int)
    else:
        seg = np.asarray(segments, dtype=int).ravel()
        if len(seg) != len(data):
            raise HybridEnsembleError(
                f"segments has {len(seg)} entries for {len(data)} rows")

    starts: list[int] = []
    ends: list[int] = []
    for i in range(len(data)):
        if i + 1 < sequence_length:
            continue
        if seg[i - sequence_length + 1] != seg[i]:
            # the window would cross a segment boundary -> not a valid sample
            continue
        starts.append(i - sequence_length + 1)
        ends.append(i)
    if not ends:
        empty = np.empty((0, sequence_length, data.shape[1]), dtype=float)
        return empty, np.empty((0,), dtype=int), np.empty((0,), dtype=int)
    idx = np.asarray(ends, dtype=int)
    windows = np.stack([data[s:e + 1] for s, e in zip(starts, ends)])
    return windows, y[idx], idx


def _scored_mask(seg: np.ndarray, sequence_length: int) -> np.ndarray:
    """Rows that can be scored by *all* base learners.

    A row is scorable when a full sequence window of ``sequence_length``
    rows ends at it *inside its own segment* (the LSTM requirement) and
    when its engineered feature vector is complete (guaranteed by the
    caller, which drops the head rows first).
    """
    mask = np.zeros(len(seg), dtype=bool)
    for value in np.unique(seg):
        idx = np.flatnonzero(seg == value)
        if len(idx) >= sequence_length:
            mask[idx[sequence_length - 1:]] = True
    return mask


def _sequences_for_rows(
    x_full: np.ndarray,
    rows: np.ndarray,
    sequence_length: int,
    seg_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequences of the full matrix that *end* at one of ``rows``.

    The window may use rows that precede ``rows`` (history only, no labels),
    which is what the LSTM needs when a block of rows is scored on its own.
    """
    seq_x, _y, idx = build_sequences(
        x_full, np.zeros(len(x_full), dtype=int), sequence_length,
        segments=seg_full)
    if len(idx) == 0:
        return seq_x, idx
    mask = np.isin(idx, rows)
    return seq_x[mask], idx[mask]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _f1(truth: np.ndarray, pred: np.ndarray) -> float:
    tp = float(np.sum((truth == 1) & (pred == 1)))
    fp = float(np.sum((truth == 0) & (pred == 1)))
    fn = float(np.sum((truth == 1) & (pred == 0)))
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def evaluate_attack(
    y_true: Sequence[int],
    proba: Sequence[float],
    *,
    threshold: float = FIXED_THRESHOLD,
) -> dict[str, Any]:
    """Attack-positive metric bundle at a **fixed** threshold.

    The bundle contains the average precision of the attack class (the area
    under the precision/recall curve), the Matthews correlation coefficient,
    the *unadjusted* balanced accuracy (computed at ``threshold``, never
    optimised), ROC-AUC and the usual attack-class precision/recall/F1 so
    that the numbers stay comparable with the paper.
    """
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, confusion_matrix,
                                 matthews_corrcoef, precision_recall_curve,
                                 precision_score, recall_score, roc_auc_score)

    truth = np.asarray(y_true).astype(int).ravel()
    scores = np.asarray(proba, dtype=float).ravel()
    if len(truth) != len(scores):
        raise HybridEnsembleError(
            f"evaluate_attack got {len(truth)} labels and {len(scores)} scores")
    if len(truth) == 0:
        raise HybridEnsembleError("evaluate_attack got an empty sample")

    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()

    precision_curve, recall_curve, _ = precision_recall_curve(truth, scores)
    if precision_curve.size > 1:
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        pr_auc = float(trapezoid(precision_curve[::-1], recall_curve[::-1]))
    else:  # pragma: no cover - degenerate input
        pr_auc = float("nan")

    if len(np.unique(truth)) < 2:
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(truth, scores))

    return {
        "schema": METRIC_SCHEMA,
        "threshold": float(threshold),
        "threshold_tuned": False,
        "n_samples": int(len(truth)),
        "n_positive": int(truth.sum()),
        "n_negative": int((truth == 0).sum()),
        "average_precision": float(average_precision_score(truth, scores)),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "matthews_corrcoef": float(matthews_corrcoef(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "accuracy": float(accuracy_score(truth, pred)),
        "precision_attack": float(precision_score(truth, pred, zero_division=0)),
        "recall_attack": float(recall_score(truth, pred, zero_division=0)),
        "f1_attack": float(_f1(truth, pred)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# --------------------------------------------------------------------------
# LSTM
# --------------------------------------------------------------------------
def _build_lstm_net(input_size: int, hidden: int, dropout: float):
    torch = _import_optional("torch")
    from torch import nn

    class _LSTMNet(nn.Module):
        """Single LSTM layer + dropout + sigmoid output (paper Sec. IV)."""

        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.dense = nn.Linear(hidden, 1)
            self.activation = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.lstm(x)
            probs = self.activation(self.dense(self.dropout(out[:, -1, :])))
            return probs.squeeze(-1)

    return _LSTMNet()


def _class_weights(labels: np.ndarray) -> np.ndarray:
    """Class weights inversely proportional to the class frequencies."""
    counts = np.bincount(labels, minlength=2).astype(float)
    total = max(counts.sum(), 1.0)
    return np.where(counts > 0, total / (2.0 * np.maximum(counts, 1.0)), 0.0)


def _train_lstm(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> Any:
    torch = _import_optional("torch")
    from torch import nn

    _seed_everything(seed)
    net = _build_lstm_net(x.shape[2], hidden, dropout)
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.BCELoss(reduction="none")
    weights = torch.tensor(_class_weights(y), dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y.astype(np.float32), dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=min(batch_size, len(dataset)), shuffle=True,
        generator=generator, drop_last=False)
    net.train()
    for _ in range(max(1, epochs)):
        for xb, yb in loader:
            optimiser.zero_grad()
            probs = net(xb).clamp(1e-7, 1.0 - 1e-7)
            loss = (criterion(probs, yb) * weights[yb.long()]).mean()
            loss.backward()
            optimiser.step()
    net.eval()
    return net


def _lstm_predict(net: Any, x: np.ndarray) -> np.ndarray:
    torch = _import_optional("torch")
    if len(x) == 0:
        return np.empty((0,), dtype=float)
    with torch.no_grad():
        out = net(torch.tensor(x, dtype=torch.float32))
    return out.detach().cpu().numpy().ravel().astype(float)


# --------------------------------------------------------------------------
# the ensemble
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PredictionResult:
    """Predictions of the ensemble together with the rows they belong to."""

    index: pd.Index
    proba: np.ndarray
    dropped_rows: int
    dropped_indices: np.ndarray

    def __len__(self) -> int:
        return int(len(self.proba))


class HybridEnsemble:
    """Hybrid stacked ensemble of Random Forest, XGBoost and LSTM."""

    def __init__(self, config: HybridEnsembleConfig | None = None) -> None:
        self.config = config or HybridEnsembleConfig()
        self.fitted_ = False
        self.selection_: FeatureSelection | None = None
        self.feature_columns_: list[str] = []
        self.raw_columns_: list[str] = []
        self.resample_records_: list[dict[str, Any]] = []
        self.fold_assignments_: np.ndarray | None = None
        self.oof_probabilities_: np.ndarray | None = None
        self.fit_audit_: dict[str, Any] = {}
        self.provenance_: dict[str, Any] = {}

    # -- internals ---------------------------------------------------------
    def _prepare(self, frame: pd.DataFrame,
                 segments: Sequence[int] | None
                 ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Index]:
        """Return the raw frame, its segments and the completeness mask.

        The engineered features are computed on the *whole* frame (the
        rolling windows must see the real history) and the rows whose
        engineered vector is undefined are dropped afterwards -- never
        back-filled.
        """
        cfg = self.config
        original_index = frame.index
        work = frame.reset_index(drop=True)
        if segments is None:
            seg = np.zeros(len(work), dtype=int)
        else:
            seg = np.asarray(segments, dtype=int).ravel()
            if len(seg) != len(work):
                raise HybridEnsembleError(
                    f"segments has {len(seg)} entries for {len(work)} rows")
        complete = np.zeros(len(work), dtype=bool)
        for value in np.unique(seg):
            idx = np.flatnonzero(seg == value)
            if len(idx) > cfg.min_history:
                # the head of every segment lacks the rolling history
                complete[idx[cfg.min_history:]] = True
        return work, seg, complete, original_index

    def _engineer(self, frame: pd.DataFrame, seg: np.ndarray) -> pd.DataFrame:
        """Engineer the temporal features of the whole frame."""
        cfg = self.config
        engineered = engineer_temporal_features(
            frame, self.selection_.features, lags=cfg.lags,
            rolling_window=cfg.rolling_window, segments=seg)
        if self.feature_columns_:
            missing = [c for c in self.feature_columns_
                       if c not in engineered.columns]
            if missing:
                raise HybridEnsembleError(
                    f"engineered frame is missing feature column(s): {missing}")
            engineered = engineered[self.feature_columns_]
        return engineered

    def _make_folds(self, y: np.ndarray,
                    seg: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Out-of-fold splits *inside* the training partition."""
        cfg = self.config
        n = len(y)
        k = min(cfg.n_folds, n)
        if k < 2:
            raise HybridEnsembleError(
                "not enough training rows for out-of-fold stacking")
        if cfg.fold_strategy == "stratified":
            from sklearn.model_selection import StratifiedKFold
            splitter = StratifiedKFold(n_splits=k, shuffle=True,
                                       random_state=cfg.seed)
            return [(tr, va) for tr, va in splitter.split(np.zeros(n), y)]
        if cfg.fold_strategy == "chronological":
            folds: list[tuple[np.ndarray, np.ndarray]] = []
            edges = np.linspace(0, n, k + 1).astype(int)
            for i in range(k):
                va = np.arange(edges[i], edges[i + 1])
                tr = np.arange(0, edges[i])
                if len(tr) == 0 or len(va) == 0:
                    continue
                folds.append((tr, va))
            if not folds:
                raise HybridEnsembleError(
                    "chronological folds are empty; use more training rows")
            return folds
        # blocked: the contiguous runs of equal labels are kept together
        # where possible, so that an attack block is not shredded across
        # folds.  A run is only split when there are fewer runs than folds,
        # which keeps both classes present in every training fold.
        fold_of_row = np.zeros(n, dtype=int)
        for value in np.unique(seg):
            idx = np.flatnonzero(seg == value)
            labels = y[idx]
            bounds = np.flatnonzero(np.diff(labels) != 0) + 1
            starts = np.concatenate(([0], bounds))
            ends = np.concatenate((bounds, [len(idx)]))
            for cls in (0, 1):
                pieces: list[np.ndarray] = []
                for s, e in zip(starts, ends):
                    if labels[s] != cls:
                        continue
                    pieces.append(idx[s:e])
                if not pieces:
                    continue
                while len(pieces) < k:
                    pieces.sort(key=len, reverse=True)
                    biggest = pieces.pop(0)
                    middle = len(biggest) // 2
                    if middle == 0:
                        pieces.append(biggest)
                        break
                    pieces.extend([biggest[:middle], biggest[middle:]])
                for b, piece in enumerate(pieces):
                    fold_of_row[piece] = b % k
        folds = []
        for b in range(k):
            va = np.flatnonzero(fold_of_row == b)
            tr = np.flatnonzero(fold_of_row != b)
            if len(tr) == 0 or len(va) == 0:
                continue
            folds.append((tr, va))
        if not folds:
            raise HybridEnsembleError("blocked folds are empty")
        return folds

    def _resample(self, x: np.ndarray, y: np.ndarray,
                  scope: str) -> tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE to a *training* matrix (never to an eval partition)."""
        cfg = self.config
        record: dict[str, Any] = {
            "scope": scope, "method": "none", "smote_applied": False,
            "n_in": int(len(y)), "n_out": int(len(y)),
            "n_minority_in": int(np.sum(y == 1)),
            "n_minority_out": int(np.sum(y == 1)),
        }
        minority = int(np.sum(y == 1))
        if not cfg.use_smote:
            record["reason"] = "SMOTE disabled by configuration"
            self.resample_records_.append(record)
            return x, y
        if minority < 2:
            record["reason"] = ("too few minority samples for SMOTE; the "
                                "partition is trained unbalanced")
            self.resample_records_.append(record)
            return x, y
        from imblearn.over_sampling import SMOTE
        k = max(1, min(cfg.smote_k_neighbors, minority - 1))
        sampler = SMOTE(random_state=cfg.seed, k_neighbors=k)
        xs, ys = sampler.fit_resample(x, y)
        record.update({"method": "SMOTE", "smote_applied": True,
                       "k_neighbors": int(k), "n_out": int(len(ys)),
                       "n_minority_out": int(np.sum(ys == 1))})
        self.resample_records_.append(record)
        return xs, ys

    def _fit_base_models(self, engineered: pd.DataFrame, seg: np.ndarray,
                         y: np.ndarray, tab_rows: np.ndarray,
                         lstm_rows: np.ndarray, seed: int, scope: str
                         ) -> tuple[dict[str, Any], Any, Any]:
        """Fit RF, XGB and LSTM on a training partition.

        The StandardScaler is fit on the tabular training rows, the
        MinMaxScaler on the rows the LSTM is trained on -- never on an
        evaluation row.  SMOTE is applied to the *scaled* tabular training
        matrix; the LSTM is trained on the original temporal sequences with
        class weights.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        cfg = self.config
        values = engineered.to_numpy(dtype=float)
        scaler = StandardScaler().fit(values[tab_rows])
        minmax = MinMaxScaler().fit(values[lstm_rows])
        x_tab = scaler.transform(values)
        x_lstm = minmax.transform(values)

        models: dict[str, Any] = {}
        if "rf" in cfg.base_models or "xgb" in cfg.base_models:
            x_tab_res, y_tab_res = self._resample(x_tab[tab_rows],
                                                  y[tab_rows], scope)
            if "rf" in cfg.base_models:
                rf = RandomForestClassifier(
                    n_estimators=cfg.rf_n_estimators, random_state=seed,
                    n_jobs=1)
                rf.fit(x_tab_res, y_tab_res)
                models["rf"] = rf
            if "xgb" in cfg.base_models:
                xgb = _import_optional("xgboost")
                xgb_model = xgb.XGBClassifier(
                    n_estimators=cfg.xgb_n_estimators,
                    max_depth=cfg.xgb_max_depth,
                    learning_rate=cfg.xgb_learning_rate,
                    random_state=seed, n_jobs=1, verbosity=0,
                    tree_method="hist")
                xgb_model.fit(x_tab_res, y_tab_res)
                models["xgb"] = xgb_model
        if "lstm" in cfg.base_models:
            seq_x, seq_idx = _sequences_for_rows(
                x_lstm, lstm_rows, cfg.sequence_length, seg)
            seq_y = y[seq_idx]
            if len(seq_x) == 0:
                raise HybridEnsembleError(
                    "no LSTM sequences could be built from the training "
                    "partition")
            models["lstm"] = _train_lstm(
                seq_x, seq_y, hidden=cfg.lstm_hidden,
                dropout=cfg.lstm_dropout, epochs=cfg.lstm_epochs,
                batch_size=cfg.lstm_batch_size,
                learning_rate=cfg.lstm_learning_rate, seed=seed)
        return models, scaler, minmax

    @staticmethod
    def _positive_proba(model: Any, x: np.ndarray) -> np.ndarray:
        """Attack-class probability of a fitted base learner.

        A learner that saw a single class exposes a one-column
        ``predict_proba``; the attack probability is then 0 (or 1) by
        definition.
        """
        proba = model.predict_proba(x)
        if proba.shape[1] == 1:
            classes = list(getattr(model, "classes_", [0]))
            return np.full(len(x), 1.0 if classes[0] == 1 else 0.0)
        return proba[:, 1]

    def _base_proba(self, models: dict[str, Any], engineered: pd.DataFrame,
                    seg: np.ndarray, x_tab: np.ndarray, x_lstm: np.ndarray,
                    rows: np.ndarray) -> np.ndarray:
        """Attack probabilities of the base learners for ``rows``."""
        cfg = self.config
        columns: list[np.ndarray] = []
        if "rf" in cfg.base_models:
            columns.append(self._positive_proba(models["rf"], x_tab[rows]))
        if "xgb" in cfg.base_models:
            columns.append(self._positive_proba(models["xgb"], x_tab[rows]))
        if "lstm" in cfg.base_models:
            seq_x, seq_idx = _sequences_for_rows(
                x_lstm, rows, cfg.sequence_length, seg)
            probs = _lstm_predict(models["lstm"], seq_x)
            if len(probs) != len(rows):
                raise HybridEnsembleError(
                    f"LSTM produced {len(probs)} predictions for "
                    f"{len(rows)} rows")
            columns.append(probs)
        return np.column_stack(columns)

    # -- public API --------------------------------------------------------
    def fit(self, frame: pd.DataFrame, y: Sequence[int], *,
            segments: Sequence[int] | None = None) -> "HybridEnsemble":
        """Fit the ensemble on a **training** partition.

        ``frame`` holds the raw sensor columns in chronological order and
        ``y`` the binary attack labels.  Nothing outside this call is used
        for fitting; validation/test partitions are only ever passed to
        :meth:`predict_proba`.
        """
        from sklearn.linear_model import LogisticRegression

        cfg = self.config
        _seed_everything(cfg.seed)
        labels = np.asarray(y).astype(int).ravel()
        if len(labels) != len(frame):
            raise HybridEnsembleError(
                f"fit got {len(frame)} rows and {len(labels)} labels")

        work, seg, complete, _index = self._prepare(frame, segments)
        y_work = labels[complete]
        if len(np.unique(y_work)) < 2:
            raise HybridEnsembleError(
                "the training partition must contain both classes")

        self.raw_columns_ = [str(c) for c in work.columns]
        matrix_rows = np.flatnonzero(complete)
        self.selection_ = select_top_features(
            work.iloc[matrix_rows].reset_index(drop=True), y_work,
            n_top=cfg.n_selected_features,
            n_estimators=cfg.rf_n_estimators, seed=cfg.seed)
        engineered_full = self._engineer(work, seg)
        engineered = engineered_full.iloc[matrix_rows].reset_index(drop=True)
        seg = seg[complete]
        self.feature_columns_ = [str(c) for c in engineered.columns]

        # ---- out-of-fold base predictions (SMOTE inside the folds only) --
        # The meta-learner only sees rows that all three base learners can
        # score, i.e. rows with a complete engineered feature vector *and* a
        # complete sequence window.
        n_rows = len(engineered)
        scored = _scored_mask(seg, cfg.sequence_length)
        scored_rows = np.flatnonzero(scored)
        y_scored = y_work[scored_rows]

        folds = self._make_folds(y_scored, seg[scored_rows])
        n_base = len(cfg.base_models)
        oof = np.zeros((len(scored_rows), n_base), dtype=float)
        fold_assignment = np.full(len(scored_rows), -1, dtype=int)
        fold_details: list[dict[str, int]] = []
        for k, (tr_s, va_s) in enumerate(folds):
            fold_assignment[va_s] = k
            va_rows = scored_rows[va_s]
            # the tabular branch trains on every engineered row that is not
            # being validated; the LSTM additionally needs a full window
            tab_rows = np.setdiff1d(np.arange(n_rows), va_rows)
            lstm_rows = np.setdiff1d(scored_rows, va_rows)
            fold_details.append({"train": int(len(tab_rows)),
                                 "validation": int(len(va_rows)),
                                 "sequence_train": int(len(lstm_rows))})
            fold_models, fold_scaler, fold_minmax = self._fit_base_models(
                engineered, seg, y_work, tab_rows, lstm_rows,
                seed=cfg.seed + k, scope="oof_fold_train")
            values = engineered.to_numpy(dtype=float)
            x_tab = fold_scaler.transform(values)
            x_lstm = fold_minmax.transform(values)
            oof[va_s] = self._base_proba(
                fold_models, engineered, seg, x_tab, x_lstm, va_rows)
        if np.any(fold_assignment < 0):
            raise HybridEnsembleError(
                "some training rows were never used as an out-of-fold "
                "validation row")
        self.fold_assignments_ = fold_assignment
        self.oof_probabilities_ = oof

        if cfg.ensemble_mode == "mean":
            self.meta_ = None
        else:
            meta = LogisticRegression(max_iter=1000, random_state=cfg.seed)
            meta.fit(oof, y_scored)
            self.meta_ = meta

        # ---- final base models on the whole training partition -----------
        models, scaler, minmax = self._fit_base_models(
            engineered, seg, y_work, np.arange(n_rows), scored_rows,
            seed=cfg.seed, scope="final_train")
        self.models_ = models
        self.scaler_ = scaler
        self.minmax_ = minmax

        self.fit_audit_ = {
            "train_frame_digest": frame_digest(
                work.iloc[np.flatnonzero(complete)].reset_index(drop=True)),
            "labels_digest": array_digest(y_work),
            "n_rows": int(len(y_work)),
            "n_scored_rows": int(len(scored_rows)),
            "n_dropped_head_rows": int(len(work) - len(y_work)),
            "n_folds": int(len(folds)),
            "fold_sizes": fold_details,
        }
        self.provenance_ = {
            "schema": BUNDLE_SCHEMA,
            "seed": int(cfg.seed),
            "config": cfg.to_dict(),
            "adaptations": [dict(a) for a in ADAPTATIONS],
            "feature_selection": self.selection_.to_dict(),
            "feature_columns": list(self.feature_columns_),
            "raw_columns": list(self.raw_columns_),
            "base_models": list(cfg.base_models),
            "threshold": {"value": float(cfg.threshold), "tuned": False},
            "resampling": list(self.resample_records_),
            "fit_audit": dict(self.fit_audit_),
            "library_versions": _library_versions(),
        }
        self.fitted_ = True
        return self

    def predict_proba(self, frame: pd.DataFrame, *,
                      segments: Sequence[int] | None = None
                      ) -> PredictionResult:
        """Attack probability for every scored row of ``frame``.

        The rows without history (the first ``min_history`` rows) are
        dropped, never back-filled.
        """
        if not self.fitted_:
            raise HybridEnsembleError("predict_proba called before fit")
        cfg = self.config
        work, seg, complete, index = self._prepare(frame, segments)
        engineered_full = self._engineer(work, seg)
        matrix_rows = np.flatnonzero(complete)
        engineered = engineered_full.iloc[matrix_rows].reset_index(drop=True)
        seg = seg[complete]
        values = engineered.to_numpy(dtype=float)
        x_tab = self.scaler_.transform(values)
        x_lstm = self.minmax_.transform(values)
        scored = _scored_mask(seg, cfg.sequence_length)
        rows = np.flatnonzero(scored)
        base = self._base_proba(self.models_, engineered, seg, x_tab, x_lstm,
                                rows)
        if self.config.ensemble_mode == "mean":
            proba = base.mean(axis=1)
        else:
            proba = self.meta_.predict_proba(base)[:, 1]
        kept_positions = matrix_rows[rows]
        all_rows = np.arange(len(frame))
        dropped_indices = np.setdiff1d(all_rows, kept_positions)
        return PredictionResult(index=index[kept_positions],
                                proba=np.asarray(proba),
                                dropped_rows=int(len(dropped_indices)),
                                dropped_indices=dropped_indices)

    def predict(self, frame: pd.DataFrame, *,
                segments: Sequence[int] | None = None,
                threshold: float | None = None) -> PredictionResult:
        """Hard predictions at the fixed threshold (never tuned)."""
        result = self.predict_proba(frame, segments=segments)
        cut = self.config.threshold if threshold is None else float(threshold)
        labels = (result.proba >= cut).astype(int)
        return PredictionResult(index=result.index, proba=labels,
                                dropped_rows=result.dropped_rows,
                                dropped_indices=result.dropped_indices)

    # -- persistence -------------------------------------------------------
    def _sklearn_state(self) -> dict[str, Any]:
        models = {k: v for k, v in self.models_.items() if k != "lstm"}
        return {
            "models": models,
            "meta": self.meta_,
            "scaler": self.scaler_,
            "minmax": self.minmax_,
            "selection": self.selection_,
            "feature_columns": list(self.feature_columns_),
            "raw_columns": list(self.raw_columns_),
            "fold_assignments": self.fold_assignments_,
            "oof_probabilities": self.oof_probabilities_,
            "resample_records": list(self.resample_records_),
            "fit_audit": dict(self.fit_audit_),
            "provenance": self.provenance_,
        }

    def save(self, out_dir: str | Path) -> Path:
        """Persist the fitted ensemble into ``out_dir``.

        The bundle is deterministic: two fits with the same seed and the
        same data produce byte-identical files.
        """
        if not self.fitted_:
            raise HybridEnsembleError("save called before fit")
        torch = _import_optional("torch")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest = dict(self.provenance_)
        manifest["bundle_schema"] = BUNDLE_SCHEMA
        manifest["metric_schema"] = METRIC_SCHEMA
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        with open(out / "model.pkl", "wb") as handle:
            pickle.dump(self._sklearn_state(), handle, protocol=4)
        if "lstm" in self.models_:
            torch.save(self.models_["lstm"].state_dict(), out / "lstm.pt")
        return out

    @classmethod
    def load(cls, out_dir: str | Path) -> "HybridEnsemble":
        """Load a persisted ensemble."""
        torch = _import_optional("torch")
        out = Path(out_dir)
        manifest = json.loads(
            (out / "manifest.json").read_text(encoding="utf-8"))
        config = HybridEnsembleConfig.from_dict(manifest["config"])
        ensemble = cls(config)
        with open(out / "model.pkl", "rb") as handle:
            state = pickle.load(handle)
        ensemble.models_ = dict(state["models"])
        ensemble.meta_ = state["meta"]
        ensemble.scaler_ = state["scaler"]
        ensemble.minmax_ = state["minmax"]
        ensemble.selection_ = state["selection"]
        ensemble.feature_columns_ = list(state["feature_columns"])
        ensemble.raw_columns_ = list(state["raw_columns"])
        ensemble.fold_assignments_ = state["fold_assignments"]
        ensemble.oof_probabilities_ = state["oof_probabilities"]
        ensemble.resample_records_ = list(state["resample_records"])
        ensemble.fit_audit_ = dict(state["fit_audit"])
        ensemble.provenance_ = dict(state["provenance"])
        if (out / "lstm.pt").exists():
            net = _build_lstm_net(len(ensemble.feature_columns_),
                                  config.lstm_hidden, config.lstm_dropout)
            net.load_state_dict(torch.load(out / "lstm.pt", weights_only=True))
            net.eval()
            ensemble.models_["lstm"] = net
        ensemble.fitted_ = True
        return ensemble

    def state_digest(self) -> str:
        """Fingerprint of the fitted state.

        Used to prove that ``predict`` never refits or tunes anything.
        """
        torch = _import_optional("torch")
        buffer = pickle.dumps(self._sklearn_state(), protocol=4)
        state_dict = (self.models_["lstm"].state_dict()
                      if "lstm" in self.models_ else {})
        buffer += pickle.dumps(
            {k: v.detach().cpu().numpy().tobytes()
             for k, v in state_dict.items()}, protocol=4)
        return _sha256_bytes(buffer)


# --------------------------------------------------------------------------
# native BATADAL runner
# --------------------------------------------------------------------------
@dataclass
class BatadalRunResult:
    """Outcome of :func:`run_batadal`."""

    ensemble: HybridEnsemble
    metrics_validation: dict[str, Any]
    metrics_test: dict[str, Any] | None
    probabilities_validation: np.ndarray
    probabilities_test: np.ndarray | None
    labels_validation: np.ndarray
    labels_test: np.ndarray | None
    provenance: dict[str, Any]
    artifacts: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics_validation": self.metrics_validation,
            "metrics_test": self.metrics_test,
            "provenance": self.provenance,
            "artifacts": self.artifacts,
        }


def _load_batadal_dataset04(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Raw BATADAL dataset 04 -> (frame, labels).

    ``ATT_FLAG`` is 1 during an attack and -999 ("no attack") otherwise.
    """
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame.rename(columns=_canonical_columns(list(frame.columns)))
    if "att_flag" not in frame.columns:
        raise HybridEnsembleError(
            f"{path}: no ATT_FLAG column, cannot build supervised labels")
    labels = (frame["att_flag"].astype(float) == 1.0).astype(int).to_numpy()
    frame = frame.drop(columns=[c for c in ("att_flag", "datetime")
                               if c in frame.columns])
    return frame.reset_index(drop=True), labels


def _file_digests(paths: Sequence[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in paths:
        p = Path(path)
        out[str(p)] = _sha256_bytes(p.read_bytes()) if p.exists() else "missing"
    return out


def run_batadal(
    *,
    data_root: str | Path = "data",
    topology: str = "ctown",
    config: HybridEnsembleConfig | None = None,
    raw_dataset04: str | Path | None = None,
    calibration_mode: str = "append_normal",
    score_official_test: bool = True,
    output_dir: str | Path | None = None,
) -> BatadalRunResult:
    """Run the hybrid ensemble on the native BATADAL material.

    * the **supervised** source is the raw ``BATADAL_dataset04.csv``
      (``ATT_FLAG`` -> attack label);
    * the **calibration** source is the normalized dataset-03 calibration
      frame committed under ``data/<topology>/dataset/calibration``; with
      ``calibration_mode="append_normal"`` its rows join the *training*
      side only, as extra normal-operation rows in their own segment;
    * the **official test** set (``data/<topology>/dataset/evaluation``) is
      never fitted on and never used to tune a threshold.  Its digest is
      recorded before and after the run to prove it was not modified.

    Returns
    -------
    BatadalRunResult
    """
    from dataio.loader import load_topology
    from dataio.manifest import load_dataset_manifest

    cfg = config or HybridEnsembleConfig()
    if calibration_mode not in ("append_normal", "exclude"):
        raise HybridEnsembleError(
            "calibration_mode must be 'append_normal' or 'exclude', got "
            f"{calibration_mode!r}")
    root = Path(data_root)
    dataset = load_topology(topology, data_root=root)

    dataset_dir = root / topology / "dataset"
    manifest = load_dataset_manifest(dataset_dir / "dataset_manifest.yaml")
    eval_files = [dataset_dir / rel for rel in manifest["files"]["evaluation"]]
    digests_before = _file_digests(eval_files)

    raw_path = Path(raw_dataset04) if raw_dataset04 is not None else (
        root / topology / "raw" / "BATADAL_dataset04.csv")
    if not raw_path.exists():
        raise HybridEnsembleError(
            f"raw BATADAL dataset 04 not found: {raw_path}")
    supervised, labels = _load_batadal_dataset04(raw_path)

    calibration = pd.concat(dataset.calibration_frames,
                            ignore_index=True).reset_index(drop=True)

    # ---- chronological outer split of the labelled series -----------------
    n = len(supervised)
    cut = int(round(n * (1.0 - cfg.validation_fraction)))
    cut = min(max(cut, 1), n - 1)
    train_frame = supervised.iloc[:cut].reset_index(drop=True)
    val_frame = supervised.iloc[cut:].reset_index(drop=True)
    y_train = labels[:cut]
    y_val = labels[cut:]

    # ---- training side: supervised rows (+ the calibration segment) -------
    frames = [train_frame]
    ys = [y_train]
    segments = [np.zeros(len(train_frame), dtype=int)]
    if calibration_mode == "append_normal" and len(calibration) > 0:
        frames.append(calibration.reindex(columns=train_frame.columns))
        ys.append(np.zeros(len(calibration), dtype=int))
        segments.append(np.ones(len(calibration), dtype=int))
    fit_frame = pd.concat(frames, ignore_index=True)
    y_fit = np.concatenate(ys)
    fit_segments = np.concatenate(segments)

    ensemble = HybridEnsemble(cfg)
    ensemble.fit(fit_frame, y_fit, segments=fit_segments)

    # ---- validation (chronological tail of dataset 04) --------------------
    val_result = ensemble.predict_proba(val_frame)
    val_positions = val_result.index.to_numpy()
    metrics_validation = evaluate_attack(y_val[val_positions],
                                         val_result.proba,
                                         threshold=cfg.threshold)
    metrics_validation["dropped_rows"] = int(val_result.dropped_rows)
    metrics_validation["partition"] = "validation_dataset04_tail"

    # ---- official test (read-only, never fitted on) ----------------------
    metrics_test = None
    proba_test = None
    labels_test = None
    if score_official_test and dataset.eval_scenarios:
        scenario = dataset.eval_scenarios[0]
        test_frame = scenario.frame.reindex(columns=train_frame.columns)
        test_result = ensemble.predict_proba(test_frame)
        labels_test = np.asarray(scenario.labels)[test_result.index.to_numpy()]
        proba_test = test_result.proba
        metrics_test = evaluate_attack(labels_test, proba_test,
                                       threshold=cfg.threshold)
        metrics_test["dropped_rows"] = int(test_result.dropped_rows)
        metrics_test["partition"] = f"official_test:{scenario.name}"

    digests_after = _file_digests(eval_files)

    artifacts = {
        "raw_dataset04": str(raw_path),
        "raw_dataset04_sha256": _sha256_bytes(raw_path.read_bytes()),
        "calibration_rows": int(len(calibration)),
        "supervised_rows": int(n),
        "train_rows": int(len(y_fit)),
        "validation_rows": int(len(y_val)),
        "split_cut_row": int(cut),
        "calibration_mode": calibration_mode,
        "official_test_files": {str(k): v for k, v in digests_before.items()},
        "official_test_sha256_after": {str(k): v
                                       for k, v in digests_after.items()},
        "official_test_unchanged": digests_before == digests_after,
        "official_test_scored": bool(metrics_test is not None),
    }
    provenance = {
        "schema": "baselines/hybrid_ensemble/run/v1",
        "seed": int(cfg.seed),
        "config": cfg.to_dict(),
        "adaptations": [dict(a) for a in ADAPTATIONS],
        "fit_audit": dict(ensemble.fit_audit_),
        "resampling": list(ensemble.resample_records_),
        "feature_selection": ensemble.selection_.to_dict(),
        "library_versions": _library_versions(),
        "artifacts": artifacts,
    }
    result = BatadalRunResult(
        ensemble=ensemble,
        metrics_validation=metrics_validation,
        metrics_test=metrics_test,
        probabilities_validation=val_result.proba,
        probabilities_test=proba_test,
        labels_validation=y_val[val_positions],
        labels_test=labels_test,
        provenance=provenance,
        artifacts=artifacts,
    )
    if output_dir is not None:
        out = Path(output_dir)
        ensemble.save(out)
        (out / "run_result.json").write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        np.savez(out / "validation_predictions.npz", probabilities=val_result.proba, labels=result.labels_validation)
        if proba_test is not None:
            np.savez(out / "test_predictions.npz", probabilities=proba_test, labels=labels_test)
    return result
