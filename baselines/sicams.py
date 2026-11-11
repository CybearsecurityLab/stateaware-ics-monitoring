"""SICAMS statistical baseline, implemented from Melnikov et al. (2025).

Reference: O. Melnikov, Y. Dorofieiev, Y. Shakhnovskiy, H. Truong,
V. Degeler, "A Multivariate Statistical Framework for Detection,
Classification and Pre-localization of Anomalies in Water Distribution
Networks", arXiv:2512.15685v1, 17 Dec 2025.

This module keeps the pointwise detection core: per-temporal-cluster normal
mean/covariance estimation, Cholesky whitening, Hotelling's T^2 with
finite-sample F rescaling, causal moving-average scores and optional
hysteresis.  The paper's later stages (PELT change-point segmentation,
Algorithm 2 post-processing, sensor-fault/hydraulic discrimination, Laplacian
pre-localization and leak-volume regression) live in
:mod:`baselines.sicams_stages` so that legacy detection-only callers keep the
same API and numerical semantics.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import solve_triangular
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
)

SCHEMA = "baselines/sicams/v1"
PAPER = (
    "Melnikov, Dorofieiev, Shakhnovskiy, Truong, Degeler (2025), "
    "arXiv:2512.15685v1"
)
PAPER_URL = "https://arxiv.org/abs/2512.15685v1"

SCHEME_HOUR_OF_DAY = "hour_of_day"
SCHEME_TWO_HOUR_BLOCKS = "two_hour_blocks"
SCHEME_DAY_OF_WEEK = "day_of_week"
SCHEME_MONTH = "month"
SCHEME_HOUR_OF_WEEK = "hour_of_week"
SCHEME_LTOWN_AREA_A = "ltown_area_a"
SCHEME_LTOWN_AREA_C = "ltown_area_c"
SCHEME_LTOWN_AREA_B = "ltown_area_b"

NATIVE_LTOWN_SCHEMES = frozenset(
    {SCHEME_LTOWN_AREA_A, SCHEME_LTOWN_AREA_C, SCHEME_LTOWN_AREA_B}
)
TEMPORAL_SCHEMES = frozenset(
    {
        SCHEME_HOUR_OF_DAY,
        SCHEME_TWO_HOUR_BLOCKS,
        SCHEME_DAY_OF_WEEK,
        SCHEME_MONTH,
        SCHEME_HOUR_OF_WEEK,
        *NATIVE_LTOWN_SCHEMES,
    }
)

METRIC_KEYS = frozenset(
    {
        "mcc",
        "balanced_accuracy",
        "balanced_accuracy_adjusted",
        "attack_average_precision",
        "positive_class",
        "score_for_ap",
        "scope",
        "tp",
        "fp",
        "tn",
        "fn",
        "n_samples",
        "n_positive",
    }
)


class SicamsError(RuntimeError):
    """Base class for explicit SICAMS failures."""


class NotFittedError(SicamsError):
    """Raised when prediction is attempted before fitting."""


class InsufficientClusterSamplesError(SicamsError):
    """Raised when a temporal cluster violates the paper's n_k > s + 2 rule."""

    def __init__(self, cluster_id: int, n_k: int, s: int) -> None:
        self.cluster_id = int(cluster_id)
        self.n_k = int(n_k)
        self.s = int(s)
        super().__init__(
            f"temporal cluster {self.cluster_id} has n_k={self.n_k} "
            f"observations for s={self.s} sensors; SICAMS requires "
            f"n_k > s + 2 (need at least {self.s + 3})"
        )


class SingularCovarianceError(SicamsError):
    """Raised when a cluster covariance is singular or not positive definite."""

    def __init__(
        self,
        cluster_id: int,
        n_k: int,
        s: int,
        rank: int,
        eigenvalues: np.ndarray,
        reason: str,
    ) -> None:
        self.cluster_id = int(cluster_id)
        self.n_k = int(n_k)
        self.s = int(s)
        self.rank = int(rank)
        self.eigenvalues = np.asarray(eigenvalues, dtype=float).copy()
        self.reason = str(reason)
        super().__init__(
            f"temporal cluster {self.cluster_id} covariance is {self.reason} "
            f"(n_k={self.n_k}, s={self.s}, rank={self.rank}, "
            f"eigenvalues={self.eigenvalues.tolist()}). "
            "SICAMS refuses to whiten with an invertible-looking approximation; "
            "remove collinear/constant sensors or supply more data."
        )


class UnknownClusterError(SicamsError):
    """Raised when prediction requests a temporal cluster never seen in fit."""


class FeatureDimensionError(SicamsError):
    """Raised when prediction sensors differ from the fitted sensor set."""


@dataclass(frozen=True)
class SicamsConfig:
    """Declared SICAMS settings.

    ``significance`` is the pre-declared alpha used in
    theta1 = F_{1-alpha}(s, n_k - s). ``moving_average_window`` is the causal
    trailing window L from Algorithm 1. ``hysteresis`` toggles the theta0 reset
    rule. ``temporal_scheme`` is used only when timestamps, rather than explicit
    cluster ids, are supplied.
    """

    significance: float = 0.05
    moving_average_window: int = 12
    temporal_scheme: str = SCHEME_HOUR_OF_DAY
    hysteresis: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < float(self.significance) < 1.0:
            raise ValueError("significance must lie strictly between 0 and 1")
        if int(self.moving_average_window) < 1:
            raise ValueError("moving_average_window must be >= 1")
        if self.temporal_scheme not in TEMPORAL_SCHEMES:
            raise ValueError(
                f"unknown temporal_scheme {self.temporal_scheme!r}; "
                f"expected one of {sorted(TEMPORAL_SCHEMES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "significance": float(self.significance),
            "moving_average_window": int(self.moving_average_window),
            "temporal_scheme": str(self.temporal_scheme),
            "hysteresis": bool(self.hysteresis),
        }


@dataclass(frozen=True)
class SicamsTuningGrid:
    """Pre-declared validation-only candidate settings."""

    significances: tuple[float, ...] = (0.01, 0.05, 0.10)
    moving_average_windows: tuple[int, ...] = (1, 6, 12)
    selection_metric: str = "mcc"

    def __post_init__(self) -> None:
        if not self.significances:
            raise ValueError("significances must not be empty")
        if not self.moving_average_windows:
            raise ValueError("moving_average_windows must not be empty")
        for alpha in self.significances:
            if not 0.0 < float(alpha) < 1.0:
                raise ValueError("grid significances must lie between 0 and 1")
        for window in self.moving_average_windows:
            if int(window) < 1:
                raise ValueError("grid moving-average windows must be >= 1")
        if self.selection_metric not in {
            "mcc",
            "balanced_accuracy",
            "attack_average_precision",
        }:
            raise ValueError(
                "selection_metric must be one of mcc, balanced_accuracy, "
                "attack_average_precision"
            )


@dataclass(frozen=True)
class ClusterModel:
    cluster_id: int
    n_k: int
    s: int
    mean: np.ndarray
    covariance: np.ndarray
    cholesky: np.ndarray
    covariance_rank: int
    condition_number: float
    upper_threshold: float
    lower_threshold: float


@dataclass
class SicamsResult:
    t2: np.ndarray
    tf2: np.ndarray
    moving_average: np.ndarray
    flags: np.ndarray
    upper_threshold: np.ndarray
    lower_threshold: np.ndarray
    cluster_ids: np.ndarray
    config: SicamsConfig
    provenance: dict[str, Any]
    feature_names: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "t2": _floats(self.t2),
            "tf2": _floats(self.tf2),
            "moving_average": _floats(self.moving_average),
            "flags": _ints(self.flags),
            "upper_threshold": _floats(self.upper_threshold),
            "lower_threshold": _floats(self.lower_threshold),
            "cluster_ids": _ints(self.cluster_ids),
        }


@dataclass(frozen=True)
class ValidationSelection:
    config: SicamsConfig
    metric: str
    score: float | None
    candidates: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]


def temporal_cluster_labels(
    timestamps: Iterable[Any], scheme: str
) -> np.ndarray:
    """Map timestamps to declared temporal clusters.

    ``hour_of_day`` / ``ltown_area_a``: clock hour 0..23.
    ``two_hour_blocks`` / ``ltown_area_c``: two-hour blocks 0..11.
    ``day_of_week``: Monday=0 .. Sunday=6.
    ``month``: January=1 .. December=12 (kept as-is, not zero-based).
    ``hour_of_week``: ``day_of_week * 24 + hour`` (0..167).
    ``ltown_area_b``: one cluster.

    These are the calendar keys the paper names (hour of day, day of week,
    month) plus the two L-Town native groupings.  Statistical merging of
    calendar keys is a separate, training-only stage in
    :mod:`baselines.sicams_stages`.
    """
    if scheme not in TEMPORAL_SCHEMES:
        raise ValueError(
            f"unknown temporal scheme {scheme!r}; expected one of "
            f"{sorted(TEMPORAL_SCHEMES)}"
        )
    index = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
    if scheme in (SCHEME_HOUR_OF_DAY, SCHEME_LTOWN_AREA_A):
        return index.hour.to_numpy(dtype=np.int64)
    if scheme in (SCHEME_TWO_HOUR_BLOCKS, SCHEME_LTOWN_AREA_C):
        return (index.hour // 2).to_numpy(dtype=np.int64)
    if scheme == SCHEME_DAY_OF_WEEK:
        return index.dayofweek.to_numpy(dtype=np.int64)
    if scheme == SCHEME_MONTH:
        return index.month.to_numpy(dtype=np.int64)
    if scheme == SCHEME_HOUR_OF_WEEK:
        return (
            index.dayofweek.to_numpy(dtype=np.int64) * 24
            + index.hour.to_numpy(dtype=np.int64)
        )
    return np.zeros(len(index), dtype=np.int64)


def causal_moving_average(values: Sequence[float] | np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average; entry i uses only values[: i + 1]."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("causal_moving_average expects a one-dimensional series")
    window = int(window)
    if window < 1:
        raise ValueError("window must be >= 1")
    out = np.empty_like(values, dtype=float)
    for i in range(values.size):
        start = max(0, i - window + 1)
        out[i] = float(np.mean(values[start : i + 1]))
    return out


def hysteresis_state(
    scores: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Apply Algorithm 1's upper/lower hysteresis rule."""
    scores = np.asarray(scores, dtype=float)
    upper = np.asarray(upper, dtype=float)
    lower = np.asarray(lower, dtype=float)
    if not (scores.shape == upper.shape == lower.shape):
        raise ValueError("scores, upper and lower must have the same shape")
    flags = np.zeros(scores.size, dtype=int)
    detected = False
    for i in range(scores.size):
        if not detected:
            if scores[i] > upper[i]:
                detected = True
        elif scores[i] < lower[i]:
            detected = False
        flags[i] = int(detected)
    return flags


def pointwise_metrics(
    labels: Sequence[int] | np.ndarray,
    flags: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Pointwise binary detection metrics with attack as the positive class."""
    labels = np.asarray(labels, dtype=int)
    flags = np.asarray(flags, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if not (labels.ndim == flags.ndim == scores.ndim == 1):
        raise ValueError("labels, flags and scores must be one-dimensional")
    if not (labels.size == flags.size == scores.size):
        raise ValueError("labels, flags and scores must have equal length")
    if labels.size == 0:
        raise ValueError("metrics require at least one observation")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("labels must be binary 0/1")
    if not np.all(np.isin(flags, (0, 1))):
        raise ValueError("flags must be binary 0/1")

    tn, fp, fn, tp = confusion_matrix(labels, flags, labels=[0, 1]).ravel()
    n_positive = int(labels.sum())
    both_classes = 0 < n_positive < labels.size
    mcc = float(matthews_corrcoef(labels, flags)) if both_classes else None
    balanced = (
        float(balanced_accuracy_score(labels, flags)) if both_classes else None
    )
    ap = (
        float(average_precision_score(labels, scores))
        if n_positive > 0
        else None
    )
    return {
        "mcc": mcc,
        "balanced_accuracy": balanced,
        "balanced_accuracy_adjusted": False,
        "attack_average_precision": ap,
        "positive_class": "attack",
        "score_for_ap": "causal_moving_average_tf2",
        "scope": "pointwise_binary_detection_only",
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_samples": int(labels.size),
        "n_positive": n_positive,
    }


class SicamsBaseline:
    """Whitening + Hotelling T^2 detector fitted on normal training data."""

    def __init__(self, config: SicamsConfig | None = None) -> None:
        self.config = config or SicamsConfig()
        self.clusters_: dict[int, ClusterModel] = {}
        self.n_features_: int | None = None
        self.feature_names_: list[str] | None = None
        self.assignment_: str | None = None
        self.scheme_: str | None = None

    @property
    def fitted(self) -> bool:
        return self.n_features_ is not None

    def fit(
        self,
        X: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
        cluster_ids: Sequence[int] | np.ndarray | None = None,
        *,
        timestamps: Iterable[Any] | None = None,
        temporal_scheme: str | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> "SicamsBaseline":
        matrix, inferred_names = _as_matrix(X)
        n, s = matrix.shape
        scheme = temporal_scheme or self.config.temporal_scheme
        ids, assignment = _resolve_cluster_ids(
            n, cluster_ids, timestamps, scheme
        )
        names = list(feature_names) if feature_names is not None else inferred_names
        if names is not None and len(names) != s:
            raise ValueError("feature_names length does not match sensor count")

        alpha = float(self.config.significance)
        fitted: dict[int, ClusterModel] = {}
        for cluster_id in np.unique(ids):
            key = int(cluster_id)
            mask = ids == key
            n_k = int(mask.sum())
            if n_k <= s + 2:
                raise InsufficientClusterSamplesError(key, n_k, s)
            block = matrix[mask]
            mean = block.mean(axis=0)
            centered = block - mean
            covariance = np.atleast_2d(centered.T @ centered / (n_k - 1))
            rank = int(np.linalg.matrix_rank(covariance))
            eigenvalues = np.linalg.eigvalsh(covariance)
            try:
                cholesky = np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as exc:
                reason = (
                    "singular" if rank < s else "not positive definite"
                )
                raise SingularCovarianceError(
                    key, n_k, s, rank, eigenvalues, reason
                ) from exc
            if rank < s:
                raise SingularCovarianceError(
                    key, n_k, s, rank, eigenvalues, "rank deficient"
                )
            upper = float(stats.f.ppf(1.0 - alpha, s, n_k - s))
            lower = float((n_k - s) / (n_k - s - 2))
            condition = float(np.linalg.cond(covariance))
            fitted[key] = ClusterModel(
                cluster_id=key,
                n_k=n_k,
                s=s,
                mean=mean.copy(),
                covariance=covariance.copy(),
                cholesky=cholesky.copy(),
                covariance_rank=rank,
                condition_number=condition,
                upper_threshold=upper,
                lower_threshold=lower,
            )

        self.clusters_ = fitted
        self.n_features_ = s
        self.feature_names_ = names
        self.assignment_ = assignment
        self.scheme_ = scheme if assignment != "explicit_cluster_ids" else None
        return self

    def predict(
        self,
        X: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
        cluster_ids: Sequence[int] | np.ndarray | None = None,
        *,
        timestamps: Iterable[Any] | None = None,
        temporal_scheme: str | None = None,
    ) -> SicamsResult:
        if not self.fitted:
            raise NotFittedError("SicamsBaseline.predict called before fit")
        matrix, _ = _as_matrix(X)
        if matrix.shape[1] != self.n_features_:
            raise FeatureDimensionError(
                f"prediction has {matrix.shape[1]} sensors; model was fitted "
                f"with {self.n_features_}"
            )
        if cluster_ids is None and timestamps is None:
            if self.assignment_ != "single_cluster":
                raise ValueError(
                    "fitted temporal regimes require timestamps or cluster_ids; "
                    "predict must not silently replace them with cluster zero"
                )
        scheme = temporal_scheme or self.config.temporal_scheme
        ids, assignment = _resolve_cluster_ids(
            matrix.shape[0], cluster_ids, timestamps, scheme
        )
        unknown = sorted(set(int(k) for k in np.unique(ids)) - set(self.clusters_))
        if unknown:
            raise UnknownClusterError(
                f"prediction uses temporal clusters {unknown} that were not "
                f"present during fit ({sorted(self.clusters_)})"
            )

        t2 = np.empty(matrix.shape[0], dtype=float)
        upper = np.empty(matrix.shape[0], dtype=float)
        lower = np.empty(matrix.shape[0], dtype=float)
        tf2 = np.empty(matrix.shape[0], dtype=float)
        s = self.n_features_
        for cluster_id in np.unique(ids):
            key = int(cluster_id)
            stat = self.clusters_[key]
            mask = ids == key
            residual = matrix[mask] - stat.mean
            whitened = solve_triangular(
                stat.cholesky, residual.T, lower=True, check_finite=True
            )
            t2[mask] = np.einsum("ij,ij->j", whitened, whitened)
            scale = (stat.n_k - s) / (s * (stat.n_k - 1))
            tf2[mask] = scale * t2[mask]
            upper[mask] = stat.upper_threshold
            lower[mask] = stat.lower_threshold

        moving = causal_moving_average(tf2, self.config.moving_average_window)
        if self.config.hysteresis:
            flags = hysteresis_state(moving, upper, lower)
        else:
            flags = (moving > upper).astype(int)

        provenance = self._provenance(assignment)
        return SicamsResult(
            t2=t2,
            tf2=tf2,
            moving_average=moving,
            flags=flags,
            upper_threshold=upper,
            lower_threshold=lower,
            cluster_ids=ids,
            config=self.config,
            provenance=provenance,
            feature_names=list(self.feature_names_) if self.feature_names_ else None,
        )

    def cluster_table(self) -> list[dict[str, Any]]:
        if not self.fitted:
            raise NotFittedError("SicamsBaseline.cluster_table called before fit")
        return [
            {
                "cluster_id": stat.cluster_id,
                "n_k": stat.n_k,
                "s": stat.s,
                "covariance_rank": stat.covariance_rank,
                "condition_number": stat.condition_number,
                "upper_threshold": stat.upper_threshold,
                "lower_threshold": stat.lower_threshold,
            }
            for stat in sorted(self.clusters_.values(), key=lambda c: c.cluster_id)
        ]

    def _provenance(self, assignment: str) -> dict[str, Any]:
        table = self.cluster_table()
        return {
            "schema": SCHEMA,
            "method": "SICAMS",
            "paper": {
                "citation": PAPER,
                "url": PAPER_URL,
                "equations_implemented": {
                    "whitening": "eq. (5), Cholesky solve with W^T W = S^-1",
                    "t2": "eq. (9), sample covariance with n_k - 1 denominator",
                    "f_rescaling": "eq. (10), TF^2 = (n_k-s)/(s(n_k-1)) T^2",
                    "upper_threshold": "eq. (13), F_{1-alpha}(s, n_k-s)",
                    "lower_threshold": "eq. (14), (n_k-s)/(n_k-s-2)",
                    "moving_average": "Algorithm 1, causal trailing window",
                    "hysteresis": "Algorithm 1, theta1 entry / theta0 exit",
                },
            },
            "implementation": {
                "whitening": (
                    "lower Cholesky L of S, z = L^-1 (x - mu); the preprint's "
                    "W = L^T notation is treated as the equivalent inverse-"
                    "factor convention W^T W = S^-1"
                ),
                "covariance": "unbiased sample covariance, ddof=1",
                "singularity_policy": "raise SingularCovarianceError with rank",
                "feature_count": self.n_features_,
                "feature_names": (
                    list(self.feature_names_) if self.feature_names_ else None
                ),
                "config": self.config.to_dict(),
                "cluster_assignment": assignment,
            },
            "temporal_clusters": {
                "assignment": assignment,
                "declared_scheme": self.scheme_,
                "n_clusters": len(table),
                "clusters": table,
            },
            "native_vs_adaptation": {
                "supplied_scheme": self.config.temporal_scheme,
                "scheme_is_native_ltown": (
                    self.config.temporal_scheme in NATIVE_LTOWN_SCHEMES
                ),
                "native_sicams": {
                    "ltown_area_a": (
                        "hourly temporal clusters (24); 29 pressure + 2 flow + "
                        "1 tank-level features"
                    ),
                    "ltown_area_c": (
                        "two-hour temporal clusters (12); 3 pressure + tank-level "
                        "features"
                    ),
                    "ltown_area_b": "single temporal cluster; scalar pressure",
                    "thresholds": "theoretical F quantiles, not tuned on test",
                },
                "adaptations": {
                    "anytown_batadal": (
                        "hour-of-day temporal clusters over one spatial cluster; "
                        "dataset labels are attacks/anomalies, not native leaks"
                    ),
                    "pointwise_evaluation": (
                        "binary pointwise comparison only; no event-level metrics"
                    ),
                    "spatial_clusters": (
                        "the caller must supply one area/sensor set at a time; no "
                        "spatial clustering algorithm is implemented"
                    ),
                },
            },
            "missing_original_settings": [
                "The preprint does not report the significance level alpha used "
                "for its L-Town results; callers must supply it.",
                "Exact leak-free training intervals for L-Town Areas A/B/C are "
                "not enumerated; callers must supply a calibration interval.",
                "The retraining schedule for coexisting/background leaks is only "
                "described qualitatively.",
                "PELT penalty and the Student t-test variant are not reported; "
                "the stages module exposes both as explicit inputs.",
                "The slope threshold delta is not reported; redundancy removal "
                "is disabled unless the caller supplies it.",
                "The spatial partitioning algorithm is assumed supplied; the "
                "stages module offers a training-only correlation clustering "
                "adaptation but does not claim it is the paper's method.",
                "Temporal aggregation beyond Area A hourly, Area C two-hour and "
                "Area B single is not specified.",
                "EPANET simulation settings (10% demand noise, number of runs) "
                "are not used for SCADA adaptation.",
            ],
            "omitted_native_components": [
                "event-level leak-hours and detection-delay metrics",
                "exact paper PELT/Student/slope hyperparameters (not reported)",
                "original L-Town spatial partition and sensor-to-node mapping",
                "Wilson-Hilferty display transform is not applied to scores",
            ],
            "extended_stages_module": {
                "module": "baselines.sicams_stages",
                "implemented": [
                    "PELT structural change-point segmentation",
                    "Algorithm 2 change-point classification/post-processing",
                    "training-only calendar/statistical temporal partitioning",
                    "supplied or training-only spatial sensor partitioning",
                    "sensor-fault versus hydraulic anomaly discrimination",
                    "Laplacian pre-localization (Eq. 16-17)",
                    "leak-volume linear/log-linear regression (Eq. 19-20 forms)",
                ],
                "detection_only_api": "SicamsBaseline.predict / SicamsFullPipeline.detect",
                "full_api": "SicamsFullPipeline.run_full (supplied inputs only)",
            },
            "determinism": {
                "json_sort_keys": True,
                "random_state": None,
                "refit_on_evaluation": False,
                "threshold_tuning": "validation_only",
            },
        }


def select_config_on_validation(
    train_X: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
    validation_X: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
    validation_labels: Sequence[int] | np.ndarray,
    *,
    train_clusters: Sequence[int] | np.ndarray | None = None,
    validation_clusters: Sequence[int] | np.ndarray | None = None,
    train_timestamps: Iterable[Any] | None = None,
    validation_timestamps: Iterable[Any] | None = None,
    temporal_scheme: str | None = None,
    base_config: SicamsConfig | None = None,
    grid: SicamsTuningGrid | None = None,
) -> ValidationSelection:
    """Pick a pre-declared config using validation data only.

    The model is fitted on ``train_*`` for every candidate and scored on
    ``validation_*``. ``validation_labels`` are never used for fitting, and no
    test data enter this function. Ties are resolved by the declared grid order.
    """
    base = base_config if base_config is not None else SicamsConfig()
    grid = grid or SicamsTuningGrid()
    scheme = temporal_scheme or base.temporal_scheme
    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_config: SicamsConfig | None = None

    for alpha in grid.significances:
        for window in grid.moving_average_windows:
            config = replace(
                base,
                significance=float(alpha),
                moving_average_window=int(window),
                temporal_scheme=scheme,
            )
            model = SicamsBaseline(config).fit(
                train_X,
                train_clusters,
                timestamps=train_timestamps,
                temporal_scheme=scheme,
            )
            result = model.predict(
                validation_X,
                validation_clusters,
                timestamps=validation_timestamps,
                temporal_scheme=scheme,
            )
            metrics = pointwise_metrics(
                validation_labels, result.flags, result.moving_average
            )
            score = metrics[grid.selection_metric]
            record = {
                "significance": float(alpha),
                "moving_average_window": int(window),
                "selection_metric": grid.selection_metric,
                "validation_score": score,
                "validation_metrics": metrics,
            }
            candidates.append(record)
            if best is None or _better(score, best["validation_score"]):
                best = record
                best_config = config

    assert best is not None and best_config is not None
    provenance = {
        "selection_data": "validation_only",
        "test_data_used": False,
        "labels_used_for_fit": False,
        "selection_metric": grid.selection_metric,
        "grid": {
            "significances": [float(a) for a in grid.significances],
            "moving_average_windows": [
                int(w) for w in grid.moving_average_windows
            ],
        },
        "selected": {
            "significance": best["significance"],
            "moving_average_window": best["moving_average_window"],
            "validation_score": best["validation_score"],
        },
    }
    return ValidationSelection(
        config=best_config,
        metric=grid.selection_metric,
        score=best["validation_score"],
        candidates=tuple(candidates),
        provenance=provenance,
    )


def build_artifact(
    result: SicamsResult,
    *,
    labels: Sequence[int] | np.ndarray | None = None,
    dataset: dict[str, Any] | None = None,
    extra_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provenance = dict(result.provenance)
    if extra_provenance:
        provenance["run"] = dict(extra_provenance)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "config": result.config.to_dict(),
        "provenance": provenance,
        "predictions": result.to_dict(),
    }
    if labels is not None:
        payload["metrics"] = pointwise_metrics(
            labels, result.flags, result.moving_average
        )
    if dataset is not None:
        payload["dataset"] = dict(dataset)
    return payload


def artifact_json(
    result: SicamsResult,
    *,
    labels: Sequence[int] | np.ndarray | None = None,
    dataset: dict[str, Any] | None = None,
    extra_provenance: dict[str, Any] | None = None,
    indent: int = 2,
) -> str:
    payload = build_artifact(
        result,
        labels=labels,
        dataset=dataset,
        extra_provenance=extra_provenance,
    )
    return json.dumps(
        _jsonable(payload), sort_keys=True, indent=indent, allow_nan=False
    )


def write_artifact(
    path: str | Path,
    result: SicamsResult,
    *,
    labels: Sequence[int] | np.ndarray | None = None,
    dataset: dict[str, Any] | None = None,
    extra_provenance: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        artifact_json(
            result,
            labels=labels,
            dataset=dataset,
            extra_provenance=extra_provenance,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def _as_matrix(
    X: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
) -> tuple[np.ndarray, list[str] | None]:
    names: list[str] | None = None
    if isinstance(X, pd.DataFrame):
        names = [str(c) for c in X.columns]
        matrix = X.to_numpy(dtype=float)
    else:
        matrix = np.asarray(X, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError("X must be a 2-D array or DataFrame")
    if matrix.shape[0] == 0:
        raise ValueError("X must contain at least one observation")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("X contains NaN or infinite values")
    return matrix, names


def _resolve_cluster_ids(
    n: int,
    cluster_ids: Sequence[int] | np.ndarray | None,
    timestamps: Iterable[Any] | None,
    scheme: str,
) -> tuple[np.ndarray, str]:
    if cluster_ids is not None:
        ids = np.asarray(cluster_ids)
        if ids.ndim != 1 or ids.size != n:
            raise ValueError("cluster_ids must be one-dimensional and match X")
        if not np.all(np.isfinite(ids)):
            raise ValueError("cluster_ids must be finite")
        return ids.astype(np.int64), "explicit_cluster_ids"
    if timestamps is not None:
        ids = temporal_cluster_labels(timestamps, scheme)
        if ids.size != n:
            raise ValueError("timestamps length must match X")
        return ids, f"timestamps:{scheme}"
    return np.zeros(n, dtype=np.int64), "single_cluster"


def _better(candidate: float | None, incumbent: float | None) -> bool:
    if candidate is None:
        return False
    if incumbent is None:
        return True
    if math.isnan(candidate):
        return False
    if math.isnan(incumbent):
        return True
    return candidate > incumbent


def _floats(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float).tolist()]


def _ints(values: np.ndarray) -> list[int]:
    return [int(v) for v in np.asarray(values, dtype=int).tolist()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
