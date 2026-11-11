"""SICAMS stages omitted from the detection-only core.

Reference: O. Melnikov, Y. Dorofieiev, Y. Shakhnovskiy, H. Truong,
V. Degeler, "A Multivariate Statistical Framework for Detection,
Classification and Pre-localization of Anomalies in Water Distribution
Networks", Expert Systems With Applications 313 (2026) 131450,
doi:10.1016/j.eswa.2026.131450.

The paper's Stage 1/2 detector is implemented in :mod:`baselines.sicams`.
This module adds the later stages *only where the paper specifies them*:

* structural change-point (SCP) segmentation with PELT and Algorithm 2
  (Section 3.4, Algorithm 2);
* temporal partitioning by calendar keys and training-only statistical
  merging (Section 3.3, Step 1b);
* spatial sensor partitioning, supplied by the caller or learned from the
  training rows only (Section 3.3, Step 1a);
* sensor-fault versus hydraulic-anomaly discrimination (Section 5.3,
  Eqs. 15-16 and the out-of-range count M_a);
* Laplacian pre-localization (Section 3.5, Eqs. 16-17);
* leak-volume regression in the paper's linear and log-linear forms
  (Section 6.1, Eqs. 19-20).

Every quantity that the paper leaves unspecified is an explicit input or a
declared, documented adaptation.  In particular, significance levels, the
normal calibration interval, sensor-to-node mappings, operational ranges,
PELT penalties, slope thresholds and regression targets are never inferred
from benchmark leak labels or test rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from baselines.sicams import (
    SCHEME_HOUR_OF_DAY,
    SCHEME_LTOWN_AREA_B,
    SCHEME_LTOWN_AREA_C,
    FeatureDimensionError,
    NotFittedError,
    SicamsBaseline,
    SicamsConfig,
    SicamsError,
    SicamsResult,
    UnknownClusterError,
    causal_moving_average,
    temporal_cluster_labels,
)

STAGES_SCHEMA = "baselines/sicams/stages/v2"

RESERVED_COLUMNS = frozenset(
    {
        "time_seconds",
        "timestamp",
        "attack_active",
        "attack_class",
        "physical_impact",
        "delivery_or_command_modified",
    }
)
TRUTH_PREFIXES = ("leak_", "demand_", "emitter_")

# Paper Eqs. (19)-(20) report coefficients fitted to L-Town 2018 only,
# excluding leaks p257 and p427.  They are recorded for provenance and are
# never applied automatically: a run must fit its own coefficients from
# supplied training targets.
PAPER_VOLUME_LINEAR: dict[str, Any] = {
    "intercept": -4.038,
    "slope": 15.087,
    "r2": 0.846,
    "se_intercept": 0.110,
    "se_slope": 0.069,
    "source": "Eq. (19), p.18, L-Town 2018 excluding p257/p427",
}
PAPER_VOLUME_LOG_LINEAR: dict[str, Any] = {
    "intercept": 2.197,
    "slope": 1.576,
    "r2": 0.923,
    "se_intercept": 0.004,
    "se_slope": 0.005,
    "source": "Eq. (20), p.18, L-Town 2018 excluding p257/p427",
}


class SicamsStageError(SicamsError):
    """Base class for explicit later-stage failures."""


class MissingSuppliedInputError(SicamsStageError):
    """Raised when a stage needs an input the caller did not supply."""


class InvalidPartitionError(SicamsStageError):
    """Raised when a temporal/spatial partition is malformed."""


# ---------------------------------------------------------------------------
# shared small helpers
# ---------------------------------------------------------------------------


def _as_matrix(
    X: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str] | None]:
    names: list[str] | None = None
    if isinstance(X, pd.DataFrame):
        if feature_names is not None:
            missing = [c for c in feature_names if c not in X.columns]
            if missing:
                raise FeatureDimensionError(
                    f"frame is missing fitted feature columns: {missing}"
                )
            frame = X.loc[:, list(feature_names)]
            names = [str(c) for c in frame.columns]
            matrix = frame.to_numpy(dtype=float)
        else:
            names = [str(c) for c in X.columns]
            matrix = X.to_numpy(dtype=float)
    else:
        matrix = np.asarray(X, dtype=float)
        if feature_names is not None:
            names = [str(c) for c in feature_names]
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError("X must be a 2-D array or DataFrame")
    if matrix.shape[0] == 0:
        raise ValueError("X must contain at least one observation")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("X contains NaN or infinite values")
    if names is not None and len(names) != matrix.shape[1]:
        raise ValueError("feature_names length does not match X")
    return matrix, names


def _as_timestamps(timestamps: Iterable[Any] | None) -> pd.DatetimeIndex | None:
    if timestamps is None:
        return None
    return pd.DatetimeIndex(pd.to_datetime(list(timestamps)))


def select_observed_columns(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    exclude_truth: bool = True,
    drop_constant: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """Select observed sensor columns without touching truth/label columns.

    When ``feature_columns`` is supplied it is used verbatim (after checking
    presence).  Otherwise numeric, non-boolean columns are selected, excluding
    ``RESERVED_COLUMNS`` and, by default, ``leak_*``/``demand_*``/``emitter_*``
    truth columns.  Constants are dropped only when ``drop_constant`` is true.
    The returned report records every exclusion.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("select_observed_columns expects a DataFrame")
    if feature_columns is not None:
        names = [str(c) for c in feature_columns]
        missing = [c for c in names if c not in frame.columns]
        if missing:
            raise FeatureDimensionError(f"frame is missing columns: {missing}")
        return names, {
            "source": "supplied_feature_columns",
            "excluded": [],
            "dropped_constant": [],
        }

    selected: list[str] = []
    excluded: list[dict[str, str]] = []
    for column in frame.columns:
        name = str(column)
        if name in RESERVED_COLUMNS:
            excluded.append({"column": name, "reason": "reserved label/provenance"})
            continue
        if exclude_truth and name.startswith(TRUTH_PREFIXES):
            excluded.append({"column": name, "reason": "benchmark truth prefix"})
            continue
        series = frame[column]
        if pd.api.types.is_bool_dtype(series):
            excluded.append({"column": name, "reason": "boolean column"})
            continue
        if not pd.api.types.is_numeric_dtype(series):
            excluded.append({"column": name, "reason": "non-numeric column"})
            continue
        selected.append(name)

    dropped: list[str] = []
    if drop_constant:
        kept: list[str] = []
        for name in selected:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            if np.all(np.isfinite(values)) and np.ptp(values) == 0.0:
                dropped.append(name)
            else:
                kept.append(name)
        selected = kept
    if not selected:
        raise FeatureDimensionError("no observed numeric sensor columns remain")
    return selected, {
        "source": "observed_numeric_columns",
        "excluded": excluded,
        "dropped_constant": dropped,
    }


# ---------------------------------------------------------------------------
# temporal partitioning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalPartition:
    scheme: str
    source: str
    n_clusters: int
    base_scheme: str | None = None
    label_map: Mapping[int, int] | None = None
    merge_significance: float | None = None

    def apply(
        self,
        timestamps: Iterable[Any] | None = None,
        cluster_ids: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        if cluster_ids is not None:
            ids = np.asarray(cluster_ids, dtype=np.int64)
            if ids.ndim != 1:
                raise ValueError("cluster_ids must be one-dimensional")
            return ids
        if timestamps is None:
            raise ValueError("timestamps or cluster_ids are required")
        index = _as_timestamps(timestamps)
        assert index is not None
        if self.label_map is None:
            return temporal_cluster_labels(index, self.scheme)
        base = temporal_cluster_labels(index, str(self.base_scheme))
        mapped = np.empty(base.size, dtype=np.int64)
        for i, value in enumerate(base):
            try:
                mapped[i] = int(self.label_map[int(value)])
            except KeyError as exc:
                raise UnknownClusterError(
                    f"temporal base cluster {int(value)} was not present during "
                    "training; refusing to invent a mapping"
                ) from exc
        return mapped

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "source": self.source,
            "n_clusters": int(self.n_clusters),
            "base_scheme": self.base_scheme,
            "label_map": (
                {int(k): int(v) for k, v in sorted(self.label_map.items())}
                if self.label_map is not None
                else None
            ),
            "merge_significance": self.merge_significance,
        }


def _hotelling_two_sample_pvalue(a: np.ndarray, b: np.ndarray) -> float | None:
    """Two-sample Hotelling T^2 p-value for equal covariance."""
    n1, n2 = a.shape[0], b.shape[0]
    s = a.shape[1]
    if n1 < 2 or n2 < 2:
        return None
    df2 = n1 + n2 - s - 1
    if df2 <= 0:
        return None
    diff = a.mean(axis=0) - b.mean(axis=0)
    s1 = np.atleast_2d(np.cov(a, rowvar=False, ddof=1))
    s2 = np.atleast_2d(np.cov(b, rowvar=False, ddof=1))
    pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    try:
        inv = np.linalg.inv(pooled)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(pooled)
    t2 = float(n1 * n2 / (n1 + n2) * diff @ inv @ diff)
    f_stat = float(df2 / (s * (n1 + n2 - 2)) * t2)
    return float(stats.f.sf(f_stat, s, df2))


def learn_temporal_partition(
    timestamps: Iterable[Any],
    X: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
    *,
    base_scheme: str = SCHEME_HOUR_OF_DAY,
    significance: float = 0.05,
) -> TemporalPartition:
    """Merge adjacent calendar clusters that are statistically similar.

    The paper names calendar structure and statistical similarity as the two
    inputs to Step 1b but does not give a merging algorithm.  This is a
    declared training-only adaptation: adjacent base clusters are merged when
    a two-sample Hotelling T^2 test on the *training rows* has ``p >= alpha``.
    """
    if not 0.0 < float(significance) < 1.0:
        raise ValueError("significance must lie strictly between 0 and 1")
    matrix, _ = _as_matrix(X)
    index = _as_timestamps(timestamps)
    assert index is not None
    if index.size != matrix.shape[0]:
        raise ValueError("timestamps length must match X")
    base = temporal_cluster_labels(index, base_scheme)
    labels = sorted(int(v) for v in np.unique(base))

    groups: list[tuple[list[int], np.ndarray]] = []
    for label in labels:
        rows = matrix[base == label]
        if not groups:
            groups.append(([label], rows))
            continue
        previous_labels, previous_rows = groups[-1]
        p_value = _hotelling_two_sample_pvalue(previous_rows, rows)
        if p_value is None or p_value >= float(significance):
            groups[-1] = (previous_labels + [label], np.vstack([previous_rows, rows]))
        else:
            groups.append(([label], rows))

    label_map: dict[int, int] = {}
    for new_label, (old_labels, _) in enumerate(groups):
        for old in old_labels:
            label_map[int(old)] = int(new_label)
    return TemporalPartition(
        scheme=f"learned:{base_scheme}",
        source="training_only_hotelling_merge",
        n_clusters=len(groups),
        base_scheme=base_scheme,
        label_map=label_map,
        merge_significance=float(significance),
    )


# ---------------------------------------------------------------------------
# spatial partitioning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialPartition:
    clusters: Mapping[str, tuple[int, ...]]
    feature_names: tuple[str, ...]
    source: str
    correlation_threshold: float | None = None

    def indices(self, cluster: str) -> tuple[int, ...]:
        return self.clusters[cluster]

    def names(self, cluster: str) -> tuple[str, ...]:
        return tuple(self.feature_names[i] for i in self.clusters[cluster])

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "correlation_threshold": self.correlation_threshold,
            "clusters": {
                name: list(self.names(name)) for name in sorted(self.clusters)
            },
        }


def build_spatial_partition(
    feature_names: Sequence[str],
    *,
    supplied: Mapping[str, Sequence[str]] | None = None,
    train_X: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame | None = None,
    correlation_threshold: float | None = None,
) -> SpatialPartition:
    """Build a disjoint spatial partition of the sensor index set.

    ``supplied`` maps cluster names to sensor names and is validated to cover
    every feature exactly once.  Otherwise, when training rows and a
    correlation threshold are supplied, connected components of the training
    absolute-correlation graph are used.  The paper says the partition may rely
    on the physical layout or a clustering algorithm; the correlation variant
    is labelled as an adaptation, not as the paper's unspecified algorithm.
    """
    names = [str(n) for n in feature_names]
    if len(set(names)) != len(names):
        raise InvalidPartitionError("feature_names contains duplicates")
    position = {name: i for i, name in enumerate(names)}

    if supplied is not None:
        clusters: dict[str, tuple[int, ...]] = {}
        seen: set[int] = set()
        for cluster, members in supplied.items():
            idx: list[int] = []
            for member in members:
                if member not in position:
                    raise InvalidPartitionError(
                        f"spatial cluster {cluster!r} names unknown sensor {member!r}"
                    )
                i = position[member]
                if i in seen:
                    raise InvalidPartitionError(
                        f"sensor {member!r} appears in more than one spatial cluster"
                    )
                seen.add(i)
                idx.append(i)
            if not idx:
                raise InvalidPartitionError(f"spatial cluster {cluster!r} is empty")
            clusters[str(cluster)] = tuple(sorted(idx))
        missing = sorted(set(range(len(names))) - seen)
        if missing:
            raise InvalidPartitionError(
                "supplied spatial partition does not cover sensors: "
                + ", ".join(names[i] for i in missing)
            )
        return SpatialPartition(
            clusters=clusters,
            feature_names=tuple(names),
            source="supplied_topology_mapping",
        )

    if train_X is not None and correlation_threshold is not None:
        matrix, _ = _as_matrix(train_X)
        if matrix.shape[1] != len(names):
            raise ValueError("train_X width does not match feature_names")
        threshold = float(correlation_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("correlation_threshold must lie in [0, 1]")
        if len(names) == 1:
            return SpatialPartition(
                clusters={"spatial_0": (0,)},
                feature_names=tuple(names),
                source="learned_training_correlation",
                correlation_threshold=threshold,
            )
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(matrix, rowvar=False)
        adjacency = np.abs(corr) >= threshold
        adjacency = np.nan_to_num(adjacency, nan=False).astype(bool)
        np.fill_diagonal(adjacency, False)
        remaining = set(range(len(names)))
        components: list[tuple[int, ...]] = []
        while remaining:
            seed = min(remaining)
            stack = [seed]
            component: set[int] = set()
            while stack:
                node = stack.pop()
                if node in component:
                    continue
                component.add(node)
                stack.extend(
                    int(neighbour)
                    for neighbour in np.flatnonzero(adjacency[node])
                    if int(neighbour) in remaining and int(neighbour) not in component
                )
            remaining -= component
            components.append(tuple(sorted(component)))
        clusters = {
            f"spatial_{i}": component
            for i, component in enumerate(sorted(components))
        }
        return SpatialPartition(
            clusters=clusters,
            feature_names=tuple(names),
            source="learned_training_correlation",
            correlation_threshold=threshold,
        )

    return SpatialPartition(
        clusters={"spatial_0": tuple(range(len(names)))},
        feature_names=tuple(names),
        source="single_cluster",
    )


# ---------------------------------------------------------------------------
# PELT and Algorithm 2
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChangePointSegment:
    start: int
    end: int
    mean: float
    slope: float | None
    label: str
    direction: str | None
    p_value: float | None
    test: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": int(self.start),
            "end": int(self.end),
            "mean": float(self.mean),
            "slope": self.slope,
            "label": self.label,
            "direction": self.direction,
            "p_value": self.p_value,
            "test": self.test,
        }


@dataclass(frozen=True)
class ClassifiedChangePoint:
    index: int
    left_interval: tuple[int, int]
    right_interval: tuple[int, int]
    left_mean: float
    right_mean: float
    p_value: float | None
    label: str
    redundant: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "left_interval": [int(self.left_interval[0]), int(self.left_interval[1])],
            "right_interval": [int(self.right_interval[0]), int(self.right_interval[1])],
            "left_mean": float(self.left_mean),
            "right_mean": float(self.right_mean),
            "p_value": self.p_value,
            "label": self.label,
            "redundant": bool(self.redundant),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ChangePointResult:
    change_points: tuple[int, ...]
    raw_change_points: tuple[int, ...]
    segments: tuple[ChangePointSegment, ...]
    classified: tuple[ClassifiedChangePoint, ...]
    removed_redundant: tuple[int, ...]
    penalty: float
    penalty_rule: str
    significance: float
    slope_threshold: float | None
    online: bool
    latency: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_points": [int(v) for v in self.change_points],
            "raw_change_points": [int(v) for v in self.raw_change_points],
            "segments": [segment.to_dict() for segment in self.segments],
            "classified": [point.to_dict() for point in self.classified],
            "removed_redundant": [int(v) for v in self.removed_redundant],
            "penalty": float(self.penalty),
            "penalty_rule": self.penalty_rule,
            "significance": float(self.significance),
            "slope_threshold": self.slope_threshold,
            "online": False,
            "latency": self.latency,
        }


def _pelt(
    values: np.ndarray,
    *,
    penalty: float | None,
    min_segment: int,
) -> tuple[tuple[int, ...], float, str, float]:
    y = np.asarray(values, dtype=float)
    n = y.size
    if n == 0:
        raise ValueError("PELT requires a non-empty series")
    if min_segment < 1:
        raise ValueError("min_segment must be >= 1")
    if penalty is None:
        variance = float(np.var(y, ddof=1)) if n > 1 else 0.0
        if not math.isfinite(variance) or variance <= 0.0:
            penalty_value = 0.0
        else:
            penalty_value = variance * math.log(n)
        penalty_rule = "bic_variance_log_n"
    else:
        penalty_value = float(penalty)
        if penalty_value < 0.0 or not math.isfinite(penalty_value):
            raise ValueError("penalty must be a finite non-negative value")
        penalty_rule = "supplied"

    if n < 2 * min_segment:
        cost = float(np.sum((y - y.mean()) ** 2)) if n else 0.0
        return (), penalty_value, penalty_rule, cost

    prefix = np.concatenate(([0.0], np.cumsum(y)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(y * y)))

    def segment_cost(start: int, stop: int) -> float:
        count = stop - start
        total = float(prefix[stop] - prefix[start])
        total_sq = float(prefix_sq[stop] - prefix_sq[start])
        return total_sq - total * total / count

    objective = np.full(n + 1, np.inf, dtype=float)
    objective[0] = -penalty_value
    last = np.full(n + 1, -1, dtype=np.int64)
    candidates: list[int] = [0]

    for stop in range(1, n + 1):
        best_value = math.inf
        best_start = -1
        remaining: list[int] = []
        for start in candidates:
            if stop - start < min_segment:
                # Not old enough to form a legal new segment, but it may be
                # legal at a later stop, so keep it as a candidate.
                remaining.append(start)
                continue
            candidate_value = objective[start] + segment_cost(start, stop)
            if candidate_value < best_value:
                best_value = candidate_value
                best_start = start
        objective[stop] = (
            math.inf if best_start < 0 else best_value + penalty_value
        )
        last[stop] = best_start
        if best_start >= 0:
            if min_segment == 1:
                # PELT pruning: drop a candidate only when adding the segment
                # penalty cannot improve on the current optimum.
                remaining = [
                    start
                    for start in candidates
                    if objective[start] + segment_cost(start, stop)
                    <= objective[stop]
                ]
            else:
                # With a minimum segment length, plain PELT pruning is not
                # exact: a candidate pruned at ``stop`` can still be optimal
                # for a future stop that is closer than ``min_segment``. Keep
                # every candidate and pay the O(n^2) cost for correctness.
                remaining = list(candidates)
        if stop < n and math.isfinite(objective[stop]):
            remaining.append(stop)
        candidates = remaining

    if n >= min_segment and not math.isfinite(objective[n]):
        raise SicamsStageError("PELT could not find a feasible segmentation")

    change_points: list[int] = []
    cursor = n
    while cursor > 0:
        previous = int(last[cursor])
        if previous < 0:
            raise SicamsStageError("PELT backtracking hit an infeasible boundary")
        if previous > 0:
            change_points.append(previous)
        cursor = previous
    change_points.sort()
    total_cost = float(objective[n])
    return tuple(change_points), penalty_value, penalty_rule, total_cost


def _mean_test_pvalue(
    left: np.ndarray,
    right: np.ndarray,
    method: str,
) -> tuple[float | None, str | None]:
    if left.size < 2 or right.size < 2:
        return None, None
    if method == "paired" or (method == "auto" and left.size == right.size):
        # Algorithm 2 bisects an interval, which gives equal halves for even
        # lengths and halves that differ by one for odd lengths.  A paired
        # test needs aligned samples, so compare the common prefix; at most one
        # boundary sample is dropped.
        size = min(left.size, right.size)
        result = stats.ttest_rel(left[:size], right[:size])
        return float(result.pvalue), "paired_student_t"
    result = stats.ttest_ind(left, right, equal_var=False)
    return float(result.pvalue), "welch_student_t"


def _linear_slope(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    x = np.arange(values.size, dtype=float)
    slope = np.polyfit(x, values, 1)[0]
    return float(slope)


def detect_change_points(
    series: Sequence[float] | np.ndarray,
    *,
    penalty: float | None = None,
    min_segment: int = 1,
    significance: float = 0.05,
    slope_threshold: float | None = None,
    t_test: str = "paired",
) -> ChangePointResult:
    """PELT segmentation plus Algorithm 2 post-processing.

    PELT is applied to the supplied series (in the paper, the moving average of
    TF2).  Algorithm 2 bisects every inter-SCP interval, labels it increasing or
    stable, labels each SCP as anomaly start/end from the adjacent interval
    means, and removes a redundant SCP when two adjacent increasing intervals
    have slopes closer than ``slope_threshold`` (Algorithm 2's pseudocode).
    The paper text and pseudocode disagree on this last condition; the
    pseudocode is followed and the discrepancy is recorded in the provenance.
    """
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError("detect_change_points expects a one-dimensional series")
    if not np.all(np.isfinite(values)):
        raise ValueError("series contains NaN or infinite values")
    if not 0.0 < float(significance) < 1.0:
        raise ValueError("significance must lie strictly between 0 and 1")
    if t_test not in {"auto", "paired", "welch"}:
        raise ValueError("t_test must be auto, paired or welch")

    n = values.size
    raw, penalty_value, penalty_rule, _cost = _pelt(
        values, penalty=penalty, min_segment=min_segment
    )
    boundaries = [0, *raw, n]
    segments: list[ChangePointSegment] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        block = values[start:stop]
        midpoint = start + block.size // 2
        left = values[start:midpoint]
        right = values[midpoint:stop]
        p_value, test = _mean_test_pvalue(left, right, t_test)
        if p_value is None:
            label = "insufficient"
            direction = None
        elif p_value < float(significance):
            label = "increasing"
            direction = "up" if right.mean() > left.mean() else "down"
        else:
            label = "stable"
            direction = None
        segments.append(
            ChangePointSegment(
                start=int(start),
                end=int(stop),
                mean=float(block.mean()),
                slope=_linear_slope(block),
                label=label,
                direction=direction,
                p_value=p_value,
                test=test,
            )
        )

    classified: list[ClassifiedChangePoint] = []
    redundant: set[int] = set()
    for i, index in enumerate(raw):
        left_segment = segments[i]
        right_segment = segments[i + 1]
        left = values[left_segment.start : left_segment.end]
        right = values[right_segment.start : right_segment.end]
        p_value, _test = _mean_test_pvalue(left, right, t_test)
        if p_value is not None and p_value < float(significance):
            label = "anomaly_start" if right.mean() > left.mean() else "anomaly_end"
            reason = None
        else:
            label = "inconclusive"
            reason = "adjacent interval means are not significantly different"
        is_redundant = False
        if (
            slope_threshold is not None
            and left_segment.label == "increasing"
            and right_segment.label == "increasing"
            and left_segment.slope is not None
            and right_segment.slope is not None
        ):
            if abs(left_segment.slope - right_segment.slope) < float(slope_threshold):
                is_redundant = True
                redundant.add(int(index))
                reason = "adjacent increasing slopes are closer than slope_threshold"
        classified.append(
            ClassifiedChangePoint(
                index=int(index),
                left_interval=(int(left_segment.start), int(left_segment.end)),
                right_interval=(int(right_segment.start), int(right_segment.end)),
                left_mean=float(left.mean()),
                right_mean=float(right.mean()),
                p_value=p_value,
                label=label,
                redundant=is_redundant,
                reason=reason,
            )
        )

    final = tuple(int(v) for v in raw if int(v) not in redundant)
    return ChangePointResult(
        change_points=final,
        raw_change_points=tuple(int(v) for v in raw),
        segments=tuple(segments),
        classified=tuple(classified),
        removed_redundant=tuple(sorted(redundant)),
        penalty=float(penalty_value),
        penalty_rule=penalty_rule,
        significance=float(significance),
        slope_threshold=(None if slope_threshold is None else float(slope_threshold)),
        online=False,
        latency="retrospective_full_series",
    )


# ---------------------------------------------------------------------------
# leak-volume regression (paper Eqs. 19-20 model forms)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeakVolumeRegression:
    model: str
    intercept: float
    slope: float
    r2: float | None
    n: int
    window: int
    target_name: str
    published_reference: Mapping[str, Any]
    target_source: str = "supplied_training_targets"

    def predict(self, tf2: Sequence[float] | np.ndarray) -> np.ndarray:
        feature = causal_moving_average(tf2, self.window)
        if self.model == "linear":
            return self.intercept + self.slope * feature
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.expm1(self.intercept + self.slope * np.log(feature))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "intercept": float(self.intercept),
            "slope": float(self.slope),
            "r2": self.r2,
            "n": int(self.n),
            "window": int(self.window),
            "target_name": self.target_name,
            "target_source": self.target_source,
            "published_reference": dict(self.published_reference),
            "published_reference_applied": False,
            "online": True,
            "latency_samples": int(self.window - 1),
        }


def fit_leak_volume_regression(
    tf2: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    *,
    model: str = "linear",
    window: int = 12,
    target_name: str = "total_leak_volume_m3",
) -> LeakVolumeRegression:
    """Fit Eq. (19) linear or Eq. (20) log-linear regression.

    ``targets`` must be training-only leak volumes aligned with ``tf2``.  The
    paper's published coefficients are recorded as a reference but never
    applied; the returned model always uses coefficients learned from the
    supplied targets.
    """
    values = np.asarray(tf2, dtype=float)
    y = np.asarray(targets, dtype=float)
    if values.ndim != 1 or y.ndim != 1:
        raise ValueError("tf2 and targets must be one-dimensional")
    if values.size != y.size:
        raise ValueError("tf2 and targets must have equal length")
    if values.size == 0:
        raise ValueError("regression requires at least one training row")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(y)):
        raise ValueError("tf2 and targets must be finite")
    if np.any(y < 0.0):
        raise ValueError("leak-volume targets must be non-negative")
    if model not in {"linear", "log_linear"}:
        raise ValueError("model must be 'linear' or 'log_linear'")
    if int(window) < 1:
        raise ValueError("window must be >= 1")

    feature = causal_moving_average(values, int(window))
    if model == "linear":
        design = np.column_stack([np.ones_like(feature), feature])
        response = y
        reference = PAPER_VOLUME_LINEAR
    else:
        if np.any(feature <= 0.0):
            raise ValueError("log-linear regression requires a positive moving average")
        design = np.column_stack([np.ones_like(feature), np.log(feature)])
        response = np.log1p(y)
        reference = PAPER_VOLUME_LOG_LINEAR

    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    fitted = design @ coefficients
    residual = float(np.sum((response - fitted) ** 2))
    total = float(np.sum((response - response.mean()) ** 2))
    r2 = None if total <= 0.0 else float(1.0 - residual / total)
    return LeakVolumeRegression(
        model=model,
        intercept=float(coefficients[0]),
        slope=float(coefficients[1]),
        r2=r2,
        n=int(values.size),
        window=int(window),
        target_name=str(target_name),
        published_reference=reference,
    )


# ---------------------------------------------------------------------------
# sensor-wise contributions, discrimination and pre-localization
# ---------------------------------------------------------------------------


def _cluster_ids_for_model(
    model: SicamsBaseline,
    n: int,
    cluster_ids: Sequence[int] | np.ndarray | None,
    timestamps: Iterable[Any] | None,
    temporal_scheme: str | None,
) -> np.ndarray:
    if cluster_ids is not None:
        ids = np.asarray(cluster_ids, dtype=np.int64)
        if ids.ndim != 1 or ids.size != n:
            raise ValueError("cluster_ids must be one-dimensional and match X")
        return ids
    if timestamps is not None:
        scheme = temporal_scheme or model.config.temporal_scheme
        return temporal_cluster_labels(timestamps, scheme)
    return np.zeros(n, dtype=np.int64)


def z_score_matrix(
    model: SicamsBaseline,
    X: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
    *,
    cluster_ids: Sequence[int] | np.ndarray | None = None,
    timestamps: Iterable[Any] | None = None,
    temporal_scheme: str | None = None,
) -> np.ndarray:
    """Eq. (16): per-cluster sensor z-normalization from training moments."""
    if not model.fitted:
        raise NotFittedError("z_score_matrix requires a fitted model")
    matrix, _ = _as_matrix(X)
    if matrix.shape[1] != model.n_features_:
        raise FeatureDimensionError("X width does not match the fitted model")
    ids = _cluster_ids_for_model(
        model, matrix.shape[0], cluster_ids, timestamps, temporal_scheme
    )
    unknown = sorted(set(int(v) for v in np.unique(ids)) - set(model.clusters_))
    if unknown:
        raise UnknownClusterError(f"unknown temporal clusters: {unknown}")
    z = np.empty_like(matrix, dtype=float)
    for key in np.unique(ids):
        stat = model.clusters_[int(key)]
        mask = ids == key
        scale = np.sqrt(np.diag(stat.covariance))
        if np.any(scale <= 0.0):
            raise SicamsStageError(
                f"cluster {int(key)} has a zero-variance sensor; Eq. (16) is undefined"
            )
        z[mask] = (matrix[mask] - stat.mean) / scale
    return z


def sensor_contributions(
    model: SicamsBaseline,
    X: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
    *,
    cluster_ids: Sequence[int] | np.ndarray | None = None,
    timestamps: Iterable[Any] | None = None,
    temporal_scheme: str | None = None,
) -> np.ndarray:
    """Eq. (15): ``c_j = (x - mean)_j * (S^-1 (x - mean))_j``.

    Contributions can be negative because of off-diagonal covariance terms.
    """
    if not model.fitted:
        raise NotFittedError("sensor_contributions requires a fitted model")
    matrix, _ = _as_matrix(X)
    if matrix.shape[1] != model.n_features_:
        raise FeatureDimensionError("X width does not match the fitted model")
    ids = _cluster_ids_for_model(
        model, matrix.shape[0], cluster_ids, timestamps, temporal_scheme
    )
    unknown = sorted(set(int(v) for v in np.unique(ids)) - set(model.clusters_))
    if unknown:
        raise UnknownClusterError(f"unknown temporal clusters: {unknown}")
    out = np.empty_like(matrix, dtype=float)
    for key in np.unique(ids):
        stat = model.clusters_[int(key)]
        mask = ids == key
        residual = matrix[mask] - stat.mean
        solved = np.linalg.solve(stat.covariance, residual.T).T
        out[mask] = residual * solved
    return out


def contribution_shares(z: Sequence[float] | np.ndarray) -> np.ndarray:
    """Section 5.3: ``s_j = z_j^2 / sum_j z_j^2`` for one observation."""
    values = np.asarray(z, dtype=float)
    if values.ndim != 1:
        raise ValueError("contribution_shares expects a one-dimensional row")
    squared = values * values
    total = float(np.sum(squared))
    if total <= 0.0:
        return np.zeros_like(values)
    return squared / total


@dataclass(frozen=True)
class AnomalyDiscrimination:
    status: str
    reason: str | None
    out_of_range_sensors: tuple[str, ...]
    out_of_range_count: int
    max_share: float | None
    shares: Mapping[str, float]
    rule: str
    online: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "out_of_range_sensors": list(self.out_of_range_sensors),
            "out_of_range_count": int(self.out_of_range_count),
            "max_share": self.max_share,
            "shares": {k: float(v) for k, v in self.shares.items()},
            "rule": self.rule,
            "online": bool(self.online),
        }


def discriminate_anomaly(
    raw_values: Sequence[float] | np.ndarray | None,
    z_values: Sequence[float] | np.ndarray,
    feature_names: Sequence[str],
    *,
    detected: bool = True,
    operational_ranges: Mapping[str, tuple[float, float]] | None = None,
    z_threshold: float | None = None,
    concentration_threshold: float | None = None,
    max_out_of_range: int = 1,
) -> AnomalyDiscrimination:
    """Paper Section 5.3: sensor bias versus hydraulic (leak-like) anomaly.

    The primary rule is the paper's ``M_a``: if Hotelling detects an anomaly but
    only one sensor is outside its admissible range, a sensor fault is likely;
    several out-of-range sensors indicate a hydraulic event.  A supplied
    concentration threshold adds the alternative ``s_j`` rule.  If neither
    operational ranges nor a statistical z threshold are supplied, the stage is
    explicitly unavailable rather than guessed.
    """
    z = np.asarray(z_values, dtype=float)
    names = [str(n) for n in feature_names]
    if z.ndim != 1 or z.size != len(names):
        raise ValueError("z_values must match feature_names")
    shares = contribution_shares(z)
    share_map = {name: float(value) for name, value in zip(names, shares)}
    max_share = float(np.max(shares)) if shares.size else None
    if not detected:
        return AnomalyDiscrimination(
            status="not_anomalous",
            reason="detector flag is 0",
            out_of_range_sensors=(),
            out_of_range_count=0,
            max_share=max_share,
            shares=share_map,
            rule="not_applicable",
        )

    out_of_range: list[str] = []
    rule: str
    if operational_ranges is None and z_threshold is None and concentration_threshold is not None:
        if max_share is None:
            raise ValueError("concentration discrimination requires nonzero z values")
        status = (
            "sensor_fault"
            if max_share >= float(concentration_threshold)
            else "hydraulic_anomaly"
        )
        return AnomalyDiscrimination(
            status=status,
            reason=f"single-sensor share {max_share:.3f} compared with {float(concentration_threshold):.3f}",
            out_of_range_sensors=(),
            out_of_range_count=0,
            max_share=max_share,
            shares=share_map,
            rule="adapted_sj_concentration_threshold",
        )
    if operational_ranges is not None:
        raw = None if raw_values is None else np.asarray(raw_values, dtype=float)
        if raw is not None and raw.size != len(names):
            raise ValueError("raw_values must match feature_names")
        missing_ranges: list[str] = []
        for i, name in enumerate(names):
            bounds = operational_ranges.get(name)
            if bounds is None:
                missing_ranges.append(name)
                continue
            low, high = float(bounds[0]), float(bounds[1])
            value = float(z[i]) if raw is None else float(raw[i])
            if raw is None:
                # No raw value: compare the z score against supplied bounds is
                # meaningless, so refuse instead of mixing scales.
                raise MissingSuppliedInputError(
                    "operational_ranges requires raw sensor values"
                )
            if value < low or value > high:
                out_of_range.append(name)
        if missing_ranges:
            return AnomalyDiscrimination(
                status="unavailable",
                reason=f"operational ranges are missing for {sorted(missing_ranges)}",
                out_of_range_sensors=(),
                out_of_range_count=0,
                max_share=max_share,
                shares=share_map,
                rule="paper_M_a_requires_complete_range_coverage",
            )
        rule = (
            "paper_M_a_with_adapted_sj_tiebreak"
            if concentration_threshold is not None
            else "paper_M_a_operational_range"
        )
    elif z_threshold is not None:
        threshold = float(z_threshold)
        if threshold < 0.0:
            raise ValueError("z_threshold must be non-negative")
        out_of_range = [
            name for name, value in zip(names, z) if abs(float(value)) > threshold
        ]
        rule = "statistical_z_range_proxy"
    else:
        return AnomalyDiscrimination(
            status="unavailable",
            reason=(
                "no operational ranges or z threshold supplied; sensor-fault "
                "versus hydraulic discrimination is not identifiable"
            ),
            out_of_range_sensors=(),
            out_of_range_count=0,
            max_share=max_share,
            shares=share_map,
            rule="unavailable",
        )

    count = len(out_of_range)
    if count == 0:
        status = "inconclusive"
        reason = "no sensor exceeds its supplied range"
    elif count <= int(max_out_of_range):
        status = "sensor_fault"
        reason = (
            f"{count} out-of-range sensor(s) <= max_out_of_range="
            f"{int(max_out_of_range)}"
        )
    elif concentration_threshold is not None and max_share is not None:
        if max_share >= float(concentration_threshold):
            status = "sensor_fault"
            reason = (
                f"single-sensor share {max_share:.3f} >= concentration threshold "
                f"{float(concentration_threshold):.3f}"
            )
        else:
            status = "hydraulic_anomaly"
            reason = "several sensors exceed range with a diffuse share pattern"
    else:
        status = "hydraulic_anomaly"
        reason = "several sensors exceed their supplied ranges"
    return AnomalyDiscrimination(
        status=status,
        reason=reason,
        out_of_range_sensors=tuple(out_of_range),
        out_of_range_count=count,
        max_share=max_share,
        shares=share_map,
        rule=rule,
    )


def _normalise_adjacency(
    adjacency: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    neighbours: dict[str, set[str]] = {}
    for node, adjacent in adjacency.items():
        key = str(node)
        neighbours.setdefault(key, set())
        for other in adjacent:
            other_key = str(other)
            neighbours[key].add(other_key)
            neighbours.setdefault(other_key, set()).add(key)
    return {
        node: tuple(sorted(values)) for node, values in sorted(neighbours.items())
    }


@dataclass(frozen=True)
class PrelocalizationResult:
    field: Mapping[str, float]
    sensor_values: Mapping[str, float]
    top_node: str
    top_value: float
    support_nodes: tuple[str, ...]
    method: str = "laplacian_interpolation_eq17"
    online: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": {k: float(v) for k, v in sorted(self.field.items())},
            "sensor_values": {k: float(v) for k, v in sorted(self.sensor_values.items())},
            "top_node": self.top_node,
            "top_value": float(self.top_value),
            "support_nodes": list(self.support_nodes),
            "method": self.method,
            "online": bool(self.online),
        }


def laplacian_prelocalization(
    adjacency: Mapping[str, Sequence[str]],
    sensor_values: Mapping[str, float],
    *,
    tolerance: float = 1e-9,
    max_iter: int = 10000,
) -> PrelocalizationResult:
    """Solve Eq. (17) with fixed sensor-node values.

    The graph and the sensor-to-node mapping are supplied by the caller.  A
    connected component with no sensor makes the Laplace solution non-unique;
    that case raises instead of returning an arbitrary field.
    """
    if not sensor_values:
        raise MissingSuppliedInputError("no sensor-node values supplied")
    graph = _normalise_adjacency(adjacency)
    unknown_sensors = [node for node in sensor_values if node not in graph]
    if unknown_sensors:
        raise InvalidPartitionError(
            f"sensor nodes absent from adjacency: {unknown_sensors}"
        )
    nodes = list(graph)
    fixed = {str(node): float(value) for node, value in sensor_values.items()}
    unknown = [node for node in nodes if node not in fixed]
    field: dict[str, float] = dict(fixed)

    if unknown:
        position = {node: i for i, node in enumerate(unknown)}
        size = len(unknown)
        matrix = np.zeros((size, size), dtype=float)
        rhs = np.zeros(size, dtype=float)
        for node in unknown:
            row = position[node]
            neighbours = graph[node]
            if not neighbours:
                raise InvalidPartitionError(f"node {node!r} has no neighbours")
            matrix[row, row] = 1.0
            for neighbour in neighbours:
                weight = 1.0 / len(neighbours)
                if neighbour in position:
                    matrix[row, position[neighbour]] -= weight
                else:
                    rhs[row] += weight * fixed[neighbour]
        try:
            solved = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError as exc:
            raise SicamsStageError(
                "Laplacian system is singular; a graph component has no sensor"
            ) from exc
        residual = float(np.max(np.abs(matrix @ solved - rhs))) if size else 0.0
        if residual > float(tolerance) * max(1.0, float(np.max(np.abs(rhs)))):
            raise SicamsStageError(
                f"Laplacian solve did not converge (residual={residual:.3e})"
            )
        for node, value in zip(unknown, solved):
            field[node] = float(value)

    top_node = max(field, key=lambda node: field[node])
    return PrelocalizationResult(
        field=field,
        sensor_values=fixed,
        top_node=top_node,
        top_value=float(field[top_node]),
        support_nodes=tuple(sorted(fixed)),
    )


# ---------------------------------------------------------------------------
# later-stage configuration and full pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SicamsStageConfig:
    """Explicit settings for the paper's later stages.

    ``significance`` is the Algorithm 2 test level.  ``pelt_penalty=None`` uses
    a BIC penalty ``variance * log(n)`` and records that choice.  Redundancy
    removal is disabled unless ``slope_threshold`` is supplied because the
    paper does not report delta.
    """

    significance: float = 0.05
    pelt_penalty: float | None = None
    pelt_min_segment: int = 1
    t_test: str = "paired"
    slope_threshold: float | None = None
    sensor_z_threshold: float | None = None
    concentration_threshold: float | None = None
    max_out_of_range: int = 1
    volume_model: str = "linear"
    volume_window: int = 12
    prelocalization_tolerance: float = 1e-9
    prelocalization_max_iter: int = 10000

    def __post_init__(self) -> None:
        if not 0.0 < float(self.significance) < 1.0:
            raise ValueError("significance must lie strictly between 0 and 1")
        if self.pelt_penalty is not None and (
            not math.isfinite(float(self.pelt_penalty))
            or float(self.pelt_penalty) < 0.0
        ):
            raise ValueError("pelt_penalty must be finite and non-negative")
        if int(self.pelt_min_segment) < 1:
            raise ValueError("pelt_min_segment must be >= 1")
        if self.t_test not in {"auto", "paired", "welch"}:
            raise ValueError("t_test must be auto, paired or welch")
        if self.slope_threshold is not None and float(self.slope_threshold) < 0.0:
            raise ValueError("slope_threshold must be non-negative")
        if self.sensor_z_threshold is not None and float(self.sensor_z_threshold) < 0.0:
            raise ValueError("sensor_z_threshold must be non-negative")
        if self.concentration_threshold is not None and not (
            0.0 <= float(self.concentration_threshold) <= 1.0
        ):
            raise ValueError("concentration_threshold must lie in [0, 1]")
        if int(self.max_out_of_range) < 0:
            raise ValueError("max_out_of_range must be non-negative")
        if self.volume_model not in {"linear", "log_linear"}:
            raise ValueError("volume_model must be linear or log_linear")
        if int(self.volume_window) < 1:
            raise ValueError("volume_window must be >= 1")
        if float(self.prelocalization_tolerance) <= 0.0:
            raise ValueError("prelocalization_tolerance must be positive")
        if int(self.prelocalization_max_iter) < 1:
            raise ValueError("prelocalization_max_iter must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "significance": float(self.significance),
            "pelt_penalty": self.pelt_penalty,
            "pelt_min_segment": int(self.pelt_min_segment),
            "t_test": self.t_test,
            "slope_threshold": self.slope_threshold,
            "sensor_z_threshold": self.sensor_z_threshold,
            "concentration_threshold": self.concentration_threshold,
            "max_out_of_range": int(self.max_out_of_range),
            "volume_model": self.volume_model,
            "volume_window": int(self.volume_window),
            "prelocalization_tolerance": float(self.prelocalization_tolerance),
            "prelocalization_max_iter": int(self.prelocalization_max_iter),
        }


@dataclass
class SicamsDetectionResult:
    t2: np.ndarray
    tf2: np.ndarray
    moving_average: np.ndarray
    flags: np.ndarray
    upper_threshold: np.ndarray
    lower_threshold: np.ndarray
    cluster_ids: np.ndarray
    feature_names: list[str]
    clusters: dict[str, SicamsResult]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "t2": [float(v) for v in self.t2],
            "tf2": [float(v) for v in self.tf2],
            "moving_average": [float(v) for v in self.moving_average],
            "flags": [int(v) for v in self.flags],
            "upper_threshold": [float(v) for v in self.upper_threshold],
            "lower_threshold": [float(v) for v in self.lower_threshold],
            "cluster_ids": [int(v) for v in self.cluster_ids],
            "feature_names": list(self.feature_names),
            "clusters": {
                name: result.to_dict() for name, result in sorted(self.clusters.items())
            },
        }


@dataclass
class SicamsFullResult:
    detection: SicamsDetectionResult
    change_points: dict[str, ChangePointResult]
    discrimination: dict[str, Any]
    prelocalization: dict[str, Any]
    leak_volume: dict[str, Any]
    latency: dict[str, Any]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection": self.detection.to_dict(),
            "change_points": {
                name: result.to_dict()
                for name, result in sorted(self.change_points.items())
            },
            "discrimination": self.discrimination,
            "prelocalization": self.prelocalization,
            "leak_volume": self.leak_volume,
            "latency": self.latency,
            "provenance": self.provenance,
        }


class SicamsFullPipeline:
    """Detection-only and full supplied-input SICAMS pipelines.

    ``fit`` estimates only normal calibration moments.  ``detect`` is the
    detection-only path and is online/causal: row ``i`` uses the trailing
    moving average of TF2 rows ``i-L+1..i`` and never a future row.  ``run_full``
    adds the later stages.  PELT/Algorithm 2 are retrospective and are reported
    as offline; pre-localization and discrimination are per-row and causal.
    """

    def __init__(
        self,
        detection_config: SicamsConfig | None = None,
        stage_config: SicamsStageConfig | None = None,
    ) -> None:
        self.detection_config = detection_config or SicamsConfig()
        self.stage_config = stage_config or SicamsStageConfig()
        self.feature_names_: list[str] | None = None
        self.spatial_partition_: SpatialPartition | None = None
        self.temporal_partition_: TemporalPartition | None = None
        self.models_: dict[str, SicamsBaseline] = {}
        self.calibration_: dict[str, Any] = {}
        self.volume_model_: LeakVolumeRegression | None = None
        self.volume_unavailable_reason_: str | None = None
        self.column_report_: dict[str, Any] | None = None
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        X: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
        *,
        timestamps: Iterable[Any] | None = None,
        cluster_ids: Sequence[int] | np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
        calibration_interval: tuple[Any, Any] | None = None,
        spatial_clusters: Mapping[str, Sequence[str]] | None = None,
        spatial_correlation_threshold: float | None = None,
        temporal_base_scheme: str | None = None,
        learn_temporal: bool = False,
        leak_volume_targets: Sequence[float] | np.ndarray | None = None,
        volume_training_X: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame | None = None,
        volume_training_cluster_ids: Sequence[int] | np.ndarray | None = None,
        volume_training_timestamps: Iterable[Any] | None = None,
        volume_model: str | None = None,
        volume_window: int | None = None,
    ) -> "SicamsFullPipeline":
        """Fit normal-calibration moments and optional later-stage inputs.

        No label, test or attack-class argument is accepted.  If a DataFrame is
        supplied, truth columns are excluded by :func:`select_observed_columns`.
        """
        index = _as_timestamps(timestamps)
        names: list[str] | None = None
        report: dict[str, Any] | None = None
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                selected, report = select_observed_columns(X)
                names = selected
            else:
                names = [str(c) for c in feature_names]
                missing = [c for c in names if c not in X.columns]
                if missing:
                    raise FeatureDimensionError(f"frame is missing columns: {missing}")
            matrix = X.loc[:, names].to_numpy(dtype=float)
        else:
            matrix, inferred = _as_matrix(X, feature_names)
            names = inferred
        if names is None:
            names = [f"x{i}" for i in range(matrix.shape[1])]
        if len(names) != matrix.shape[1]:
            raise ValueError("feature_names length does not match X")
        if index is not None and index.size != matrix.shape[0]:
            raise ValueError("timestamps length must match X")

        calibration: dict[str, Any] = {
            "supplied_rows": int(matrix.shape[0]),
            "source": "all_supplied_training_rows",
        }
        training_mask = np.ones(matrix.shape[0], dtype=bool)
        if calibration_interval is not None:
            if index is None:
                raise MissingSuppliedInputError(
                    "calibration_interval requires timestamps"
                )
            start = pd.Timestamp(calibration_interval[0])
            end = pd.Timestamp(calibration_interval[1])
            if end < start:
                raise ValueError("calibration_interval end precedes start")
            mask = np.asarray((index >= start) & (index <= end), dtype=bool)
            if not np.any(mask):
                raise ValueError("calibration_interval selects no training rows")
            training_mask = mask
            matrix = matrix[mask]
            index = index[mask]
            if cluster_ids is not None:
                cluster_ids = np.asarray(cluster_ids, dtype=np.int64)[mask]
            if isinstance(X, pd.DataFrame) and feature_names is None:
                selected, report = select_observed_columns(X.iloc[np.flatnonzero(mask)])
                names = selected
                matrix = X.iloc[np.flatnonzero(mask)].loc[:, names].to_numpy(dtype=float)
            calibration.update(
                {
                    "source": "supplied_normal_interval",
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "rows_in_interval": int(matrix.shape[0]),
                    "excluded_rows": int(mask.size - int(mask.sum())),
                }
            )
        elif index is not None:
            calibration.update(
                {
                    "start": index.min().isoformat(),
                    "end": index.max().isoformat(),
                    "rows_in_interval": int(matrix.shape[0]),
                    "excluded_rows": 0,
                }
            )
        else:
            calibration["timestamps"] = "not_supplied"

        base_scheme = temporal_base_scheme or self.detection_config.temporal_scheme
        if learn_temporal:
            if index is None:
                raise MissingSuppliedInputError("learn_temporal requires timestamps")
            partition = learn_temporal_partition(
                index,
                matrix,
                base_scheme=base_scheme,
                significance=self.stage_config.significance,
            )
            resolved_ids = partition.apply(index)
        elif cluster_ids is not None:
            resolved_ids = np.asarray(cluster_ids, dtype=np.int64)
            partition = TemporalPartition(
                scheme="explicit_cluster_ids",
                source="supplied",
                n_clusters=int(len(np.unique(resolved_ids))),
            )
        elif index is not None:
            partition = TemporalPartition(
                scheme=base_scheme,
                source="calendar",
                n_clusters=int(len(np.unique(temporal_cluster_labels(index, base_scheme)))),
            )
            resolved_ids = partition.apply(index)
        else:
            resolved_ids = np.zeros(matrix.shape[0], dtype=np.int64)
            partition = TemporalPartition(
                scheme="single_cluster",
                source="default",
                n_clusters=1,
            )
        if resolved_ids.size != matrix.shape[0]:
            raise ValueError("cluster_ids length must match X")

        spatial = build_spatial_partition(
            names,
            supplied=spatial_clusters,
            train_X=matrix,
            correlation_threshold=spatial_correlation_threshold,
        )
        models: dict[str, SicamsBaseline] = {}
        for cluster, indices in spatial.clusters.items():
            model = SicamsBaseline(self.detection_config).fit(
                matrix[:, indices],
                cluster_ids=resolved_ids,
                feature_names=[names[i] for i in indices],
            )
            models[cluster] = model

        self.feature_names_ = list(names)
        self.spatial_partition_ = spatial
        self.temporal_partition_ = partition
        self.models_ = models
        self.calibration_ = calibration
        self.column_report_ = report
        self.volume_model_ = None
        self.volume_unavailable_reason_ = None
        self._fitted = True

        if leak_volume_targets is not None:
            if volume_training_X is None:
                raise MissingSuppliedInputError(
                    "volume_training_X is required: regression pairs must be "
                    "independent of nominal-moment calibration data"
                )
            volume_matrix = self._resolve_input(volume_training_X)
            targets = np.asarray(leak_volume_targets, dtype=float)
            if targets.ndim != 1 or targets.size != volume_matrix.shape[0]:
                raise ValueError(
                    "leak_volume_targets must align with volume_training_X"
                )
            detection = self.detect(
                volume_matrix,
                cluster_ids=volume_training_cluster_ids,
                timestamps=volume_training_timestamps,
            )
            self.volume_model_ = fit_leak_volume_regression(
                detection.tf2,
                targets,
                model=volume_model or self.stage_config.volume_model,
                window=volume_window or self.stage_config.volume_window,
            )
            self.volume_model_ = replace(
                self.volume_model_,
                target_source="independent_regression_training_pairs",
            )
        else:
            self.volume_unavailable_reason_ = (
                "no training-only leak-volume targets supplied; regression is "
                "unavailable and no leak_* column was read automatically"
            )
        return self

    def _resolve_input(
        self,
        X: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        if not self.fitted:
            raise NotFittedError("SicamsFullPipeline used before fit")
        assert self.feature_names_ is not None
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_names_ if c not in X.columns]
            if missing:
                raise FeatureDimensionError(f"frame is missing fitted columns: {missing}")
            matrix = X.loc[:, self.feature_names_].to_numpy(dtype=float)
        else:
            matrix, _ = _as_matrix(X, feature_names)
        if matrix.shape[1] != len(self.feature_names_):
            raise FeatureDimensionError(
                f"prediction has {matrix.shape[1]} sensors; model was fitted "
                f"with {len(self.feature_names_)}"
            )
        return matrix

    def detect(
        self,
        X: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
        *,
        timestamps: Iterable[Any] | None = None,
        cluster_ids: Sequence[int] | np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> SicamsDetectionResult:
        """Detection-only, online/causal path."""
        matrix = self._resolve_input(X, feature_names)
        assert self.temporal_partition_ is not None
        index = _as_timestamps(timestamps)
        if cluster_ids is not None:
            ids = np.asarray(cluster_ids, dtype=np.int64)
            if ids.ndim != 1 or ids.size != matrix.shape[0]:
                raise ValueError("cluster_ids must be one-dimensional and match X")
        elif index is not None:
            ids = self.temporal_partition_.apply(index)
        elif self.temporal_partition_.scheme == "single_cluster":
            ids = np.zeros(matrix.shape[0], dtype=np.int64)
        else:
            raise MissingSuppliedInputError(
                "this fitted pipeline used temporal clusters; supply timestamps or "
                "cluster_ids for prediction"
            )

        cluster_results: dict[str, SicamsResult] = {}
        for cluster, model in self.models_.items():
            assert self.spatial_partition_ is not None
            indices = self.spatial_partition_.indices(cluster)
            cluster_results[cluster] = model.predict(matrix[:, indices], cluster_ids=ids)

        names = sorted(cluster_results)
        stack_t2 = np.vstack([cluster_results[n].t2 for n in names])
        stack_tf2 = np.vstack([cluster_results[n].tf2 for n in names])
        stack_ma = np.vstack([cluster_results[n].moving_average for n in names])
        stack_flags = np.vstack([cluster_results[n].flags for n in names])
        stack_upper = np.vstack([cluster_results[n].upper_threshold for n in names])
        stack_lower = np.vstack([cluster_results[n].lower_threshold for n in names])

        provenance = {
            "schema": STAGES_SCHEMA,
            "detection": {
                "online": True,
                "moving_average": "trailing causal window",
                "window": int(self.detection_config.moving_average_window),
                "latency_samples": int(self.detection_config.moving_average_window - 1),
                "combination": "max_envelope_adaptation; per-spatial-cluster results are retained separately",
                "spatial_partition": self.spatial_partition_.to_dict(),
                "temporal_partition": self.temporal_partition_.to_dict(),
                "calibration": dict(self.calibration_),
                "column_selection": self.column_report_,
            },
        }
        return SicamsDetectionResult(
            t2=np.max(stack_t2, axis=0),
            tf2=np.max(stack_tf2, axis=0),
            moving_average=np.max(stack_ma, axis=0),
            flags=np.any(stack_flags, axis=0).astype(int),
            upper_threshold=np.max(stack_upper, axis=0),
            lower_threshold=np.min(stack_lower, axis=0),
            cluster_ids=ids,
            feature_names=list(self.feature_names_ or []),
            clusters=cluster_results,
            provenance=provenance,
        )

    def _z_matrix(self, detection: SicamsDetectionResult, matrix: np.ndarray) -> np.ndarray:
        assert self.spatial_partition_ is not None
        z = np.empty_like(matrix, dtype=float)
        for cluster, model in self.models_.items():
            indices = self.spatial_partition_.indices(cluster)
            z[:, indices] = z_score_matrix(
                model, matrix[:, indices], cluster_ids=detection.cluster_ids
            )
        return z

    def run_full(
        self,
        X: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
        *,
        timestamps: Iterable[Any] | None = None,
        cluster_ids: Sequence[int] | np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
        sensor_nodes: Mapping[str, str] | None = None,
        adjacency: Mapping[str, Sequence[str]] | None = None,
        operational_ranges: Mapping[str, tuple[float, float]] | None = None,
        sensor_z_threshold: float | None = None,
        concentration_threshold: float | None = None,
        pelt_penalty: float | None = None,
        slope_threshold: float | None = None,
    ) -> SicamsFullResult:
        """Full supplied-input path; missing inputs are reported, never faked."""
        matrix = self._resolve_input(X, feature_names)
        detection = self.detect(
            matrix,
            timestamps=timestamps,
            cluster_ids=cluster_ids,
        )

        penalty = (
            self.stage_config.pelt_penalty if pelt_penalty is None else pelt_penalty
        )
        delta = (
            self.stage_config.slope_threshold
            if slope_threshold is None
            else slope_threshold
        )
        change_points: dict[str, ChangePointResult] = {}
        for cluster, result in detection.clusters.items():
            change_points[cluster] = detect_change_points(
                result.moving_average,
                penalty=penalty,
                min_segment=self.stage_config.pelt_min_segment,
                significance=self.stage_config.significance,
                slope_threshold=delta,
                t_test=self.stage_config.t_test,
            )
        change_points["combined"] = detect_change_points(
            detection.moving_average,
            penalty=penalty,
            min_segment=self.stage_config.pelt_min_segment,
            significance=self.stage_config.significance,
            slope_threshold=delta,
            t_test=self.stage_config.t_test,
        )

        z_matrix = self._z_matrix(detection, matrix)
        threshold = (
            self.stage_config.sensor_z_threshold
            if sensor_z_threshold is None
            else sensor_z_threshold
        )
        concentration = (
            self.stage_config.concentration_threshold
            if concentration_threshold is None
            else concentration_threshold
        )
        per_row: list[dict[str, Any]] = []
        status_counts: dict[str, int] = {}
        anomalous_counts: dict[str, int] = {}
        for i in range(matrix.shape[0]):
            outcome = discriminate_anomaly(
                matrix[i],
                z_matrix[i],
                self.feature_names_ or [],
                detected=bool(detection.flags[i]),
                operational_ranges=operational_ranges,
                z_threshold=threshold,
                concentration_threshold=concentration,
                max_out_of_range=self.stage_config.max_out_of_range,
            )
            status_counts[outcome.status] = status_counts.get(outcome.status, 0) + 1
            if detection.flags[i]:
                anomalous_counts[outcome.status] = (
                    anomalous_counts.get(outcome.status, 0) + 1
                )
            per_row.append(outcome.to_dict())
        if not np.any(detection.flags):
            discrimination_status = "not_anomalous"
            discrimination_reason = "no row was flagged by the detector"
        elif "unavailable" in anomalous_counts:
            discrimination_status = "unavailable"
            discrimination_reason = (
                "detection ran, but sensor-fault versus hydraulic discrimination "
                "needs supplied operational ranges or a z threshold"
            )
        else:
            discrimination_status = max(
                anomalous_counts, key=lambda key: anomalous_counts[key]
            )
            discrimination_reason = None
        discrimination = {
            "status": discrimination_status,
            "reason": discrimination_reason,
            "counts": status_counts,
            "anomalous_counts": anomalous_counts,
            "per_row": per_row,
            "online": True,
            "rule": (
                "paper_M_a_operational_range" if operational_ranges is not None
                else "statistical_z_range_proxy" if threshold is not None
                else "adapted_sj_concentration_threshold" if concentration is not None
                else "unavailable_missing_operational_ranges"
            ),
        }

        prelocalization: dict[str, Any]
        if sensor_nodes is None or adjacency is None:
            prelocalization = {
                "status": "unavailable",
                "reason": (
                    "sensor_nodes and adjacency must be supplied; the full-signal "
                    "adapter does not define a pressure-sensor-to-node mapping"
                ),
                "online": False,
            }
        else:
            flagged = np.flatnonzero(detection.flags)
            row = int(flagged[np.argmax(detection.moving_average[flagged])]) if flagged.size else int(np.argmax(detection.moving_average))
            mapped = {
                name: sensor_nodes[name]
                for name in (self.feature_names_ or [])
                if name in sensor_nodes
            }
            missing = [
                name for name in (self.feature_names_ or []) if name not in sensor_nodes
            ]
            if not mapped:
                prelocalization = {
                    "status": "unavailable",
                    "reason": "no fitted sensor has a supplied sensor_nodes mapping",
                    "online": False,
                }
            else:
                # Section 3.5, Eq. (16)-(17): the *squared* z-scores are the
                # values propagated by Laplacian interpolation.
                values = {
                    node: float(z_matrix[row, (self.feature_names_ or []).index(name)] ** 2)
                    for name, node in mapped.items()
                }
                try:
                    result = laplacian_prelocalization(
                        adjacency,
                        values,
                        tolerance=self.stage_config.prelocalization_tolerance,
                        max_iter=self.stage_config.prelocalization_max_iter,
                    )
                except SicamsStageError as exc:
                    prelocalization = {
                        "status": "unavailable",
                        "reason": str(exc),
                        "online": False,
                    }
                else:
                    prelocalization = {
                        "status": "ok",
                        "row": int(row),
                        "result": result.to_dict(),
                        "unmapped_sensors": missing,
                        "online": False,
                        "selection": "retrospective_peak_row",
                    }

        if self.volume_model_ is None:
            leak_volume = {
                "status": "unavailable",
                "reason": self.volume_unavailable_reason_,
                "online": True,
            }
        else:
            leak_volume = {
                "status": "ok",
                "model": self.volume_model_.to_dict(),
                "predictions": [
                    float(v) for v in self.volume_model_.predict(detection.tf2)
                ],
                "online": True,
            }

        latency = {
            "detection": {
                "mode": "online_causal",
                "window": int(self.detection_config.moving_average_window),
                "delay_samples": int(self.detection_config.moving_average_window - 1),
                "alignment": "row i uses tf2[i-window+1 : i+1]",
            },
            "change_points": {
                "mode": "offline_retrospective",
                "online_available": False,
                "latency": "unbounded; PELT and Algorithm 2 need future samples",
            },
            "discrimination": {"mode": "per_row_online", "online_available": True},
            "prelocalization": {"mode": "offline_retrospective", "online_available": False},
            "leak_volume": {
                "mode": "online_causal",
                "window": int(self.volume_model_.window) if self.volume_model_ else None,
                "delay_samples": (
                    int(self.volume_model_.window - 1) if self.volume_model_ else None
                ),
            },
        }
        provenance = {
            "schema": STAGES_SCHEMA,
            "detection": detection.provenance,
            "stages": {
                "config": self.stage_config.to_dict(),
                "pelt_pseudocode_note": (
                    "Algorithm 2 removes a redundant SCP when |m_a-m_b| < delta; "
                    "the surrounding text says 'statistically significant'. The "
                    "pseudocode is followed."
                ),
                "prelocalization": "Eqs. (16)-(17), supplied graph and sensor mapping",
                "volume_regression": "Eqs. (19)-(20) model forms, fitted on supplied targets",
                "targets": "training-only; benchmark leak_* columns are never read",
            },
        }
        return SicamsFullResult(
            detection=detection,
            change_points=change_points,
            discrimination=discrimination,
            prelocalization=prelocalization,
            leak_volume=leak_volume,
            latency=latency,
            provenance=provenance,
        )

    def verify_online_causality(
        self,
        X: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
        *,
        timestamps: Iterable[Any] | None = None,
        cluster_ids: Sequence[int] | np.ndarray | None = None,
        prefix: int | None = None,
    ) -> dict[str, Any]:
        """Check that detection row i is unchanged by appending future rows."""
        matrix = self._resolve_input(X)
        if timestamps is not None:
            index = _as_timestamps(timestamps)
            assert index is not None
            if prefix is None:
                prefix = max(1, matrix.shape[0] // 2)
            full = self.detect(matrix, timestamps=index)
            part = self.detect(matrix[:prefix], timestamps=index[:prefix])
            return {
                "online_causal": bool(
                    np.array_equal(full.flags[:prefix], part.flags)
                    and np.allclose(full.moving_average[:prefix], part.moving_average)
                ),
                "prefix": int(prefix),
                "window": int(self.detection_config.moving_average_window),
                "alignment": "trailing window [i-window+1, i]",
                "max_abs_diff": float(
                    np.max(np.abs(full.moving_average[:prefix] - part.moving_average))
                ),
            }
        if cluster_ids is not None:
            ids = np.asarray(cluster_ids, dtype=np.int64)
            if prefix is None:
                prefix = max(1, matrix.shape[0] // 2)
            full = self.detect(matrix, cluster_ids=ids)
            part = self.detect(matrix[:prefix], cluster_ids=ids[:prefix])
            return {
                "online_causal": bool(
                    np.array_equal(full.flags[:prefix], part.flags)
                    and np.allclose(full.moving_average[:prefix], part.moving_average)
                ),
                "prefix": int(prefix),
                "window": int(self.detection_config.moving_average_window),
                "alignment": "trailing window [i-window+1, i]",
                "max_abs_diff": float(
                    np.max(np.abs(full.moving_average[:prefix] - part.moving_average))
                ),
            }
        raise MissingSuppliedInputError(
            "verify_online_causality needs timestamps or cluster_ids"
        )


__all__ = [
    "AnomalyDiscrimination",
    "ChangePointResult",
    "ClassifiedChangePoint",
    "ChangePointSegment",
    "InvalidPartitionError",
    "LeakVolumeRegression",
    "MissingSuppliedInputError",
    "PAPER_VOLUME_LINEAR",
    "PAPER_VOLUME_LOG_LINEAR",
    "PrelocalizationResult",
    "SicamsDetectionResult",
    "SicamsFullPipeline",
    "SicamsFullResult",
    "SicamsStageConfig",
    "SicamsStageError",
    "SpatialPartition",
    "TemporalPartition",
    "build_spatial_partition",
    "contribution_shares",
    "detect_change_points",
    "discriminate_anomaly",
    "fit_leak_volume_regression",
    "laplacian_prelocalization",
    "learn_temporal_partition",
    "select_observed_columns",
    "sensor_contributions",
    "z_score_matrix",
]
