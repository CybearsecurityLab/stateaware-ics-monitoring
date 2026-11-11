"""Thesis-inspired IM-DISCO invariant-rule baseline.

The implementation follows Saraiva's IM-DISCO paper and 2023 thesis where
their behaviour is explicit, and records every adaptation in
:meth:`IMDISCO.provenance`.  The pipeline is:

1. **Predicate generation** from expert-supplied operational modes: mode-local
   sensor MinMax, ordinary Gradient, entry-only TransitionGradient and
   SteadyTime predicates, plus actuator State items shared across modes.
2. **Invariant rule mining**: one transaction per timestep plus a separate
   mode item.  The default is an exact maximal-rule adaptation using frequent
   closed transaction patterns and subsumption pruning.  Setting
   ``prune_subsumed_rules=False`` instead materializes uncapped mlxtend
   FP-Growth itemsets and association rules (``minsup=0.05``, ``minconf=1.0``).
3. **Runtime inference**: evaluate rules in a documented deterministic order
   and take the first matching mode.  No rule matches -> anomaly.

The thesis reports redundancy removal after association-rule generation, and
the paper reports one maximal conjunction per mode, but neither supplies the
reduction algorithm or complete code.  The exact closed-pattern path is
therefore an adaptation, not an official-code reproduction.  It computes the
same retained rules as uncapped FP-Growth followed by :func:`_prune_subsumed`
without materializing the potentially exponential intermediate rule set.

The paper's formulas print strict inequalities, while its examples use ``<=``.
Inclusive bounds are the default; ``inclusive_bounds=False`` restores the
strictly printed formulas.  ``mlxtend`` is required only by the uncapped arm.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import warnings
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "IMDISCO",
    "IMDISCOConfig",
    "IMDISCOError",
    "MiningTooLargeError",
    "MissingMinerError",
    "InvariantRule",
    "Predicate",
    "PredictionResult",
    "anomaly_metrics",
    "bounds_artifact",
    "derive_predicates",
]

SCHEMA_VERSION = "imdisco-baseline-v2"
MODE_ITEM_PREFIX = "mode:"


class IMDISCOError(Exception):
    """Base error for the IM-DISCO baseline."""


class MissingMinerError(IMDISCOError, ImportError):
    """Raised when the required mlxtend FP-Growth miner is unavailable."""


class MiningTooLargeError(IMDISCOError):
    """Raised when frequent-itemset extraction exceeds the safety cap."""


# ---------------------------------------------------------------------------
# Deterministic value encoding


def _value_key(value: Any) -> str | None:
    """Return a deterministic, type-tagged string key for a scalar value.

    Missing values map to ``None`` so they never match a learned actuator
    state.  The type tag prevents ``1`` and ``"1"`` from colliding.
    """
    if value is None:
        return "none"
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return "bool:true" if bool(value) else "bool:false"
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return None
        if number.is_integer():
            return f"float:{int(number)}"
        return f"float:{number!r}"
    return f"str:{value}"


def _display(value: Any) -> str:
    """Human-readable, JSON-safe label for a mode/state value."""
    if isinstance(value, str):
        return str(value)
    key = _value_key(value)
    return str(value) if key is None else key


def _jsonable(value: Any) -> Any:
    """Keep primitive mode values as-is; stringify everything else."""
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return _display(value)
        return value
    return _display(value)


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _mode_sort_key(value: Any) -> tuple[str, str]:
    return (_display(value), _value_key(value) or "")


def _predicate_item(
    kind: str,
    variable: str,
    mode_key: str | None,
    lo: float | None,
    hi: float | None,
    state_key: str | None,
    previous_mode_key: str | None = None,
) -> str:
    return json.dumps(
        [kind, variable, mode_key, lo, hi, state_key, previous_mode_key],
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _mode_item(mode_key: str) -> str:
    return MODE_ITEM_PREFIX + mode_key


# ---------------------------------------------------------------------------
# Configuration


@dataclass(frozen=True)
class IMDISCOConfig:
    """Deterministic IM-DISCO configuration.

    ``sensor_columns`` / ``actuator_columns`` should normally be supplied by the
    caller.  When both are ``None`` the roles are inferred deterministically:
    numeric non-integer (and higher-cardinality integer) columns are sensors;
    boolean, categorical, and binary integer columns are actuators.  The
    attack-label column (``label``) and ``mode_column`` are never used as
    features.

    ``max_itemset_len`` defaults to ``None``.  It is forwarded to FP-Growth
    only in the uncapped arm (``prune_subsumed_rules=False``).  The exact
    maximal-rule arm rejects a length cap rather than silently narrowing the
    rule family.  ``max_frequent_itemsets`` is a tractability guard for the
    uncapped arm only; the maximal arm never materializes that itemset family.
    """

    sensor_columns: tuple[str, ...] | None = None
    actuator_columns: tuple[str, ...] | None = None
    mode_column: str | None = None
    minsup: float = 0.05
    minconf: float = 1.0
    inclusive_bounds: bool = True
    max_itemset_len: int | None = None
    prune_subsumed_rules: bool = True
    mode_order: tuple[Any, ...] | None = None
    max_frequent_itemsets: int | None = 2_000_000

    def __post_init__(self) -> None:
        if not (0.0 < float(self.minsup) <= 1.0):
            raise ValueError("minsup must be in (0, 1] (mlxtend requirement)")
        if not (0.0 <= float(self.minconf) <= 1.0):
            raise ValueError("minconf must be in [0, 1]")
        if self.max_itemset_len is not None and int(self.max_itemset_len) < 1:
            raise ValueError("max_itemset_len must be >= 1 or None")
        if self.max_frequent_itemsets is not None and int(self.max_frequent_itemsets) < 1:
            raise ValueError("max_frequent_itemsets must be >= 1 or None")
        for name in ("sensor_columns", "actuator_columns", "mode_order"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor_columns": None if self.sensor_columns is None else list(self.sensor_columns),
            "actuator_columns": None if self.actuator_columns is None else list(self.actuator_columns),
            "mode_column": self.mode_column,
            "minsup": float(self.minsup),
            "minconf": float(self.minconf),
            "inclusive_bounds": bool(self.inclusive_bounds),
            "max_itemset_len": self.max_itemset_len,
            "prune_subsumed_rules": bool(self.prune_subsumed_rules),
            "mode_order": None if self.mode_order is None else [_display(m) for m in self.mode_order],
            "max_frequent_itemsets": self.max_frequent_itemsets,
        }


# ---------------------------------------------------------------------------
# Predicates and rules


@dataclass(frozen=True)
class Predicate:
    """A sensor/actuator predicate.

    State predicates have ``mode_key=None`` and are shared across modes.  All
    other predicates are mode-local.  TransitionGradient additionally records
    the previous mode and is valid only at that mode boundary.
    """

    kind: str
    variable: str
    mode: Any
    mode_key: str | None
    lo: float | None = None
    hi: float | None = None
    state_key: str | None = None
    n: int = 0
    inclusive: bool = True
    previous_mode: Any = None
    previous_mode_key: str | None = None

    @property
    def item(self) -> str:
        return _predicate_item(
            self.kind,
            self.variable,
            self.mode_key,
            self.lo,
            self.hi,
            self.state_key,
            self.previous_mode_key,
        )

    def evaluate(
        self,
        *,
        values: Mapping[str, Any],
        gradients: Mapping[str, Any],
        run_lengths: Mapping[str, Any],
        previous_mode_key: str | None = None,
        frame_start: bool = False,
    ) -> bool:
        if self.kind in ("minmax", "gradient", "transitiongradient"):
            if self.kind == "gradient" and (
                frame_start
                or (previous_mode_key is not None and previous_mode_key != self.mode_key)
            ):
                return False
            if self.kind == "transitiongradient" and (
                frame_start
                or previous_mode_key != self.previous_mode_key
                or previous_mode_key == self.mode_key
            ):
                return False
            raw = values.get(self.variable) if self.kind == "minmax" else gradients.get(self.variable)
            try:
                number = float(raw)
            except (TypeError, ValueError):
                return False
            if math.isnan(number) or self.lo is None or self.hi is None:
                return False
            if self.inclusive:
                return self.lo <= number <= self.hi
            return self.lo < number < self.hi

        current = _value_key(values.get(self.variable))
        if current is None or current != self.state_key:
            return False
        if self.kind == "state":
            return True
        if self.kind == "steadytime":
            run = run_lengths.get(self.variable)
            try:
                length = float(run)
            except (TypeError, ValueError):
                return False
            if math.isnan(length) or self.lo is None or self.hi is None:
                return False
            if self.inclusive:
                return self.lo <= length <= self.hi
            return self.lo < length < self.hi
        return False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "variable": self.variable,
            "mode": _jsonable(self.mode),
            "mode_key": self.mode_key,
            "lo": None if self.lo is None else _finite_or_none(self.lo),
            "hi": None if self.hi is None else _finite_or_none(self.hi),
            "state": self.state_key,
            "n": int(self.n),
            "inclusive": bool(self.inclusive),
            "previous_mode": _jsonable(self.previous_mode),
            "previous_mode_key": self.previous_mode_key,
            "item": self.item,
        }
        return out


@dataclass(frozen=True)
class InvariantRule:
    """An mined implication ``antecedent => mode`` with perfect confidence."""

    mode: Any
    mode_key: str
    antecedent: tuple[Predicate, ...]
    support: float
    confidence: float
    lift: float = float("nan")

    @property
    def antecedent_items(self) -> tuple[str, ...]:
        return tuple(p.item for p in self.antecedent)

    @property
    def signature(self) -> str:
        return " && ".join(self.antecedent_items)

    @property
    def rule_id(self) -> str:
        return f"{self.mode_key} | {self.signature}"

    def evaluate(
        self,
        *,
        values: Mapping[str, Any],
        gradients: Mapping[str, Any],
        run_lengths: Mapping[str, Any],
        previous_mode_key: str | None = None,
        frame_start: bool = False,
    ) -> bool:
        return all(
            p.evaluate(
                values=values,
                gradients=gradients,
                run_lengths=run_lengths,
                previous_mode_key=previous_mode_key,
                frame_start=frame_start,
            )
            for p in self.antecedent
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": _jsonable(self.mode),
            "mode_key": self.mode_key,
            "antecedent": [p.to_dict() for p in self.antecedent],
            "antecedent_items": list(self.antecedent_items),
            "support": _finite_or_none(self.support),
            "confidence": _finite_or_none(self.confidence),
            "lift": None if math.isnan(self.lift) else _finite_or_none(self.lift),
            "rule_id": self.rule_id,
        }


@dataclass(frozen=True)
class PredictionResult:
    """Binary flags plus the information needed for deterministic auditing."""

    predicted_modes: tuple[Any | None, ...]
    anomaly_flags: np.ndarray
    anomaly_scores: np.ndarray
    matched_rule_ids: tuple[str | None, ...]

    def __len__(self) -> int:
        return len(self.anomaly_flags)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_modes": [_jsonable(m) for m in self.predicted_modes],
            "anomaly_flags": [int(v) for v in self.anomaly_flags],
            "anomaly_scores": [float(v) for v in self.anomaly_scores],
            "matched_rule_ids": list(self.matched_rule_ids),
        }


# ---------------------------------------------------------------------------
# Frame / column handling


def _as_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    return frame.reset_index(drop=True).copy()


def _infer_columns(
    frame: pd.DataFrame,
    mode_column: str | None,
    max_actuator_states: int = 2,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    excluded = {"label"}
    if mode_column is not None:
        excluded.add(mode_column)
    sensors: list[str] = []
    actuators: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        series = frame[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            actuators.append(str(column))
            continue
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series) and series.nunique(dropna=True) <= max_actuator_states:
                actuators.append(str(column))
            else:
                sensors.append(str(column))
    return tuple(sensors), tuple(actuators)


def _resolve_columns(
    frame: pd.DataFrame,
    config: IMDISCOConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    sensors = config.sensor_columns
    actuators = config.actuator_columns
    if sensors is None and actuators is None:
        sensors, actuators = _infer_columns(frame, config.mode_column)
    else:
        if sensors is None:
            inferred_sensors, _ = _infer_columns(frame, config.mode_column)
            actuators = tuple(actuators or ())
            sensors = tuple(c for c in inferred_sensors if c not in set(actuators))
        if actuators is None:
            _, inferred_actuators = _infer_columns(frame, config.mode_column)
            sensors = tuple(sensors or ())
            actuators = tuple(c for c in inferred_actuators if c not in set(sensors))
    sensors = tuple(dict.fromkeys(str(c) for c in (sensors or ())))
    actuators = tuple(dict.fromkeys(str(c) for c in (actuators or ())))

    forbidden = {"label"}
    if config.mode_column is not None:
        forbidden.add(config.mode_column)
    overlap = set(sensors) & set(actuators)
    if overlap:
        raise ValueError(f"columns cannot be both sensors and actuators: {sorted(overlap)}")
    used_forbidden = (set(sensors) | set(actuators)) & forbidden
    if used_forbidden:
        raise ValueError(
            "attack labels and the mode column must not be used as features: "
            f"{sorted(used_forbidden)}"
        )
    missing = [c for c in (*sensors, *actuators) if c not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    if not sensors and not actuators:
        raise ValueError("no sensor or actuator columns were supplied or inferred")
    return sensors, actuators


def _resolve_modes(
    frame: pd.DataFrame,
    modes: Sequence[Any] | pd.Series | str | None,
    mode_column: str | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if modes is None:
        if mode_column is None:
            raise ValueError(
                "expert mode labels are required; pass modes=... or mode_column=... "
                "(the baseline never infers modes from attack labels)"
            )
        if mode_column not in frame.columns:
            raise ValueError(f"mode column {mode_column!r} is not in the frame")
        modes = frame[mode_column]
    elif isinstance(modes, str):
        if modes not in frame.columns:
            raise ValueError(f"mode column {modes!r} is not in the frame")
        modes = frame[modes]

    values = np.asarray(modes)
    if values.ndim != 1 or len(values) != len(frame):
        raise ValueError("expert mode labels must be one-dimensional and match frame length")
    if len(values) == 0:
        raise ValueError("cannot fit or derive predicates from an empty frame")
    keys = tuple(_value_key(v) for v in values)
    if any(k is None for k in keys):
        raise ValueError("expert mode labels must be non-null")
    return values, keys


# ---------------------------------------------------------------------------
# Predicate generation


def _gradient_frame(frame: pd.DataFrame, sensor_columns: Sequence[str]) -> pd.DataFrame:
    if not sensor_columns:
        return pd.DataFrame(index=frame.index)
    numeric = frame.loc[:, list(sensor_columns)].apply(pd.to_numeric, errors="coerce")
    gradient = numeric.diff()
    if len(gradient):
        gradient.iloc[0] = 0.0
    return gradient.astype(float)


def _completed_run_records(
    values: Iterable[Any],
    mode_keys: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return total durations for completed runs and their explicit provenance.

    A completed run's total duration is assigned retrospectively to every row
    of that run, following the thesis examples.  The final ongoing run has no
    completed duration.  During training, a run spanning multiple expert modes
    is excluded from both duration learning and training transactions.
    """
    values = list(values)
    if mode_keys is not None and len(mode_keys) != len(values):
        raise ValueError("mode_keys must match actuator values")
    durations = np.full(len(values), np.nan, dtype=float)
    records: list[dict[str, Any]] = []
    start = 0
    while start < len(values):
        state_key = _value_key(values[start])
        end = start + 1
        while end < len(values) and _value_key(values[end]) == state_key:
            end += 1
        if state_key is not None:
            completed = end < len(values)
            run_modes = (
                sorted(set(mode_keys[start:end])) if mode_keys is not None else []
            )
            crosses_modes = len(run_modes) > 1
            eligible = completed and not crosses_modes
            exclusion_reason = None
            if not completed:
                exclusion_reason = "final_ongoing_run"
            elif crosses_modes:
                exclusion_reason = "crosses_mode_boundary"
            if eligible:
                durations[start:end] = float(end - start)
            records.append({
                "start_row": int(start),
                "end_row": int(end - 1),
                "state_key": state_key,
                "duration": int(end - start),
                "completed": bool(completed),
                "mode_keys": run_modes,
                "crosses_mode_boundary": bool(crosses_modes),
                "eligible_for_training": bool(eligible),
                "exclusion_reason": exclusion_reason,
            })
        start = end
    return durations, records


def _run_lengths(values: Iterable[Any]) -> np.ndarray:
    """Completed-run total durations, NaN on the final ongoing/missing runs."""
    durations, _ = _completed_run_records(values)
    return durations


def _steadytime_provenance(
    frame: pd.DataFrame,
    actuator_columns: Sequence[str],
    mode_keys: Sequence[str],
) -> dict[str, Any]:
    per_actuator: dict[str, Any] = {}
    for column in actuator_columns:
        _, records = _completed_run_records(frame[column].tolist(), mode_keys)
        per_actuator[column] = {
            "completed_runs": sum(record["completed"] for record in records),
            "eligible_runs": sum(record["eligible_for_training"] for record in records),
            "excluded_final_ongoing_runs": sum(
                record["exclusion_reason"] == "final_ongoing_run" for record in records
            ),
            "excluded_mode_crossing_runs": sum(
                record["exclusion_reason"] == "crosses_mode_boundary" for record in records
            ),
            "runs": records,
        }
    return {
        "duration_assignment": "completed run total assigned retrospectively to every run row",
        "causal": False,
        "final_ongoing_run": "excluded; duration is NaN",
        "mode_crossing_runs": "excluded from training durations and transactions",
        "actuators": per_actuator,
    }


def _transition_provenance(
    mode_values: np.ndarray,
    mode_keys: Sequence[str],
) -> dict[str, Any]:
    transitions = [
        {
            "row": int(index),
            "previous_mode": _jsonable(mode_values[index - 1]),
            "previous_mode_key": mode_keys[index - 1],
            "mode": _jsonable(mode_values[index]),
            "mode_key": mode_keys[index],
        }
        for index in range(1, len(mode_keys))
        if mode_keys[index] != mode_keys[index - 1]
    ]
    return {
        "ordinary_gradient": "within-mode rows only; mode-entry and frame-start rows excluded",
        "transitiongradient": "first row entering a new mode only; keyed by current and previous mode",
        "runtime_previous_mode": "previous row's predicted mode; no transition at frame start or after an unmatched row",
        "transitions": transitions,
    }


def derive_predicates(
    frame: pd.DataFrame,
    modes: Sequence[Any] | pd.Series | str | None,
    sensor_columns: Sequence[str],
    actuator_columns: Sequence[str],
    *,
    mode_column: str | None = None,
    inclusive_bounds: bool = True,
) -> list[Predicate]:
    """Generate thesis-inspired predicates without requiring mlxtend.

    MinMax, Gradient, TransitionGradient and SteadyTime are mode-local.  State
    items are deliberately global so identical actuator states shared by modes
    have one item and can produce cross-mode confidence below one.
    """
    frame = _as_frame(frame)
    mode_values, mode_keys = _resolve_modes(frame, modes, mode_column)
    sensors = tuple(str(c) for c in sensor_columns)
    actuators = tuple(str(c) for c in actuator_columns)
    missing = [c for c in (*sensors, *actuators) if c not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")

    gradient = _gradient_frame(frame, sensors)
    ordinary_gradient = np.asarray([
        index > 0 and mode_keys[index] == mode_keys[index - 1]
        for index in range(len(frame))
    ])
    transition_groups: dict[tuple[str, str], list[int]] = {}
    for index in range(1, len(frame)):
        if mode_keys[index] != mode_keys[index - 1]:
            transition_groups.setdefault((mode_keys[index], mode_keys[index - 1]), []).append(index)

    predicates: list[Predicate] = []
    mode_representatives = {
        key: value for value, key in zip(mode_values, mode_keys)
    }
    for mode_key in sorted(set(mode_keys)):
        indices = np.flatnonzero(np.asarray([key == mode_key for key in mode_keys]))
        mode_value = mode_representatives[mode_key]

        for column in sensors:
            values = pd.to_numeric(frame.iloc[indices][column], errors="coerce").to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            if len(values):
                predicates.append(Predicate(
                    kind="minmax", variable=column, mode=mode_value, mode_key=mode_key,
                    lo=float(values.min()), hi=float(values.max()), n=int(len(values)),
                    inclusive=bool(inclusive_bounds),
                ))
            ordinary_indices = indices[ordinary_gradient[indices]]
            gradients = gradient.iloc[ordinary_indices][column].to_numpy(dtype=float)
            gradients = gradients[~np.isnan(gradients)]
            if len(gradients):
                predicates.append(Predicate(
                    kind="gradient", variable=column, mode=mode_value, mode_key=mode_key,
                    lo=float(gradients.min()), hi=float(gradients.max()),
                    n=int(len(gradients)), inclusive=bool(inclusive_bounds),
                ))

        for column in actuators:
            _, records = _completed_run_records(frame[column].tolist(), mode_keys)
            eligible_by_state: dict[str, list[int]] = {}
            for record in records:
                if record["eligible_for_training"] and record["mode_keys"] == [mode_key]:
                    eligible_by_state.setdefault(record["state_key"], []).append(record["duration"])
            for state_key, run_durations in sorted(eligible_by_state.items()):
                predicates.append(Predicate(
                    kind="steadytime", variable=column, mode=mode_value, mode_key=mode_key,
                    lo=float(min(run_durations)), hi=float(max(run_durations)),
                    state_key=state_key, n=int(len(run_durations)),
                    inclusive=bool(inclusive_bounds),
                ))

    for (mode_key, previous_mode_key), indices in sorted(transition_groups.items()):
        for column in sensors:
            gradients = gradient.iloc[indices][column].to_numpy(dtype=float)
            gradients = gradients[~np.isnan(gradients)]
            if len(gradients):
                predicates.append(Predicate(
                    kind="transitiongradient", variable=column,
                    mode=mode_representatives[mode_key], mode_key=mode_key,
                    previous_mode=mode_representatives[previous_mode_key],
                    previous_mode_key=previous_mode_key,
                    lo=float(gradients.min()), hi=float(gradients.max()),
                    n=int(len(gradients)), inclusive=bool(inclusive_bounds),
                ))

    # Thesis Table 4.3: State is a shared transaction item, not a mode-keyed one.
    for column in actuators:
        column_values = frame[column].tolist()
        state_keys = sorted({
            key for key in (_value_key(value) for value in column_values) if key is not None
        })
        for state_key in state_keys:
            predicates.append(Predicate(
                kind="state", variable=column, mode=None, mode_key=None,
                state_key=state_key,
                n=sum(_value_key(value) == state_key for value in column_values),
            ))

    predicates.sort(key=lambda predicate: predicate.item)
    return predicates


def bounds_artifact(predicates: Sequence[Predicate]) -> dict[str, Any]:
    """Group predicates into a deterministic, JSON-safe bounds report."""
    out: dict[str, Any] = {
        "minmax": {},
        "gradient": {},
        "transitiongradient": {},
        "steadytime": {},
        "actuator_states": {},
    }
    for predicate in predicates:
        mode = _display(predicate.mode)
        if predicate.kind in ("minmax", "gradient", "transitiongradient"):
            bucket = out[predicate.kind].setdefault(mode, {})
            if predicate.kind == "transitiongradient":
                bucket = bucket.setdefault(_display(predicate.previous_mode), {})
            bucket[predicate.variable] = {
                "lo": None if predicate.lo is None else _finite_or_none(predicate.lo),
                "hi": None if predicate.hi is None else _finite_or_none(predicate.hi),
                "n": int(predicate.n),
                "inclusive": bool(predicate.inclusive),
            }
        elif predicate.kind == "state":
            bucket = out["actuator_states"].setdefault("*", {}).setdefault(predicate.variable, [])
            bucket.append(predicate.state_key)
        elif predicate.kind == "steadytime":
            out["steadytime"].setdefault(mode, {}).setdefault(predicate.variable, {})[
                predicate.state_key
            ] = {
                "lo": None if predicate.lo is None else _finite_or_none(predicate.lo),
                "hi": None if predicate.hi is None else _finite_or_none(predicate.hi),
                "n": int(predicate.n),
                "inclusive": bool(predicate.inclusive),
            }

    for per_mode in out["actuator_states"].values():
        for column in per_mode:
            per_mode[column] = sorted(per_mode[column], key=lambda key: key or "")
    return out


# ---------------------------------------------------------------------------
# Mining


def _import_frequent_patterns() -> tuple[Any, Any]:
    from mlxtend.frequent_patterns import association_rules, fpgrowth

    return fpgrowth, association_rules


def _mlxtend_info() -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec("mlxtend")
    except (ImportError, ValueError):
        spec = None
    version: str | None = None
    if spec is not None:
        try:
            from importlib.metadata import version as package_version

            version = package_version("mlxtend")
        except Exception:
            version = None
    return {"available": spec is not None, "version": version}


def _load_mlxtend() -> SimpleNamespace:
    try:
        fpgrowth, association_rules = _import_frequent_patterns()
    except Exception as exc:  # pragma: no cover - exercised through monkeypatch
        raise MissingMinerError(
            "IM-DISCO requires the mlxtend FP-Growth and association_rules "
            "implementations for prune_subsumed_rules=False (paper: FP-Growth "
            "min_support=0.05, association_rules confidence>=1.0). mlxtend is "
            "not installed or cannot be imported. Install mlxtend for the "
            "uncapped arm, or use the default exact maximal-rule adaptation."
        ) from exc
    return SimpleNamespace(fpgrowth=fpgrowth, association_rules=association_rules)


def _transactions(
    frame: pd.DataFrame,
    modes_values: np.ndarray,
    mode_keys: Sequence[str],
    predicates: Sequence[Predicate],
    gradient: pd.DataFrame,
    runs: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, list[str]]:
    by_mode: dict[str, list[Predicate]] = {}
    shared: list[Predicate] = []
    for predicate in predicates:
        if predicate.mode_key is None:
            shared.append(predicate)
        else:
            by_mode.setdefault(predicate.mode_key, []).append(predicate)

    rows: list[set[str]] = []
    for i in range(len(frame)):
        values = {column: frame[column].iat[i] for column in frame.columns}
        gradients = gradient.iloc[i].to_dict()
        run_lengths = {column: values_series[i] for column, values_series in runs.items()}
        previous_mode_key = mode_keys[i - 1] if i > 0 else None
        applicable = [*shared, *by_mode.get(mode_keys[i], ())]
        items = {
            predicate.item
            for predicate in applicable
            if predicate.evaluate(
                values=values,
                gradients=gradients,
                run_lengths=run_lengths,
                previous_mode_key=previous_mode_key,
                frame_start=i == 0,
            )
        }
        items.add(_mode_item(mode_keys[i]))
        rows.append(items)

    columns = sorted({item for row in rows for item in row})
    matrix = np.zeros((len(rows), len(columns)), dtype=bool)
    positions = {item: index for index, item in enumerate(columns)}
    for row_index, row in enumerate(rows):
        for item in row:
            matrix[row_index, positions[item]] = True
    return pd.DataFrame(matrix, columns=columns), columns


def _row_bitsets(transactions: pd.DataFrame, columns: Sequence[str]) -> list[int]:
    """Encode transaction rows as Python integer bitsets, one bit per column."""
    if not columns:
        return [0] * len(transactions)
    matrix = transactions.loc[:, list(columns)].to_numpy(dtype=bool)
    rows: list[int] = []
    for row in matrix:
        bitset = 0
        for position in np.flatnonzero(row):
            bitset |= 1 << int(position)
        rows.append(bitset)
    return rows


def _closed_patterns(patterns: Iterable[int]) -> set[int]:
    """Compute the intersection-closed family of distinct transaction patterns."""
    closed = set(patterns)
    for bitset in tuple(closed):
        closed |= {candidate & bitset for candidate in tuple(closed)}
    return closed


def _maximal_patterns(patterns: Iterable[int]) -> list[int]:
    candidates = set(patterns)
    return sorted(
        (
            candidate for candidate in candidates
            if not any(
                candidate != other and candidate & ~other == 0
                for other in candidates
            )
        ),
        key=lambda bitset: (-bitset.bit_count(), bitset),
    )


def _subset_union_count(maximal_patterns: Sequence[int], *, include_empty: bool) -> int:
    """Count a union of subset lattices exactly by inclusion--exclusion."""
    if not maximal_patterns:
        return 1 if include_empty else 0
    total = 0
    for size in range(1, len(maximal_patterns) + 1):
        sign = 1 if size % 2 else -1
        for combination in itertools.combinations(maximal_patterns, size):
            intersection = combination[0]
            for candidate in combination[1:]:
                intersection &= candidate
            total += sign * (1 << intersection.bit_count())
    return total if include_empty else total - 1


def _count_frequent_itemsets(
    transactions: pd.DataFrame,
    *,
    min_support_count: int,
) -> tuple[int, dict[str, int]]:
    """Count uncapped frequent itemsets from maximal frequent closed patterns."""
    columns = list(transactions.columns)
    pattern_counts: Counter[int] = Counter(_row_bitsets(transactions, columns))
    closed = _closed_patterns(pattern_counts)

    def support(candidate: int) -> int:
        return sum(
            count for bitset, count in pattern_counts.items()
            if candidate & ~bitset == 0
        )

    frequent_closed = {
        candidate for candidate in closed if support(candidate) >= min_support_count
    }
    maximal = _maximal_patterns(frequent_closed)
    count = _subset_union_count(maximal, include_empty=False)
    return count, {
        "n_closed_itemsets_including_empty": len(closed),
        "n_frequent_closed_itemsets_including_empty": len(frequent_closed),
        "n_maximal_frequent_closed_itemsets_including_empty": len(maximal),
    }


def _maximal_rules_from_transactions(
    transactions: pd.DataFrame,
    predicate_by_item: Mapping[str, Predicate],
    mode_by_item: Mapping[str, Any],
    *,
    mode_key_by_item: Mapping[str, str],
    minsup: float,
    minconf: float,
) -> tuple[list[InvariantRule], dict[str, Any]]:
    """Exact maximal mode rules without FP-Growth/ARG materialization.

    A non-closed antecedent is always subsumed by its closure, which has the
    same total and mode-joint supports.  Consequently the maximal retained
    rules are precisely the maximal qualifying frequent closed antecedents.
    Shared State items are handled by total (cross-mode) antecedent support.
    """
    n_transactions = len(transactions)
    min_support_count = int(math.ceil(float(minsup) * n_transactions))
    predicate_columns = [column for column in transactions.columns if column not in mode_by_item]
    mode_items = list(mode_by_item)
    row_predicates = _row_bitsets(transactions, predicate_columns)
    row_modes = _row_bitsets(transactions, mode_items)
    mode_position = {
        index: mode_key_by_item[item] for index, item in enumerate(mode_items)
    }
    pattern_counts: Counter[int] = Counter()
    pattern_mode_counts: Counter[tuple[int, str]] = Counter()
    for predicates, modes in zip(row_predicates, row_modes):
        mode_positions = [index for index in mode_position if modes & (1 << index)]
        if len(mode_positions) != 1:
            raise ValueError("each transaction must contain exactly one mode item")
        pattern_counts[predicates] += 1
        pattern_mode_counts[(predicates, mode_position[mode_positions[0]])] += 1

    closed = _closed_patterns(pattern_counts)
    mode_row_counts = {
        mode_key: sum(
            count for (pattern, key), count in pattern_mode_counts.items() if key == mode_key
        )
        for mode_key in set(mode_position.values())
    }

    def total_support(candidate: int) -> int:
        return sum(
            count for pattern, count in pattern_counts.items()
            if candidate & ~pattern == 0
        )

    def mode_support(candidate: int, mode_key: str) -> int:
        return sum(
            count for (pattern, key), count in pattern_mode_counts.items()
            if key == mode_key and candidate & ~pattern == 0
        )

    frequent_closed = {
        candidate for candidate in closed
        if candidate and total_support(candidate) >= min_support_count
    }
    maximal_by_mode: dict[str, list[int]] = {}
    qualified_closed_rules = 0
    for mode_key in sorted(mode_row_counts):
        qualified: list[int] = []
        for candidate in frequent_closed:
            antecedent_count = total_support(candidate)
            joint_count = mode_support(candidate, mode_key)
            if joint_count < min_support_count:
                continue
            confidence = joint_count / antecedent_count
            if confidence >= float(minconf) - 1e-12:
                qualified.append(candidate)
        qualified_closed_rules += len(qualified)
        maximal_by_mode[mode_key] = _maximal_patterns(qualified)

    rules: list[InvariantRule] = []
    mode_value_by_key = {
        mode_key_by_item[item]: value for item, value in mode_by_item.items()
    }
    for mode_key, maximal in maximal_by_mode.items():
        for candidate in maximal:
            antecedent_count = total_support(candidate)
            joint_count = mode_support(candidate, mode_key)
            items = [
                predicate_columns[position]
                for position in range(len(predicate_columns))
                if candidate & (1 << position)
            ]
            predicates = tuple(predicate_by_item[item] for item in sorted(items))
            rules.append(InvariantRule(
                mode=mode_value_by_key[mode_key],
                mode_key=mode_key,
                antecedent=predicates,
                support=joint_count / n_transactions,
                confidence=joint_count / antecedent_count,
                lift=(
                    joint_count * n_transactions
                    / (antecedent_count * mode_row_counts[mode_key])
                ),
            ))

    frequent_itemset_count, frequent_stats = _count_frequent_itemsets(
        transactions, min_support_count=min_support_count
    )
    stats: dict[str, Any] = {
        "miner": "exact frequent closed transaction patterns + subsumption pruning",
        "minsup": float(minsup),
        "minconf": float(minconf),
        "min_support_count": min_support_count,
        "n_closed_antecedent_patterns_including_empty": len(closed),
        "n_frequent_closed_antecedents": len(frequent_closed),
        "n_qualified_closed_rules": int(qualified_closed_rules),
        "n_frequent_itemsets": int(frequent_itemset_count),
        "n_rules_before_pruning": None,
        "n_rules_before_pruning_status": (
            "not materialized or estimated: shared State items make confidence "
            "non-monotone under antecedent subsets; the qualified-closed-rule "
            "intermediate count and final maximal count are reported separately"
        ),
        "n_rules": len(rules),
        "frequent_itemsets_materialized": False,
        "unpruned_rules_materialized": False,
        "unpruned_count_method": "not used by the exact maximal-rule adaptation",
        "association_rules_num_itemsets_kwarg": False,
        **frequent_stats,
    }
    return rules, stats


def _rule_dataframe(
    miner: SimpleNamespace,
    frequent_itemsets: pd.DataFrame,
    *,
    minconf: float,
    n_transactions: int,
) -> tuple[pd.DataFrame, bool]:
    if frequent_itemsets is None or len(frequent_itemsets) == 0:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"]), False
    used_num_itemsets = True
    try:
        rules = miner.association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=float(minconf),
            num_itemsets=int(n_transactions),
        )
    except TypeError:
        used_num_itemsets = False
        rules = miner.association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=float(minconf),
        )
    return rules, used_num_itemsets


def _rules_from_dataframe(
    rules: pd.DataFrame,
    predicate_by_item: Mapping[str, Predicate],
    mode_by_item: Mapping[str, Any],
    *,
    mode_key_by_item: Mapping[str, str],
    minconf: float,
) -> list[InvariantRule]:
    if rules is None or len(rules) == 0:
        return []

    deduped: dict[tuple[str, tuple[str, ...]], InvariantRule] = {}
    for _, row in rules.iterrows():
        consequent = set(row.get("consequents", ()))
        if len(consequent) != 1:
            continue
        consequent_item = next(iter(consequent))
        if consequent_item not in mode_by_item:
            continue
        antecedent = set(row.get("antecedents", ()))
        if any(item in mode_by_item for item in antecedent):
            continue
        try:
            confidence = float(row.get("confidence", float("nan")))
        except (TypeError, ValueError):
            confidence = float("nan")
        if math.isnan(confidence) or confidence < float(minconf) - 1e-12:
            continue
        predicates: list[Predicate] = []
        for item in sorted(antecedent):
            predicate = predicate_by_item.get(item)
            if predicate is None:
                predicates = []
                break
            predicates.append(predicate)
        if not predicates:
            continue
        support = float(row.get("support", float("nan")))
        lift = float(row.get("lift", float("nan"))) if "lift" in row else float("nan")
        mode_key = mode_key_by_item[consequent_item]
        rule = InvariantRule(
            mode=mode_by_item[consequent_item],
            mode_key=mode_key,
            antecedent=tuple(predicates),
            support=support,
            confidence=confidence,
            lift=lift,
        )
        key = (rule.mode_key, rule.antecedent_items)
        previous = deduped.get(key)
        if previous is None or (rule.support, rule.confidence) > (previous.support, previous.confidence):
            deduped[key] = rule
    return list(deduped.values())


def _prune_subsumed(rules: Sequence[InvariantRule]) -> list[InvariantRule]:
    kept: list[InvariantRule] = []
    ordered = sorted(
        rules,
        key=lambda rule: (
            rule.mode_key,
            -len(rule.antecedent_items),
            -rule.support,
            -rule.confidence,
            rule.signature,
        ),
    )
    for rule in ordered:
        items = set(rule.antecedent_items)
        if any(
            items < set(other.antecedent_items)
            for other in kept
            if other.mode_key == rule.mode_key
        ):
            continue
        kept.append(rule)
    return kept


# ---------------------------------------------------------------------------
# Metrics


def anomaly_metrics(
    y_true: Sequence[int] | np.ndarray,
    anomaly_flags: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Binary anomaly metrics with coarse AP, MCC, and unadjusted BA.

    ``scores`` defaults to the binary flags, which is the deliberately coarse
    score available from rule matching.
    """
    truth = np.asarray(y_true, dtype=int).reshape(-1)
    predicted = np.asarray(anomaly_flags, dtype=int).reshape(-1)
    if len(truth) != len(predicted):
        raise ValueError("y_true and anomaly_flags must have the same length")
    if scores is None:
        score_values = predicted.astype(float)
    else:
        score_values = np.asarray(scores, dtype=float).reshape(-1)
        if len(score_values) != len(truth):
            raise ValueError("scores must have the same length as y_true")

    tp = int(np.sum((truth == 1) & (predicted == 1)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    n = int(len(truth))

    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        f1_score,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mcc = float(matthews_corrcoef(truth, predicted)) if n else float("nan")
        balanced = float(balanced_accuracy_score(truth, predicted)) if n else float("nan")
        precision = float(precision_score(truth, predicted, zero_division=0)) if n else float("nan")
        recall = float(recall_score(truth, predicted, zero_division=0)) if n else float("nan")
        f1 = float(f1_score(truth, predicted, zero_division=0)) if n else float("nan")
        both_classes = n > 0 and len(np.unique(truth)) == 2 and len(np.unique(predicted)) >= 1
        if both_classes and score_values.size and not np.all(np.isnan(score_values)):
            average_precision = float(average_precision_score(truth, score_values))
        else:
            average_precision = float("nan")

    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "balanced_accuracy": balanced,
        "unadjusted_balanced_accuracy": balanced,
        "average_precision": average_precision,
        "average_precision_kind": "coarse_binary_rule_match",
        "positive_rate": float(np.mean(truth)) if n else float("nan"),
        "predicted_positive_rate": float(np.mean(predicted)) if n else float("nan"),
    }


# ---------------------------------------------------------------------------
# Model


class IMDISCO:
    """Fit mode-characterising invariant rules and flag rule violations.

    ``fit`` consumes *expert* operational-mode labels supplied by the caller.
    ``predict`` consumes only telemetry; it never reads an attack label.
    """

    def __init__(self, config: IMDISCOConfig | None = None) -> None:
        self.config = config or IMDISCOConfig()
        self._reset()

    def _reset(self) -> None:
        self._fitted = False
        self._sensor_columns: tuple[str, ...] = ()
        self._actuator_columns: tuple[str, ...] = ()
        self._mode_order: tuple[Any, ...] = ()
        self._mode_order_keys: tuple[str, ...] = ()
        self._predicates: tuple[Predicate, ...] = ()
        self._predicate_by_item: dict[str, Predicate] = {}
        self._bounds: dict[str, Any] = {}
        self._steadytime: dict[str, Any] = {}
        self._transitions: dict[str, Any] = {}
        self._rules: tuple[InvariantRule, ...] = ()
        self._rules_by_mode_key: dict[str, tuple[InvariantRule, ...]] = {}
        self._mode_counts: dict[str, int] = {}
        self._training_rows = 0
        self._mining: dict[str, Any] = {}
        self._association_rules_num_itemsets = False

    # -- fitting ---------------------------------------------------------

    def fit(
        self,
        frame: pd.DataFrame,
        modes: Sequence[Any] | pd.Series | str | None = None,
    ) -> "IMDISCO":
        self._reset()
        frame = _as_frame(frame)
        mode_values, mode_keys = _resolve_modes(frame, modes, self.config.mode_column)
        sensors, actuators = _resolve_columns(frame, self.config)
        mode_order = self._resolve_mode_order(mode_values, mode_keys)

        predicates = derive_predicates(
            frame,
            mode_values,
            sensors,
            actuators,
            inclusive_bounds=self.config.inclusive_bounds,
        )
        predicate_by_item = {predicate.item: predicate for predicate in predicates}
        mode_by_item = {_mode_item(key): value for value, key in zip(mode_values, mode_keys)}
        mode_key_by_item = {_mode_item(key): key for key in mode_keys}

        gradient = _gradient_frame(frame, sensors)
        runs = {
            column: _completed_run_records(frame[column].tolist(), mode_keys)[0]
            for column in actuators
        }
        transactions, columns = _transactions(
            frame, mode_values, mode_keys, predicates, gradient, runs
        )

        if self.config.prune_subsumed_rules:
            if self.config.max_itemset_len is not None:
                raise ValueError(
                    "the exact maximal-rule arm requires max_itemset_len=None; "
                    "use prune_subsumed_rules=False for a caller-chosen FP-Growth cap"
                )
            rules, mining_stats = _maximal_rules_from_transactions(
                transactions,
                predicate_by_item,
                mode_by_item,
                mode_key_by_item=mode_key_by_item,
                minsup=self.config.minsup,
                minconf=self.config.minconf,
            )
            num_itemsets_used = False
            estimated_itemsets = mining_stats["n_frequent_itemsets"]
        else:
            # The closed-pattern count is an exact guard, not a substitute for
            # the explicitly requested uncapped FP-Growth/ARG execution.
            estimated_itemsets, closed_count_stats = _count_frequent_itemsets(
                transactions,
                min_support_count=int(math.ceil(self.config.minsup * len(frame))),
            )
            if (
                self.config.max_itemset_len is None
                and self.config.max_frequent_itemsets is not None
                and estimated_itemsets > int(self.config.max_frequent_itemsets)
            ):
                raise MiningTooLargeError(
                    f"uncapped FP-Growth has exactly {estimated_itemsets} frequent "
                    f"itemsets (cap {self.config.max_frequent_itemsets}). Use the "
                    "default exact maximal-rule adaptation, set max_itemset_len "
                    "explicitly, or raise the cap explicitly."
                )

            miner = _load_mlxtend()
            mining_kwargs: dict[str, Any] = {
                "min_support": float(self.config.minsup),
                "use_colnames": True,
            }
            if self.config.max_itemset_len is not None:
                mining_kwargs["max_len"] = int(self.config.max_itemset_len)
            frequent_itemsets = miner.fpgrowth(transactions, **mining_kwargs)
            if self.config.max_frequent_itemsets is not None and len(frequent_itemsets) > int(
                self.config.max_frequent_itemsets
            ):
                raise MiningTooLargeError(
                    f"FP-Growth produced {len(frequent_itemsets)} frequent itemsets "
                    f"(cap {self.config.max_frequent_itemsets}). Set "
                    "IMDISCOConfig.max_itemset_len or raise the cap explicitly."
                )

            rule_frame, num_itemsets_used = _rule_dataframe(
                miner,
                frequent_itemsets,
                minconf=self.config.minconf,
                n_transactions=len(transactions),
            )
            rules = _rules_from_dataframe(
                rule_frame,
                predicate_by_item,
                mode_by_item,
                mode_key_by_item=mode_key_by_item,
                minconf=self.config.minconf,
            )
            mining_stats = {
                "miner": "mlxtend.frequent_patterns.fpgrowth + association_rules",
                "n_frequent_itemsets": int(len(frequent_itemsets)),
                "n_rules_before_pruning": len(rules),
                "frequent_itemsets_materialized": True,
                "unpruned_rules_materialized": True,
                **closed_count_stats,
            }

        ordered_keys = tuple(_value_key(mode) for mode in mode_order)
        mode_index = {key: index for index, key in enumerate(ordered_keys)}
        rules.sort(
            key=lambda rule: (
                mode_index.get(rule.mode_key, len(mode_index)),
                -len(rule.antecedent_items),
                -rule.support,
                rule.signature,
            )
        )
        rules_by_mode: dict[str, list[InvariantRule]] = {}
        for rule in rules:
            rules_by_mode.setdefault(rule.mode_key, []).append(rule)

        counts_by_key: dict[str, int] = {}
        representative = {key: value for value, key in zip(mode_values, mode_keys)}
        for key in mode_keys:
            counts_by_key[key] = counts_by_key.get(key, 0) + 1
        counts = {
            _display(representative[key]): count
            for key, count in counts_by_key.items()
        }

        self._sensor_columns = sensors
        self._actuator_columns = actuators
        self._mode_order = mode_order
        self._mode_order_keys = tuple(_value_key(mode) for mode in mode_order)
        self._predicates = tuple(predicates)
        self._predicate_by_item = predicate_by_item
        self._bounds = bounds_artifact(predicates)
        self._steadytime = _steadytime_provenance(frame, actuators, mode_keys)
        self._transitions = _transition_provenance(mode_values, mode_keys)
        self._rules = tuple(rules)
        self._rules_by_mode_key = {key: tuple(value) for key, value in rules_by_mode.items()}
        self._mode_counts = counts
        self._training_rows = int(len(frame))
        self._association_rules_num_itemsets = bool(num_itemsets_used)
        self._mining = {
            **mining_stats,
            "mlxtend": _mlxtend_info(),
            "minsup": float(self.config.minsup),
            "minconf": float(self.config.minconf),
            "max_itemset_len": self.config.max_itemset_len,
            "uncapped_frequent_itemsets_exact_count": int(estimated_itemsets),
            "prune_subsumed_rules": bool(self.config.prune_subsumed_rules),
            "n_transactions": int(len(transactions)),
            "n_items": int(len(columns)),
            "n_rules": int(len(rules)),
            "association_rules_num_itemsets_kwarg": bool(num_itemsets_used),
        }
        self._fitted = True
        return self

    def _resolve_mode_order(
        self,
        mode_values: np.ndarray,
        mode_keys: Sequence[str],
    ) -> tuple[Any, ...]:
        unique: dict[str, Any] = {}
        for value, key in zip(mode_values, mode_keys):
            unique.setdefault(key, value)
        if self.config.mode_order is not None:
            ordered = tuple(self.config.mode_order)
            ordered_keys = tuple(_value_key(mode) for mode in ordered)
            if len(set(ordered_keys)) != len(ordered_keys):
                raise ValueError("mode_order contains duplicate modes")
            if set(ordered_keys) != set(unique):
                missing = sorted(set(unique) - set(ordered_keys))
                extra = sorted(set(ordered_keys) - set(unique))
                raise ValueError(
                    "mode_order must contain exactly the training modes; "
                    f"missing={missing}, extra={extra}"
                )
            return ordered
        return tuple(sorted(unique.values(), key=_mode_sort_key))

    # -- prediction ------------------------------------------------------

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise IMDISCOError("IMDISCO must be fitted before this operation")

    def predict(self, frame: pd.DataFrame) -> PredictionResult:
        self._require_fitted()
        frame = _as_frame(frame)
        missing = [
            column
            for column in (*self._sensor_columns, *self._actuator_columns)
            if column not in frame.columns
        ]
        if missing:
            raise ValueError(f"missing feature columns: {missing}")

        gradient = _gradient_frame(frame, self._sensor_columns)
        runs = {
            column: _run_lengths(frame[column].tolist())
            for column in self._actuator_columns
        }

        predicted_modes: list[Any | None] = []
        anomaly_flags = np.zeros(len(frame), dtype=int)
        anomaly_scores = np.zeros(len(frame), dtype=float)
        matched_rule_ids: list[str | None] = []
        previous_mode_key: str | None = None

        for i in range(len(frame)):
            values = {column: frame[column].iat[i] for column in frame.columns}
            gradients = gradient.iloc[i].to_dict()
            run_lengths = {column: values_series[i] for column, values_series in runs.items()}
            chosen: InvariantRule | None = None
            chosen_mode: Any | None = None
            for mode_key in self._mode_order_keys:
                for rule in self._rules_by_mode_key.get(mode_key, ()):
                    if rule.evaluate(
                        values=values,
                        gradients=gradients,
                        run_lengths=run_lengths,
                        previous_mode_key=previous_mode_key,
                        frame_start=i == 0,
                    ):
                        chosen = rule
                        chosen_mode = rule.mode
                        break
                if chosen is not None:
                    break
            if chosen is None:
                anomaly_flags[i] = 1
                anomaly_scores[i] = 1.0
                predicted_modes.append(None)
                matched_rule_ids.append(None)
            else:
                anomaly_flags[i] = 0
                anomaly_scores[i] = 0.0
                predicted_modes.append(chosen_mode)
                matched_rule_ids.append(chosen.rule_id)
            previous_mode_key = _value_key(chosen_mode) if chosen is not None else None

        return PredictionResult(
            predicted_modes=tuple(predicted_modes),
            anomaly_flags=anomaly_flags,
            anomaly_scores=anomaly_scores,
            matched_rule_ids=tuple(matched_rule_ids),
        )

    def evaluate(
        self,
        frame: pd.DataFrame,
        y_true_anomaly: Sequence[int] | np.ndarray,
    ) -> dict[str, Any]:
        result = self.predict(frame)
        return {
            "predictions": result.to_dict(),
            "metrics": anomaly_metrics(
                y_true_anomaly, result.anomaly_flags, result.anomaly_scores
            ),
        }

    # -- artefacts / provenance -----------------------------------------

    def provenance(self) -> dict[str, Any]:
        info = _mlxtend_info()
        return {
            "schema_version": SCHEMA_VERSION,
            "paper": "Saraiva, Apolinario & Pardal, IM-DISCO (paper and 2023 thesis)",
            "method": (
                "mode-local MinMax/Gradient/TransitionGradient/SteadyTime + shared State; "
                "default exact maximal closed-pattern rules, optional uncapped mlxtend "
                "FP-Growth + association_rules with mode consequent"
            ),
            "miner": self._mining.get(
                "miner", "exact maximal by default; uncapped mlxtend when pruning is false"
            ),
            "mlxtend_available": info["available"],
            "mlxtend_version": info["version"],
            "original_rule_ordering_available": False,
            "missing_original_rule_ordering": True,
            "original_artifacts_available": False,
            "complete_original_implementation_available": False,
            "partial_source_snippets_available": True,
            "artifact_availability": {
                "paper_code": "partial source snippets in thesis Appendix A, not complete code",
                "complete_paper_code": False,
                "paper_rule_files": False,
                "paper_training_data": False,
                "paper_prediction_order": False,
                "thesis": "https://scholar.tecnico.ulisboa.pt/records/LmNvJf4G1Zrcca0CmVSN_A7JNLQxYDwTC6SA",
            },
            "rule_order_policy": (
                "mode order: explicit IMDISCOConfig.mode_order when supplied, otherwise "
                "lexicographic ascending by mode label; within a mode: longer antecedent "
                "first, then higher support, then antecedent signature. The paper does "
                "not publish its original rule ordering, so this order is a documented "
                "deterministic reconstruction."
            ),
            "rule_pruning": {
                "default": "exact maximal-rule thesis-inspired adaptation",
                "thesis_evidence": (
                    "Section 4.4.2 and Listing A.3 specify redundancy removal after ARG: "
                    "antecedents contained in other antecedents are removed"
                ),
                "paper_evidence": (
                    "Section 5.4 reports one maximal conjunction per mode but does not "
                    "specify the reduction algorithm"
                ),
                "exactness": (
                    "frequent closed antecedents plus strict-subsumption pruning compute "
                    "the same final rules as uncapped FP-Growth/ARG followed by "
                    "_prune_subsumed; unpruned FP-Growth remains potentially exponential"
                ),
                "uncapped_arm": "set prune_subsumed_rules=False",
                "official_code_reproduction": False,
            },
            "predicate_identity": {
                "mode_local": ["minmax", "gradient", "transitiongradient", "steadytime"],
                "shared_across_modes": ["state"],
                "cross_mode_confidence": (
                    "shared State support is computed across all modes, so State=>mode "
                    "confidence can be below one; the mode item remains a separate consequent"
                ),
            },
            "mode_order_policy": (
                "caller-supplied mode_order, else lexicographic mode-label ascending"
            ),
            "bound_conventions": {
                "inclusive_by_default": bool(self.config.inclusive_bounds),
                "paper_note": (
                    "Section 3.3 prints strict inequalities; Section 5.3 example rules "
                    "use <=. Inclusive bounds are the default so observed extrema remain "
                    "valid; set inclusive_bounds=False to restore the printed formulas."
                ),
                "first_gradient": (
                    "no ordinary Gradient at frame start or mode entry; raw diff still "
                    "defines first-row scratch value as 0.0"
                ),
                "gradient": "ordinary within-mode differences only",
                "transitiongradient": (
                    "separate current/previous-mode-keyed predicate at the first row "
                    "entering a mode only, never subsequent rows of that mode"
                ),
                "steadytime": (
                    "completed run total duration is assigned retrospectively to every "
                    "row of the run; final ongoing runs are excluded; mode-crossing runs "
                    "are excluded from training with per-run provenance"
                ),
                "steadytime_causal": False,
                "missing_values": "missing sensor/actuator values never satisfy a predicate",
            },
            "runtime_conventions": {
                "first_matching_mode": True,
                "unmatched": "anomalous (binary flag 1, predicted mode None)",
                "out_of_training": "anomalous when no invariant rule matches",
                "anomaly_score": "coarse binary: 0.0 matched, 1.0 unmatched",
                "gradient_and_run_lengths_reset_at_frame_start": True,
                "steadytime_retrospective_noncausal": True,
                "transition_previous_mode": "previous predicted mode, not an attack or expert label",
            },
            "reproduction_gaps": [
                "the original rule ordering is not reported",
                "the thesis provides partial source snippets, not the complete implementation, rule files, or ride data",
                "the thesis specifies redundancy removal after ARG but not its reduction algorithm",
                "strict vs inclusive bound notation differs between formulas and examples",
                "mode-crossing SteadyTime runs are excluded as an explicit adaptation",
                "completed-run totals are retrospective and noncausal at runtime",
                "TransitionGradient is entry-only here and runtime mode boundaries use the previous predicted mode",
                "the default maximal arm avoids exponential materialization; the original uncapped FP-Growth/ARG arm remains potentially exponential",
                "average precision is coarse because matched/unmatched is binary",
            ],
            "training": (
                None
                if not self._fitted
                else {
                    "n_rows": int(self._training_rows),
                    "mode_counts": dict(sorted(self._mode_counts.items())),
                }
            ),
        }

    def artifacts(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "schema_version": SCHEMA_VERSION,
            "config": self.config.to_dict(),
            "provenance": self.provenance(),
            "columns": {
                "sensors": list(self._sensor_columns),
                "actuators": list(self._actuator_columns),
            },
            "mode_order": [_display(mode) for mode in self._mode_order],
            "training": {
                "n_rows": int(self._training_rows),
                "mode_counts": dict(sorted(self._mode_counts.items())),
            },
            "bounds": self._bounds,
            "steadytime": self._steadytime,
            "transitions": self._transitions,
            "predicates": [predicate.to_dict() for predicate in self._predicates],
            "rules": [rule.to_dict() for rule in self._rules],
            "mining": dict(self._mining),
        }

    def artifacts_json(self) -> str:
        """Canonical JSON, suitable for byte-for-byte determinism checks."""
        return json.dumps(self.artifacts(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    # -- introspection helpers ------------------------------------------

    @property
    def rules(self) -> tuple[InvariantRule, ...]:
        self._require_fitted()
        return self._rules

    @property
    def predicates(self) -> tuple[Predicate, ...]:
        self._require_fitted()
        return self._predicates

    @property
    def mode_order(self) -> tuple[Any, ...]:
        self._require_fitted()
        return self._mode_order

    def first_matching_mode(self, matched_modes: Iterable[Any]) -> Any | None:
        """Return the first mode in this model's documented order.

        ``matched_modes`` may contain any modes; values outside the training
        order are ignored, so unknown modes can never be predicted.
        """
        self._require_fitted()
        matched_keys = {_value_key(mode) for mode in matched_modes}
        for mode in self._mode_order:
            if _value_key(mode) in matched_keys:
                return mode
        return None
