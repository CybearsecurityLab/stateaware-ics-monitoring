"""IM-DISCO invariant-rule baseline (Saraiva, Apolinario & Pardal).

This module is a from-scratch implementation of the full IM-DISCO paper
(``imdisco-paper.pdf``), not of the aggregate results tables.  The pipeline is:

1. **Predicate generation** per expert-supplied operational mode:
   * sensor ``MinMax`` bounds (``min(V_m) <= x <= max(V_m)``),
   * first-gradient bounds (``diff(x)``, first row defined as ``0.0``),
   * actuator state predicates, and
   * actuator ``SteadyTime`` bounds over consecutive runs of one state.
2. **Invariant rule mining**: one transaction per timestep containing the
   predicates of that timestep's expert mode plus the mode item; FP-Growth
   (``minsup=0.05``) and mlxtend association rules (``minconf=1.0``) with the
   mode as consequent.
3. **Runtime inference**: evaluate rules in a documented deterministic order
   and take the first matching mode.  No rule matches (including behaviour
   outside the training bounds) -> anomaly.

The paper's formulas print strict inequalities, while its own example rules
(Section 5.3) use ``<=``.  This implementation defaults to inclusive bounds;
``IMDISCOConfig.inclusive_bounds=False`` restores strictly printed formulas.

The original paper does **not** publish code, rule files, the training ride
data, or its rule ordering.  Those gaps are recorded in
:meth:`IMDISCO.provenance` rather than guessed silently.

``mlxtend`` is required for rule mining.  If it is absent, ``fit`` raises
:class:`MissingMinerError`; no alternate miner is bundled.
"""

from __future__ import annotations

import importlib.util
import json
import math
import warnings
from dataclasses import dataclass, field
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

SCHEMA_VERSION = "imdisco-baseline-v1"
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
    mode_key: str,
    lo: float | None,
    hi: float | None,
    state_key: str | None,
) -> str:
    return json.dumps(
        [kind, variable, mode_key, lo, hi, state_key],
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

    ``max_itemset_len`` defaults to ``None``: Section 4 of the paper specifies
    all frequent itemsets with ``minsup=0.05`` and no maximum length.  An
    explicit integer imposes a caller-chosen cap; that cap is forwarded to
    FP-Growth, never replaced by a different miner or a silent length limit.
    ``max_frequent_itemsets`` is the explicit tractability guard: exceeding it
    raises :class:`MiningTooLargeError` instead of substituting another method.
    """

    sensor_columns: tuple[str, ...] | None = None
    actuator_columns: tuple[str, ...] | None = None
    mode_column: str | None = None
    minsup: float = 0.05
    minconf: float = 1.0
    inclusive_bounds: bool = True
    max_itemset_len: int | None = None
    prune_subsumed_rules: bool = False
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
    """A mode-scoped sensor/actuator predicate."""

    kind: str
    variable: str
    mode: Any
    mode_key: str
    lo: float | None = None
    hi: float | None = None
    state_key: str | None = None
    n: int = 0
    inclusive: bool = True

    @property
    def item(self) -> str:
        return _predicate_item(
            self.kind, self.variable, self.mode_key, self.lo, self.hi, self.state_key
        )

    def evaluate(
        self,
        *,
        values: Mapping[str, Any],
        gradients: Mapping[str, Any],
        run_lengths: Mapping[str, Any],
    ) -> bool:
        if self.kind in ("minmax", "gradient"):
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
    ) -> bool:
        return all(
            p.evaluate(values=values, gradients=gradients, run_lengths=run_lengths)
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


def _run_lengths(values: Iterable[Any]) -> np.ndarray:
    values = list(values)
    out = np.zeros(len(values), dtype=float)
    previous: str | None = None
    run = 0
    for i, value in enumerate(values):
        current = _value_key(value)
        if current is None:
            run = 0
            out[i] = 0.0
        elif previous == current:
            run += 1
            out[i] = float(run)
        else:
            run = 1
            out[i] = 1.0
        previous = current
    return out


def derive_predicates(
    frame: pd.DataFrame,
    modes: Sequence[Any] | pd.Series | str | None,
    sensor_columns: Sequence[str],
    actuator_columns: Sequence[str],
    *,
    mode_column: str | None = None,
    inclusive_bounds: bool = True,
) -> list[Predicate]:
    """Generate mode-scoped MinMax, Gradient, State, and SteadyTime predicates.

    This function is deliberately usable without mlxtend so that bound tests do
    not need a working miner.
    """
    frame = _as_frame(frame)
    mode_values, mode_keys = _resolve_modes(frame, modes, mode_column)
    sensors = tuple(str(c) for c in sensor_columns)
    actuators = tuple(str(c) for c in actuator_columns)
    missing = [c for c in (*sensors, *actuators) if c not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")

    gradient = _gradient_frame(frame, sensors)
    runs = {column: _run_lengths(frame[column].tolist()) for column in actuators}

    predicates: list[Predicate] = []
    for mode_key in sorted(set(mode_keys)):
        indices = np.flatnonzero(np.array([key == mode_key for key in mode_keys]))
        if len(indices) == 0:
            continue
        mode_value = mode_values[int(indices[0])]

        for column in sensors:
            values = pd.to_numeric(frame.iloc[indices][column], errors="coerce").to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            if len(values):
                predicates.append(
                    Predicate(
                        kind="minmax",
                        variable=column,
                        mode=mode_value,
                        mode_key=mode_key,
                        lo=float(values.min()),
                        hi=float(values.max()),
                        n=int(len(values)),
                        inclusive=bool(inclusive_bounds),
                    )
                )
            gradients = gradient.iloc[indices][column].to_numpy(dtype=float)
            gradients = gradients[~np.isnan(gradients)]
            if len(gradients):
                predicates.append(
                    Predicate(
                        kind="gradient",
                        variable=column,
                        mode=mode_value,
                        mode_key=mode_key,
                        lo=float(gradients.min()),
                        hi=float(gradients.max()),
                        n=int(len(gradients)),
                        inclusive=bool(inclusive_bounds),
                    )
                )

        for column in actuators:
            column_values = frame.iloc[indices][column].tolist()
            state_keys = sorted(
                {key for key in (_value_key(v) for v in column_values) if key is not None}
            )
            for state_key in state_keys:
                state_rows = [
                    int(index)
                    for index in indices
                    if _value_key(frame.iloc[int(index)][column]) == state_key
                ]
                predicates.append(
                    Predicate(
                        kind="state",
                        variable=column,
                        mode=mode_value,
                        mode_key=mode_key,
                        state_key=state_key,
                        n=len(state_rows),
                    )
                )
                durations = [runs[column][index] for index in state_rows if runs[column][index] > 0]
                if durations:
                    predicates.append(
                        Predicate(
                            kind="steadytime",
                            variable=column,
                            mode=mode_value,
                            mode_key=mode_key,
                            lo=float(min(durations)),
                            hi=float(max(durations)),
                            state_key=state_key,
                            n=int(len(durations)),
                            inclusive=bool(inclusive_bounds),
                        )
                    )

    predicates.sort(key=lambda predicate: predicate.item)
    return predicates


def bounds_artifact(predicates: Sequence[Predicate]) -> dict[str, Any]:
    """Group predicates into a deterministic, JSON-safe bounds report."""
    out: dict[str, Any] = {
        "minmax": {},
        "gradient": {},
        "steadytime": {},
        "actuator_states": {},
    }
    for predicate in predicates:
        mode = _display(predicate.mode)
        if predicate.kind in ("minmax", "gradient"):
            out[predicate.kind].setdefault(mode, {})[predicate.variable] = {
                "lo": None if predicate.lo is None else _finite_or_none(predicate.lo),
                "hi": None if predicate.hi is None else _finite_or_none(predicate.hi),
                "n": int(predicate.n),
                "inclusive": bool(predicate.inclusive),
            }
        elif predicate.kind == "state":
            bucket = out["actuator_states"].setdefault(mode, {}).setdefault(predicate.variable, [])
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
            "implementations (paper: FP-Growth min_support=0.05, "
            "association_rules confidence>=1.0). mlxtend is not installed or "
            "cannot be imported; no alternate miner is bundled. "
            "Install mlxtend and re-run."
        ) from exc
    return SimpleNamespace(fpgrowth=fpgrowth, association_rules=association_rules)


def _estimate_frequent_itemsets(
    predicates: Sequence[Predicate],
    mode_keys: Sequence[str],
    *,
    minsup: float,
    max_itemset_len: int | None,
) -> int:
    """Conservative upper bound for own-mode transaction itemsets.

    Every predicate derived for a mode holds on every training row of that
    mode (inclusive bounds), so all subsets of a frequent mode co-occur.  Modes
    below ``minsup`` contribute no frequent items.
    """
    if not mode_keys:
        return 0
    per_mode: dict[str, int] = {}
    for predicate in predicates:
        per_mode[predicate.mode_key] = per_mode.get(predicate.mode_key, 0) + 1
    row_counts: dict[str, int] = {}
    for key in mode_keys:
        row_counts[key] = row_counts.get(key, 0) + 1
    n_rows = len(mode_keys)
    total = 0
    for key, predicate_count in per_mode.items():
        if row_counts.get(key, 0) / n_rows < float(minsup):
            continue
        item_count = predicate_count + 1  # predicates plus the mode item
        top = item_count if max_itemset_len is None else min(int(max_itemset_len), item_count)
        total += sum(math.comb(item_count, size) for size in range(1, top + 1))
    return total


def _transactions(
    frame: pd.DataFrame,
    modes_values: np.ndarray,
    mode_keys: Sequence[str],
    predicates: Sequence[Predicate],
    gradient: pd.DataFrame,
    runs: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, list[str]]:
    by_mode: dict[str, list[Predicate]] = {}
    for predicate in predicates:
        by_mode.setdefault(predicate.mode_key, []).append(predicate)

    rows: list[set[str]] = []
    for i in range(len(frame)):
        values = frame.iloc[i].to_dict()
        gradients = gradient.iloc[i].to_dict()
        run_lengths = {column: values_series[i] for column, values_series in runs.items()}
        items = {
            predicate.item
            for predicate in by_mode.get(mode_keys[i], ())
            if predicate.evaluate(values=values, gradients=gradients, run_lengths=run_lengths)
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
        runs = {column: _run_lengths(frame[column].tolist()) for column in actuators}
        transactions, columns = _transactions(
            frame, mode_values, mode_keys, predicates, gradient, runs
        )

        estimated_itemsets = _estimate_frequent_itemsets(
            predicates,
            mode_keys,
            minsup=self.config.minsup,
            max_itemset_len=self.config.max_itemset_len,
        )
        if (
            self.config.max_frequent_itemsets is not None
            and estimated_itemsets > int(self.config.max_frequent_itemsets)
        ):
            raise MiningTooLargeError(
                f"FP-Growth is estimated to produce {estimated_itemsets} frequent "
                f"itemsets (cap {self.config.max_frequent_itemsets}) because "
                "mode-local predicates co-occur. Lower IMDISCOConfig.max_itemset_len "
                "or raise max_frequent_itemsets explicitly."
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
        before_pruning = len(rules)
        if self.config.prune_subsumed_rules:
            rules = _prune_subsumed(rules)

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
        self._rules = tuple(rules)
        self._rules_by_mode_key = {key: tuple(value) for key, value in rules_by_mode.items()}
        self._mode_counts = counts
        self._training_rows = int(len(frame))
        self._association_rules_num_itemsets = bool(num_itemsets_used)
        self._mining = {
            "miner": "mlxtend.frequent_patterns.fpgrowth + association_rules",
            "mlxtend": _mlxtend_info(),
            "minsup": float(self.config.minsup),
            "minconf": float(self.config.minconf),
            "max_itemset_len": self.config.max_itemset_len,
            "estimated_frequent_itemsets_upper_bound": int(estimated_itemsets),
            "prune_subsumed_rules": bool(self.config.prune_subsumed_rules),
            "n_transactions": int(len(transactions)),
            "n_items": int(len(columns)),
            "n_frequent_itemsets": int(len(frequent_itemsets)),
            "n_rules_before_pruning": int(before_pruning),
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

        for i in range(len(frame)):
            values = frame.iloc[i].to_dict()
            gradients = gradient.iloc[i].to_dict()
            run_lengths = {column: values_series[i] for column, values_series in runs.items()}
            chosen: InvariantRule | None = None
            chosen_mode: Any | None = None
            for mode_key in self._mode_order_keys:
                for rule in self._rules_by_mode_key.get(mode_key, ()):
                    if rule.evaluate(
                        values=values, gradients=gradients, run_lengths=run_lengths
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
            "paper": "Saraiva, Apolinario & Pardal, IM-DISCO (full paper)",
            "method": "MinMax + first-gradient + SteadyTime predicates; "
            "FP-Growth (minsup=0.05) + association rules (minconf=1.0) with mode consequent",
            "miner": "mlxtend.frequent_patterns.fpgrowth + association_rules",
            "mlxtend_available": info["available"],
            "mlxtend_version": info["version"],
            "original_rule_ordering_available": False,
            "missing_original_rule_ordering": True,
            "original_artifacts_available": False,
            "artifact_availability": {
                "paper_code": False,
                "paper_rule_files": False,
                "paper_training_data": False,
                "paper_prediction_order": False,
            },
            "rule_order_policy": (
                "mode order: explicit IMDISCOConfig.mode_order when supplied, otherwise "
                "lexicographic ascending by mode label; within a mode: longer antecedent "
                "first, then higher support, then antecedent signature. The paper does "
                "not publish its original rule ordering, so this order is a documented "
                "deterministic reconstruction."
            ),
            "rule_pruning": (
                "disabled by default: the paper selects rules only by minconf, so "
                "no subsumption pruning is applied unless the caller explicitly "
                "sets IMDISCOConfig.prune_subsumed_rules=True"
            ),
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
                "first_gradient": "0.0 (the paper does not define the first row's gradient)",
                "steadytime": (
                    "run length counts consecutive identical actuator values over the "
                    "supplied frame; the first row starts at 1 and mode-local bounds are "
                    "computed from rows labelled with that mode"
                ),
                "missing_values": "missing sensor/actuator values never satisfy a predicate",
            },
            "runtime_conventions": {
                "first_matching_mode": True,
                "unmatched": "anomalous (binary flag 1, predicted mode None)",
                "out_of_training": "anomalous when no invariant rule matches",
                "anomaly_score": "coarse binary: 0.0 matched, 1.0 unmatched",
                "gradient_and_run_lengths_reset_at_frame_start": True,
            },
            "reproduction_gaps": [
                "the original rule ordering is not reported in the paper",
                "the original code, rule files, and ride data are not published",
                "strict vs inclusive bound notation differs between Sections 3.3 and 5.3",
                "the first-row gradient and run-length boundary rules are unspecified",
                "the paper does not specify how a SteadyTime run crossing a mode boundary is attributed",
                "the paper specifies all frequent itemsets with minsup=0.05 and no maximum length; this implementation defaults to no length cap and fails explicitly through max_frequent_itemsets when tractability is exceeded",
                "runtime gradients and SteadyTime runs restart at the first row of each prediction frame",
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
