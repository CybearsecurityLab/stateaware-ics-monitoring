"""Topology-driven residual detector.

Time comes from a finite, strictly increasing ``time_seconds`` column when
present; otherwise the index must be numeric seconds. Flow channels ``f_ID``
are instantaneous rates in m^3/s. Optional ``volume_ID`` channels are signed
interval-integrated volumes in m^3 (positive along link orientation).

The detector is evaluated row by row, so every *static* fact about the
network (element lookups, incident links, channel names, pattern objects,
curve points, ...) is resolved once during
:meth:`TopologyStlDetector._compile` and reused for every sample.  The
per-sample work is then limited to reading the observed channels and
evaluating the residual equations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from hydraulics.model import ChannelSpec
from hydraulics.templates import TemplateStatus, compile_templates
from monitor.protocol import StepFlags
from . import residuals
from .topology import channel_column


@dataclass(frozen=True)
class FormulaResult:
    key: str
    family: str
    required_channels: tuple[str, ...]
    unit: str
    status: str
    reason: str = ""


@dataclass(frozen=True)
class Tolerance:
    epsilon: float
    unit: str
    percentile: float
    additive: float


# --------------------------------------------------------------------------
# sample access
# --------------------------------------------------------------------------

def _column_values(series: "pd.Series"):
    """Return the values of *series* as a numpy array, preserving element types.

    The elements must look exactly like the ones a ``DataFrame.iterrows()``
    row would carry, so that ``float(value)`` and its error messages are
    unchanged.
    """
    kind = getattr(series.dtype, "kind", "O")
    if kind in "biuf":  # bool, signed/unsigned int, float
        return series.to_numpy()
    return series.to_numpy(dtype=object)


class _SampleView:
    """Read access to the observed channels of one sample.

    Two implementations exist: :class:`_FrameView` reads a frame
    column-wise (the fast path) and :class:`_SeriesView` reads one
    ``pandas.Series`` row (the legacy, single-row interface).
    """

    __slots__ = ()

    def has(self, col: str) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def raw(self, col: str, index: int):
        raise NotImplementedError  # pragma: no cover - interface

    def number(self, col: str, index: int) -> float:
        """Return the finite float carried by *col* (raise when unusable)."""
        if not self.has(col):
            raise KeyError(f"missing channel {col}")
        value = float(self.raw(col, index))
        if not math.isfinite(value):
            raise ValueError(f"non-finite channel {col}")
        return value

    def multiplier(self, pattern_id, pattern, time: float) -> float:
        """Return the pattern multiplier at *time*."""
        return pattern.multiplier_at(time)


class _FrameView(_SampleView):
    """Column-oriented access to the values of one frame."""

    __slots__ = ("arrays", "duplicates", "present", "_multipliers", "_values", "_index")

    def __init__(self, frame: "pd.DataFrame", needed: Sequence[str]):
        counts: dict[str, int] = {}
        for col in frame.columns:
            counts[col] = counts.get(col, 0) + 1
        self.arrays = {}
        self.duplicates = {}
        for col in needed:
            occurrences = counts.get(col, 0)
            if occurrences == 1:
                self.arrays[col] = _column_values(frame[col])
            elif occurrences > 1:
                # A duplicated column behaves like pandas does: the row value
                # is a Series, so ``float(...)`` fails with pandas' message.
                self.duplicates[col] = frame[col]
        self.present = frozenset(self.arrays) | frozenset(self.duplicates)
        self._multipliers: dict[str, float] = {}
        self._values: dict[str, float] = {}
        self._index = -1

    def has(self, col: str) -> bool:
        return col in self.present

    def raw(self, col: str, index: int):
        array = self.arrays.get(col)
        if array is not None:
            return array[index]
        duplicate = self.duplicates.get(col)
        if duplicate is None:
            raise KeyError(col)
        return duplicate.iloc[index]

    def number(self, col: str, index: int) -> float:
        """Return the finite float carried by *col* (cached for the current sample)."""
        current = index == self._index
        if current:
            cached = self._values.get(col)
            if cached is not None:
                return cached
        array = self.arrays.get(col)
        if array is not None:
            raw = array[index]
        else:
            duplicate = self.duplicates.get(col)
            if duplicate is None:
                raise KeyError(f"missing channel {col}")
            raw = duplicate.iloc[index]
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"non-finite channel {col}")
        if current:
            self._values[col] = value
        return value

    def multiplier(self, pattern_id, pattern, time: float) -> float:
        cached = self._multipliers.get(pattern_id)
        if cached is None:
            cached = pattern.multiplier_at(time)
            self._multipliers[pattern_id] = cached
        return cached

    def new_sample(self, index: int) -> None:
        """Start a new sample: drop the per-sample caches."""
        self._multipliers.clear()
        self._values.clear()
        self._index = index


class _SeriesView(_SampleView):
    """Access to the values of one ``pandas.Series`` row."""

    __slots__ = ("row",)

    def __init__(self, row: "pd.Series"):
        self.row = row

    def has(self, col: str) -> bool:
        return col in self.row.index

    def raw(self, col: str, index: int):
        return self.row[col]


# --------------------------------------------------------------------------
# compiled plans
# --------------------------------------------------------------------------

class _Head:
    """How the head (m) of one node is obtained at a given time."""

    __slots__ = ("kind", "elevation", "base_head", "pattern", "pattern_id", "column", "error")

    def __init__(self, kind: str, *, elevation: float = 0.0, base_head: float = 0.0,
                 pattern=None, pattern_id=None, column: str = "", error=None):
        self.kind = kind
        self.elevation = elevation
        self.base_head = base_head
        self.pattern = pattern
        self.pattern_id = pattern_id
        self.column = column
        self.error = error

    def value(self, view: _SampleView, index: int, time: float) -> float:
        kind = self.kind
        if kind == "reservoir":
            pattern = self.pattern
            if pattern is None:
                return self.base_head
            return self.base_head * view.multiplier(self.pattern_id, pattern, time)
        if kind in ("tank", "junction"):
            return self.elevation + view.number(self.column, index)
        raise self.error

    def column_name(self) -> str:
        return self.column


class _Plan:
    """One compiled template, ready to be evaluated sample by sample."""

    __slots__ = ("key", "base", "status", "reason", "unit", "tol_keys", "element_missing")

    family = ""

    def __init__(self, key: str, base: dict, status: TemplateStatus, reason: str, unit: str):
        self.key = key
        self.base = base
        self.status = status
        self.reason = reason
        self.unit = unit
        self.tol_keys: dict[tuple[str, str], str] = {}
        self.element_missing = False

    def tolerance_key(self, regime: str, unit: str) -> str:
        """Return the tolerance key of *regime*/*unit* (precomputed where possible)."""
        by_unit = self.tol_keys.get(regime)
        if by_unit is None:
            by_unit = {}
            self.tol_keys[regime] = by_unit
        key = by_unit.get(unit)
        if key is None:
            key = f"{self.key}|{regime}|{unit}"
            by_unit[unit] = key
        return key

    def needed_columns(self) -> set[str]:
        """Return the columns this plan may read."""
        return set()

    def evaluate(self, view: _SampleView, index: int, time: float, times):  # pragma: no cover
        if self.element_missing:
            raise StopIteration()
        raise NotImplementedError


class _JunctionPlan(_Plan):
    """Continuity residual of one junction."""

    __slots__ = ("flow_cols", "terms", "demand_col", "leak_col", "emitter_col",
                 "pressure_col", "is_pdd", "demand_multiplier", "demand_model",
                 "pressure_exponent", "required_pressure", "minimum_pressure",
                 "emitter_coefficient", "emitter_exponent")

    family = "junction_continuity"

    def needed_columns(self) -> set[str]:
        columns = {col for col, _ in self.flow_cols}
        columns.update((self.demand_col, self.leak_col, self.emitter_col, self.pressure_col))
        return columns

    def evaluate(self, view: _SampleView, index: int, time: float, times):
        incoming: list[float] = []
        outgoing: list[float] = []
        for col, is_outgoing in self.flow_cols:
            flow = view.number(col, index)
            if is_outgoing:
                outgoing.append(flow)
            else:
                incoming.append(flow)
        terms = []
        for base_demand, pattern_id, pattern, error in self.terms:
            if error is not None:
                raise error
            multiplier = view.multiplier(pattern_id, pattern, time) if pattern is not None else 1.0
            terms.append((base_demand, multiplier))
        has_demand = view.has(self.demand_col)
        has_leak = view.has(self.leak_col)
        needs_pressure = (
            (not has_demand and self.is_pdd)
            or (not has_leak and self.emitter_coefficient > 0)
        )
        pressure = view.number(self.pressure_col, index) if needs_pressure else 0.0
        demand = (
            view.number(self.demand_col, index)
            if has_demand
            else residuals.delivered_demand(
                terms, self.demand_multiplier, self.demand_model, pressure,
                self.pressure_exponent, self.required_pressure, self.minimum_pressure,
            )
        )
        leak = (
            view.number(self.leak_col, index)
            if has_leak
            else residuals.emitter_withdrawal(
                self.emitter_coefficient, pressure, self.emitter_exponent
            )
        )
        emitter_col = self.emitter_col
        emitter = view.number(emitter_col, index) if view.has(emitter_col) else 0.0
        value = residuals.junction_continuity_residual(
            inflows=incoming, outflows=outgoing, delivered_demand=demand,
            emitter_withdrawals=(leak, emitter),
        )
        return value, "observed", self.unit


class _PipePlan(_Plan):
    """Energy residual of one pipe (a closed pipe only checks its flow)."""

    __slots__ = ("flow_col", "status_col", "length", "diameter", "roughness",
                 "minor_loss", "head_start", "head_end")

    family = "pipe_energy"

    def needed_columns(self) -> set[str]:
        columns = {self.flow_col, self.status_col}
        columns.update((self.head_start.column_name(), self.head_end.column_name()))
        return columns

    def evaluate(self, view: _SampleView, index: int, time: float, times):
        status = TopologyStlDetector._status(view.raw(self.status_col, index))
        if status == "CLOSED":
            return view.number(self.flow_col, index), "closed", "m^3/s"
        flow = view.number(self.flow_col, index)
        value = residuals.pipe_energy_residual(
            flow=flow, length=self.length, diameter=self.diameter,
            roughness=self.roughness,
            head_start=self.head_start.value(view, index, time),
            head_end=self.head_end.value(view, index, time),
            minor_loss=self.minor_loss,
        )
        return value, "observed", self.unit


class _TankPlan(_Plan):
    """Volume residual of one tank over the preceding interval."""

    __slots__ = ("level_col", "links", "diameter", "min_volume", "min_level", "volume_curve")

    family = "tank_volume"

    def needed_columns(self) -> set[str]:
        columns = {self.level_col}
        for volume_col, flow_col, _ in self.links:
            columns.add(flow_col)
            if volume_col is not None:
                columns.add(volume_col)
        return columns

    def evaluate(self, view: _SampleView, index: int, time: float, times):
        if index == 0:
            raise LookupError("initial sample has no preceding interval")
        dt = time - float(times[index - 1])
        net = 0.0
        for volume_col, flow_col, is_inflow in self.links:
            if volume_col is not None and view.has(volume_col):
                quantity = view.number(volume_col, index)
            else:
                quantity = view.number(flow_col, index - 1) * dt
            net += quantity if is_inflow else -quantity
        value = residuals.tank_volume_residual(
            level_start=view.number(self.level_col, index - 1),
            level_end=view.number(self.level_col, index),
            net_inflow=net, dt_s=1.0, diameter=self.diameter,
            min_volume=self.min_volume, min_level=self.min_level,
            volume_curve=self.volume_curve,
        )
        return value, "interval", self.unit


class _PumpPlan(_Plan):
    """Head-gain residual of one pump."""

    __slots__ = ("status_col", "flow_col", "speed_col", "curve_points",
                 "head_start", "head_end", "backend")

    family = "pump_head"

    def needed_columns(self) -> set[str]:
        columns = {self.status_col, self.flow_col, self.speed_col}
        columns.update((self.head_start.column_name(), self.head_end.column_name()))
        return columns

    def evaluate(self, view: _SampleView, index: int, time: float, times):
        status = TopologyStlDetector._status(view.raw(self.status_col, index))
        flow = view.number(self.flow_col, index)
        speed = view.number(self.speed_col, index) if status == "OPEN" else None
        dispatch = residuals.pump_dispatch(
            flow=flow, status=status, speed=speed, curve_points=self.curve_points,
            head_start=self.head_start.value(view, index, time) if status == "OPEN" else None,
            head_end=self.head_end.value(view, index, time) if status == "OPEN" else None,
            backend=self.backend,
        )
        if dispatch.residual is None:
            raise KeyError(dispatch.reason)
        return dispatch.residual, dispatch.regime, dispatch.unit


class _ValvePlan(_Plan):
    """Dispatch residual of one valve."""

    __slots__ = ("status_col", "flow_col", "setting_col", "kind", "setting",
                 "diameter", "minor_loss", "headloss_curve", "head_start", "head_end",
                 "elevation_start", "elevation_end", "elevation_start_error",
                 "elevation_end_error")

    family = "valve_dispatch"

    def needed_columns(self) -> set[str]:
        columns = {self.status_col, self.flow_col, self.setting_col}
        columns.update((self.head_start.column_name(), self.head_end.column_name()))
        return columns

    def evaluate(self, view: _SampleView, index: int, time: float, times):
        status = TopologyStlDetector._status(view.raw(self.status_col, index))
        flow = view.number(self.flow_col, index)
        kind = self.kind
        setting = (
            view.number(self.setting_col, index)
            if kind != "GPV" and status == "ACTIVE"
            else self.setting
        )
        needs_head = status == "OPEN" or (status == "ACTIVE" and kind in ("PRV", "PSV", "TCV", "GPV"))
        head_start = self.head_start.value(view, index, time) if needs_head else None
        head_end = self.head_end.value(view, index, time) if needs_head else None
        active_pressure_valve = status == "ACTIVE" and kind in ("PRV", "PSV")
        if active_pressure_valve:
            if self.elevation_start_error is not None:
                raise self.elevation_start_error
            if self.elevation_end_error is not None:
                raise self.elevation_end_error
            elevation_start = self.elevation_start
            elevation_end = self.elevation_end
        else:
            elevation_start = 0.0
            elevation_end = 0.0
        dispatch = residuals.valve_dispatch(
            valve_type=kind, flow=flow, setting=setting, diameter=self.diameter,
            minor_loss=self.minor_loss, headloss_curve=self.headloss_curve,
            head_start=head_start, head_end=head_end,
            elevation_start=elevation_start, elevation_end=elevation_end,
            status=status,
        )
        if dispatch.residual is None:
            raise KeyError(dispatch.reason)
        return dispatch.residual, dispatch.regime, dispatch.unit


def _head_of(heads: Mapping[str, _Head], node_id: str) -> _Head:
    """Return the head plan of *node_id* (an unresolvable node raises)."""
    head = heads.get(node_id)
    if head is None:
        head = _Head("error", error=StopIteration())
        heads[node_id] = head
    return head


class TopologyStlDetector:
    name = "topology_stl"

    def __init__(self, network, config: Mapping[str, Any] | None = None):
        self.network = network
        self.config = dict(config or {})
        self.percentile = float(self.config.get("percentile", 99.0))
        raw = self.config.get("additive", 0.0)
        if isinstance(raw, Mapping):
            self.additives = {str(k): float(v) for k, v in raw.items()}
        else:
            scalar = float(raw)
            if scalar != 0:
                raise ValueError("nonzero additive margins must be configured by residual unit")
            self.additives = {"*": scalar}
        if not math.isfinite(self.percentile) or not 0 <= self.percentile <= 100:
            raise ValueError("percentile must be finite and in [0,100]")
        if any(not math.isfinite(v) or v < 0 for v in self.additives.values()):
            raise ValueError("additive margins must be finite and nonnegative")
        self.closed_flow_epsilon = float(self.config.get("closed_flow_epsilon", 1e-6))
        if not math.isfinite(self.closed_flow_epsilon) or self.closed_flow_epsilon < 0:
            raise ValueError("closed_flow_epsilon must be finite and nonnegative")
        self.unknown_policy = str(self.config.get("unknown_policy", "raise"))
        if self.unknown_policy not in ("raise", "report"):
            raise ValueError("unknown_policy must be 'raise' or 'report'")
        self.last_unknown: list[dict[str, Any]] = []
        self.templates = ()
        self.formulas: dict[str, FormulaResult] = {}
        self.tolerances: dict[str, Tolerance] = {}
        self._fitted = False
        self._plans: tuple[_Plan, ...] = ()
        self._needed_columns: frozenset[str] = frozenset()
        self._heads: dict[str, _Head] = {}
        self._times_current = None
        self._previous_row = None

    # -- static helpers ---------------------------------------------------
    @staticmethod
    def _times(frame):
        values = frame["time_seconds"].to_numpy() if "time_seconds" in frame else frame.index.to_numpy()
        try: times = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc: raise ValueError("time must be numeric seconds") from exc
        if not np.isfinite(times).all() or (len(times) > 1 and np.any(np.diff(times) <= 0)):
            raise ValueError("time must be finite and strictly increasing seconds")
        return times

    @staticmethod
    def _number(row, col):
        if col not in row.index: raise KeyError(f"missing channel {col}")
        value = float(row[col])
        if not math.isfinite(value): raise ValueError(f"non-finite channel {col}")
        return value

    @staticmethod
    def _status(value):
        names = {0:"CLOSED", 1:"OPEN", 2:"ACTIVE"}
        if isinstance(value, str):
            text=value.strip().upper()
            if text in names.values(): return text
            try: value=float(text)
            except ValueError: raise ValueError(f"invalid status {value!r}")
        try: number=float(value)
        except (TypeError, ValueError): raise ValueError(f"invalid status {value!r}")
        if not math.isfinite(number) or number not in names: raise ValueError(f"invalid status code {value!r}")
        return names[int(number)]

    # -- compilation ------------------------------------------------------
    def _compile(self, frames):
        cols = set().union(*(set(f.columns) for f in frames)) if frames else set()
        available = [ChannelSpec(k, e).key for k in ("flow", "pressure", "level", "status", "speed", "setting")
                     for e in self.network.node_ids() + self.network.link_ids() if channel_column(ChannelSpec(k, e)) in cols]
        # Actuator requirements depend on the observed state.  Defer these
        # channels to runtime so a closed unit need not carry open-state inputs.
        available.extend(ChannelSpec("speed", p.id).key for p in self.network.pumps)
        available.extend(ChannelSpec("setting", v.id).key for v in self.network.valves)
        node_ids=set(self.network.node_ids())
        available.extend(ChannelSpec("pressure", n).key for n in node_ids)
        available.extend(ChannelSpec("level", n).key for n in node_ids)
        self.templates = compile_templates(self.network, available)
        units = {"junction_continuity":"m^3/s", "pipe_energy":"m", "tank_volume":"m^3", "pump_head":"m", "valve_dispatch":"m"}
        self.formulas = {f"{t.family.value}:{t.key}": FormulaResult(t.key,t.family.value,tuple(s.key for s in t.required_channels),units[t.family.value],t.status.value,t.reason) for t in self.templates}
        self._plans, self._needed_columns = self._build_plans(self.templates, units)

    # -- plan construction ------------------------------------------------
    def _build_plans(self, templates, units):
        """Resolve every static fact about the templates once.

        Returns the compiled plans and the set of columns the plans may read.
        """
        network = self.network
        links_by_id = {}
        for link in network.links():
            links_by_id.setdefault(link.id, link)
        groups = {
            "junction_continuity": network.junctions,
            "pipe_energy": network.pipes,
            "tank_volume": network.tanks,
            "pump_head": network.pumps,
            "valve_dispatch": network.valves,
        }
        elements = {}
        for family, group in groups.items():
            table = {}
            for element in group:
                table.setdefault(element.id, element)
            elements[family] = table
        incident = {}
        for family, group in groups.items():
            for element in group:
                incident[(family, element.id)] = tuple(
                    (link_id, links_by_id[link_id])
                    for link_id in network.incident_links(element.id)
                    if link_id in links_by_id
                )
        heads = {node_id: self._head_plan(node_id) for node_id in network.node_ids()}
        self._heads = heads

        plans = []
        needed: set[str] = set()
        for template in templates:
            family = template.family.value
            key = f"{family}:{template.key}"
            base = asdict(self.formulas[key])
            unit = units[family]
            if template.status is not TemplateStatus.COMPILED:
                plans.append(_Plan(key, base, template.status, template.reason, unit))
                continue
            element = elements[family].get(template.key)
            if element is None:
                plan = _Plan(key, base, template.status, template.reason, unit)
                plan.element_missing = True
                plans.append(plan)
                continue
            if family == "junction_continuity":
                plan = self._junction_plan(template, base, element, incident[(family, element.id)], unit)
            elif family == "pipe_energy":
                plan = self._pipe_plan(template, base, element, heads, unit)
            elif family == "tank_volume":
                plan = self._tank_plan(template, base, element, incident[(family, element.id)], unit)
            elif family == "pump_head":
                plan = self._pump_plan(template, base, element, heads, unit)
            else:
                plan = self._valve_plan(template, base, element, heads, unit)
            plans.append(plan)
            needed.update(plan.needed_columns())
        return tuple(plans), frozenset(needed)

    def _head_plan(self, node_id: str) -> _Head:
        """Return how the head of *node_id* is obtained."""
        for reservoir in self.network.reservoirs:
            if reservoir.id == node_id:
                pattern = None
                error = None
                pattern_id = reservoir.head_pattern_id
                try:
                    pattern = self.network.pattern(pattern_id)
                except Exception as exc:  # unresolved pattern: raise it per sample
                    error = exc
                return _Head("reservoir", base_head=reservoir.base_head,
                             pattern=pattern, pattern_id=pattern_id, error=error)
        for tank in self.network.tanks:
            if tank.id == node_id:
                return _Head("tank", elevation=tank.elevation,
                             column=channel_column(ChannelSpec("level", node_id)))
        for junction in self.network.junctions:
            if junction.id == node_id:
                return _Head("junction", elevation=junction.elevation,
                             column=channel_column(ChannelSpec("pressure", node_id)))
        return _Head("error", error=StopIteration())

    def _junction_plan(self, template, base, element, incident, unit) -> _JunctionPlan:
        flow_cols = []
        for link_id, link in incident:
            flow_cols.append((channel_column(ChannelSpec("flow", link_id)),
                              link.start_node == element.id))
        terms = []
        for demand in element.demands:
            pattern = None
            error = None
            pattern_id = demand.pattern_id
            try:
                pattern = self.network.pattern(pattern_id)
            except Exception as exc:
                error = exc
            terms.append((demand.base_demand, pattern_id, pattern, error))
        demand = self.network.demand
        plan = _JunctionPlan(
            f"junction_continuity:{element.id}", base, template.status, template.reason, unit,
        )
        plan.flow_cols = tuple(flow_cols)
        plan.terms = tuple(terms)
        plan.demand_col = f"demand_{element.id}"
        plan.leak_col = f"leak_{element.id}"
        plan.emitter_col = f"emitter_{element.id}"
        plan.pressure_col = element.id
        plan.is_pdd = demand.model.upper() in ("PDD", "PDA")
        plan.demand_multiplier = demand.multiplier
        plan.demand_model = demand.model
        plan.pressure_exponent = element.pressure_exponent
        plan.required_pressure = element.required_pressure
        plan.minimum_pressure = element.minimum_pressure
        plan.emitter_coefficient = element.emitter_coefficient
        plan.emitter_exponent = element.emitter_exponent
        return plan

    def _pipe_plan(self, template, base, element, heads, unit) -> _PipePlan:
        plan = _PipePlan(f"pipe_energy:{element.id}", base, template.status, template.reason, unit)
        plan.flow_col = channel_column(ChannelSpec("flow", element.id))
        plan.status_col = channel_column(ChannelSpec("status", element.id))
        plan.length = element.length
        plan.diameter = element.diameter
        plan.roughness = element.roughness
        plan.minor_loss = element.minor_loss
        plan.head_start = _head_of(heads, element.start_node)
        plan.head_end = _head_of(heads, element.end_node)
        return plan

    def _tank_plan(self, template, base, element, incident, unit) -> _TankPlan:
        links = []
        for link_id, link in incident:
            links.append((f"volume_{link_id}",
                          channel_column(ChannelSpec("flow", link_id)),
                          link.end_node == element.id))
        plan = _TankPlan(f"tank_volume:{element.id}", base, template.status, template.reason, unit)
        plan.level_col = element.id
        plan.links = tuple(links)
        plan.diameter = element.diameter
        plan.min_volume = element.min_volume
        plan.min_level = element.min_level
        plan.volume_curve = element.volume_curve
        return plan

    def _pump_plan(self, template, base, element, heads, unit) -> _PumpPlan:
        plan = _PumpPlan(f"pump_head:{element.id}", base, template.status, template.reason, unit)
        plan.status_col = channel_column(ChannelSpec("status", element.id))
        plan.flow_col = channel_column(ChannelSpec("flow", element.id))
        plan.speed_col = channel_column(ChannelSpec("speed", element.id))
        plan.curve_points = element.curve.points if element.curve else None
        plan.head_start = _head_of(heads, element.start_node)
        plan.head_end = _head_of(heads, element.end_node)
        plan.backend = self.network.provenance.simulator_backend
        return plan

    def _valve_plan(self, template, base, element, heads, unit) -> _ValvePlan:
        plan = _ValvePlan(f"valve_dispatch:{element.id}", base, template.status, template.reason, unit)
        plan.status_col = channel_column(ChannelSpec("status", element.id))
        plan.flow_col = channel_column(ChannelSpec("flow", element.id))
        plan.setting_col = channel_column(ChannelSpec("setting", element.id))
        plan.kind = element.valve_type.upper()
        plan.setting = element.setting
        plan.diameter = element.diameter
        plan.minor_loss = element.minor_loss
        plan.headloss_curve = element.headloss_curve
        plan.head_start = _head_of(heads, element.start_node)
        plan.head_end = _head_of(heads, element.end_node)
        elevation_start, elevation_end = 0.0, 0.0
        error_start = error_end = None
        try:
            elevation_start = self.network.node_elevation(element.start_node)
        except Exception as exc:
            error_start = exc
        try:
            elevation_end = self.network.node_elevation(element.end_node)
        except Exception as exc:
            error_end = exc
        plan.elevation_start = elevation_start
        plan.elevation_end = elevation_end
        plan.elevation_start_error = error_start
        plan.elevation_end_error = error_end
        return plan

    # -- evaluation -------------------------------------------------------
    def _view(self, frame) -> _FrameView:
        return _FrameView(frame, sorted(self._needed_columns))

    def _evaluate_view(self, view: _SampleView, index: int, time: float, times):
        out = {}
        for plan in self._plans:
            base = plan.base
            try:
                if plan.status is not TemplateStatus.COMPILED:
                    if "missing channel" in plan.reason: raise KeyError(plan.reason)
                    out[plan.key]={**base,"residual":None,"epsilon":None,"robustness":None,"regime":plan.status.value,"reason":plan.reason}; continue
                val, regime, unit = plan.evaluate(view, index, time, times)
                val=float(val)
                if not math.isfinite(val): raise ValueError("computed residual is non-finite")
            except KeyError as e: out[plan.key]={**base,"residual":None,"epsilon":None,"robustness":None,"regime":"missing","reason":str(e)}
            except LookupError as e: out[plan.key]={**base,"residual":None,"epsilon":None,"robustness":None,"regime":"pending","reason":str(e)}
            except Exception as e: out[plan.key]={**base,"residual":None,"epsilon":None,"robustness":None,"regime":"error","reason":f"{type(e).__name__}: {e}"}
            else:
                tk=plan.tolerance_key(regime,unit); tol=self.tolerances.get(tk)
                if tol is None and regime=="closed": tol=Tolerance(self.closed_flow_epsilon,unit,self.percentile,0.0)
                out[plan.key]={**base,"unit":unit,"residual":val,"epsilon":tol.epsilon if tol else None,"robustness":None if tol is None else tol.epsilon-abs(val),"regime":regime if tol else "uncalibrated","operating_regime":regime,"reason":"" if tol else f"uncalibrated regime/unit: {regime}/{unit}"}
        return out

    def _head(self, node, row, time):
        """Return the head (m) of *node* for the sample *row* (legacy helper)."""
        head = self._heads.get(node)
        if head is None:
            head = self._head_plan(node)
            self._heads[node] = head
        return head.value(_SeriesView(row), 0, time)

    def _evaluate(self, row, time, idx):
        """Evaluate every template for one sample given as a ``pandas.Series``."""
        return self._evaluate_view(_SeriesView(row), idx, time, self._times_current)

    # -- public API -------------------------------------------------------
    @staticmethod
    def _tolerance_key(key, regime, unit): return f"{key}|{regime}|{unit}"

    def fit(self, nominal_training_frames):
        frames=list(nominal_training_frames)
        if not frames or not any(len(f) for f in frames): raise ValueError("fit requires nonempty nominal frames")
        self._compile(frames); samples={}; self.tolerances={}
        for frame in frames:
            if not len(frame): continue
            times=self._times(frame); self._times_current=times; self._previous_row=None
            view=self._view(frame)
            for i in range(len(frame)):
                view.new_sample(i)
                for key,d in self._evaluate_view(view,i,float(times[i]),times).items():
                    if d["regime"] in ("error", "missing"): raise ValueError(f"invalid calibration sample for {key}: {d['reason']}")
                    if d["residual"] is not None: samples.setdefault(self._tolerance_key(key,d.get("operating_regime",d["regime"]),d["unit"]),[]).append(abs(d["residual"]))
        for tk,values in samples.items():
            key,regime,unit=tk.rsplit("|",2); margin=self.additives.get(unit,self.additives.get("*",0.))
            epsilon=float(np.percentile(values,self.percentile))+margin
            if regime=="closed": epsilon=max(epsilon,self.closed_flow_epsilon)
            if not math.isfinite(epsilon): raise ValueError(f"non-finite tolerance for {tk}")
            self.tolerances[tk]=Tolerance(epsilon,unit,self.percentile,margin)
        if not self.tolerances: raise ValueError("fit produced no usable residuals")
        self._fitted=True; self._previous_row=None
        return self

    def predict_details(self, frame):
        if not self._fitted: raise RuntimeError("fit must be called before predict")
        times=self._times(frame); self._times_current=times; rows=[]; self._previous_row=None
        view=self._view(frame)
        for i in range(len(frame)):
            view.new_sample(i)
            rows.append(self._evaluate_view(view,i,float(times[i]),times))
        return rows

    def predict(self, frame):
        details=self.predict_details(frame); scores=[]
        self.last_unknown = []
        for row in details:
            unknown=[d for d in row.values() if d["robustness"] is None and d["regime"] in ("missing","error","uncalibrated")]
            if unknown and (self.unknown_policy == "raise" or any(d["regime"] in ("missing","error") for d in unknown)):
                raise ValueError("prediction contains unknown residuals: "+"; ".join(d["reason"] for d in unknown))
            self.last_unknown.extend(unknown)
            scores.append(sum(d["robustness"] is not None and d["robustness"] < 0 for d in row.values()))
        return StepFlags(flags=np.asarray(scores,dtype=bool),scores=np.asarray(scores,dtype=float))

    def to_artifact(self):
        coverage={k:sum(1 for tol in self.tolerances if tol.startswith(k+"|")) for k in sorted(self.formulas)}
        return {"schema":"topology-stl-v2","percentile":self.percentile,"additive":dict(sorted(self.additives.items())),"formulas":{k:asdict(v) for k,v in sorted(self.formulas.items())},"tolerances":{k:asdict(v) for k,v in sorted(self.tolerances.items())},"coverage":coverage}

    def to_json(self): return json.dumps(self.to_artifact(),sort_keys=True,separators=(",",":"))
