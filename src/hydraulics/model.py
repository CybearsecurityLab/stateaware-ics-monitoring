"""Typed hydraulic-network contract (SI units) for topology-driven monitoring.

Every quantity in this module is SI: metres (m), seconds (s), cubic metres per
second (m^3/s) and cubic metres (m^3).  Pressures are *pressure heads* in
metres, never gauge values in other units.

The contract is deliberately frozen and free of any dataset/channel concern:
it describes the water network only.  Channel binding (which dataset column
carries which quantity) lives in :mod:`stl.topology`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple


class HydraulicsError(Exception):
    """Base class for every hydraulics failure."""


class InpParseError(HydraulicsError):
    """Raised when an INP file cannot be parsed into the contract."""


class UnsupportedNetworkError(HydraulicsError):
    """Raised when an INP file uses a construct the contract cannot represent."""


#: Channel kinds a template may require.
CHANNEL_KINDS: Tuple[str, ...] = ("flow", "pressure", "level", "status", "speed", "setting")


@dataclass(frozen=True)
class ChannelSpec:
    """A dataset-agnostic request for one observable quantity.

    ``kind`` is one of :data:`CHANNEL_KINDS`; ``element_id`` is the INP node or
    link identifier the quantity belongs to.
    """

    kind: str
    element_id: str

    def __post_init__(self) -> None:
        if self.kind not in CHANNEL_KINDS:
            raise HydraulicsError(
                f"unknown channel kind {self.kind!r}; expected one of {CHANNEL_KINDS}"
            )
        if not self.element_id:
            raise HydraulicsError("channel element_id must not be empty")

    @property
    def key(self) -> str:
        """Stable, sortable identifier of the requested quantity."""
        return f"{self.kind}:{self.element_id}"


@dataclass(frozen=True)
class Pattern:
    """A dimensionless EPANET multiplier pattern."""

    id: str
    multipliers: Tuple[float, ...]
    timestep_s: float = 3600.0
    start_s: float = 0.0

    def multiplier_at(self, time_s: float) -> float:
        """Return the pattern multiplier at *time_s* (seconds).

        EPANET applies one multiplier per pattern timestep, wrapping around at
        the end of the pattern.
        """
        if not self.multipliers:
            return 1.0
        if self.timestep_s <= 0:
            raise HydraulicsError(
                f"pattern {self.id!r} has a non-positive timestep ({self.timestep_s})"
            )
        index = int((time_s + self.start_s) // self.timestep_s)
        return self.multipliers[index % len(self.multipliers)]


@dataclass(frozen=True)
class DemandTerm:
    """One demand category of a junction.

    ``base_demand`` is the SI base demand (m^3/s) of the category and
    ``pattern_id`` the pattern (if any) that scales it.
    """

    category: str
    base_demand: float
    pattern_id: Optional[str] = None


@dataclass(frozen=True)
class Junction:
    """A demand node."""

    id: str
    elevation: float
    demands: Tuple[DemandTerm, ...] = ()
    emitter_coefficient: float = 0.0
    emitter_exponent: float = 0.5
    pressure_exponent: float = 0.5
    required_pressure: float = 0.0
    minimum_pressure: float = 0.0

    @property
    def base_demand(self) -> float:
        """Total SI base demand of all categories (m^3/s)."""
        return sum(term.base_demand for term in self.demands)


@dataclass(frozen=True)
class Reservoir:
    """An infinite source with a (possibly patterned) head."""

    id: str
    base_head: float
    head_pattern_id: Optional[str] = None
    demand: float = 0.0


@dataclass(frozen=True)
class Tank:
    """A finite storage volume.

    ``volume_curve`` holds ``(level, volume)`` pairs in metres and cubic
    metres; when absent the tank is treated as a cylinder of ``diameter``.
    """

    id: str
    elevation: float
    init_level: float
    min_level: float
    max_level: float
    diameter: float
    min_volume: float = 0.0
    volume_curve: Optional[Tuple[Tuple[float, float], ...]] = None


@dataclass(frozen=True)
class Pipe:
    """A pipe (or hose) link."""

    id: str
    start_node: str
    end_node: str
    length: float
    diameter: float
    roughness: float
    minor_loss: float = 0.0


@dataclass(frozen=True)
class PumpCurve:
    """A pump head curve: ``(flow [m^3/s], head [m])`` pairs."""

    id: str
    points: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class Pump:
    """A pump link.

    ``pump_type`` is ``"HEAD"`` for a head curve pump and ``"POWER"`` for a
    constant-power pump (which this contract can carry but not evaluate).
    """

    id: str
    start_node: str
    end_node: str
    pump_type: str
    curve: Optional[PumpCurve] = None
    power: Optional[float] = None
    speed: float = 1.0
    speed_pattern_id: Optional[str] = None
    status: str = "OPEN"


@dataclass(frozen=True)
class Valve:
    """A valve link.

    ``setting`` is interpreted per ``valve_type``: a pressure setting (m) for
    PRV/PSV, a head loss setting (m) for PBV, a flow setting (m^3/s) for FCV
    and a loss coefficient for TCV.  GPV valves carry a ``headloss_curve`` of
    ``(flow [m^3/s], head loss [m])`` pairs instead of a scalar setting.
    """

    id: str
    start_node: str
    end_node: str
    valve_type: str
    setting: Optional[float] = None
    headloss_curve: Optional[Tuple[Tuple[float, float], ...]] = None
    diameter: float = 0.0
    minor_loss: float = 0.0
    status: str = "OPEN"


@dataclass(frozen=True)
class Control:
    """A simple (non-rule) EPANET control.

    ``control_type`` is ``"OPEN"``, ``"CLOSED"`` or ``"SETTING"``; the latter
    carries the commanded ``setting`` (e.g. a PRV set point).
    """

    id: str
    link_id: str
    control_type: str
    setting: Optional[float] = None
    sensor_node: Optional[str] = None
    sensor_attribute: Optional[str] = None
    relation: Optional[str] = None
    threshold: Optional[float] = None
    description: str = ""


@dataclass(frozen=True)
class TimeOptions:
    """Simulation time options, in seconds."""

    duration_s: float
    hydraulic_timestep_s: float
    report_timestep_s: float
    pattern_timestep_s: float
    pattern_start_s: float
    quality_timestep_s: float
    rule_timestep_s: float
    report_start_s: float
    statistic: str = "NONE"


@dataclass(frozen=True)
class DemandOptions:
    """Demand modelling options.

    ``model`` is ``"DDA"`` or ``"PDD"``/``"PDA"`` for pressure-dependent demand.
    """

    model: str
    multiplier: float = 1.0
    pressure_exponent: float = 0.5
    required_pressure: float = 0.0
    minimum_pressure: float = 0.0
    emitter_exponent: float = 0.5


@dataclass(frozen=True)
class Provenance:
    """Input provenance and the intended simulation backend (not the parser)."""

    inp_path: str
    inp_filename: str
    sha256: str
    size_bytes: int
    flow_units: str
    headloss_formula: str
    simulator_backend: str = "wntr"


@dataclass(frozen=True)
class HydraulicNetwork:
    """The full typed network contract.

    Element collections keep the order in which they appear in the INP file so
    that every derived artefact is deterministic.
    """

    provenance: Provenance
    junctions: Tuple[Junction, ...]
    reservoirs: Tuple[Reservoir, ...]
    tanks: Tuple[Tank, ...]
    pipes: Tuple[Pipe, ...]
    pumps: Tuple[Pump, ...]
    valves: Tuple[Valve, ...]
    patterns: Mapping[str, Pattern]
    controls: Tuple[Control, ...]
    time: TimeOptions
    demand: DemandOptions
    notes: Tuple[str, ...] = ()

    # -- lookups -----------------------------------------------------------
    def node_elevation(self, node_id: str) -> float:
        """Return the elevation of *node_id* (m)."""
        for junction in self.junctions:
            if junction.id == node_id:
                return junction.elevation
        for reservoir in self.reservoirs:
            if reservoir.id == node_id:
                return reservoir.elevation
        for tank in self.tanks:
            if tank.id == node_id:
                return tank.elevation
        raise HydraulicsError(f"unknown node {node_id!r}")

    def pattern(self, pattern_id: Optional[str]) -> Optional[Pattern]:
        """Return the pattern with *pattern_id*, or ``None``."""
        if pattern_id is None:
            return None
        try:
            return self.patterns[pattern_id]
        except KeyError as exc:
            raise HydraulicsError(f"unknown pattern {pattern_id!r}") from exc

    def links(self) -> Tuple[Pipe | Pump | Valve, ...]:
        """All links, pipes first, then pumps, then valves (INP order)."""
        return tuple(self.pipes) + tuple(self.pumps) + tuple(self.valves)

    def link_nodes(self, link_id: str) -> Tuple[str, str]:
        """Return ``(start_node, end_node)`` of *link_id*."""
        for link in self.links():
            if link.id == link_id:
                return link.start_node, link.end_node
        raise HydraulicsError(f"unknown link {link_id!r}")

    def incident_links(self, node_id: str) -> Tuple[str, ...]:
        """Return the ids of all links incident to *node_id*, in link order."""
        return tuple(link.id for link in self.links() if node_id in (link.start_node, link.end_node))

    def node_ids(self) -> Tuple[str, ...]:
        """All node ids (junctions, reservoirs, tanks) in contract order."""
        return (
            tuple(j.id for j in self.junctions)
            + tuple(r.id for r in self.reservoirs)
            + tuple(t.id for t in self.tanks)
        )

    def link_ids(self) -> Tuple[str, ...]:
        """All link ids in contract order."""
        return tuple(link.id for link in self.links())


def pattern_multipliers(patterns: Mapping[str, Pattern], pattern_id: Optional[str]) -> Tuple[float, ...]:
    """Return the multipliers of *pattern_id* (empty tuple when absent)."""
    pattern = patterns.get(pattern_id) if pattern_id is not None else None
    return pattern.multipliers if pattern is not None else ()


def sorted_channel_specs(specs: Iterable[ChannelSpec]) -> Tuple[ChannelSpec, ...]:
    """Return *specs* in a deterministic (kind, element_id) order."""
    return tuple(sorted(specs, key=lambda spec: (spec.kind, spec.element_id)))


def unique_channel_specs(specs: Sequence[ChannelSpec]) -> Tuple[ChannelSpec, ...]:
    """Return *specs* de-duplicated, keeping the first occurrence order."""
    seen: set[str] = set()
    out: list[ChannelSpec] = []
    for spec in specs:
        if spec.key in seen:
            continue
        seen.add(spec.key)
        out.append(spec)
    return tuple(out)
