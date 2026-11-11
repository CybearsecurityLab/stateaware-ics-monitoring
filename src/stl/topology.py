"""Topology-driven binding between the hydraulic contract and STL channels.

The Stage-3 STL monitor works on a flat table of channels (one column per
observable).  This module binds the typed network contract of
:mod:`hydraulics.model` onto those columns:

* which column carries the flow of a link, the pressure of a node, the level
  of a tank or the status of an actuator,
* which of those columns are actually present in a dataset, and
* how the sampling interval relates to the STL horizon.

The sampling interval and the STL horizon are deliberately separate: the
interval is the spacing of the samples, the horizon is the window a formula is
evaluated over.  A horizon that is not an integer multiple of the interval is
rejected instead of being rounded silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hydraulics.model import ChannelSpec, HydraulicNetwork
from hydraulics.templates import CompiledTemplate, TemplateStatus, compile_templates


def flow_column(element_id: str) -> str:
    """Column carrying the flow of a link."""
    return f"f_{element_id}"


def pressure_column(element_id: str) -> str:
    """Column carrying the pressure head of a node."""
    return element_id


def level_column(element_id: str) -> str:
    """Column carrying the level of a tank."""
    return element_id


def status_column(element_id: str) -> str:
    """Column carrying the status of an actuator."""
    return f"s_{element_id}"


def channel_column(spec: ChannelSpec) -> str:
    """Return the dataset column that carries *spec*."""
    if spec.kind == "flow":
        return flow_column(spec.element_id)
    if spec.kind == "pressure":
        return pressure_column(spec.element_id)
    if spec.kind == "level":
        return level_column(spec.element_id)
    if spec.kind == "status":
        return status_column(spec.element_id)
    if spec.kind == "speed":
        return f"speed_{spec.element_id}"
    if spec.kind == "setting":
        return f"setting_{spec.element_id}"
    raise ValueError(f"unknown channel kind {spec.kind!r}")


@dataclass(frozen=True)
class TimingPlan:
    """Sampling interval and STL horizon of a monitoring run.

    ``sample_interval_s`` is the spacing of the samples; ``stl_horizon_s`` is
    the window a formula is evaluated over.  They are independent quantities,
    but the horizon must cover a whole number of samples.
    """

    sample_interval_s: float
    stl_horizon_s: float

    def __post_init__(self) -> None:
        if self.sample_interval_s <= 0.0:
            raise ValueError(
                f"the sampling interval must be positive, got {self.sample_interval_s}"
            )
        if self.stl_horizon_s <= 0.0:
            raise ValueError(f"the STL horizon must be positive, got {self.stl_horizon_s}")
        ratio = self.stl_horizon_s / self.sample_interval_s
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(
                f"the STL horizon ({self.stl_horizon_s} s) must be an integer "
                f"multiple of the sampling interval ({self.sample_interval_s} s)"
            )

    @property
    def horizon_samples(self) -> int:
        """Number of sample intervals spanned by the STL horizon."""
        return int(round(self.stl_horizon_s / self.sample_interval_s))

    def horizon_is_complete(self, index: int) -> bool:
        """Whether the horizon ending at sample *index* is complete."""
        return index >= self.horizon_samples


@dataclass(frozen=True)
class ElementChannels:
    """The channels bound to one network element."""

    element_id: str
    columns: Mapping[str, str]

    def get(self, kind: str) -> Optional[str]:
        """Return the column bound to *kind*, or ``None``."""
        return self.columns.get(kind)


@dataclass(frozen=True)
class TopologyBinding:
    """The channels of a network that a dataset provides."""

    name: str
    tanks: Mapping[str, ElementChannels]
    pumps: Mapping[str, ElementChannels]
    valves: Mapping[str, ElementChannels]
    junctions: Mapping[str, ElementChannels]
    links: Mapping[str, ElementChannels]
    missing: Tuple[str, ...]
    templates: Tuple[CompiledTemplate, ...]

    def available_channel_keys(self) -> Tuple[str, ...]:
        """Return the channel keys that the dataset provides."""
        keys = []
        for group in (self.tanks, self.pumps, self.valves, self.junctions, self.links):
            for element in group.values():
                keys.extend(
                    f"{kind}:{element.element_id}" for kind in element.columns
                )
        return tuple(sorted(set(keys)))

    def compiled_templates(self) -> Tuple[CompiledTemplate, ...]:
        """Return the templates that could be compiled for this dataset."""
        return tuple(
            template
            for template in self.templates
            if template.status is TemplateStatus.COMPILED
        )


def build_binding(
    network: HydraulicNetwork,
    columns: Iterable[str],
    name: str = "",
) -> TopologyBinding:
    """Bind *network* onto the dataset columns in *columns*.

    Every element gets the channels it can have; the columns that the dataset
    does not provide are reported in :attr:`TopologyBinding.missing` instead of
    being silently ignored.
    """
    present = set(columns)

    def bind(element_id: str, kinds: Sequence[str]) -> Tuple[Dict[str, str], List[str]]:
        bound: Dict[str, str] = {}
        absent: List[str] = []
        for kind in kinds:
            column = channel_column(ChannelSpec(kind, element_id))
            if column in present:
                bound[kind] = column
            else:
                absent.append(f"{kind}:{element_id}")
        return bound, absent

    missing: List[str] = []
    tanks: Dict[str, ElementChannels] = {}
    for tank in network.tanks:
        bound, absent = bind(tank.id, ("level",))
        missing.extend(absent)
        tanks[tank.id] = ElementChannels(tank.id, bound)

    pumps: Dict[str, ElementChannels] = {}
    for pump in network.pumps:
        bound, absent = bind(pump.id, ("flow", "status", "speed"))
        missing.extend(absent)
        pumps[pump.id] = ElementChannels(pump.id, bound)

    valves: Dict[str, ElementChannels] = {}
    for valve in network.valves:
        bound, absent = bind(valve.id, ("flow", "status") if valve.valve_type.upper() == "GPV" else ("flow", "status", "setting"))
        missing.extend(absent)
        valves[valve.id] = ElementChannels(valve.id, bound)

    junctions: Dict[str, ElementChannels] = {}
    for junction in network.junctions:
        bound, absent = bind(junction.id, ("pressure",))
        missing.extend(absent)
        junctions[junction.id] = ElementChannels(junction.id, bound)

    links: Dict[str, ElementChannels] = {}
    for link in network.links():
        bound, absent = bind(link.id, ("flow", "status"))
        missing.extend(absent)
        links[link.id] = ElementChannels(link.id, bound)

    binding = TopologyBinding(
        name=name or network.provenance.inp_filename,
        tanks=tanks,
        pumps=pumps,
        valves=valves,
        junctions=junctions,
        links=links,
        missing=tuple(sorted(set(missing))),
        templates=(),
    )
    templates = compile_templates(network, binding.available_channel_keys())
    return TopologyBinding(
        name=binding.name,
        tanks=binding.tanks,
        pumps=binding.pumps,
        valves=binding.valves,
        junctions=binding.junctions,
        links=binding.links,
        missing=binding.missing,
        templates=templates,
    )
