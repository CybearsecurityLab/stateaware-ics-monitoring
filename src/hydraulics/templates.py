"""The five hydraulic template families and their deterministic compilation.

A *template* is one physics-based check over one network element (a junction,
a pipe, a tank, a pump or a valve).  Compiling a template answers three
questions:

* which observable channels does it need (:class:`ChannelSpec`),
* can the physics be evaluated at all (``unsupported`` + reason), and
* are the needed channels present in the dataset (``unobservable`` + reason).

Compilation never silently drops anything: every candidate template is
returned, carrying either ``compiled`` or an explicit status with a reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple

from .model import ChannelSpec, HydraulicNetwork, unique_channel_specs


class TemplateFamily(str, Enum):
    """The five hydraulic template families."""

    JUNCTION_CONTINUITY = "junction_continuity"
    PIPE_ENERGY = "pipe_energy"
    TANK_VOLUME = "tank_volume"
    PUMP_HEAD = "pump_head"
    VALVE_DISPATCH = "valve_dispatch"


class TemplateStatus(str, Enum):
    """Outcome of compiling one template."""

    COMPILED = "compiled"
    UNSUPPORTED = "unsupported"
    UNOBSERVABLE = "unobservable"


@dataclass(frozen=True)
class CompiledTemplate:
    """One compiled template (or an explicit refusal to compile it)."""

    family: TemplateFamily
    key: str
    required_channels: Tuple[ChannelSpec, ...]
    status: TemplateStatus
    reason: str = ""
    expression: str = ""

    def __post_init__(self) -> None:
        if self.status is not TemplateStatus.COMPILED and not self.reason:
            raise ValueError(
                f"template {self.family.value}:{self.key} is {self.status.value} "
                "but carries no reason"
            )

    @property
    def missing_channels(self) -> Tuple[ChannelSpec, ...]:
        """Placeholder kept for symmetry; use :func:`missing_channels`."""
        return ()


def missing_channels(
    template: CompiledTemplate, available_channels: Iterable[str]
) -> Tuple[ChannelSpec, ...]:
    """Return the required channels of *template* absent from *available_channels*."""
    available = set(available_channels)
    return tuple(spec for spec in template.required_channels if spec.key not in available)


def compile_templates(
    network: HydraulicNetwork, available_channels: Iterable[str]
) -> Tuple[CompiledTemplate, ...]:
    """Compile every candidate template of *network*.

    *available_channels* is the collection of available channel keys (see
    :attr:`hydraulics.model.ChannelSpec.key`).  The result is deterministic:
    families are returned in :class:`TemplateFamily` order and, inside a
    family, by ascending element id.
    """
    available = set(available_channels)
    compiled: list[CompiledTemplate] = []
    compiled.extend(_junction_continuity_templates(network, available))
    compiled.extend(_pipe_energy_templates(network, available))
    compiled.extend(_tank_volume_templates(network, available))
    compiled.extend(_pump_head_templates(network, available))
    compiled.extend(_valve_dispatch_templates(network, available))
    return tuple(compiled)


def _head_channels(network: HydraulicNetwork, node_ids: Iterable[str]) -> list[ChannelSpec]:
    """Return the pressure channels needed for *node_ids*.

    A reservoir head is known from the network contract (its base head and
    head pattern), so it never needs an observed pressure channel.
    """
    reservoir_ids = {reservoir.id for reservoir in network.reservoirs}
    tank_ids = {tank.id for tank in network.tanks}
    return [
        ChannelSpec("level" if node_id in tank_ids else "pressure", node_id)
        for node_id in node_ids
        if node_id not in reservoir_ids
    ]


def _finalise(
    family: TemplateFamily,
    key: str,
    required: Iterable[ChannelSpec],
    available: set[str],
    reason: str = "",
    expression: str = "",
) -> CompiledTemplate:
    """Build a template, applying the unsupported/unobservable precedence."""
    specs = unique_channel_specs(required)
    if reason:
        return CompiledTemplate(
            family=family,
            key=key,
            required_channels=specs,
            status=TemplateStatus.UNSUPPORTED,
            reason=reason,
        )
    missing = tuple(spec for spec in specs if spec.key not in available)
    if missing:
        return CompiledTemplate(
            family=family,
            key=key,
            required_channels=specs,
            status=TemplateStatus.UNOBSERVABLE,
            reason="missing channel(s): " + ", ".join(spec.key for spec in missing),
        )
    return CompiledTemplate(
        family=family,
        key=key,
        required_channels=specs,
        status=TemplateStatus.COMPILED,
        expression=expression,
    )


def _junction_continuity_templates(
    network: HydraulicNetwork, available: set[str]
) -> list[CompiledTemplate]:
    out = []
    for junction in sorted(network.junctions, key=lambda item: item.id):
        required = [ChannelSpec("pressure", junction.id)]
        required.extend(
            ChannelSpec("flow", link_id) for link_id in network.incident_links(junction.id)
        )
        out.append(
            _finalise(
                TemplateFamily.JUNCTION_CONTINUITY,
                junction.id,
                required,
                available,
                expression=(
                    "sum(flow_in) - sum(flow_out) - delivered_demand - emitter_withdrawal"
                ),
            )
        )
    return out


def _pipe_energy_templates(
    network: HydraulicNetwork, available: set[str]
) -> list[CompiledTemplate]:
    out = []
    headloss = network.provenance.headloss_formula.upper()
    for pipe in sorted(network.pipes, key=lambda item: item.id):
        required = _head_channels(network, (pipe.start_node, pipe.end_node))
        required.extend((ChannelSpec("flow", pipe.id), ChannelSpec("status", pipe.id)))
        reason = ""
        if headloss not in ("", "H-W"):
            reason = (
                f"head loss formula {headloss!r} is not supported; pipe energy "
                "templates require Hazen-Williams"
            )
        elif pipe.diameter <= 0.0 or pipe.roughness <= 0.0:
            reason = (
                f"pipe {pipe.id!r} has non-positive geometry "
                f"(diameter={pipe.diameter}, roughness={pipe.roughness})"
            )
        out.append(
            _finalise(
                TemplateFamily.PIPE_ENERGY,
                pipe.id,
                required,
                available,
                reason=reason,
                expression="(head_start - head_end) - hazen_williams_headloss(flow)",
            )
        )
    return out


def _tank_volume_templates(
    network: HydraulicNetwork, available: set[str]
) -> list[CompiledTemplate]:
    out = []
    for tank in sorted(network.tanks, key=lambda item: item.id):
        required = [ChannelSpec("level", tank.id)]
        required.extend(
            ChannelSpec("flow", link_id) for link_id in network.incident_links(tank.id)
        )
        reason = ""
        if tank.volume_curve is None and tank.diameter <= 0.0:
            reason = (
                f"tank {tank.id!r} has neither a volume curve nor a positive diameter; "
                "its volume cannot be evaluated"
            )
        elif tank.volume_curve is not None and len(tank.volume_curve) < 2:
            reason = f"tank {tank.id!r} has a volume curve with fewer than two points"
        out.append(
            _finalise(
                TemplateFamily.TANK_VOLUME,
                tank.id,
                required,
                available,
                reason=reason,
                expression="volume(level_end) - volume(level_start) - net_inflow * dt",
            )
        )
    return out


def _pump_head_templates(
    network: HydraulicNetwork, available: set[str]
) -> list[CompiledTemplate]:
    out = []
    for pump in sorted(network.pumps, key=lambda item: item.id):
        required = [ChannelSpec("flow", pump.id), ChannelSpec("status", pump.id), ChannelSpec("speed", pump.id)]
        required.extend(_head_channels(network, (pump.start_node, pump.end_node)))
        reason = ""
        if pump.pump_type.upper() != "HEAD":
            reason = (
                f"pump {pump.id!r} is a constant-power pump; power/flow affinity is "
                "not defined by the head-curve model and is explicitly unsupported"
            )
        elif pump.curve is None or not pump.curve.points:
            reason = f"pump {pump.id!r} has no usable head curve"
        out.append(
            _finalise(
                TemplateFamily.PUMP_HEAD,
                pump.id,
                required,
                available,
                reason=reason,
                expression="(head_end - head_start) - pump_head(flow, speed)",
            )
        )
    return out


def _valve_dispatch_templates(
    network: HydraulicNetwork, available: set[str]
) -> list[CompiledTemplate]:
    out = []
    supported_types = ("PRV", "PSV", "PBV", "FCV", "TCV", "GPV")
    for valve in sorted(network.valves, key=lambda item: item.id):
        required = [ChannelSpec("flow", valve.id), ChannelSpec("status", valve.id)]
        if valve.valve_type.upper() != "GPV":
            required.append(ChannelSpec("setting", valve.id))
        required.extend(_head_channels(network, (valve.start_node, valve.end_node)))
        reason = ""
        valve_type = valve.valve_type.upper()
        if valve_type not in supported_types:
            reason = f"valve {valve.id!r} has unsupported valve type {valve_type!r}"
        elif valve_type == "PBV":
            reason = f"pressure breaker valve {valve.id!r} has an unsupported active branch"
        elif valve_type == "GPV":
            if not valve.headloss_curve or len(valve.headloss_curve) < 2:
                reason = (
                    f"general purpose valve {valve.id!r} has no head loss curve; "
                    "its dispatch cannot be evaluated"
                )
        out.append(
            _finalise(
                TemplateFamily.VALVE_DISPATCH,
                valve.id,
                required,
                available,
                reason=reason,
                expression="observed_dispatch - expected_dispatch",
            )
        )
    return out
