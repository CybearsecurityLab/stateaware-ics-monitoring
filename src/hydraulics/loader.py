"""Load an EPANET INP file into the typed :mod:`hydraulics.model` contract.

WNTR is imported lazily so that the contract (and the residual maths) can be
imported — and unit tested — without WNTR installed.

Unit handling: WNTR works in SI internally (m, s, m^3/s, m^3), so every value
exposed by this module is SI.  Note that an INP file written in metric flow
units stores pipe diameters in **millimetres**; WNTR performs that conversion,
and this module exposes the converted metre value.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .model import (
    Control,
    DemandOptions,
    DemandTerm,
    HydraulicNetwork,
    HydraulicsError,
    InpParseError,
    Junction,
    Pattern,
    Pipe,
    Provenance,
    Pump,
    PumpCurve,
    Reservoir,
    Tank,
    TimeOptions,
    Valve,
)


def load_network(path: str | Path, *, simulator_backend: str = "wntr") -> HydraulicNetwork:
    """Parse the INP file at *path* into a :class:`HydraulicNetwork`.

    ``simulator_backend`` records the intended simulator, either ``wntr`` or
    ``epanet``; WNTR is used to parse the INP in both cases.
    Raises :class:`InpParseError` when the file cannot be parsed and
    :class:`HydraulicsError` when WNTR is unavailable.
    """
    if simulator_backend not in ("wntr", "epanet"):
        raise ValueError(f"unknown simulator backend {simulator_backend!r}")
    inp_path = Path(path)
    if not inp_path.is_file():
        raise InpParseError(f"INP file not found: {inp_path}")
    raw = inp_path.read_bytes()

    try:
        import wntr  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise HydraulicsError(
            "WNTR is required to load an INP file; install the 'hydraulics' extra"
        ) from exc

    try:
        wn = wntr.network.WaterNetworkModel(str(inp_path))
    except Exception as exc:  # pragma: no cover - depends on the INP file
        raise InpParseError(f"failed to parse INP {inp_path}: {exc}") from exc

    provenance = Provenance(
        inp_path=str(inp_path),
        inp_filename=inp_path.name,
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        flow_units=str(getattr(wn.options.hydraulic, "inpfile_units", "") or ""),
        headloss_formula=str(getattr(wn.options.hydraulic, "headloss", "") or ""),
        simulator_backend=simulator_backend,
    )
    return _network_from_wn(wn, provenance)


def _network_from_wn(
    wn: Any, provenance: Provenance
) -> HydraulicNetwork:
    notes: List[str] = []

    patterns: Dict[str, Pattern] = {}
    timestep = float(getattr(wn.options.time, "pattern_timestep", 3600.0) or 3600.0)
    pattern_start = float(getattr(wn.options.time, "pattern_start", 0.0) or 0.0)
    for name in wn.pattern_name_list:
        pattern = wn.get_pattern(name)
        multipliers = tuple(float(value) for value in pattern.multipliers)
        patterns[name] = Pattern(
            id=name,
            multipliers=multipliers,
            timestep_s=timestep,
            start_s=pattern_start,
        )

    junctions = tuple(
        _junction_from_wn(wn.get_node(name), wn) for name in wn.junction_name_list
    )

    reservoirs = []
    for name in wn.reservoir_name_list:
        reservoir = wn.get_node(name)
        reservoirs.append(
            Reservoir(
                id=name,
                base_head=float(reservoir.base_head),
                head_pattern_id=reservoir.head_pattern_name or None,
            )
        )

    tanks = tuple(_tank_from_wn(wn.get_node(name), wn) for name in wn.tank_name_list)

    pipes = tuple(
        Pipe(
            id=name,
            start_node=wn.get_link(name).start_node_name,
            end_node=wn.get_link(name).end_node_name,
            length=float(wn.get_link(name).length),
            diameter=float(wn.get_link(name).diameter),
            roughness=float(wn.get_link(name).roughness),
            minor_loss=float(wn.get_link(name).minor_loss),
        )
        for name in wn.pipe_name_list
    )

    pumps = tuple(_pump_from_wn(wn, name) for name in wn.pump_name_list)
    valves = tuple(_valve_from_wn(wn, name) for name in wn.valve_name_list)

    controls = tuple(_control_from_wn(wn))

    time = TimeOptions(
        duration_s=float(wn.options.time.duration),
        hydraulic_timestep_s=float(wn.options.time.hydraulic_timestep),
        report_timestep_s=float(wn.options.time.report_timestep),
        pattern_timestep_s=timestep,
        pattern_start_s=pattern_start,
        quality_timestep_s=float(wn.options.time.quality_timestep),
        rule_timestep_s=float(wn.options.time.rule_timestep),
        report_start_s=float(wn.options.time.report_start),
        statistic=str(wn.options.time.statistic),
    )

    hyd = wn.options.hydraulic
    demand_options = DemandOptions(
        model=str(getattr(hyd, "demand_model", "DDA") or "DDA"),
        multiplier=float(hyd.demand_multiplier if hyd.demand_multiplier is not None else 1.0),
        pressure_exponent=float(getattr(hyd, "pressure_exponent", 0.5) or 0.5),
        required_pressure=float(getattr(hyd, "required_pressure", 0.0) or 0.0),
        minimum_pressure=float(getattr(hyd, "minimum_pressure", 0.0) or 0.0),
        emitter_exponent=float(getattr(hyd, "emitter_exponent", 0.5) or 0.5),
    )

    return HydraulicNetwork(
        provenance=provenance,
        junctions=junctions,
        reservoirs=tuple(reservoirs),
        tanks=tanks,
        pipes=pipes,
        pumps=pumps,
        valves=valves,
        patterns=patterns,
        controls=controls,
        time=time,
        demand=demand_options,
        notes=tuple(notes),
    )


def _junction_from_wn(junction: Any, wn: Any) -> Junction:
    demands: List[DemandTerm] = []
    for timeseries in junction.demand_timeseries_list:
        category = getattr(timeseries, "category", None) or "base"
        pattern_name = getattr(timeseries, "pattern_name", None)
        if pattern_name in ('""', "''"):
            pattern_name = None
        demands.append(
            DemandTerm(
                category=str(category),
                base_demand=float(timeseries.base_value),
                pattern_id=str(pattern_name) if pattern_name else None,
            )
        )
    return Junction(
        id=junction.name,
        elevation=float(junction.elevation),
        demands=tuple(demands),
        emitter_coefficient=float(getattr(junction, "emitter_coefficient", 0.0) or 0.0),
        emitter_exponent=float(getattr(wn.options.hydraulic, "emitter_exponent", 0.5) or 0.5),
        pressure_exponent=float(getattr(wn.options.hydraulic, "pressure_exponent", 0.5) or 0.5),
        required_pressure=float(getattr(wn.options.hydraulic, "required_pressure", 0.0) or 0.0),
        minimum_pressure=float(getattr(wn.options.hydraulic, "minimum_pressure", 0.0) or 0.0),
    )


def _tank_from_wn(tank: Any, wn: Any) -> Tank:
    curve_points: Optional[Tuple[Tuple[float, float], ...]] = None
    curve_ref = getattr(tank, "vol_curve", None)
    if curve_ref is not None:
        curve_name = getattr(curve_ref, "name", None) or str(curve_ref)
        curve = wn.get_curve(str(curve_name))
        curve_points = tuple((float(x), float(y)) for x, y in curve.points)
    return Tank(
        id=tank.name,
        elevation=float(tank.elevation),
        init_level=float(tank.init_level),
        min_level=float(tank.min_level),
        max_level=float(tank.max_level),
        diameter=float(tank.diameter),
        min_volume=float(tank.min_vol),
        volume_curve=curve_points,
    )


def _pump_from_wn(wn: Any, name: str) -> Pump:
    pump = wn.get_link(name)
    curve: Optional[PumpCurve] = None
    curve_name = getattr(pump, "pump_curve_name", None)
    if curve_name:
        wn_curve = wn.get_curve(str(curve_name))
        curve = PumpCurve(
            id=str(curve_name),
            points=tuple((float(x), float(y)) for x, y in wn_curve.points),
        )
    speed_series = getattr(pump, "speed_timeseries", None)
    base_speed = getattr(speed_series, "base_value", None)
    speed = float(base_speed if base_speed is not None else 1.0)
    speed_pattern = getattr(speed_series, "pattern_name", None)
    power = getattr(pump, "power", None)
    return Pump(
        id=name,
        start_node=pump.start_node_name,
        end_node=pump.end_node_name,
        pump_type=str(getattr(pump, "pump_type", "HEAD") or "HEAD"),
        curve=curve,
        power=float(power) if power is not None else None,
        speed=speed,
        speed_pattern_id=str(speed_pattern) if speed_pattern else None,
        status=_status_name(pump),
    )


def _valve_from_wn(wn: Any, name: str) -> Valve:
    valve = wn.get_link(name)
    curve_points: Optional[Tuple[Tuple[float, float], ...]] = None
    curve_name = getattr(valve, "headloss_curve_name", None)
    if curve_name:
        wn_curve = wn.get_curve(str(curve_name))
        curve_points = tuple((float(x), float(y)) for x, y in wn_curve.points)
    setting = getattr(valve, "initial_setting", None)
    valve_type = str(valve.valve_type).upper()
    if valve_type == "GPV":
        # A general purpose valve is set by a head loss curve, not a scalar.
        setting = None
    return Valve(
        id=name,
        start_node=valve.start_node_name,
        end_node=valve.end_node_name,
        valve_type=valve_type,
        setting=float(setting) if setting is not None else None,
        headloss_curve=curve_points,
        diameter=float(valve.diameter),
        minor_loss=float(valve.minor_loss),
        status=_status_name(valve),
    )


def _status_name(link: Any) -> str:
    """Return ``"OPEN"``/``"CLOSED"`` for the status a link was declared with."""
    for attribute in ("_initial_status", "_user_status", "status"):
        status = getattr(link, attribute, None)
        if status is None:
            continue
        name = getattr(status, "name", None)
        if name is None:
            name = str(status).rsplit(".", 1)[-1]
        name = str(name).upper()
        if name in ("OPEN", "CLOSED", "ACTIVE"):
            return name
    return "OPEN"


def _control_from_wn(wn: Any) -> List[Control]:
    controls: List[Control] = []
    for entry in wn.controls():
        if isinstance(entry, tuple):
            control_id, control = entry
        else:  # pragma: no cover - defensive, WNTR returns (name, control)
            control_id, control = getattr(control, "name", ""), entry
        actions = control.actions()
        if not actions:
            continue
        action = actions[0]
        target = getattr(action, "_target_obj", None)
        link_id = str(getattr(target, "name", ""))
        attribute = str(getattr(action, "_attribute", "status"))
        value = getattr(action, "_value", None)
        if attribute == "status":
            control_type = "OPEN" if value else "CLOSED"
            setting = None
        else:
            control_type = "SETTING"
            setting = float(value) if isinstance(value, (int, float)) else None
        condition = getattr(control, "condition", None)
        sensor_node = None
        sensor_attribute = None
        relation = None
        threshold = None
        if condition is not None:
            source = getattr(condition, "_source_obj", None)
            sensor_node = str(getattr(source, "name", "")) or None
            sensor_attribute = str(getattr(condition, "_source_attr", "") or "") or None
            relation = str(getattr(getattr(condition, "_relation", ""), "name", "") or "") or None
            raw_threshold = getattr(condition, "_threshold", None)
            threshold = float(raw_threshold) if isinstance(raw_threshold, (int, float)) else None
        controls.append(
            Control(
                id=str(control_id),
                link_id=link_id,
                control_type=control_type,
                setting=setting,
                sensor_node=sensor_node,
                sensor_attribute=sensor_attribute,
                relation=relation,
                threshold=threshold,
                description=str(condition) if condition is not None else "",
            )
        )
    return controls
