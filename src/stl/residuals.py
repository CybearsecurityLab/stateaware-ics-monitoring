"""Pure numerical hydraulic residuals for the Stage-3 STL monitor.

Every function here is a plain float computation: no ``rtamt``, no dataset
access, no I/O.  A *residual* is the signed difference between an observed
quantity and the quantity the water-network physics predicts; a residual of
zero means the observation is consistent with the model.

Conventions
-----------
* SI units: metres (m), seconds (s), cubic metres per second (m^3/s) and
  cubic metres (m^3).
* Pressures are pressure heads in metres; a node head is
  ``elevation + pressure_head``.
* Flows are positive in the direction of the link definition
  (``start_node`` -> ``end_node``).
* ``None`` means *unknown* — the sample is initial, the window is incomplete or
  the operating regime cannot be decided.  It is never silently replaced by
  zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

#: Standard gravitational acceleration (m/s^2).
GRAVITY = 9.81

#: Hazen-Williams friction factor coefficient (SI).
HW_COEFFICIENT = 10.666829500036352

#: Hazen-Williams flow exponent.
HW_FLOW_EXPONENT = 1.852

#: Hazen-Williams roughness exponent.
HW_ROUGHNESS_EXPONENT = 1.852

#: Hazen-Williams diameter exponent.
HW_DIAMETER_EXPONENT = 4.871


def node_head(elevation: float, pressure_head: float) -> float:
    """Return the node head (m) for *elevation* and *pressure_head*."""
    return elevation + pressure_head


def interpolate_curve(
    points: Sequence[Tuple[float, float]], x: float
) -> float:
    """Interpolate *points* (``(x, y)`` pairs) at *x*.

    Linear interpolation inside the curve and linear extrapolation outside it
    using the first/last segment, which is what EPANET does for pump and
    head-loss curves.
    """
    if len(points) < 2:
        raise ValueError("a curve needs at least two points")
    ordered = sorted(points, key=lambda point: point[0])
    if x <= ordered[0][0]:
        (x0, y0), (x1, y1) = ordered[0], ordered[1]
    elif x >= ordered[-1][0]:
        (x0, y0), (x1, y1) = ordered[-2], ordered[-1]
    else:
        (x0, y0), (x1, y1) = ordered[0], ordered[1]
        for left, right in zip(ordered, ordered[1:]):
            if left[0] <= x <= right[0]:
                (x0, y0), (x1, y1) = left, right
                break
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def hazen_williams_headloss(
    flow: float,
    length: float,
    diameter: float,
    roughness: float,
    minor_loss: float = 0.0,
) -> float:
    """Signed head loss (m) of a pipe carrying *flow* (m^3/s).

    The friction part follows the Hazen-Williams formula in SI units and the
    minor-loss part follows ``K v^2 / (2 g)``.  The result is signed: it is
    negative when the flow runs from ``end_node`` to ``start_node``. WNTR's
    low-flow polynomial smoothing is not implemented.
    """
    if diameter <= 0.0:
        raise ValueError(f"pipe diameter must be positive, got {diameter}")
    if roughness <= 0.0:
        raise ValueError(f"pipe roughness must be positive, got {roughness}")
    if flow == 0.0:
        return 0.0
    magnitude = abs(flow)
    friction = (
        HW_COEFFICIENT
        * length
        * magnitude**HW_FLOW_EXPONENT
        / (roughness**HW_ROUGHNESS_EXPONENT * diameter**HW_DIAMETER_EXPONENT)
    )
    velocity = flow / link_area(diameter)
    minor = minor_loss * velocity * velocity / (2.0 * GRAVITY)
    sign = 1.0 if flow > 0.0 else -1.0
    return sign * (friction + abs(minor))


def link_area(diameter: float) -> float:
    """Cross-sectional area (m^2) of a circular link of *diameter*."""
    return math.pi * diameter * diameter / 4.0


def velocity(flow: float, diameter: float) -> float:
    """Mean velocity (m/s) of *flow* (m^3/s) in a link of *diameter*."""
    return flow / link_area(diameter)


def minor_head_loss(flow: float, diameter: float, minor_loss: float) -> float:
    """Signed minor head loss (m) of *flow* through a link."""
    if flow == 0.0:
        return 0.0
    speed = velocity(flow, diameter)
    magnitude = minor_loss * speed * speed / (2.0 * GRAVITY)
    return magnitude if flow > 0.0 else -magnitude


def pipe_energy_residual(
    *,
    flow: float,
    length: float,
    diameter: float,
    roughness: float,
    head_start: float,
    head_end: float,
    minor_loss: float = 0.0,
) -> float:
    """Energy residual (m) of a pipe.

    ``residual = (head_start - head_end) - hazen_williams_headloss(flow)``; a
    zero residual means the observed heads match the pipe geometry.
    """
    expected = hazen_williams_headloss(
        flow, length, diameter, roughness, minor_loss=minor_loss
    )
    return (head_start - head_end) - expected


def emitter_withdrawal(
    emitter_coefficient: float, pressure_head: float, exponent: float = 0.5
) -> float:
    """Emitter/leak withdrawal (m^3/s) at *pressure_head* (m).

    EPANET models an emitter as ``q = C * p^n`` with ``n = 0.5`` by default;
    the withdrawal vanishes when the pressure is not positive.
    """
    if emitter_coefficient <= 0.0 or pressure_head <= 0.0:
        return 0.0
    return emitter_coefficient * pressure_head**exponent


def delivered_demand(
    terms: Sequence[Tuple[float, float]],
    demand_multiplier: float = 1.0,
    model: str = "DDA",
    pressure_head: Optional[float] = None,
    pressure_exponent: float = 0.5,
    required_pressure: float = 0.0,
    minimum_pressure: float = 0.0,
) -> float:
    """Delivered demand (m^3/s) of one node.

    *terms* is a sequence of ``(base_demand, pattern_multiplier)`` pairs.  In
    demand-driven analysis (``model == "DDA"``) the delivered demand is
    ``sum(base * multiplier) * demand_multiplier``.  Under pressure-dependent
    demand (``model == "PDD"`` or WNTR's ``"PDA"``) positive total demand is scaled between the
    minimum and required pressures, and capped at full demand above that range.
    Nonpositive demand is not pressure dependent.
    """
    pdd = model.upper() in ("PDD", "PDA")
    total = sum(base * multiplier for base, multiplier in terms) * demand_multiplier
    if not pdd:
        return total
    if required_pressure <= minimum_pressure or pressure_exponent <= 0:
        raise ValueError("invalid pressure-dependent demand interval or exponent")
    if total <= 0:
        return total
    if pressure_head is None:
        raise ValueError("pressure-dependent demand needs a pressure head")
    if pressure_head <= minimum_pressure:
        return 0.0
    if pressure_head >= required_pressure:
        return total
    return total * ((pressure_head - minimum_pressure) / (required_pressure - minimum_pressure)) ** pressure_exponent


def junction_continuity_residual(
    *,
    inflows: Sequence[float],
    outflows: Sequence[float],
    delivered_demand: float,
    emitter_withdrawals: Sequence[float] = (),
) -> float:
    """Continuity residual (m^3/s) of a junction.

    ``residual = sum(inflows) - sum(outflows) - delivered_demand - withdrawals``;
    emitter/leak withdrawals are kept separate from the billed demand.
    """
    return (
        sum(inflows)
        - sum(outflows)
        - delivered_demand
        - sum(emitter_withdrawals)
    )


def tank_volume(
    level: float,
    diameter: float,
    min_volume: float = 0.0,
    min_level: float = 0.0,
    volume_curve: Optional[Sequence[Tuple[float, float]]] = None,
) -> float:
    """Stored volume (m^3) of a tank at *level* (m).

    With a volume curve (a sequence of ``(level, volume)`` pairs) the volume is
    interpolated linearly and clamped at the endpoints; otherwise the tank is
    a cylinder of *diameter*. WNTR's absolute-volume calculation ignores
    *min_volume* and *min_level*.
    """
    if volume_curve is not None:
        ordered = sorted(volume_curve)
        return interpolate_curve(ordered, max(ordered[0][0], min(level, ordered[-1][0])))
    area = link_area(diameter)
    return area * level


def tank_volume_residual(
    *,
    level_start: float,
    level_end: float,
    net_inflow: float,
    dt_s: float,
    diameter: float,
    min_volume: float = 0.0,
    min_level: float = 0.0,
    volume_curve: Optional[Sequence[Tuple[float, float]]] = None,
) -> float:
    """Volume residual (m^3) of a tank.

    ``residual = V(level_end) - V(level_start) - net_inflow * dt``; a zero
    residual means the observed level change matches the integrated net flow.
    """
    delta_volume = tank_volume(
        level_end,
        diameter,
        min_volume=min_volume,
        min_level=min_level,
        volume_curve=volume_curve,
    ) - tank_volume(
        level_start,
        diameter,
        min_volume=min_volume,
        min_level=min_level,
        volume_curve=volume_curve,
    )
    return delta_volume - net_inflow * dt_s


def pump_curve_head(
    points: Sequence[Tuple[float, float]], flow: float, backend: str = "wntr"
) -> float:
    """Curve head at nominal speed using WNTR 1.5 or EPANET 2.2 semantics.

    WNTR uses its two-point coefficients and a regression for >=3 points.
    EPANET uses a power curve for one point or three with a zero-flow first
    point, and piecewise lines for other curves. Solver low-flow smoothing and
    operating limits are outside this ideal curve calculation.
    """
    if backend not in ("wntr", "epanet"):
        raise ValueError(f"unknown simulator backend {backend!r}")
    if flow < 0.0:
        raise ValueError(f"pump flow must not be negative, got {flow}")
    if not points:
        raise ValueError("pump curve is empty")
    if len(points) == 1:
        q, h = points[0]
        if q <= 0 or h <= 0:
            raise ValueError("invalid pump design point")
        if backend == "wntr":
            return 4 * h / 3 - h * flow**2 / (3 * q**2)
        h0 = 1.33334 * h
        exponent = math.log(h0 / (h0 - h)) / math.log(2.0)
        return h0 - (h0 - h) * (flow / q)**exponent
    if backend == "epanet":
        if len(points) != 3 or points[0][0] != 0:
            return interpolate_curve(points, flow)
        (_, h0), (q1, h1), (q2, h2) = points
        if not (h0 > h1 > h2 and q2 > q1 > 0):
            raise ValueError("invalid EPANET three-point pump curve")
        exponent = math.log((h0 - h2) / (h0 - h1)) / math.log(q2 / q1)
        if exponent > 20:
            raise ValueError("invalid EPANET pump curve exponent")
        return h0 - (h0 - h1) * (flow / q1)**exponent
    if len(points) == 2:
        (q0, h0), (q1, h1) = points
        b = -(h1 - h0) / (q1**2 - q0**2)
        a = h0 + b * q0**2
        if a <= 0 or b < 0:
            raise ValueError("invalid pump curve coefficients")
        return a - b * flow
    from scipy.optimize import curve_fit
    import numpy as np
    q = np.asarray([p[0] for p in points])
    h = np.asarray([p[1] for p in points])
    c = math.log((h[0] - h[1]) / (h[0] - h[-1])) / math.log(q[1] / q[-1])
    b = (h[0] - h[1]) / q[1]**c
    (a, b, c), _ = curve_fit(lambda x, a, b, c: a - b * x**c, q, h, [h[0], b, c])
    if not all(math.isfinite(x) for x in (a, b, c)) or a <= 0 or b < 0 or c <= 0:
        raise ValueError("invalid pump curve coefficients")
    return float(a - b * flow**c)


def pump_head(
    points: Sequence[Tuple[float, float]], flow: float, speed: float = 1.0,
    backend: str = "wntr",
) -> float:
    """Head (m) of a pump running at *speed* (relative to the curve speed).

    The affinity law scales the chosen backend's curve as ``s^2 * H(Q / s)``.
    """
    if speed <= 0.0:
        raise ValueError(f"pump speed must be positive, got {speed}")
    if flow < 0.0:
        raise ValueError(f"pump flow must not be negative, got {flow}")
    return speed * speed * pump_curve_head(points, flow / speed, backend=backend)


def pump_energy_residual(
    *,
    flow: float,
    speed: float,
    curve_points: Sequence[Tuple[float, float]],
    head_start: float,
    head_end: float,
    backend: str = "wntr",
) -> float:
    """Head-gain residual (m) of a pump.

    This open-pump helper uses the selected backend's curve. Status must be
    checked by the caller or via :func:`pump_dispatch`.
    """
    return (head_end - head_start) - pump_head(curve_points, flow, speed, backend)


@dataclass(frozen=True)
class PumpDispatch:
    """Pump residual and units for the observed operating regime."""

    residual: Optional[float]
    unit: str
    regime: str
    reason: str = ""


def pump_dispatch(
    *, flow: float, status: Optional[str], speed: Optional[float] = None,
    curve_points: Optional[Sequence[Tuple[float, float]]] = None,
    head_start: Optional[float] = None, head_end: Optional[float] = None,
    backend: str = "wntr",
) -> PumpDispatch:
    """Use observed status: closed checks flow; open checks backend head gain.

    Missing status or open-pump inputs, including nonpositive speed, return an
    unknown result. A closed pump does not use speed, heads, or its curve.
    """
    state = str(status).upper() if status is not None else ""
    if state == "CLOSED":
        return PumpDispatch(flow, "m^3/s", "closed")
    if state != "OPEN":
        return PumpDispatch(None, "", "unknown", "pump status is missing or unknown")
    if speed is None or speed <= 0 or curve_points is None or head_start is None or head_end is None:
        return PumpDispatch(None, "m", "unknown", "open pump needs positive speed, curve and heads")
    return PumpDispatch(pump_energy_residual(
        flow=flow, speed=speed, curve_points=curve_points, head_start=head_start,
        head_end=head_end, backend=backend,
    ), "m", "open")


@dataclass(frozen=True)
class ValveDispatch:
    """Outcome of dispatching one valve.

    ``residual`` is ``None`` when inputs or the operating regime are unknown
    or unsupported. Closed valves return the observed flow as the residual.
    """

    residual: Optional[float]
    unit: str
    regime: str
    reason: str = ""


def valve_dispatch(
    *,
    valve_type: str,
    flow: float,
    setting: Optional[float] = None,
    diameter: float = 0.0,
    minor_loss: float = 0.0,
    headloss_curve: Optional[Sequence[Tuple[float, float]]] = None,
    head_start: Optional[float] = None,
    head_end: Optional[float] = None,
    elevation_start: float = 0.0,
    elevation_end: float = 0.0,
    tolerance: float = 1e-9,
    flow_tolerance: float = 1e-12,
    status: Optional[str] = None,
) -> ValveDispatch:
    """Dispatch one valve and return its residual.

    Dispatch uses the observed status, never a state inferred from flow or head.
    GPV uses its curve whenever not closed; PBV active behavior is unsupported.
    """
    kind = valve_type.upper()
    if kind not in ("PRV", "PSV", "PBV", "TCV", "FCV", "GPV"):
        return ValveDispatch(None, "m", "unsupported", f"unsupported valve type {valve_type!r}")
    state = str(status).upper() if status is not None else ""
    if state not in ("OPEN", "CLOSED", "ACTIVE"):
        return ValveDispatch(None, "", "unknown", "valve status is missing or unknown")
    if state == "CLOSED":
        return ValveDispatch(flow, "m^3/s", "closed")
    if kind == "GPV":
        if not headloss_curve or len(headloss_curve) < 2:
            return ValveDispatch(None, "m", "unsupported", "a general purpose valve needs a head loss curve")
        if head_start is None or head_end is None:
            return ValveDispatch(None, "m", "unknown", "general purpose valve needs heads")
        loss = (1.0 if flow > 0 else -1.0) * interpolate_curve(headloss_curve, abs(flow)) if flow else 0.0
        return ValveDispatch(head_start - head_end - loss, "m", state.lower())
    if state == "OPEN":
        if head_start is None or head_end is None or (diameter <= 0 and flow != 0 and minor_loss != 0):
            return ValveDispatch(None, "m", "unknown", "open valve needs heads and diameter")
        loss = minor_head_loss(flow, diameter, minor_loss) if minor_loss else 0.0
        return ValveDispatch(head_start - head_end - loss, "m", "open")
    if kind in ("PRV", "PSV"):
        if setting is None or head_start is None or head_end is None:
            return ValveDispatch(None, "m", "unknown", "active valve needs setting and heads")
        residual = head_end - elevation_end - setting if kind == "PRV" else head_start - elevation_start - setting
        return ValveDispatch(residual, "m", "active")
    if kind == "FCV":
        if setting is None:
            return ValveDispatch(None, "m^3/s", "unknown", "active valve needs setting")
        return ValveDispatch(flow - setting, "m^3/s", "active")
    if kind == "PBV":
        return ValveDispatch(None, "m", "unsupported", "PBV active branch requires EPANET verification")
    if head_start is None or head_end is None:
        return ValveDispatch(None, "m", "unknown", "active valve needs heads")
    if setting is None:
        return ValveDispatch(None, "m", "unknown", "active valve needs setting")
    else:
        if diameter <= 0:
            return ValveDispatch(None, "m", "unknown", "active valve needs diameter")
        loss = minor_head_loss(flow, diameter, setting)
    return ValveDispatch(head_start - head_end - loss, "m", "active")

def sample_difference(values: Sequence[float], index: int) -> Optional[float]:
    """First difference of *values* at *index*.

    The initial sample has no predecessor, so its difference is *unknown*
    (``None``) rather than zero.
    """
    if index <= 0:
        return None
    if index >= len(values):
        raise IndexError(f"index {index} is outside the sample series")
    return values[index] - values[index - 1]


def horizon_sum(values: Sequence[float], index: int, samples: int) -> Optional[float]:
    """Sum of the *samples* values ending at *index* (inclusive).

    The sum is *unknown* (``None``) when the window is not complete, e.g. for
    the initial samples of a series.
    """
    if samples <= 0:
        raise ValueError(f"the horizon must contain at least one sample, got {samples}")
    start = index - samples + 1
    if start < 0:
        return None
    if index >= len(values):
        raise IndexError(f"index {index} is outside the sample series")
    return sum(values[start : index + 1])
