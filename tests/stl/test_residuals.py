"""Hand-calculated tests for the pure hydraulic residuals.

Every expected value in this module was computed by hand (or from the EPANET
reference behaviour that was verified separately) and is written down as a
literal, so a change in the formulas shows up as a failing expectation.
"""

from __future__ import annotations

import math

import pytest

from stl.residuals import (
    GRAVITY,
    delivered_demand,
    emitter_withdrawal,
    hazen_williams_headloss,
    horizon_sum,
    interpolate_curve,
    junction_continuity_residual,
    node_head,
    pipe_energy_residual,
    pump_curve_head,
    pump_dispatch,
    pump_energy_residual,
    pump_head,
    sample_difference,
    tank_volume,
    tank_volume_residual,
    valve_dispatch,
    velocity,
)

# A pipe of 100 m, 300 mm diameter and Hazen-Williams roughness 100 that
# carries 0.1 m^3/s.
PIPE_LENGTH = 100.0
PIPE_DIAMETER = 0.3
PIPE_ROUGHNESS = 100.0
PIPE_FLOW = 0.1

# WNTR's Hazen-Williams coefficients give 1.0446666286 m.
EXPECTED_FRICTION = 1.0446666286017248
# velocity: 0.1 / (pi * 0.3**2 / 4) = 1.4147106 m/s
EXPECTED_VELOCITY = 1.4147106
# minor loss: 0.5 * v**2 / (2 * 9.81) = 0.0510042 m
EXPECTED_MINOR = 0.0510042328399278


def test_node_head_adds_elevation_to_pressure_head():
    assert node_head(100.0, 20.0) == pytest.approx(120.0)
    assert node_head(105.0, 0.0) == pytest.approx(105.0)


def test_hazen_williams_head_loss_is_hand_calculated():
    loss = hazen_williams_headloss(PIPE_FLOW, PIPE_LENGTH, PIPE_DIAMETER, PIPE_ROUGHNESS)
    assert loss == pytest.approx(EXPECTED_FRICTION, abs=1e-6)


def test_hazen_williams_includes_the_minor_loss():
    loss = hazen_williams_headloss(
        PIPE_FLOW,
        PIPE_LENGTH,
        PIPE_DIAMETER,
        PIPE_ROUGHNESS,
        minor_loss=0.5,
    )
    assert loss == pytest.approx(EXPECTED_FRICTION + EXPECTED_MINOR, abs=1e-6)


def test_reversed_flow_flips_the_sign_of_the_head_loss():
    forward = hazen_williams_headloss(
        PIPE_FLOW, PIPE_LENGTH, PIPE_DIAMETER, PIPE_ROUGHNESS, minor_loss=0.5
    )
    backward = hazen_williams_headloss(
        -PIPE_FLOW, PIPE_LENGTH, PIPE_DIAMETER, PIPE_ROUGHNESS, minor_loss=0.5
    )
    assert backward == pytest.approx(-forward)
    assert hazen_williams_headloss(
        0.0, PIPE_LENGTH, PIPE_DIAMETER, PIPE_ROUGHNESS
    ) == 0.0


def test_pipe_energy_residual_uses_elevation_and_pressure_head():
    # J1: elevation 100 m, pressure head 20 m -> head 120 m
    # J2: elevation 105 m, pressure head 30 m -> head 135 m
    head_start = node_head(100.0, 20.0)
    head_end = node_head(105.0, 30.0)
    residual = pipe_energy_residual(
        flow=PIPE_FLOW,
        length=PIPE_LENGTH,
        diameter=PIPE_DIAMETER,
        roughness=PIPE_ROUGHNESS,
        minor_loss=0.5,
        head_start=head_start,
        head_end=head_end,
    )
    assert residual == pytest.approx(-15.0 - EXPECTED_FRICTION - EXPECTED_MINOR, abs=1e-6)


def test_pipe_energy_residual_is_zero_for_a_consistent_pipe():
    head_start = 120.0
    head_end = head_start - EXPECTED_FRICTION - EXPECTED_MINOR
    residual = pipe_energy_residual(
        flow=PIPE_FLOW,
        length=PIPE_LENGTH,
        diameter=PIPE_DIAMETER,
        roughness=PIPE_ROUGHNESS,
        minor_loss=0.5,
        head_start=head_start,
        head_end=head_end,
    )
    assert residual == pytest.approx(0.0, abs=1e-9)


def test_emitter_withdrawal_follows_the_epanet_formula():
    # 2e-6 m^3/s/sqrt(m) at 50 m of pressure: 2e-6 * sqrt(50).
    assert emitter_withdrawal(2e-6, 50.0) == pytest.approx(1.4142136e-05, abs=1e-12)
    # No withdrawal without pressure and without an emitter.
    assert emitter_withdrawal(2e-6, 0.0) == 0.0
    assert emitter_withdrawal(0.0, 50.0) == 0.0


def test_delivered_demand_with_a_changing_pattern():
    # 0.01 m^3/s base demand with the pattern multiplier 1.2.
    assert delivered_demand([(0.01, 1.2)]) == pytest.approx(0.012)
    # Two categories add up.
    assert delivered_demand([(0.01, 1.0), (0.005, 0.6)]) == pytest.approx(0.013)
    # The demand multiplier scales the result.
    assert delivered_demand([(0.01, 1.0)], demand_multiplier=1.5) == pytest.approx(0.015)


def test_delivered_demand_under_pressure_dependent_demand():
    # 0.01 m^3/s at 0.2 m of pressure with a required pressure of 0.1 m and an
    # exponent of 0.5: 0.01 * (0.2 / 0.1) ** 0.5.
    value = delivered_demand(
        [(0.01, 1.0)],
        model="PDD",
        pressure_head=0.2,
        pressure_exponent=0.5,
        required_pressure=0.1,
    )
    assert value == pytest.approx(0.01)
    # Halfway between minimum and required gives sqrt(0.5) of demand.
    assert (
        delivered_demand(
            [(0.01, 1.0)],
            model="PDD",
            pressure_head=0.05,
            pressure_exponent=0.5,
            required_pressure=0.1,
        )
        == pytest.approx(0.01 * math.sqrt(0.5))
    )


def test_junction_continuity_residual_separates_demand_and_leaks():
    residual = junction_continuity_residual(
        inflows=[0.05, 0.01],
        outflows=[0.02],
        delivered_demand=0.03,
        emitter_withdrawals=[1.4142136e-05],
    )
    assert residual == pytest.approx(0.01 - 1.4142136e-05, abs=1e-12)
    # Without the leak the residual is exactly zero.
    assert (
        junction_continuity_residual(
            inflows=[0.05, 0.01],
            outflows=[0.02],
            delivered_demand=0.04,
            emitter_withdrawals=[],
        )
        == pytest.approx(0.0, abs=1e-12)
    )


def test_tank_volume_with_a_volume_curve():
    curve = ((0.0, 0.0), (5.0, 900.0), (10.0, 1900.0))
    assert tank_volume(3.0, 12.0, volume_curve=curve) == pytest.approx(540.0)
    assert tank_volume(3.5, 12.0, volume_curve=curve) == pytest.approx(630.0)


def test_tank_volume_of_a_cylindrical_tank():
    # Area = pi * 12**2 / 4 = 113.0973355 m^2.
    assert tank_volume(3.5, 12.0) == pytest.approx(113.0973355 * 3.5, abs=1e-6)
    assert tank_volume(3.0, 12.0, min_volume=10.0, min_level=1.0) == pytest.approx(
        113.0973355 * 3.0, abs=1e-6
    )


def test_tank_volume_residual_compares_with_the_integrated_flow():
    curve = ((0.0, 0.0), (5.0, 900.0), (10.0, 1900.0))
    # The level rose by 0.5 m -> 90 m^3, but only 18 m^3 flowed in.
    residual = tank_volume_residual(
        level_start=3.0,
        level_end=3.5,
        net_inflow=0.005,
        dt_s=3600.0,
        diameter=12.0,
        volume_curve=curve,
    )
    assert residual == pytest.approx(90.0 - 18.0, abs=1e-9)
    # A consistent tank has a zero residual.
    assert (
        tank_volume_residual(
            level_start=3.0,
            level_end=3.5,
            net_inflow=90.0 / 3600.0,
            dt_s=3600.0,
            diameter=12.0,
            volume_curve=curve,
        )
        == pytest.approx(0.0, abs=1e-9)
    )


PUMP_CURVE = ((0.0, 80.0), (0.1, 60.0), (0.2, 40.0))


def test_three_point_linear_power_curve_at_nominal_speed():
    assert pump_curve_head(PUMP_CURVE, 0.0) == pytest.approx(80.0)
    assert pump_curve_head(PUMP_CURVE, 0.05) == pytest.approx(70.0)
    assert pump_curve_head(PUMP_CURVE, 0.1) == pytest.approx(60.0)
    # Beyond the curve the last segment is extrapolated linearly.
    assert pump_curve_head(PUMP_CURVE, 0.3) == pytest.approx(20.0)


def test_pump_speed_affinity_scales_the_curve():
    # H(Q, s) = s^2 * H(Q / s): 1.5^2 * H(0.05 / 1.5) = 2.25 * 73.3333.
    assert pump_head(PUMP_CURVE, 0.05, speed=1.5) == pytest.approx(165.0, abs=1e-9)
    # 0.8^2 * H(0.05 / 0.8) = 0.64 * 67.5.
    assert pump_head(PUMP_CURVE, 0.05, speed=0.8) == pytest.approx(43.2, abs=1e-9)
    assert pump_head(PUMP_CURVE, 0.05, speed=1.0) == pytest.approx(70.0)


def test_pump_energy_residual_uses_the_head_gain():
    residual = pump_energy_residual(
        flow=0.05,
        speed=1.0,
        curve_points=PUMP_CURVE,
        head_start=50.0,
        head_end=120.0,
    )
    assert residual == pytest.approx(70.0 - 70.0, abs=1e-9)


def test_valve_dispatch_prv_active_regime():
    # The PRV holds a pressure of 25 m at its downstream node (elevation 20 m
    # -> head 45 m) while the upstream head is 50 m.
    dispatch = valve_dispatch(
        valve_type="PRV",
        status="ACTIVE",
        setting=25.0,
        flow=0.005,
        diameter=0.2,
        head_start=50.0,
        head_end=45.0,
        elevation_start=0.0,
        elevation_end=20.0,
    )
    assert dispatch.regime == "active"
    assert dispatch.unit == "m"
    assert dispatch.residual == pytest.approx(0.0, abs=1e-9)
    # A downstream head above the set point is a positive residual.
    dispatch = valve_dispatch(
        valve_type="PRV",
        status="ACTIVE",
        setting=25.0,
        flow=0.005,
        diameter=0.2,
        head_start=50.0,
        head_end=50.0,
        elevation_start=0.0,
        elevation_end=20.0,
    )
    assert dispatch.residual == pytest.approx(5.0)


def test_valve_dispatch_prv_open_regime_and_tolerance_boundary():
    # The upstream head (50 m) does not exceed the set point (20 + 25 = 45 m),
    # so the valve is fully open and only its minor loss applies.
    common = dict(
        valve_type="PRV",
        status="OPEN",
        setting=25.0,
        flow=0.005,
        diameter=0.2,
        minor_loss=0.0,
        elevation_start=0.0,
        elevation_end=20.0,
    )
    dispatch = valve_dispatch(head_start=45.0, head_end=44.9, **common)
    assert dispatch.regime == "open"
    assert dispatch.residual == pytest.approx(0.1, abs=1e-9)
    # The observed status, not an upstream head comparison, selects the regime.
    boundary = valve_dispatch(
        head_start=45.0 + 1e-9, head_end=45.0, tolerance=1e-9, **common
    )
    assert boundary.regime == "open"
    active = valve_dispatch(
        head_start=45.0 + 2e-9, head_end=45.0, tolerance=1e-9,
        **{**common, "status": "ACTIVE"},
    )
    assert active.regime == "active"


def test_valve_dispatch_psv_regimes():
    common = dict(
        valve_type="PSV",
        status="ACTIVE",
        setting=25.0,
        flow=0.005,
        diameter=0.2,
        elevation_start=20.0,
        elevation_end=0.0,
    )
    # The upstream head sits at the set point -> the valve holds it.
    dispatch = valve_dispatch(head_start=45.0, head_end=40.0, **common)
    assert dispatch.regime == "active"
    assert dispatch.residual == pytest.approx(0.0, abs=1e-9)
    # The upstream head is above the set point -> the valve is fully open.
    dispatch = valve_dispatch(head_start=50.0, head_end=49.9, **{**common, "status": "OPEN"})
    assert dispatch.regime == "open"
    assert dispatch.residual == pytest.approx(0.1, abs=1e-9)


def test_valve_dispatch_pbv_active_is_explicitly_unsupported():
    dispatch = valve_dispatch(
        valve_type="PBV",
        status="ACTIVE",
        setting=15.0,
        flow=0.005,
        head_start=50.0,
        head_end=35.0,
    )
    assert dispatch.regime == "unsupported"
    assert dispatch.residual is None
    reverse = valve_dispatch(
        valve_type="PBV",
        status="OPEN",
        setting=15.0,
        flow=-0.005,
        head_start=35.0,
        head_end=50.0,
    )
    assert reverse.regime == "open"
    assert reverse.residual == pytest.approx(-15.0)


def test_valve_dispatch_tcv_uses_the_loss_coefficient():
    dispatch = valve_dispatch(
        valve_type="TCV",
        status="ACTIVE",
        setting=0.5,
        flow=0.1,
        diameter=0.3,
        head_start=120.0,
        head_end=120.0 - EXPECTED_MINOR,
    )
    assert dispatch.regime == "active"
    assert dispatch.residual == pytest.approx(0.0, abs=1e-6)
    assert velocity(0.1, 0.3) == pytest.approx(EXPECTED_VELOCITY, abs=1e-7)


def test_valve_dispatch_fcv_holds_a_flow():
    dispatch = valve_dispatch(
        valve_type="FCV",
        status="ACTIVE",
        setting=0.005,
        flow=0.005,
        head_start=50.0,
        head_end=49.9,
    )
    assert dispatch.regime == "active"
    assert dispatch.unit == "m^3/s"
    assert dispatch.residual == pytest.approx(0.0, abs=1e-12)
    closed = valve_dispatch(
        valve_type="FCV",
        status="ACTIVE",
        setting=0.005,
        flow=0.0,
        head_start=50.0,
        head_end=49.9,
    )
    assert closed.regime == "active"
    assert closed.residual == pytest.approx(-0.005)


def test_valve_dispatch_gpv_uses_the_head_loss_curve():
    # The curve holds 5 m of head loss at 0.01 m^3/s.
    dispatch = valve_dispatch(
        valve_type="GPV",
        status="ACTIVE",
        flow=0.005,
        headloss_curve=((0.0, 0.0), (0.01, 5.0)),
        head_start=50.0,
        head_end=47.5,
    )
    assert dispatch.regime == "active"
    assert dispatch.residual == pytest.approx(0.0, abs=1e-9)
    # Without a curve the dispatch is explicitly unsupported.
    unsupported = valve_dispatch(
        valve_type="GPV",
        status="ACTIVE",
        flow=0.005,
        head_start=50.0,
        head_end=47.5,
    )
    assert unsupported.regime == "unsupported"
    assert unsupported.residual is None
    assert "head loss curve" in unsupported.reason


def test_valve_dispatch_rejects_an_unknown_valve_type():
    dispatch = valve_dispatch(valve_type="XYZ", flow=0.005)
    assert dispatch.regime == "unsupported"
    assert dispatch.residual is None


def test_interpolation_extrapolates_linearly():
    points = ((0.0, 0.0), (10.0, 5.0))
    assert interpolate_curve(points, 5.0) == pytest.approx(2.5)
    assert interpolate_curve(points, 20.0) == pytest.approx(10.0)
    assert interpolate_curve(points, -5.0) == pytest.approx(-2.5)


def test_initial_and_incomplete_samples_are_unknown():
    values = [1.0, 3.0, 6.0]
    assert sample_difference(values, 0) is None
    assert sample_difference(values, 1) == pytest.approx(2.0)
    assert sample_difference(values, 2) == pytest.approx(3.0)
    # A horizon that is not complete yet is unknown as well.
    assert horizon_sum(values, 0, 3) is None
    assert horizon_sum(values, 0, 2) is None
    assert horizon_sum(values, 1, 2) == pytest.approx(4.0)
    assert horizon_sum(values, 2, 3) == pytest.approx(10.0)
    assert horizon_sum(values, 2, 1) == pytest.approx(6.0)


def test_gravity_constant_is_the_standard_one():
    assert GRAVITY == 9.81
    assert math.isclose(GRAVITY, 9.81)


def test_pdd_thresholds_negative_demand_and_invalid_interval():
    args = dict(model="PDD", minimum_pressure=10, required_pressure=30)
    assert delivered_demand([(4, 1)], pressure_head=10, **args) == 0
    assert delivered_demand([(4, 1)], pressure_head=15, **args) == pytest.approx(2)
    assert delivered_demand([(4, 1)], pressure_head=30, **args) == 4
    assert delivered_demand([(4, 1)], pressure_head=40, **args) == 4
    assert delivered_demand([(-3, 1)], pressure_head=0, **args) == -3
    assert delivered_demand([(4, 1)], demand_multiplier=0, **args) == 0
    with pytest.raises(ValueError):
        delivered_demand([(4, 1)], model="PDD", pressure_head=10, minimum_pressure=10, required_pressure=10)
    with pytest.raises(ValueError):
        delivered_demand([(4, 1)], **args)


def test_pump_nonlinear_and_one_point_affinity():
    # H=80-20Q²; design point (2, 30) gives H=40-2.5Q².
    points = ((0, 80), (1, 60), (2, 0))
    assert pump_curve_head(points, 0.5) == pytest.approx(75)
    assert pump_head(points, 1, 2) == pytest.approx(300)
    # WNTR fits a single power curve to multipoint data, not piecewise lines.
    multipoint = ((0, 100), (1, 96), (2, 84), (3, 64))
    assert pump_curve_head(multipoint, 1.5) == pytest.approx(91)
    assert pump_curve_head(((2, 30),), 1) == pytest.approx(37.5)
    assert pump_head(((2, 30),), 2, 2) == pytest.approx(150)
    with pytest.raises(ValueError):
        pump_curve_head(((0, 30),), 1)
    with pytest.raises(ValueError):
        pump_head(points, 1, 0)


def test_tank_curve_clamps_and_absolute_cylinder_volume():
    curve = ((1, 5), (2, 9))
    assert tank_volume(0, 1, volume_curve=curve) == 5
    assert tank_volume(3, 1, volume_curve=curve) == 9
    assert tank_volume(2, 2, min_level=1, min_volume=99) == pytest.approx(2 * math.pi)


def test_valve_observed_status_reverse_gpv_and_unknown():
    curve = ((0, 1), (1, 3), (2, 7))
    result = valve_dispatch(valve_type="GPV", status="ACTIVE", flow=-1.5,
                            headloss_curve=curve, head_start=10, head_end=15)
    assert result.residual == pytest.approx(0)
    assert valve_dispatch(valve_type="FCV", flow=0).regime == "unknown"
    closed = valve_dispatch(valve_type="PRV", status="CLOSED", flow=0.2)
    assert closed.unit == "m^3/s" and closed.residual == 0.2
    opened = valve_dispatch(valve_type="PRV", status="OPEN", flow=0,
                            head_start=10, head_end=10)
    assert opened.regime == "open" and opened.residual == 0


def test_epanet_two_and_four_point_curves_follow_source_segments():
    # EPANET 2.2 input2.c selects CUSTOM and hydcoeffs.c interpolates segments.
    two = ((0.0, 80.0), (0.2, 40.0))
    four = ((0.0, 100.0), (1.0, 96.0), (2.0, 84.0), (3.0, 64.0))
    assert pump_curve_head(two, 0.1, backend="epanet") == pytest.approx(60)
    assert pump_curve_head(four, 1.5, backend="epanet") == pytest.approx(90)
    assert pump_head(four, 3, 2, backend="epanet") == pytest.approx(360)
    assert pump_energy_residual(flow=1.5, speed=1, curve_points=four,
                                head_start=10, head_end=100,
                                backend="epanet") == pytest.approx(0)


def test_wntr_two_and_four_point_curves_match_installed_coefficients():
    wntr = pytest.importorskip("wntr")
    wn = wntr.network.WaterNetworkModel()
    wn.add_reservoir("r", base_head=100)
    wn.add_junction("j")
    for name, points, flow in (
        ("two", ((0.0, 80.0), (0.2, 40.0)), 0.1),
        ("four", ((0.0, 100.0), (1.0, 96.0), (2.0, 84.0), (3.0, 64.0)), 1.5),
    ):
        wn.add_curve(name, "HEAD", list(points))
        wn.add_pump(name, "r", "j", pump_type="HEAD", pump_parameter=name)
        a, b, c = wn.get_link(name).get_head_curve_coefficients()
        assert pump_curve_head(points, flow, backend="wntr") == pytest.approx(a - b * flow**c)
    assert pump_curve_head(((0, 80), (0.2, 40)), 0.1) == pytest.approx(-20)


def test_epanet_one_and_three_point_source_power_curves():
    # EPANET 2.2 uses 1.33334 times design head as shutoff head.
    one = ((2.0, 30.0),)
    assert pump_curve_head(one, 0, backend="epanet") == pytest.approx(40.0002)
    assert pump_curve_head(one, 2, backend="epanet") == pytest.approx(30)
    three = ((0, 80), (1, 60), (2, 0))
    assert pump_curve_head(three, 0.5, backend="epanet") == pytest.approx(75)
    custom = ((1, 80), (2, 60), (3, 20))
    assert pump_curve_head(custom, 2.5, backend="epanet") == pytest.approx(40)
    with pytest.raises(ValueError):
        pump_curve_head(one, 1, backend="other")


def test_pump_dispatch_closed_ignores_retained_speed_and_head_difference():
    closed = pump_dispatch(status="CLOSED", flow=0, speed=1.5,
                           head_start=10, head_end=1000)
    assert (closed.regime, closed.unit, closed.residual) == ("closed", "m^3/s", 0)
    assert pump_dispatch(status="CLOSED", flow=0.02, speed=0).residual == 0.02
    assert pump_dispatch(status=None, flow=0).regime == "unknown"
    assert pump_dispatch(status="OPEN", flow=0, speed=0).regime == "unknown"
    opened = pump_dispatch(status="OPEN", flow=0.1, speed=1,
                           curve_points=((0, 80), (0.2, 40)),
                           head_start=10, head_end=70, backend="epanet")
    assert opened.regime == "open" and opened.residual == pytest.approx(0)


def test_open_gpv_uses_curve_instead_of_generic_open_valve_loss():
    result = valve_dispatch(valve_type="GPV", status="OPEN", flow=0.005,
                            headloss_curve=((0, 0), (0.01, 5)),
                            head_start=50, head_end=47.5)
    assert result.regime == "open" and result.residual == pytest.approx(0)
