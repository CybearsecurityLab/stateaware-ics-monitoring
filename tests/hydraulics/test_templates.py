"""Tests for the deterministic compilation of the five template families."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hydraulics import (
    TemplateFamily,
    TemplateStatus,
    Valve,
    compile_templates,
    load_network,
    missing_channels,
)

pytest.importorskip("wntr", reason="the INP loader needs WNTR")

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic.inp"

ALL_CHANNELS = {
    "pressure:J1",
    "pressure:J2",
    "pressure:J3",
    "level:T1",
    "flow:P1",
    "flow:P2",
    "flow:P3",
    "flow:PU1",
    "flow:PU2",
    "flow:PU3",
    "flow:V1",
    "flow:V2",
    "flow:V3",
    "flow:V4",
    "flow:V5",
    "flow:V6",
}
ALL_CHANNELS.update(f"status:P{i}" for i in range(1, 4))
ALL_CHANNELS.update(f"{kind}:PU{i}" for kind in ("status", "speed") for i in range(1, 4))
ALL_CHANNELS.update(f"status:V{i}" for i in range(1, 7))
ALL_CHANNELS.update(f"setting:V{i}" for i in range(1, 6))


@pytest.fixture(scope="module")
def network():
    return load_network(FIXTURE)


def by_key(templates):
    return {(template.family, template.key): template for template in templates}


def test_all_five_families_are_compiled(network):
    templates = compile_templates(network, ALL_CHANNELS)
    families = {template.family for template in templates}
    assert families == set(TemplateFamily)
    assert len(families) == 5


def test_compilation_is_deterministic(network):
    first = compile_templates(network, ALL_CHANNELS)
    second = compile_templates(network, ALL_CHANNELS)
    assert first == second
    # Families come in enum order, elements are sorted by id inside a family.
    order = list(TemplateFamily)
    seen = [order.index(template.family) for template in first]
    assert seen == sorted(seen)
    for family in order:
        keys = [t.key for t in first if t.family is family]
        assert keys == sorted(keys)


def test_compiled_template_states_its_required_channels(network):
    templates = compile_templates(network, ALL_CHANNELS)
    pipe = by_key(templates)[(TemplateFamily.PIPE_ENERGY, "P1")]
    assert pipe.status is TemplateStatus.COMPILED
    assert {spec.key for spec in pipe.required_channels} == {
        "pressure:J1",
        "pressure:J2",
        "flow:P1",
        "status:P1",
    }
    assert missing_channels(pipe, ALL_CHANNELS) == ()


def test_missing_channels_make_a_template_unobservable(network):
    available = ALL_CHANNELS - {"flow:P1", "pressure:J2"}
    templates = compile_templates(network, available)
    pipe = by_key(templates)[(TemplateFamily.PIPE_ENERGY, "P1")]
    assert pipe.status is TemplateStatus.UNOBSERVABLE
    assert "flow:P1" in pipe.reason
    assert "pressure:J2" in pipe.reason
    assert {spec.key for spec in missing_channels(pipe, available)} == {
        "pressure:J2",
        "flow:P1",
    }


def test_constant_power_pumps_are_explicitly_unsupported(network):
    templates = compile_templates(network, ALL_CHANNELS)
    power_pump = by_key(templates)[(TemplateFamily.PUMP_HEAD, "PU2")]
    assert power_pump.status is TemplateStatus.UNSUPPORTED
    assert "constant-power" in power_pump.reason
    head_pump = by_key(templates)[(TemplateFamily.PUMP_HEAD, "PU1")]
    assert head_pump.status is TemplateStatus.COMPILED


def test_general_purpose_valve_without_a_curve_is_unsupported(network):
    valves = tuple(
        dataclasses.replace(valve, headloss_curve=None) if valve.id == "V6" else valve
        for valve in network.valves
    )
    broken = dataclasses.replace(network, valves=valves)
    templates = compile_templates(broken, ALL_CHANNELS)
    gpv = by_key(templates)[(TemplateFamily.VALVE_DISPATCH, "V6")]
    assert gpv.status is TemplateStatus.UNSUPPORTED
    assert "head loss curve" in gpv.reason


def test_unknown_valve_type_is_unsupported(network):
    valves = tuple(
        dataclasses.replace(valve, valve_type="XYZ") if valve.id == "V1" else valve
        for valve in network.valves
    )
    broken = dataclasses.replace(network, valves=valves)
    templates = compile_templates(broken, ALL_CHANNELS)
    unknown = by_key(templates)[(TemplateFamily.VALVE_DISPATCH, "V1")]
    assert unknown.status is TemplateStatus.UNSUPPORTED
    assert "XYZ" in unknown.reason


def test_pipe_energy_needs_hazen_williams(network):
    provenance = dataclasses.replace(network.provenance, headloss_formula="D-W")
    broken = dataclasses.replace(network, provenance=provenance)
    templates = compile_templates(broken, ALL_CHANNELS)
    pipe = by_key(templates)[(TemplateFamily.PIPE_ENERGY, "P1")]
    assert pipe.status is TemplateStatus.UNSUPPORTED
    assert "Hazen-Williams" in pipe.reason


def test_tank_without_geometry_is_unsupported(network):
    tanks = tuple(
        dataclasses.replace(tank, diameter=0.0, volume_curve=None)
        if tank.id == "T1"
        else tank
        for tank in network.tanks
    )
    broken = dataclasses.replace(network, tanks=tanks)
    templates = compile_templates(broken, ALL_CHANNELS)
    tank = by_key(templates)[(TemplateFamily.TANK_VOLUME, "T1")]
    assert tank.status is TemplateStatus.UNSUPPORTED
    assert "volume curve" in tank.reason


def test_every_candidate_template_is_reported(network):
    """Nothing is dropped silently: every element shows up in the result."""
    templates = compile_templates(network, ALL_CHANNELS)
    keys = {(template.family, template.key) for template in templates}
    assert (TemplateFamily.JUNCTION_CONTINUITY, "J1") in keys
    assert (TemplateFamily.JUNCTION_CONTINUITY, "J3") in keys
    assert (TemplateFamily.PIPE_ENERGY, "P3") in keys
    assert (TemplateFamily.TANK_VOLUME, "T1") in keys
    assert (TemplateFamily.PUMP_HEAD, "PU3") in keys
    assert (TemplateFamily.VALVE_DISPATCH, "V6") in keys
    # Three junctions, three pipes, one tank, three pumps and six valves.
    assert len(templates) == 3 + 3 + 1 + 3 + 6


def test_valve_setting_is_required_for_scalar_valves(network):
    valves = tuple(
        dataclasses.replace(valve, setting=None) if valve.id == "V5" else valve
        for valve in network.valves
    )
    broken = dataclasses.replace(network, valves=valves)
    templates = compile_templates(broken, ALL_CHANNELS)
    tcv = by_key(templates)[(TemplateFamily.VALVE_DISPATCH, "V5")]
    assert tcv.status is TemplateStatus.COMPILED
    missing = by_key(compile_templates(broken, ALL_CHANNELS - {"setting:V5"}))[(TemplateFamily.VALVE_DISPATCH, "V5")]
    assert missing.status is TemplateStatus.UNOBSERVABLE


def test_tank_connected_energy_requires_level_and_actuator_channels(network):
    pipe = dataclasses.replace(network.pipes[0], start_node="T1")
    pump = dataclasses.replace(network.pumps[0], start_node="T1")
    valve = dataclasses.replace(network.valves[0], start_node="T1")
    altered = dataclasses.replace(network, pipes=(pipe,), pumps=(pump,), valves=(valve,))
    templates = by_key(compile_templates(altered, ALL_CHANNELS))
    for family, element in ((TemplateFamily.PIPE_ENERGY, pipe), (TemplateFamily.PUMP_HEAD, pump), (TemplateFamily.VALVE_DISPATCH, valve)):
        keys = {s.key for s in templates[(family, element.id)].required_channels}
        assert "level:T1" in keys and "pressure:T1" not in keys
    assert {"status:" + pump.id, "speed:" + pump.id} <= {s.key for s in templates[(TemplateFamily.PUMP_HEAD, pump.id)].required_channels}
    assert {"status:" + valve.id, "setting:" + valve.id} <= {s.key for s in templates[(TemplateFamily.VALVE_DISPATCH, valve.id)].required_channels}


def test_one_point_pump_is_compilable_and_current_speed_required(network):
    pump = dataclasses.replace(network.pumps[0], curve=dataclasses.replace(network.pumps[0].curve, points=((0.1, 60),)))
    altered = dataclasses.replace(network, pumps=(pump,))
    template = by_key(compile_templates(altered, ALL_CHANNELS))[(TemplateFamily.PUMP_HEAD, pump.id)]
    assert template.status is TemplateStatus.COMPILED
    template = by_key(compile_templates(altered, ALL_CHANNELS - {"speed:" + pump.id}))[(TemplateFamily.PUMP_HEAD, pump.id)]
    assert template.status is TemplateStatus.UNOBSERVABLE


def test_pbv_template_is_unsupported_even_with_all_channels(network):
    pbv = by_key(compile_templates(network, ALL_CHANNELS))[(TemplateFamily.VALVE_DISPATCH, "V3")]
    assert pbv.status is TemplateStatus.UNSUPPORTED
    assert "active branch" in pbv.reason
