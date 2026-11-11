"""Contract-level tests for the INP loader (hand-checked values)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from hydraulics import InpParseError, load_network

pytest.importorskip("wntr", reason="the INP loader needs WNTR")

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic.inp"


@pytest.fixture(scope="module")
def network():
    return load_network(FIXTURE)


def test_provenance_records_the_inp_file(network):
    raw = FIXTURE.read_bytes()
    assert network.provenance.sha256 == hashlib.sha256(raw).hexdigest()
    assert network.provenance.size_bytes == len(raw)
    assert network.provenance.inp_filename == "synthetic.inp"
    assert network.provenance.flow_units == "LPS"
    assert network.provenance.headloss_formula == "H-W"


def test_si_conversion_of_the_inp_file(network):
    # The INP file is written in LPS with pipe diameters in millimetres.
    pipe = {pipe.id: pipe for pipe in network.pipes}["P1"]
    assert pipe.diameter == pytest.approx(0.3)
    assert pipe.length == pytest.approx(500.0)

    junction = {j.id: j for j in network.junctions}["J1"]
    demands = {term.category: term for term in junction.demands}
    assert demands["irrigation"].base_demand == pytest.approx(0.004)
    assert demands["domestic"].base_demand == pytest.approx(0.002)

    # Emitter coefficient: 0.002 LPS/sqrt(m) -> 2e-6 m^3/s/sqrt(m).
    emitter_junction = {j.id: j for j in network.junctions}["J3"]
    assert emitter_junction.emitter_coefficient == pytest.approx(2e-6)

    # Flow setting of the FCV: 1.0 LPS -> 0.001 m^3/s.
    valve = {valve.id: valve for valve in network.valves}["V4"]
    assert valve.setting == pytest.approx(0.001)

    # Pump curve: 100 LPS -> 0.1 m^3/s.
    pump = {pump.id: pump for pump in network.pumps}["PU1"]
    assert pump.curve is not None
    assert pump.curve.points == ((0.0, 80.0), (0.1, 60.0), (0.2, 40.0))

    # Tank diameters are metres, not millimetres.
    tank = {tank.id: tank for tank in network.tanks}["T1"]
    assert tank.diameter == pytest.approx(12.0)


def test_directed_node_and_link_ids_are_preserved(network):
    pipe = {pipe.id: pipe for pipe in network.pipes}["P1"]
    assert (pipe.start_node, pipe.end_node) == ("J1", "J2")
    assert network.incident_links("J1") == (
        "P1",
        "PU1",
        "PU2",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    )
    assert network.node_elevation("J2") == pytest.approx(105.0)
    assert network.node_elevation("T1") == pytest.approx(120.0)


def test_delivered_demand_categories_and_patterns(network):
    junction = {j.id: j for j in network.junctions}["J1"]
    assert [(term.category, term.pattern_id) for term in junction.demands] == [
        ("irrigation", "2"),
        ("domestic", "1"),
    ]
    assert junction.base_demand == pytest.approx(0.006)


def test_pattern_multipliers_wrap(network):
    pattern = network.patterns["1"]
    assert pattern.multipliers == (1.0, 1.2, 0.8)
    assert pattern.multiplier_at(0.0) == pytest.approx(1.0)
    assert pattern.multiplier_at(3600.0) == pytest.approx(1.2)
    assert pattern.multiplier_at(7200.0) == pytest.approx(0.8)
    # The pattern repeats after its last multiplier.
    assert pattern.multiplier_at(10800.0) == pytest.approx(1.0)


def test_reservoir_pattern_and_tank_geometry(network):
    reservoir = {r.id: r for r in network.reservoirs}["R1"]
    assert reservoir.base_head == pytest.approx(50.0)
    assert reservoir.head_pattern_id == "1"

    tank = {tank.id: tank for tank in network.tanks}["T1"]
    assert tank.init_level == pytest.approx(3.0)
    assert tank.min_level == pytest.approx(0.0)
    assert tank.max_level == pytest.approx(10.0)
    assert tank.volume_curve == ((0.0, 0.0), (5.0, 900.0), (10.0, 1900.0))


def test_pump_curves_speed_and_status(network):
    pumps = {pump.id: pump for pump in network.pumps}
    assert pumps["PU1"].pump_type == "HEAD"
    assert pumps["PU1"].speed == pytest.approx(1.0)
    assert pumps["PU1"].status == "OPEN"
    assert pumps["PU3"].speed == pytest.approx(1.5)
    assert pumps["PU2"].pump_type == "POWER"
    assert pumps["PU2"].curve is None
    assert pumps["PU2"].power == pytest.approx(50000.0)


def test_valve_types_settings_and_status(network):
    valves = {valve.id: valve for valve in network.valves}
    assert valves["V1"].valve_type == "PRV"
    assert valves["V1"].setting == pytest.approx(45.0)
    assert valves["V2"].valve_type == "PSV"
    assert valves["V3"].valve_type == "PBV"
    assert valves["V4"].valve_type == "FCV"
    assert valves["V5"].valve_type == "TCV"
    assert valves["V5"].setting == pytest.approx(0.5)
    assert valves["V6"].valve_type == "GPV"
    assert valves["V6"].setting is None
    assert valves["V6"].headloss_curve == ((0.0, 0.0), (0.01, 5.0))
    assert valves["V1"].status == "ACTIVE"


def test_controls_are_preserved(network):
    assert len(network.controls) == 1
    control = network.controls[0]
    assert control.link_id == "PU1"
    assert control.control_type == "OPEN"
    assert control.sensor_node == "J1"
    assert control.sensor_attribute == "pressure"
    assert control.threshold == pytest.approx(50.0)


def test_time_and_demand_options(network):
    assert network.time.duration_s == pytest.approx(86400.0)
    assert network.time.hydraulic_timestep_s == pytest.approx(1800.0)
    assert network.time.report_timestep_s == pytest.approx(3600.0)
    assert network.time.pattern_timestep_s == pytest.approx(3600.0)
    assert network.demand.model == "DDA"
    assert network.demand.emitter_exponent == pytest.approx(0.5)


def test_missing_inp_file_is_reported(tmp_path):
    with pytest.raises(InpParseError):
        load_network(tmp_path / "does-not-exist.inp")


def test_pattern_start_is_added():
    from hydraulics.model import Pattern
    pattern = Pattern("p", (1, 2, 3), timestep_s=3600, start_s=3600)
    assert pattern.multiplier_at(0) == 2
    assert pattern.multiplier_at(7200) == 1


def test_zero_values_and_numeric_reservoir_pattern(network):
    import wntr
    from hydraulics.loader import _network_from_wn
    wn = wntr.network.WaterNetworkModel(str(FIXTURE))
    wn.options.hydraulic.demand_multiplier = 0
    wn.get_link("PU1").speed_timeseries.base_value = 0
    loaded = _network_from_wn(wn, network.provenance)
    assert loaded.demand.multiplier == 0
    assert next(p for p in loaded.pumps if p.id == "PU1").speed == 0
    reservoir = next(r for r in loaded.reservoirs if r.id == "R1")
    assert reservoir.head_pattern_id == "1" and reservoir.demand == 0


def test_loaded_pdd_alias_is_pressure_dependent_at_zero_pressure(tmp_path):
    from stl.residuals import delivered_demand

    inp = tmp_path / "pdd.inp"
    inp.write_text(FIXTURE.read_text().replace(
        " Units        LPS", " Units        LPS\n Demand Model PDD\n Minimum Pressure 0\n Required Pressure 30"
    ))
    network = load_network(inp)
    junction = next(j for j in network.junctions if j.id == "J1")
    assert network.demand.model == "PDA"
    assert delivered_demand(
        [(term.base_demand, 1.0) for term in junction.demands],
        model=network.demand.model, pressure_head=0,
        minimum_pressure=network.demand.minimum_pressure,
        required_pressure=network.demand.required_pressure,
    ) == 0


def test_loader_records_requested_simulator_backend(network):
    assert network.provenance.simulator_backend == "wntr"
    assert load_network(FIXTURE, simulator_backend="epanet").provenance.simulator_backend == "epanet"
    with pytest.raises(ValueError):
        load_network(FIXTURE, simulator_backend="other")
