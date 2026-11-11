"""Tests for the topology/channel binding and the timing plan."""

from __future__ import annotations

from pathlib import Path

import pytest

from hydraulics import TemplateFamily, TemplateStatus, load_network
from stl.topology import (
    TimingPlan,
    build_binding,
    channel_column,
    flow_column,
    level_column,
    pressure_column,
    status_column,
)

pytest.importorskip("wntr", reason="the INP loader needs WNTR")

FIXTURE = Path(__file__).resolve().parents[1] / "hydraulics" / "fixtures" / "synthetic.inp"

DATASET_COLUMNS = [
    "J1",
    "J2",
    "J3",
    "T1",
    "f_P1",
    "f_P2",
    "f_P3",
    "f_PU1",
    "f_V1",
    "s_P1",
    "s_P2",
    "s_P3",
    "s_PU1",
    "s_V1",
]


@pytest.fixture(scope="module")
def network():
    return load_network(FIXTURE)


def test_column_names_follow_the_dataset_convention():
    from hydraulics.model import ChannelSpec
    assert flow_column("P1") == "f_P1"
    assert pressure_column("J1") == "J1"
    assert level_column("T1") == "T1"
    assert status_column("PU1") == "s_PU1"
    assert channel_column(ChannelSpec("speed", "PU1")) == "speed_PU1"
    assert channel_column(ChannelSpec("setting", "V1")) == "setting_V1"
    assert channel_column.__module__ == "stl.topology"


def test_binding_maps_elements_onto_the_dataset_columns(network):
    binding = build_binding(network, DATASET_COLUMNS)
    assert binding.tanks["T1"].get("level") == "T1"
    assert binding.junctions["J1"].get("pressure") == "J1"
    assert binding.pumps["PU1"].get("flow") == "f_PU1"
    assert binding.pumps["PU1"].get("status") == "s_PU1"
    assert binding.valves["V1"].get("flow") == "f_V1"
    assert binding.links["P1"].get("flow") == "f_P1"


def test_binding_reports_the_columns_that_are_missing(network):
    binding = build_binding(network, DATASET_COLUMNS)
    assert "flow:PU3" in binding.missing
    assert "flow:PU2" in binding.missing
    assert "flow:P1" not in binding.missing
    assert "flow:P3" not in binding.missing
    # The reservoir head comes from the network, not from a channel.
    assert not any(key.endswith(":R1") for key in binding.missing)


def test_binding_drives_the_template_compilation(network):
    binding = build_binding(network, DATASET_COLUMNS)
    compiled = {
        (template.family, template.key): template
        for template in binding.templates
    }
    pipe = compiled[(TemplateFamily.PIPE_ENERGY, "P1")]
    assert pipe.status is TemplateStatus.COMPILED
    missing_valve = compiled[(TemplateFamily.VALVE_DISPATCH, "V2")]
    assert missing_valve.status is TemplateStatus.UNOBSERVABLE
    assert "flow:V2" in missing_valve.reason
    # The reservoir head is part of the contract, so P3 needs no pressure
    # channel for R1.
    pipe_to_reservoir = compiled[(TemplateFamily.PIPE_ENERGY, "P3")]
    assert pipe_to_reservoir.status is TemplateStatus.COMPILED


def test_binding_is_deterministic(network):
    first = build_binding(network, DATASET_COLUMNS)
    second = build_binding(network, DATASET_COLUMNS)
    assert first.templates == second.templates
    assert first.missing == second.missing
    assert first.available_channel_keys() == second.available_channel_keys()


def test_timing_plan_keeps_interval_and_horizon_apart():
    plan = TimingPlan(sample_interval_s=3600.0, stl_horizon_s=86400.0)
    assert plan.sample_interval_s == 3600.0
    assert plan.stl_horizon_s == 86400.0
    assert plan.horizon_samples == 24
    # 24 intervals require 25 observations.
    assert plan.horizon_is_complete(22) is False
    assert plan.horizon_is_complete(23) is False
    assert plan.horizon_is_complete(24) is True


def test_timing_plan_rejects_an_inconsistent_horizon():
    with pytest.raises(ValueError):
        TimingPlan(sample_interval_s=3600.0, stl_horizon_s=5000.0)
    with pytest.raises(ValueError):
        TimingPlan(sample_interval_s=0.0, stl_horizon_s=3600.0)
    with pytest.raises(ValueError):
        TimingPlan(sample_interval_s=3600.0, stl_horizon_s=0.0)
