"""Parity between the optimised detector and the slow reference implementation.

``tests/stl/reference_topology_detector.py`` is a verbatim copy of the
pre-optimisation detector.  The optimised detector must reproduce it exactly:
same residuals (bit for bit), same regimes, same units, same robustness and
the same reason strings, for every family, every actuator status and every
kind of missing or unusable data.
"""

import json

import numpy as np
import pandas as pd
import pytest

from hydraulics.model import (
    DemandOptions,
    DemandTerm,
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
from stl.topology_detector import TopologyStlDetector

from .reference_topology_detector import ReferenceTopologyStlDetector

TIMES = (0.0, 3600.0, 7200.0)


class ParityNetwork:
    """A small network that exercises all five template families."""

    def __init__(self, demand_model: str = "DDA"):
        self.patterns = {
            "P1": Pattern("P1", (1.0, 0.5, 1.25), 3600.0, 0.0),
            "HP": Pattern("HP", (1.0, 1.1), 3600.0, 0.0),
        }
        self.reservoirs = (Reservoir("R", 30.0, "HP"),)
        if demand_model.upper() in ("PDD", "PDA"):
            junction = Junction("J1", 10.0, (DemandTerm("base", 5.0, "P1"),),
                                emitter_coefficient=2.0, pressure_exponent=0.5,
                                required_pressure=15.0, minimum_pressure=0.0)
        else:
            junction = Junction("J1", 10.0, (DemandTerm("base", 5.0, "P1"),),
                                emitter_coefficient=2.0)
        if demand_model.upper() in ("PDD", "PDA"):
            self.junctions = (junction, Junction("J2", 12.0, pressure_exponent=0.5,
                                                 required_pressure=15.0, minimum_pressure=0.0))
        else:
            self.junctions = (junction, Junction("J2", 12.0))
        self.tanks = (Tank("T", 20.0, 5.0, 0.0, 10.0, 3.0),)
        self.pipes = (
            Pipe("P1", "R", "J1", 100.0, 0.3, 100.0),
            Pipe("P2", "J1", "J2", 80.0, 0.25, 90.0),
            Pipe("P3", "J2", "T", 60.0, 0.2, 80.0),
        )
        self.pumps = (Pump("PU", "R", "J1", "HEAD", PumpCurve("C1", ((0.0, 40.0), (1.0, 20.0)))),)
        self.valves = (
            Valve("V1", "J2", "T", "PRV", 5.0, None, 0.3, 0.0),
            Valve("V2", "J1", "J2", "GPV", None, ((0.0, 0.0), (1.0, 2.0)), 0.25, 0.0),
            Valve("V3", "J2", "T", "FCV", 0.1, None, 0.2, 0.0),
            Valve("V4", "J1", "T", "TCV", 0.5, None, 0.2, 0.0),
        )
        self.controls = ()
        if demand_model.upper() in ("PDD", "PDA"):
            self.demand = DemandOptions(demand_model, 1.0, 0.5, 15.0, 0.0)
        else:
            self.demand = DemandOptions(demand_model)
        self.provenance = Provenance("", "", "", 0, "LPS", "H-W")
        self.time = TimeOptions(7200, 3600, 3600, 3600, 0, 1, 1, 0)

    # -- lookups used by the detector -----------------------------------
    def node_ids(self):
        return tuple(j.id for j in self.junctions) + tuple(r.id for r in self.reservoirs) + tuple(t.id for t in self.tanks)

    def link_ids(self):
        return tuple(link.id for link in self.links())

    def links(self):
        return tuple(self.pipes) + tuple(self.pumps) + tuple(self.valves)

    def incident_links(self, node):
        return tuple(link.id for link in self.links() if node in (link.start_node, link.end_node))

    def pattern(self, key):
        return self.patterns.get(key)

    def node_elevation(self, node_id):
        for junction in self.junctions:
            if junction.id == node_id:
                return junction.elevation
        for tank in self.tanks:
            if tank.id == node_id:
                return tank.elevation
        raise AttributeError(f"'Reservoir' object has no attribute 'elevation'")


def full_frame(pump_status=0, prv_status=1, gpv_status=1, fcv_status=0, tcv_status=1,
               pipe_status=1, index=TIMES, time_seconds=None):
    """A frame carrying every channel the compiled templates need."""
    n = len(index)
    data = {
        "J1": [40.0 + 0.5 * i for i in range(n)],
        "J2": [35.0 - 0.25 * i for i in range(n)],
        "T": [5.0 + 0.1 * i for i in range(n)],
        "f_P1": [0.4 + 0.01 * i for i in range(n)],
        "f_P2": [0.2 + 0.02 * i for i in range(n)],
        "f_P3": [0.2 + 0.02 * i for i in range(n)],
        "f_PU": [0.3 + 0.01 * i for i in range(n)],
        "f_V1": [0.05 * i for i in range(n)],
        "f_V2": [0.01 * i for i in range(n)],
        "f_V3": [0.02 * i for i in range(n)],
        "f_V4": [0.03 * i for i in range(n)],
        "s_P1": [pipe_status] * n,
        "s_P2": [pipe_status] * n,
        "s_P3": [pipe_status] * n,
        "s_PU": [pump_status] * n,
        "s_V1": [prv_status] * n,
        "s_V2": [gpv_status] * n,
        "s_V3": [fcv_status] * n,
        "s_V4": [tcv_status] * n,
        "speed_PU": [1.0] * n,
        "setting_V1": [5.0] * n,
        "setting_V3": [0.1] * n,
        "setting_V4": [0.5] * n,
    }
    frame = pd.DataFrame(data, index=pd.Index(list(index), dtype=float))
    if time_seconds is not None:
        frame = frame.assign(time_seconds=list(time_seconds))
    return frame


def _detectors(network, frames):
    """Return the optimised and the reference detector fitted on *frames*."""
    fast = TopologyStlDetector(network).fit(frames)
    slow = ReferenceTopologyStlDetector(network).fit(frames)
    return fast, slow


def _assert_identical(fast, slow, frame):
    fast_rows = fast.predict_details(frame)
    slow_rows = slow.predict_details(frame)
    assert len(fast_rows) == len(slow_rows)
    for index, (fast_row, slow_row) in enumerate(zip(fast_rows, slow_rows)):
        assert list(fast_row) == list(slow_row), f"key order differs at sample {index}"
        for key in slow_row:
            fast_value = fast_row[key]
            slow_value = slow_row[key]
            assert fast_value == slow_value, f"{key} differs at sample {index}"
            if isinstance(fast_value.get("residual"), float):
                assert repr(fast_value["residual"]) == repr(slow_value["residual"])
            assert fast_value.get("reason") == slow_value.get("reason")
    assert fast.to_json() == slow.to_json()
    assert json.dumps(fast.to_artifact(), sort_keys=True) == json.dumps(slow.to_artifact(), sort_keys=True)


def _assert_same_prediction_failure(fast, slow, frame):
    """Both detectors must fail (or succeed) in exactly the same way."""
    fast_error = slow_error = None
    try:
        fast.predict(frame)
    except Exception as exc:  # noqa: BLE001 - the two must fail the same way
        fast_error = exc
    try:
        slow.predict(frame)
    except Exception as exc:  # noqa: BLE001
        slow_error = exc
    assert type(fast_error) is type(slow_error)
    assert str(fast_error) == str(slow_error)


def test_all_five_families_are_compiled_and_evaluated():
    network = ParityNetwork()
    frame = full_frame()
    fast, slow = _detectors(network, [frame])
    assert set(fast.formulas) == {
        "junction_continuity:J1", "junction_continuity:J2",
        "pipe_energy:P1", "pipe_energy:P2", "pipe_energy:P3",
        "tank_volume:T", "pump_head:PU",
        "valve_dispatch:V1", "valve_dispatch:V2", "valve_dispatch:V3", "valve_dispatch:V4",
    }
    _assert_identical(fast, slow, frame)


@pytest.mark.parametrize("demand_model", ["DDA", "PDD"])
def test_demand_models_agree(demand_model):
    network = ParityNetwork(demand_model)
    frame = full_frame()
    fast, slow = _detectors(network, [frame])
    _assert_identical(fast, slow, frame)


@pytest.mark.parametrize("pipe_status", [0, 1, 2])
def test_pipe_statuses_agree(pipe_status):
    network = ParityNetwork()
    fast, slow = _detectors(network, [full_frame()])
    frame = full_frame(pipe_status=pipe_status)
    _assert_identical(fast, slow, frame)
    _assert_same_prediction_failure(fast, slow, frame)


@pytest.mark.parametrize("pump_status", [0, 1, 2])
def test_pump_statuses_agree(pump_status):
    network = ParityNetwork()
    fast, slow = _detectors(network, [full_frame(pump_status=0)])
    _assert_identical(fast, slow, full_frame(pump_status=pump_status))
    _assert_same_prediction_failure(fast, slow, full_frame(pump_status=pump_status))


@pytest.mark.parametrize("prv_status,gpv_status,fcv_status,tcv_status", [
    (0, 1, 0, 1), (1, 1, 0, 1), (2, 1, 0, 1),
    (1, 0, 1, 0), (1, 2, 1, 0), (2, 2, 2, 2),
])
def test_valve_statuses_agree(prv_status, gpv_status, fcv_status, tcv_status):
    network = ParityNetwork()
    fast, slow = _detectors(network, [full_frame()])
    frame = full_frame(prv_status=prv_status, gpv_status=gpv_status,
                       fcv_status=fcv_status, tcv_status=tcv_status)
    _assert_identical(fast, slow, frame)
    _assert_same_prediction_failure(fast, slow, frame)


@pytest.mark.parametrize("column", [
    "J1", "J2", "T", "f_P1", "f_P2", "f_P3", "f_PU", "f_V1", "f_V2", "f_V3", "f_V4",
    "s_P1", "s_P2", "s_P3", "s_PU", "s_V1", "s_V2", "s_V3", "s_V4", "speed_PU",
    "setting_V1", "setting_V3", "setting_V4",
])
def test_missing_channel_agrees(column):
    network = ParityNetwork()
    frame = full_frame().drop(columns=column)
    fast = TopologyStlDetector(network)
    slow = ReferenceTopologyStlDetector(network)
    fast_error = slow_error = None
    try:
        fast.fit([frame])
    except Exception as exc:  # noqa: BLE001 - the two must fail the same way
        fast_error = exc
    try:
        slow.fit([frame])
    except Exception as exc:  # noqa: BLE001
        slow_error = exc
    assert type(fast_error) is type(slow_error)
    assert str(fast_error) == str(slow_error)
    if fast_error is None:
        _assert_identical(fast, slow, frame)


def test_missing_channel_at_predict_time_agrees():
    network = ParityNetwork()
    frame = full_frame()
    fast, slow = _detectors(network, [frame])
    broken = frame.drop(columns="f_P2")
    fast_rows = fast.predict_details(broken)
    slow_rows = slow.predict_details(broken)
    for fast_row, slow_row in zip(fast_rows, slow_rows):
        assert fast_row == slow_row


def test_non_finite_channel_agrees():
    network = ParityNetwork()
    frame = full_frame()
    fast, slow = _detectors(network, [frame])
    broken = frame.copy()
    broken.loc[broken.index[1], "f_P2"] = np.inf
    fast_rows = fast.predict_details(broken)
    slow_rows = slow.predict_details(broken)
    for fast_row, slow_row in zip(fast_rows, slow_rows):
        assert fast_row == slow_row


def test_initial_sample_and_prefix_causality_agree():
    network = ParityNetwork()
    frame = full_frame()
    fast, slow = _detectors(network, [frame])
    assert fast.predict_details(frame)[0]["tank_volume:T"]["regime"] == "pending"
    _assert_identical(fast, slow, frame)
    _assert_identical(fast, slow, frame.iloc[:2])


def test_time_seconds_column_agrees():
    network = ParityNetwork()
    frame = full_frame(time_seconds=[0.0, 1800.0, 3600.0])
    fast, slow = _detectors(network, [frame])
    _assert_identical(fast, slow, frame)


def test_predictions_and_artifact_agree():
    network = ParityNetwork()
    frame = full_frame()
    fast, slow = _detectors(network, [frame])
    fast_flags = fast.predict(frame)
    slow_flags = slow.predict(frame)
    assert np.array_equal(fast_flags.flags, slow_flags.flags)
    assert np.array_equal(fast_flags.scores, slow_flags.scores)
    assert fast.to_artifact() == slow.to_artifact()
