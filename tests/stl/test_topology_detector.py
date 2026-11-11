from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hydraulics.model import DemandOptions, DemandTerm, Junction, Pipe, Provenance, Pump, PumpCurve, Reservoir, Tank, TimeOptions
from stl.topology_detector import TopologyStlDetector


class Network:
    def __init__(self):
        self.junctions = (Junction("J", 0.0),)
        self.reservoirs = (Reservoir("R", 10.0),)
        self.tanks = (Tank("T", 0.0, 1.0, 0.0, 10.0, 2.0),)
        self.pipes = (Pipe("P", "R", "J", 1.0, 1.0, 100.0), Pipe("Q", "J", "T", 1.0, 1.0, 100.0))
        self.pumps = self.valves = self.controls = ()
        self.patterns = {}
        self.demand = DemandOptions("DDA")
        self.provenance = Provenance("", "", "", 0, "LPS", "H-W")
        self.time = TimeOptions(7200, 3600, 3600, 3600, 0, 1, 1, 0)

    def node_ids(self): return ("J", "R", "T")
    def link_ids(self): return tuple(link.id for link in self.links())
    def links(self): return self.pipes + self.pumps + self.valves
    def incident_links(self, node): return tuple(p.id for p in self.links() if node in (p.start_node, p.end_node))
    def pattern(self, key): return self.patterns.get(key)


def balanced(index=(0., 3600., 7200.)):
    n = len(index)
    # R->J supplies 1 m3/s; J->T has 1 m3/s. Tank rises by 3600 s * 1 m3/s.
    return pd.DataFrame({"J": [10.] * n, "T": [1. + i * 3600 / np.pi for i in range(n)],
                         "f_P": [1.] * n, "f_Q": [1.] * n,
                         "s_P": [1] * n, "s_Q": [1] * n}, index=index)


def test_analytical_continuity_and_tank_balance_and_prefix_causality():
    frame = balanced()
    detector = TopologyStlDetector(Network()).fit([frame])
    details = detector.predict_details(frame)
    assert details[1]["junction_continuity:J"]["residual"] == pytest.approx(0)
    assert details[0]["tank_volume:T"]["regime"] == "pending"
    assert details[1]["tank_volume:T"]["residual"] == pytest.approx(0)
    assert details[:2] == detector.predict_details(frame.iloc[:2])
    pred = detector.predict(frame)
    assert np.array_equal(pred.flags, pred.scores > 0)


def test_time_seconds_overrides_index_and_is_validated():
    frame = balanced((0, 1, 2)).assign(time_seconds=[0., 3600., 7200.])
    det = TopologyStlDetector(Network()).fit([frame])
    assert det.predict_details(frame)[1]["tank_volume:T"]["residual"] == pytest.approx(0)
    with pytest.raises(ValueError, match="strictly increasing"):
        det.predict_details(frame.assign(time_seconds=[0., 1., 1.]))
    with pytest.raises(ValueError, match="numeric seconds"):
        det.predict_details(frame.assign(time_seconds=["a", "b", "c"]))


def test_previous_sample_flow_is_used_after_switch():
    frame = balanced((0., 3600.)).copy()
    frame.loc[3600., "f_Q"] = 0.0
    # End level reflects the prior sample's 1 m3/s inflow, not the new zero flow.
    det = TopologyStlDetector(Network()).fit([balanced((0., 3600.))])
    result = det.predict_details(frame)[1]["tank_volume:T"]
    assert result["residual"] == pytest.approx(0)


def test_additive_units_empty_fit_nonfinite_and_missing_are_explicit():
    with pytest.raises(ValueError, match="by residual unit"):
        TopologyStlDetector(Network(), {"additive": .1})
    with pytest.raises(ValueError, match="finite"):
        TopologyStlDetector(Network(), {"percentile": float("nan")})
    with pytest.raises(ValueError, match="nonempty"):
        TopologyStlDetector(Network()).fit([])
    det = TopologyStlDetector(Network()).fit([balanced()])
    missing = det.predict_details(balanced().drop(columns="f_P"))[1]["junction_continuity:J"]
    assert missing["regime"] == "missing"
    with pytest.raises(ValueError, match="unknown residuals"):
        det.predict(balanced().drop(columns="f_P"))


def test_artifact_is_deterministic_and_tolerances_include_regime_unit():
    det = TopologyStlDetector(Network(), {"additive": {"m^3": .01}}).fit([balanced()])
    artifact = det.to_artifact()
    assert artifact["schema"] == "topology-stl-v2"
    assert "tank_volume:T|interval|m^3" in artifact["tolerances"]
    assert det.to_json() == det.to_json()


def test_solver_delivered_demand_leak_and_emitter_channels_are_separate():
    network = Network()
    network.junctions = (Junction("J", 0., (DemandTerm("base", 9.),), emitter_coefficient=10.),)
    frame = balanced().assign(f_P=.8, f_Q=0., demand_J=.5, leak_J=.2, emitter_J=.1)
    det = TopologyStlDetector(network).fit([frame])
    assert det.predict_details(frame)[1]["junction_continuity:J"]["residual"] == pytest.approx(0)


@pytest.mark.parametrize("value,expected", [(0,"CLOSED"),(1,"OPEN"),(2,"ACTIVE"),("0","CLOSED"),("OPEN","OPEN"),("active","ACTIVE")])
def test_solver_status_codes_and_names(value, expected):
    assert TopologyStlDetector._status(value) == expected


@pytest.mark.parametrize("value", [3, -1, "3", "invalid", float("nan")])
def test_invalid_solver_status_is_rejected(value):
    with pytest.raises(ValueError, match="invalid status"):
        TopologyStlDetector._status(value)


def test_closed_pipe_checks_zero_flow_instead_of_energy():
    frame = balanced().assign(f_P=0.0, s_P=0)
    det = TopologyStlDetector(Network()).fit([frame])
    result = det.predict_details(frame)[1]["pipe_energy:P"]
    assert result["regime"] == "closed"
    assert result["unit"] == "m^3/s"
    assert result["residual"] == 0.0
    assert result["robustness"] > 0
    assert "pipe_energy:P|closed|m^3/s" in det.tolerances
    det.predict(frame)


def test_closed_pump_uses_flow_units_without_speed_and_open_regime_is_uncalibrated():
    network = Network()
    network.pumps = (Pump("PU", "R", "J", "HEAD", PumpCurve("C", ((0., 10.), (1., 5.)))),)
    frame = balanced().assign(f_P=.8, f_PU=.2, s_PU=0)
    det = TopologyStlDetector(network).fit([frame])
    closed = det.predict_details(frame)[1]["pump_head:PU"]
    assert closed["residual"] == pytest.approx(.2)
    assert closed["unit"] == "m^3/s"
    assert "pump_head:PU|closed|m^3/s" in det.tolerances
    opened = frame.assign(s_PU="OPEN", speed_PU=1.)
    result = det.predict_details(opened)[1]["pump_head:PU"]
    assert result["regime"] == "uncalibrated"
    with pytest.raises(ValueError, match="unknown residuals"):
        det.predict(opened)
    reporting = TopologyStlDetector(network, {"unknown_policy": "report"}).fit([frame])
    result = reporting.predict(opened)
    assert result.flags.shape == (3,)
    assert len(reporting.last_unknown) == 3
    assert reporting.last_unknown[0]["regime"] == "uncalibrated"
