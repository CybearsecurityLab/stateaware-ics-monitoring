"""Slow reference implementation of the topology-driven residual detector.

This module is a verbatim copy of the pre-optimisation implementation of
``stl.topology_detector.TopologyStlDetector``.  It is deliberately *not*
optimised: the regression tests use it as the oracle that the optimised
detector must reproduce exactly (residuals, regimes, units, robustness and
reason strings).

Original docstring follows.

Topology-driven residual detector.

Time comes from a finite, strictly increasing ``time_seconds`` column when
present; otherwise the index must be numeric seconds. Flow channels ``f_ID``
are instantaneous rates in m^3/s. Optional ``volume_ID`` channels are signed
interval-integrated volumes in m^3 (positive along link orientation).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from hydraulics.model import ChannelSpec
from hydraulics.templates import TemplateStatus, compile_templates
from monitor.protocol import StepFlags
from stl import residuals
from stl.topology import channel_column


@dataclass(frozen=True)
class FormulaResult:
    key: str
    family: str
    required_channels: tuple[str, ...]
    unit: str
    status: str
    reason: str = ""


@dataclass(frozen=True)
class Tolerance:
    epsilon: float
    unit: str
    percentile: float
    additive: float


class ReferenceTopologyStlDetector:
    name = "topology_stl"

    def __init__(self, network, config: Mapping[str, Any] | None = None):
        self.network = network
        self.config = dict(config or {})
        self.percentile = float(self.config.get("percentile", 99.0))
        raw = self.config.get("additive", 0.0)
        if isinstance(raw, Mapping):
            self.additives = {str(k): float(v) for k, v in raw.items()}
        else:
            scalar = float(raw)
            if scalar != 0:
                raise ValueError("nonzero additive margins must be configured by residual unit")
            self.additives = {"*": scalar}
        if not math.isfinite(self.percentile) or not 0 <= self.percentile <= 100:
            raise ValueError("percentile must be finite and in [0,100]")
        if any(not math.isfinite(v) or v < 0 for v in self.additives.values()):
            raise ValueError("additive margins must be finite and nonnegative")
        self.closed_flow_epsilon = float(self.config.get("closed_flow_epsilon", 1e-6))
        if not math.isfinite(self.closed_flow_epsilon) or self.closed_flow_epsilon < 0:
            raise ValueError("closed_flow_epsilon must be finite and nonnegative")
        self.unknown_policy = str(self.config.get("unknown_policy", "raise"))
        if self.unknown_policy not in ("raise", "report"):
            raise ValueError("unknown_policy must be 'raise' or 'report'")
        self.last_unknown: list[dict[str, Any]] = []
        self.templates = ()
        self.formulas: dict[str, FormulaResult] = {}
        self.tolerances: dict[str, Tolerance] = {}
        self._fitted = False

    @staticmethod
    def _times(frame):
        values = frame["time_seconds"].to_numpy() if "time_seconds" in frame else frame.index.to_numpy()
        try: times = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc: raise ValueError("time must be numeric seconds") from exc
        if not np.isfinite(times).all() or (len(times) > 1 and np.any(np.diff(times) <= 0)):
            raise ValueError("time must be finite and strictly increasing seconds")
        return times

    def _compile(self, frames):
        cols = set().union(*(set(f.columns) for f in frames)) if frames else set()
        available = [ChannelSpec(k, e).key for k in ("flow", "pressure", "level", "status", "speed", "setting")
                     for e in self.network.node_ids() + self.network.link_ids() if channel_column(ChannelSpec(k, e)) in cols]
        # Actuator requirements depend on the observed state.  Defer these
        # channels to runtime so a closed unit need not carry open-state inputs.
        available.extend(ChannelSpec("speed", p.id).key for p in self.network.pumps)
        available.extend(ChannelSpec("setting", v.id).key for v in self.network.valves)
        node_ids=set(self.network.node_ids())
        available.extend(ChannelSpec("pressure", n).key for n in node_ids)
        available.extend(ChannelSpec("level", n).key for n in node_ids)
        self.templates = compile_templates(self.network, available)
        units = {"junction_continuity":"m^3/s", "pipe_energy":"m", "tank_volume":"m^3", "pump_head":"m", "valve_dispatch":"m"}
        self.formulas = {f"{t.family.value}:{t.key}": FormulaResult(t.key,t.family.value,tuple(s.key for s in t.required_channels),units[t.family.value],t.status.value,t.reason) for t in self.templates}

    @staticmethod
    def _number(row, col):
        if col not in row.index: raise KeyError(f"missing channel {col}")
        value = float(row[col])
        if not math.isfinite(value): raise ValueError(f"non-finite channel {col}")
        return value

    @staticmethod
    def _status(value):
        names = {0:"CLOSED", 1:"OPEN", 2:"ACTIVE"}
        if isinstance(value, str):
            text=value.strip().upper()
            if text in names.values(): return text
            try: value=float(text)
            except ValueError: raise ValueError(f"invalid status {value!r}")
        try: number=float(value)
        except (TypeError, ValueError): raise ValueError(f"invalid status {value!r}")
        if not math.isfinite(number) or number not in names: raise ValueError(f"invalid status code {value!r}")
        return names[int(number)]

    def _head(self, node, row, time):
        for x in self.network.reservoirs:
            if x.id == node:
                p=self.network.pattern(x.head_pattern_id); return x.base_head*(p.multiplier_at(time) if p else 1.)
        for x in self.network.tanks:
            if x.id == node: return x.elevation+self._number(row, channel_column(ChannelSpec("level",node)))
        j=next(x for x in self.network.junctions if x.id==node)
        return residuals.node_head(j.elevation,self._number(row,channel_column(ChannelSpec("pressure",node))))

    def _evaluate(self, row, time, idx):
        out={}
        for t in self.templates:
            key=f"{t.family.value}:{t.key}"; base=self.formulas[key]
            try:
                if t.status is not TemplateStatus.COMPILED:
                    if "missing channel" in t.reason: raise KeyError(t.reason)
                    out[key]={**asdict(base),"residual":None,"epsilon":None,"robustness":None,"regime":t.status.value,"reason":t.reason}; continue
                fam=t.family.value
                group={"junction_continuity":self.network.junctions,"pipe_energy":self.network.pipes,"tank_volume":self.network.tanks,"pump_head":self.network.pumps,"valve_dispatch":self.network.valves}[fam]
                element=next(x for x in group if x.id==t.key); regime="observed"
                if fam=="junction_continuity":
                    incoming=[]; outgoing=[]
                    for lid in self.network.incident_links(element.id):
                        link=next(x for x in self.network.links() if x.id==lid); q=self._number(row,channel_column(ChannelSpec("flow",lid)))
                        (outgoing if link.start_node==element.id else incoming).append(q)
                    terms=[]
                    for d in element.demands:
                        p=self.network.pattern(d.pattern_id); terms.append((d.base_demand,p.multiplier_at(time) if p else 1.))
                    has_demand=f"demand_{element.id}" in row.index
                    has_leak=f"leak_{element.id}" in row.index
                    needs_pressure=(not has_demand and self.network.demand.model.upper() in ("PDD","PDA")) or (not has_leak and element.emitter_coefficient>0)
                    pressure=self._number(row,element.id) if needs_pressure else 0.
                    # Solver exports, when available, account for delivered demand and emitters explicitly.
                    demand=self._number(row,f"demand_{element.id}") if has_demand else residuals.delivered_demand(terms,self.network.demand.multiplier,self.network.demand.model,pressure,element.pressure_exponent,element.required_pressure,element.minimum_pressure)
                    leak=self._number(row,f"leak_{element.id}") if has_leak else residuals.emitter_withdrawal(element.emitter_coefficient,pressure,element.emitter_exponent)
                    emit=self._number(row,f"emitter_{element.id}") if f"emitter_{element.id}" in row.index else 0.
                    val=residuals.junction_continuity_residual(inflows=incoming,outflows=outgoing,delivered_demand=demand,emitter_withdrawals=(leak,emit))
                elif fam=="pipe_energy":
                    status=self._status(row[channel_column(ChannelSpec("status",element.id))])
                    if status=="CLOSED":
                        val=self._number(row,channel_column(ChannelSpec("flow",element.id)))
                        regime="closed"
                    else:
                        val=residuals.pipe_energy_residual(flow=self._number(row,channel_column(ChannelSpec("flow",element.id))),length=element.length,diameter=element.diameter,roughness=element.roughness,head_start=self._head(element.start_node,row,time),head_end=self._head(element.end_node,row,time),minor_loss=element.minor_loss)
                elif fam=="tank_volume":
                    if idx==0: raise LookupError("initial sample has no preceding interval")
                    dt=time-self._times_current[idx-1]; net=0.
                    for lid in self.network.incident_links(element.id):
                        link=next(x for x in self.network.links() if x.id==lid)
                        q=self._number(row,f"volume_{lid}") if f"volume_{lid}" in row.index else self._number(self._previous_row,channel_column(ChannelSpec("flow",lid)))*dt
                        net += q if link.end_node==element.id else -q
                    val=residuals.tank_volume_residual(level_start=self._number(self._previous_row,element.id),level_end=self._number(row,element.id),net_inflow=net,dt_s=1.,diameter=element.diameter,min_volume=element.min_volume,min_level=element.min_level,volume_curve=element.volume_curve); regime="interval"
                elif fam=="pump_head":
                    status=self._status(row[channel_column(ChannelSpec("status",element.id))]); q=self._number(row,channel_column(ChannelSpec("flow",element.id)))
                    speed=self._number(row,channel_column(ChannelSpec("speed",element.id))) if status=="OPEN" else None
                    d=residuals.pump_dispatch(flow=q,status=status,speed=speed,curve_points=element.curve.points if element.curve else None,head_start=self._head(element.start_node,row,time) if status=="OPEN" else None,head_end=self._head(element.end_node,row,time) if status=="OPEN" else None,backend=self.network.provenance.simulator_backend); val=d.residual; regime=d.regime
                    if val is None: raise KeyError(d.reason)
                else:
                    status=self._status(row[channel_column(ChannelSpec("status",element.id))]); q=self._number(row,channel_column(ChannelSpec("flow",element.id)))
                    kind=element.valve_type.upper(); setting=self._number(row,channel_column(ChannelSpec("setting",element.id))) if kind!="GPV" and status=="ACTIVE" else element.setting
                    needs_head=status=="OPEN" or (status=="ACTIVE" and kind in ("PRV","PSV","TCV","GPV"))
                    hs=self._head(element.start_node,row,time) if needs_head else None; he=self._head(element.end_node,row,time) if needs_head else None
                    active_pressure_valve=status=="ACTIVE" and kind in ("PRV","PSV")
                    d=residuals.valve_dispatch(valve_type=kind,flow=q,setting=setting,diameter=element.diameter,minor_loss=element.minor_loss,headloss_curve=element.headloss_curve,head_start=hs,head_end=he,elevation_start=self.network.node_elevation(element.start_node) if active_pressure_valve else 0.,elevation_end=self.network.node_elevation(element.end_node) if active_pressure_valve else 0.,status=status); val=d.residual; regime=d.regime
                    if val is None: raise KeyError(d.reason)
                val=float(val)
                if not math.isfinite(val): raise ValueError("computed residual is non-finite")
            except KeyError as e: out[key]={**asdict(base),"residual":None,"epsilon":None,"robustness":None,"regime":"missing","reason":str(e)}
            except LookupError as e: out[key]={**asdict(base),"residual":None,"epsilon":None,"robustness":None,"regime":"pending","reason":str(e)}
            except Exception as e: out[key]={**asdict(base),"residual":None,"epsilon":None,"robustness":None,"regime":"error","reason":f"{type(e).__name__}: {e}"}
            else:
                unit = d.unit if fam in ("pump_head", "valve_dispatch") else ("m^3/s" if fam=="pipe_energy" and regime=="closed" else base.unit)
                tk=self._tolerance_key(key,regime,unit); tol=self.tolerances.get(tk)
                if tol is None and regime=="closed":
                    tol=Tolerance(self.closed_flow_epsilon,unit,self.percentile,0.0)
                out[key]={**asdict(base),"unit":unit,"residual":val,"epsilon":tol.epsilon if tol else None,"robustness":None if tol is None else tol.epsilon-abs(val),"regime":regime if tol else "uncalibrated","operating_regime":regime,"reason":"" if tol else f"uncalibrated regime/unit: {regime}/{unit}"}
        return out

    @staticmethod
    def _tolerance_key(key, regime, unit): return f"{key}|{regime}|{unit}"

    def fit(self, nominal_training_frames):
        frames=list(nominal_training_frames)
        if not frames or not any(len(f) for f in frames): raise ValueError("fit requires nonempty nominal frames")
        self._compile(frames); samples={}; self.tolerances={}
        for frame in frames:
            if not len(frame): continue
            times=self._times(frame); self._times_current=times; self._previous_row=None
            for i,(_,row) in enumerate(frame.iterrows()):
                for key,d in self._evaluate(row,float(times[i]),i).items():
                    if d["regime"] in ("error", "missing"): raise ValueError(f"invalid calibration sample for {key}: {d['reason']}")
                    if d["residual"] is not None: samples.setdefault(self._tolerance_key(key,d.get("operating_regime",d["regime"]),d["unit"]),[]).append(abs(d["residual"]))
                self._previous_row=row
        for tk,values in samples.items():
            key,regime,unit=tk.rsplit("|",2); margin=self.additives.get(unit,self.additives.get("*",0.))
            epsilon=float(np.percentile(values,self.percentile))+margin
            if regime=="closed":
                epsilon=max(epsilon,self.closed_flow_epsilon)
            if not math.isfinite(epsilon): raise ValueError(f"non-finite tolerance for {tk}")
            self.tolerances[tk]=Tolerance(epsilon,unit,self.percentile,margin)
        if not self.tolerances: raise ValueError("fit produced no usable residuals")
        self._fitted=True; self._previous_row=None
        return self

    def predict_details(self, frame):
        if not self._fitted: raise RuntimeError("fit must be called before predict")
        times=self._times(frame); self._times_current=times; rows=[]; self._previous_row=None
        for i,(_,row) in enumerate(frame.iterrows()): rows.append(self._evaluate(row,float(times[i]),i)); self._previous_row=row
        return rows

    def predict(self, frame):
        details=self.predict_details(frame); scores=[]
        self.last_unknown = []
        for row in details:
            unknown=[d for d in row.values() if d["robustness"] is None and d["regime"] in ("missing","error","uncalibrated")]
            if unknown and (self.unknown_policy == "raise" or any(d["regime"] in ("missing","error") for d in unknown)):
                raise ValueError("prediction contains unknown residuals: "+"; ".join(d["reason"] for d in unknown))
            self.last_unknown.extend(unknown)
            scores.append(sum(d["robustness"] is not None and d["robustness"] < 0 for d in row.values()))
        return StepFlags(flags=np.asarray(scores,dtype=bool),scores=np.asarray(scores,dtype=float))

    def to_artifact(self):
        coverage={k:sum(1 for tol in self.tolerances if tol.startswith(k+"|")) for k in sorted(self.formulas)}
        return {"schema":"topology-stl-v2","percentile":self.percentile,"additive":dict(sorted(self.additives.items())),"formulas":{k:asdict(v) for k,v in sorted(self.formulas.items())},"tolerances":{k:asdict(v) for k,v in sorted(self.tolerances.items())},"coverage":coverage}

    def to_json(self): return json.dumps(self.to_artifact(),sort_keys=True,separators=(",",":"))
