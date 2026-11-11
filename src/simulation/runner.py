"""Run WNTR and persist explicitly scoped, SI-valued nominal traces."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wntr

from hydraulics import load_network
from stl.topology import build_binding


def _contract_hash(manifest: dict[str, Any]) -> str:
    contract = {key: manifest[key] for key in ("expected_columns", "solve_times", "row_count", "required_channels", "column_units", "unavailable_channels", "template_coverage")}
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _expected_columns(network: Any) -> list[str]:
    columns = ["time_seconds"]
    for link in network.links():
        columns.extend((f"f_{link.id}", f"s_{link.id}"))
    for node in network.junctions:
        columns.extend((node.id, f"demand_{node.id}", f"leak_{node.id}", f"emitter_{node.id}"))
    for tank in network.tanks:
        columns.extend((tank.id, f"leak_{tank.id}"))
    columns.extend(f"head_{reservoir.id}" for reservoir in network.reservoirs)
    columns.extend(f"speed_{pump.id}" for pump in network.pumps)
    columns.extend(f"setting_{valve.id}" for valve in network.valves)
    return columns


@dataclass(frozen=True)
class RunSpec:
    inp_path: str | Path
    run_id: str
    seed: int = 0
    split: str = "nominal"

    def __post_init__(self) -> None:
        if not self.run_id or not self.split or not isinstance(self.seed, int):
            raise ValueError("run_id and split must be nonempty and seed must be an integer")


@dataclass
class TraceBundle:
    observed: pd.DataFrame
    plant_truth: pd.DataFrame
    controller_view: pd.DataFrame
    events: list[dict[str, Any]]
    manifest: dict[str, Any]

    def validate(self) -> None:
        m = self.manifest
        required = {"schema_version", "run_id", "seed", "split", "inp_sha256", "wntr_version", "engine", "demand_model", "timesteps", "effective_timing", "column_units", "unavailable_channels", "limitations", "expected_columns", "solve_times", "row_count", "required_channels", "template_coverage", "scope_metadata", "contract_sha256"}
        if not isinstance(m, dict) or not required.issubset(m) or m["schema_version"] != 1:
            raise ValueError("invalid bundle manifest")
        if not isinstance(m["column_units"], dict) or not isinstance(m["unavailable_channels"], dict):
            raise ValueError("invalid channel metadata")
        if m["unavailable_channels"]:
            raise ValueError("nominal traces require every channel to be available")
        if m["contract_sha256"] != _contract_hash(m):
            raise ValueError("trace contract checksum mismatch")
        if not isinstance(self.events, list) or self.events:
            raise ValueError("nominal events must be empty")
        columns = list(m["column_units"])
        if columns != m["expected_columns"] or m["required_channels"] != columns:
            raise ValueError("invalid expected or required channel inventory")
        if m["row_count"] < 2 or m["row_count"] != len(m["solve_times"]):
            raise ValueError("invalid expected solve times")
        if m["solve_times"][0] != 0 or m["solve_times"][-1] != m["timesteps"]["duration"]:
            raise ValueError("solve times do not cover the configured duration")
        if not np.isin(np.arange(0, m["timesteps"]["duration"] + 1, m["effective_timing"]["hydraulic_timestep"]), m["solve_times"]).all():
            raise ValueError("solve times omit an effective hydraulic grid point")
        if not columns or columns[0] != "time_seconds" or len(columns) != len(set(columns)):
            raise ValueError("invalid column schema")
        if any(c not in columns or not isinstance(r, str) or not r for c, r in m["unavailable_channels"].items()):
            raise ValueError("invalid unavailable channel metadata")
        first = None
        for frame in (self.observed, self.plant_truth, self.controller_view):
            if not isinstance(frame, pd.DataFrame) or list(frame.columns) != columns or frame.empty:
                raise ValueError("trace columns do not match manifest")
            if len(frame) != m["row_count"] or not np.array_equal(frame.time_seconds.to_numpy(), np.asarray(m["solve_times"])):
                raise ValueError("trace rows do not match expected solve times")
            if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in columns):
                raise ValueError("trace values must be numeric")
            if np.isinf(frame.to_numpy(dtype=float)).any():
                raise ValueError("trace values must not be infinite")
            times = frame.time_seconds.to_numpy()
            if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
                raise ValueError("trace times must be finite and strictly increasing")
            if first is not None and not np.array_equal(first, times):
                raise ValueError("scope times differ")
            first = times
            for col in m["unavailable_channels"]:
                if not frame[col].isna().all():
                    raise ValueError(f"unavailable channel {col} contains values")
            for col in columns:
                if frame[col].isna().any() and col not in m["unavailable_channels"]:
                    raise ValueError(f"unknown values lack a reason: {col}")


def run_nominal(spec: RunSpec) -> TraceBundle:
    network = load_network(spec.inp_path, simulator_backend="wntr")
    wn = wntr.network.WaterNetworkModel(str(spec.inp_path))
    if hashlib.sha256(Path(spec.inp_path).read_bytes()).hexdigest() != network.provenance.sha256:
        raise ValueError("INP changed during loading")
    for node in network.junctions:
        if node.emitter_coefficient != 0:
            raise ValueError(f"junction {node.id}: nonzero emitter unsupported by WNTRSimulator")
    for pump in network.pumps:
        if pump.speed != 1.0 or pump.speed_pattern_id:
            raise ValueError(f"pump {pump.id}: non-unit or patterned speed unsupported by WNTRSimulator")
    for _, control in wn.controls():
        for action in control.actions():
            target, attr = action.target()
            if isinstance(target, wntr.network.Pump) and attr in ("base_speed", "speed", "speed_timeseries"):
                raise ValueError(f"pump {target.name}: controlled speed unsupported by WNTRSimulator")
    timing = wn.options.time
    original_report = timing.report_timestep
    timesteps = {key: getattr(timing, key) for key in ("duration", "hydraulic_timestep", "report_timestep", "report_start", "pattern_timestep", "pattern_start", "rule_timestep", "quality_timestep")}
    hydraulic_options = dict(vars(wn.options.hydraulic))
    h, report, start, duration = (timesteps[k] for k in ("hydraulic_timestep", "report_timestep", "report_start", "duration"))
    if not isinstance(report, (int, float)) or h <= 0 or report <= 0 or duration <= 0:
        raise ValueError("duration, hydraulic timestep and numeric report timestep must be positive")
    effective_h = min(h, report)
    if max(h, report) % effective_h:
        raise ValueError("hydraulic and report timesteps must be aligned integer multiples")
    if start < 0 or start > duration or start % effective_h:
        raise ValueError("report start must lie on the effective hydraulic grid within duration")
    if duration % effective_h:
        raise ValueError("duration must lie on the effective hydraulic grid")
    if timing.pattern_timestep % effective_h or timing.pattern_start % effective_h:
        raise ValueError("pattern changes must lie on the effective hydraulic grid")
    timing.hydraulic_timestep = effective_h
    timing.report_timestep = "ALL"
    result = wntr.sim.WNTRSimulator(wn).run_sim(convergence_error=True)
    times = np.asarray(result.time, dtype=float)
    if len(times) < 2 or times[0] != 0 or times[-1] != timing.duration:
        raise RuntimeError("WNTR returned an incomplete simulation")
    if isinstance(original_report, (int, float)):
        expected = np.arange(timesteps["report_start"], timing.duration + 1, original_report)
        if not np.isin(expected, times).all() or not np.isin(np.arange(0, duration + 1, effective_h), times).all():
            raise ValueError("WNTR hydraulic steps do not include every configured report time")
    data: dict[str, Any] = {"time_seconds": times}
    units = {"time_seconds": "s"}
    unavailable: dict[str, str] = {}

    def add(col: str, unit: str, values: Any = None, reason: str = "") -> None:
        if col in data:
            raise ValueError(f"ambiguous flat column name: {col}")
        units[col] = unit
        if reason:
            unavailable[col] = reason
            data[col] = np.full(len(times), np.nan)
        else:
            arr = np.asarray(values, dtype=float)
            if len(arr) != len(times):
                raise ValueError(f"invalid result length for {col}")
            if np.isnan(arr).any():
                raise ValueError(f"WNTR returned unknown values for required channel {col}")
            data[col] = arr

    for link in network.links():
        add(f"f_{link.id}", "m^3/s", result.link["flowrate"][link.id])
        add(f"s_{link.id}", "status code: 0=closed, 1=open, 2=active", result.link["status"][link.id])
    for node in network.junctions:
        add(node.id, "m", result.node["pressure"][node.id])
        add(f"demand_{node.id}", "m^3/s", result.node["demand"][node.id])
        add(f"leak_{node.id}", "m^3/s", result.node["leak_demand"][node.id])
        add(f"emitter_{node.id}", "m^3/s", np.zeros(len(times)))
    for tank in network.tanks:
        add(tank.id, "m", result.node["pressure"][tank.id])
        add(f"leak_{tank.id}", "m^3/s", result.node["leak_demand"][tank.id])
    for reservoir in network.reservoirs:
        add(f"head_{reservoir.id}", "m", result.node["head"][reservoir.id])
    for pump in network.pumps:
        add(f"speed_{pump.id}", "dimensionless", np.ones(len(times)))
    for valve in network.valves:
        unit = {"PRV": "m", "PSV": "m", "PBV": "m", "FCV": "m^3/s", "TCV": "dimensionless"}.get(valve.valve_type.upper(), "unknown")
        if valve.valve_type.upper() == "GPV":
            add(f"setting_{valve.id}", unit, reason="GPV setting is a curve, not a scalar")
        else:
            add(f"setting_{valve.id}", unit, result.link["setting"][valve.id])
    expected_columns = _expected_columns(network)
    if list(data) != expected_columns or unavailable:
        raise ValueError("simulation did not produce the complete expected channel inventory")
    binding = build_binding(network, (col for col in data if col not in unavailable))
    coverage = [{"family": t.family.value, "element_id": t.key, "status": t.status.value, "reason": t.reason, "required_channels": [c.key for c in t.required_channels]} for t in binding.templates]
    manifest = {"schema_version": 1, "run_id": spec.run_id, "seed": spec.seed, "split": spec.split,
                "inp_sha256": network.provenance.sha256, "inp_filename": network.provenance.inp_filename,
                "wntr_version": wntr.__version__, "engine": "WNTRSimulator", "demand_model": network.demand.model,
                "timesteps": timesteps, "hydraulic_options": hydraulic_options, "column_units": units, "unavailable_channels": unavailable,
                "limitations": ["WNTRSimulator does not model INP emitters; nonzero coefficients are rejected", "WNTRSimulator does not model variable head-pump speed; non-unit, patterned and controlled speeds are rejected", "PBV and GPV valves and non-Hazen-Williams headloss are unsupported by WNTRSimulator"],
                "effective_timing": {"hydraulic_timestep": effective_h, "report_timestep": "ALL", "report_start": 0, "policy": "all hydraulic and control solve times; configured report grid included"},
                "expected_columns": expected_columns, "required_channels": expected_columns.copy(),
                "solve_times": times.tolist(), "row_count": len(times), "template_coverage": coverage,
                "scope_metadata": {scope: "ideal full-state nominal view; not a sensor or controller visibility model" for scope in ("observed", "plant_truth", "controller_view")},
                "flow_direction": "positive from link start node to end node", "status_codes": {"0": "closed", "1": "open", "2": "active"}}
    manifest["contract_sha256"] = _contract_hash(manifest)
    frame = pd.DataFrame(data)
    bundle = TraceBundle(frame.copy(), frame.copy(), frame.copy(), [], manifest)
    bundle.validate()
    return bundle


def write_bundle(bundle: TraceBundle, path: str | Path) -> None:
    bundle.validate()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name in ("observed", "plant_truth", "controller_view"):
        getattr(bundle, name).to_csv(path / f"{name}.csv", index=False, float_format="%.17g", lineterminator="\n")
    for name in ("events", "manifest"):
        (path / f"{name}.json").write_text(json.dumps(getattr(bundle, name), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_bundle(path: str | Path) -> TraceBundle:
    path = Path(path)
    frames = [pd.read_csv(path / f"{name}.csv", float_precision="round_trip") for name in ("observed", "plant_truth", "controller_view")]
    events, manifest = [json.loads((path / f"{name}.json").read_text(encoding="utf-8")) for name in ("events", "manifest")]
    bundle = TraceBundle(*frames, events, manifest)
    bundle.validate()
    return bundle
