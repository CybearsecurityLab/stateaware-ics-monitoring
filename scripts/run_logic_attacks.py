#!/usr/bin/env python3
"""Reproducible control-logic attack corpus for Anytown, C-Town and L-Town.

The runner derives every attack window from an attack-free nominal trace by
looking only at the *controlling guard* thresholds declared in the resolved
INP.  A pump is forced to the status opposite to its guard for one scan when
that guard first transitions from false to true:

* ``gt`` (source high threshold) -> force ``OPEN`` for one scan;
* ``lt`` (source low threshold)  -> force ``CLOSED`` for one scan.

The resulting one-scan command is a control-guard violation.  Sensor
offset/freeze/replay, actuator denial of service, observation-only actuator
feedback spoofing, coordinated and bounded adaptive scenarios are anchored on
the same nominal crossings.  Magnitudes come from the source thresholds, never
from GFSM or STL detector outputs.  The runner does not read, import or invent
BattLeDIM leak labels.

Nominals
--------
Anytown and C-Town are simulated for 24 h at a 300 s scan step using their
existing resolved topology files.  L-Town is the separate 48 h Real/PDD
nominal under ``scripts/out/full_signal_pdd/ltown/nominal_two_day`` and is only
read, never rewritten.

Outputs live under ``scripts/out/logic_attacks`` by default.  Each topology
gets a ``plan.json`` with the selection rationale, a nominal reference, and one
directory per scenario containing the standard cyber bundle plus
``scenario.json`` with its attack class.  Per-row ``attack_class`` labels come
from :mod:`simulation.cyber`.

Usage
-----
    python scripts/run_logic_attacks.py --topology all --plan-only
    python scripts/run_logic_attacks.py --topology anytown
    python scripts/run_logic_attacks.py --topology ctown --scenario guard_open_PU1
    python scripts/run_logic_attacks.py --topology ltown --list-scenarios
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
DEFAULT_OUT_ROOT = SCRIPTS_ROOT / "out" / "logic_attacks"
LTOWN_NOMINAL_DIR = SCRIPTS_ROOT / "out" / "full_signal_pdd" / "ltown" / "nominal_two_day"
LTOWN_TOPOLOGY = SCRIPTS_ROOT / "out" / "full_signal_pdd" / "ltown" / "topology.inp"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

LOGIC_SCAN_STEP = 300
ANYTOWN_DURATION = 86_400
CTOWN_DURATION = 86_400
LTOWN_DURATION = 172_800
SENSOR_WINDOW_SCANS = 12
DOS_WINDOW_SCANS = 12
REPLAY_PERIOD_SCANS = {"anytown": 72, "ctown": 72, "ltown": 288}
ADAPTIVE_FRACTION = 0.01
SEED = 42


class LogicAttackError(RuntimeError):
    """Raised for invalid plans, missing nominals or unsafe output paths."""


class LogicAttackExistsError(LogicAttackError):
    """Raised when an existing artifact would be overwritten without --overwrite."""


def _load_run_full_signal():
    path = SCRIPTS_ROOT / "run_full_signal.py"
    spec = importlib.util.spec_from_file_location("run_full_signal", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise LogicAttackError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


run_full_signal = _load_run_full_signal()


@dataclass(frozen=True)
class TopologySpec:
    name: str
    duration: int
    scan_step: int
    nominal_kind: str  # "generate" or "reuse"
    nominal_dir: Path
    demand_model: Any | None = None


def topology_specs(out_root: Path, *, ltown_nominal: Path | None = None) -> dict[str, TopologySpec]:
    out_root = Path(out_root)
    reuse_dir = Path(ltown_nominal) if ltown_nominal is not None else LTOWN_NOMINAL_DIR
    return {
        "anytown": TopologySpec(
            "anytown", ANYTOWN_DURATION, LOGIC_SCAN_STEP, "generate",
            out_root / "anytown" / "nominal",
        ),
        "ctown": TopologySpec(
            "ctown", CTOWN_DURATION, LOGIC_SCAN_STEP, "generate",
            out_root / "ctown" / "nominal",
        ),
        "ltown": TopologySpec(
            "ltown", LTOWN_DURATION, LOGIC_SCAN_STEP, "reuse", reuse_dir,
        ),
    }


# ---------------------------------------------------------------------------
# input and nominal handling
# ---------------------------------------------------------------------------

def resolve_source_inp(topology: str) -> Path:
    """Return the source INP for a generated nominal."""
    if topology == "ltown":
        return LTOWN_TOPOLOGY
    path, _ = run_full_signal.resolve_input(
        topology,
        input_root=run_full_signal.DEFAULT_SIGNAL_ROOT,
        out_root=run_full_signal.DEFAULT_SIGNAL_ROOT,
    )
    return Path(path)


def build_resolved_inp(topology: str, spec: TopologySpec) -> str:
    """Return the 24 h / 300 s resolved INP for Anytown or C-Town."""
    source = resolve_source_inp(topology)
    text = source.read_text(encoding="utf-8", errors="strict")
    timing = SimpleNamespace(duration=spec.duration, scan_step=spec.scan_step)
    text = run_full_signal.patch_timing_text(text, timing)
    if spec.demand_model is not None:
        text = run_full_signal.patch_demand_text(text, spec.demand_model)
    return text


def _sha256_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _publish_text(path: Path, text: str, *, overwrite: bool) -> None:
    path = Path(path)
    payload = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() == payload:
            return
        if not overwrite:
            raise LogicAttackExistsError(
                f"refusing to replace different resolved topology {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False)
    try:
        with handle:
            handle.write(payload)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", dir=path.parent, delete=False)
    try:
        with handle:
            handle.write(text)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _check_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and path.is_dir() and any(path.iterdir()) and not overwrite:
        raise LogicAttackExistsError(
            f"refusing to overwrite existing artifact {path}; pass --overwrite")


def load_nominal(nominal_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a written cyber nominal bundle (``plant_truth.csv`` + manifest)."""
    nominal_dir = Path(nominal_dir)
    truth = nominal_dir / "plant_truth.csv"
    manifest_path = nominal_dir / "manifest.json"
    if not truth.is_file() or not manifest_path.is_file():
        raise LogicAttackError(
            f"nominal is incomplete (need plant_truth.csv and manifest.json): {nominal_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(truth, float_precision="round_trip")
    return frame, manifest


def validate_nominal(topology: str, frame: pd.DataFrame, manifest: Mapping[str, Any],
                     spec: TopologySpec) -> None:
    duration = int(manifest.get("timesteps", {}).get("duration", -1))
    scan = int(manifest.get("effective_timing", {}).get("scan_timestep", -1))
    if duration != spec.duration or scan != spec.scan_step:
        raise LogicAttackError(
            f"{topology} nominal must be {spec.duration} s at {spec.scan_step} s "
            f"(got {duration} s at {scan} s)")
    if topology == "ltown":
        demand = str(manifest.get("hydraulic_options", {}).get("demand_model", "")).upper()
        if demand not in ("PDA", "PDD"):
            raise LogicAttackError("L-Town nominal must be the Real/PDD nominal")
    if frame.empty or "time_seconds" not in frame:
        raise LogicAttackError(f"{topology} nominal has no time grid")


def ensure_nominal(topology: str, spec: TopologySpec, *, out_root: Path,
                   overwrite: bool) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    """Return ``(frame, manifest, nominal_dir)``, generating Anytown/C-Town."""
    from simulation import RunSpec
    from simulation.cyber import run_cyber, write_cyber_bundle

    if spec.nominal_kind == "reuse":
        frame, manifest = load_nominal(spec.nominal_dir)
        validate_nominal(topology, frame, manifest, spec)
        reference = Path(out_root) / topology / "nominal_reference.json"
        _write_json(reference, {
            "topology": topology,
            "kind": "existing_real_pdd_nominal",
            "path": str(spec.nominal_dir),
            "plant_truth_sha256": _sha256_file(spec.nominal_dir / "plant_truth.csv"),
            "manifest_sha256": _sha256_file(spec.nominal_dir / "manifest.json"),
            "duration": spec.duration,
            "scan_step": spec.scan_step,
            "demand_model": manifest.get("hydraulic_options", {}).get("demand_model"),
            "rewritten": False,
        })
        return frame, manifest, spec.nominal_dir

    topology_dir = Path(out_root) / topology
    topology_path = topology_dir / "topology.inp"
    nominal_dir = spec.nominal_dir
    if nominal_dir.exists() and any(nominal_dir.iterdir()) and not overwrite:
        frame, manifest = load_nominal(nominal_dir)
        validate_nominal(topology, frame, manifest, spec)
        return frame, manifest, nominal_dir
    text = build_resolved_inp(topology, spec)
    _publish_text(topology_path, text, overwrite=overwrite)
    _check_output_dir(nominal_dir, overwrite)
    nominal_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{nominal_dir.name}.", dir=nominal_dir.parent))
    try:
        with tempfile.TemporaryDirectory(prefix="logic_nominal_") as work:
            resolved = Path(work) / "topology.inp"
            resolved.write_text(text, encoding="utf-8")
            bundle = run_cyber(RunSpec(resolved, f"{topology}-logic-nominal", SEED, "calibration"))
            write_cyber_bundle(bundle, staging)
            _write_json(staging / "nominal.json", {
                "schema_version": 1,
                "topology": topology,
                "kind": "generated_24h_300s_nominal",
                "duration": spec.duration,
                "scan_step": spec.scan_step,
                "seed": SEED,
                "source_inp": str(resolve_source_inp(topology)),
                "resolved_topology": str(topology_path),
                "resolved_sha256": _sha256_file(topology_path),
                "demand_model": (
                    spec.demand_model.options() if spec.demand_model else None),
            })
        if nominal_dir.exists():
            shutil.rmtree(nominal_dir)
        os.replace(staging, nominal_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    frame, manifest = load_nominal(nominal_dir)
    validate_nominal(topology, frame, manifest, spec)
    return frame, manifest, nominal_dir


# ---------------------------------------------------------------------------
# threshold-based planning
# ---------------------------------------------------------------------------

def pump_links(manifest: Mapping[str, Any]) -> set[str]:
    """Return pump ids from the nominal manifest's ``speed_`` channels."""
    return {
        column[len("speed_"):]
        for column in manifest.get("expected_columns", [])
        if column.startswith("speed_")
    }


def control_windows(frame: pd.DataFrame, controls: Iterable[Mapping[str, Any]],
                    *, step: int, duration: int,
                    pumps: set[str]) -> list[dict[str, Any]]:
    """Return the first true guard state for each pump control."""
    times = frame["time_seconds"].astype(int).tolist()
    windows: list[dict[str, Any]] = []
    for control in controls:
        link = str(control["link"])
        if link not in pumps or control.get("relation") not in ("lt", "gt"):
            continue
        sensor = str(control["sensor"])
        if sensor not in frame.columns:
            raise LogicAttackError(f"nominal has no channel for guard sensor {sensor!r}")
        values = frame[sensor].astype(float).to_numpy()
        threshold = float(control["threshold"])
        relation = str(control["relation"])

        def holds(value: float) -> bool:
            return value < threshold if relation == "lt" else value > threshold

        previous = holds(float(values[0]))
        for index in range(len(values)):
            current = holds(float(values[index]))
            if current and (index == 0 or not previous) and times[index] + step <= duration:
                nominal_status = "OPEN" if int(control["value"]) == 1 else "CLOSED"
                forced_status = "CLOSED" if nominal_status == "OPEN" else "OPEN"
                windows.append({
                    "control_id": str(control["id"]),
                    "link": link,
                    "sensor": sensor,
                    "relation": relation,
                    "threshold": threshold,
                    "direction": "above_high" if relation == "gt" else "below_low",
                    "scan_index": index,
                    "time_seconds": int(times[index]),
                    "sensor_value": float(values[index]),
                    "nominal_status": nominal_status,
                    "forced_status": forced_status,
                })
                break
            previous = current
    return windows


def offset_magnitude(sensor: str, controls: Iterable[Mapping[str, Any]],
                     *, fraction: float = 0.5) -> float:
    """Return a threshold-derived offset; never uses detector outputs."""
    lows = [abs(float(c["threshold"])) for c in controls
            if c["sensor"] == sensor and c.get("relation") == "lt"]
    highs = [float(c["threshold"]) for c in controls
             if c["sensor"] == sensor and c.get("relation") == "gt"]
    if lows and highs and max(highs) > min(lows):
        return (max(highs) - min(lows)) * fraction
    bounds = highs or lows
    if not bounds:
        raise LogicAttackError(f"no guard threshold for sensor {sensor!r}")
    return max(abs(min(bounds)) * 0.1, 1e-9)


def _guard_event(window: Mapping[str, Any], *, scans: int, step: int,
                 attack_class: str) -> dict[str, Any]:
    start = int(window["time_seconds"])
    return {
        "kind": "actuator",
        "target": window["link"],
        "value": window["forced_status"],
        "start": start,
        "end": start + scans * step,
        "attack_class": attack_class,
    }


def _sensor_event(window: Mapping[str, Any], *, mode: str, scans: int, step: int,
                  value: Any = None, period: int | None = None,
                  attack_class: str) -> dict[str, Any]:
    start = int(window["time_seconds"])
    event: dict[str, Any] = {
        "kind": "sensor",
        "target": window["sensor"],
        "mode": mode,
        "scope": "both",
        "start": start,
        "end": start + scans * step,
        "attack_class": attack_class,
    }
    if value is not None:
        event["value"] = value
    if period is not None:
        event["period"] = period
        event["noise"] = 0.0
    return event


def _first_window(windows: Sequence[Mapping[str, Any]], *, step: int,
                  scans: int, duration: int,
                  min_time: int = 0) -> Mapping[str, Any] | None:
    for window in sorted(windows, key=lambda item: item["time_seconds"]):
        start = int(window["time_seconds"])
        if start >= min_time and start + scans * step <= duration:
            return window
    return None


def _taxonomy() -> dict[str, str]:
    from simulation.cyber import ATTACK_CLASS_TAXONOMY

    return dict(ATTACK_CLASS_TAXONOMY)


def build_plan(topology: str, frame: pd.DataFrame, manifest: Mapping[str, Any],
               spec: TopologySpec, *, period_scans: int | None = None,
               sensor_window_scans: int = SENSOR_WINDOW_SCANS,
               dos_window_scans: int = DOS_WINDOW_SCANS,
               adaptive_fraction: float = ADAPTIVE_FRACTION) -> dict[str, Any]:
    """Build the full scenario plan from nominal threshold crossings."""
    step = int(spec.scan_step)
    duration = int(spec.duration)
    controls = list(manifest.get("controls", []))
    pumps = pump_links(manifest)
    if not pumps:
        raise LogicAttackError(f"{topology} nominal manifest declares no pump channels")
    windows = control_windows(frame, controls, step=step, duration=duration, pumps=pumps)
    if not windows:
        raise LogicAttackError(
            f"{topology} nominal has no true pump guard state; cannot pre-register windows")
    period = REPLAY_PERIOD_SCANS[topology] if period_scans is None else int(period_scans)
    if period < 1:
        raise LogicAttackError("replay period must be a positive number of scans")

    scenarios: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(name: str, attack_class: str, events: list[dict[str, Any]],
            selection: Mapping[str, Any], description: str) -> None:
        if name in seen:
            return
        seen.add(name)
        scenarios.append({
            "name": name,
            "attack_class": attack_class,
            "description": description,
            "events": events,
            "selection": dict(selection),
        })

    selection_policy = {
        "rule": (
            "first nominal false-to-true transition of the resolved INP controlling "
            "guard for a pump; the forced status is the opposite of the guard action"
        ),
        "high_guard": "gt threshold -> force OPEN for one scan",
        "low_guard": "lt threshold -> force CLOSED for one scan",
        "sensor_magnitudes": "derived from the same guard thresholds only",
        "detector_results_consulted": False,
        "battledim_leak_labels_used": False,
    }

    for window in windows:
        direction = "open" if window["forced_status"] == "OPEN" else "closed"
        name = f"guard_{direction}_{window['link']}"
        add(
            name, "control_guard_violation",
            [_guard_event(window, scans=1, step=step,
                          attack_class="control_guard_violation")],
            {
                "kind": "single_scan_control_guard_violation",
                "window": dict(window),
                "magnitude": "one scan, opposite of the guard action",
                "rationale": (
                    f"nominal {window['sensor']}={window['sensor_value']:.6g} crossed "
                    f"{window['relation']} {window['threshold']:g} at "
                    f"{window['time_seconds']} s; guard would issue "
                    f"{window['nominal_status']}, attack forces "
                    f"{window['forced_status']} for one scan"),
            },
            f"one-scan {window['forced_status']} override of {window['link']} at a "
            f"nominal {window['direction']} crossing",
        )

    dos_window = _first_window(
        [w for w in windows if w["forced_status"] == "CLOSED"],
        step=step, scans=dos_window_scans, duration=duration)
    if dos_window is not None:
        add(
            f"actuator_dos_{dos_window['link']}", "actuator_dos",
            [_guard_event(dos_window, scans=dos_window_scans, step=step,
                          attack_class="actuator_dos")],
            {
                "kind": "fixed_window_actuator_dos",
                "window": dict(dos_window),
                "magnitude": f"{dos_window_scans} scans CLOSED",
                "rationale": (
                    f"start at the first nominal low-guard crossing "
                    f"({dos_window['time_seconds']} s) and hold the pump CLOSED for "
                    f"{dos_window_scans} scans"),
            },
            f"actuator denial of service on {dos_window['link']}",
        )

    sensors: list[str] = []
    for window in windows:
        if window["sensor"] not in sensors:
            sensors.append(window["sensor"])
    sensor = sensors[0]
    sensor_windows = [w for w in windows if w["sensor"] == sensor]
    offset = offset_magnitude(sensor, controls, fraction=0.5)
    adaptive = offset_magnitude(sensor, controls, fraction=adaptive_fraction)

    anchor = _first_window(sensor_windows, step=step, scans=sensor_window_scans,
                           duration=duration)
    if anchor is not None:
        add(f"sensor_offset_{sensor}", "sensor_offset",
            [_sensor_event(anchor, mode="offset", scans=sensor_window_scans, step=step,
                           value=offset, attack_class="sensor_offset")],
            {
                "kind": "threshold_derived_sensor_offset",
                "window": dict(anchor),
                "magnitude": offset,
                "magnitude_source": "half the gap between the source low and high thresholds",
                "rationale": f"offset {offset:g} on {sensor} anchored at a nominal crossing",
            },
            f"additive offset on {sensor}")
        add(f"sensor_freeze_{sensor}", "sensor_freeze",
            [_sensor_event(anchor, mode="freeze", scans=sensor_window_scans, step=step,
                           attack_class="sensor_freeze")],
            {
                "kind": "sensor_freeze",
                "window": dict(anchor),
                "magnitude": "hold activation value",
                "rationale": f"freeze {sensor} at a nominal crossing",
            },
            f"delivery freeze on {sensor}")

    replay_period = period
    replay_anchor = _first_window(
        sensor_windows, step=step, scans=sensor_window_scans, duration=duration,
        min_time=period * step)
    if replay_anchor is None:
        candidates = [
            w for w in sensor_windows
            if int(w["time_seconds"]) >= step
            and int(w["time_seconds"]) + sensor_window_scans * step <= duration
        ]
        if candidates:
            replay_anchor = max(candidates, key=lambda w: w["time_seconds"])
            replay_period = max(1, int(replay_anchor["time_seconds"]) // step - 1)
    if replay_anchor is not None:
        add(f"sensor_replay_{sensor}", "sensor_replay",
            [_sensor_event(replay_anchor, mode="replay", scans=sensor_window_scans,
                           step=step, period=replay_period,
                           attack_class="sensor_replay")],
            {
                "kind": "recipient_history_replay",
                "window": dict(replay_anchor),
                "magnitude": f"period {replay_period} scans",
                "rationale": (
                    f"replay each recipient's own delivery from {replay_period} scans "
                    f"earlier, anchored at a nominal crossing with sufficient "
                    f"pre-attack history"),
            },
            f"recipient-specific replay of {sensor}")

    feedback_window = _first_window(windows, step=step, scans=1, duration=duration)
    if feedback_window is not None:
        add(f"actuator_feedback_spoof_{feedback_window['link']}",
            "actuator_feedback_spoof",
            [{
                "kind": "actuator_feedback",
                "target": feedback_window["link"],
                "value": feedback_window["forced_status"],
                "scope": "scada",
                "start": int(feedback_window["time_seconds"]),
                "end": int(feedback_window["time_seconds"]) + step,
                "attack_class": "actuator_feedback_spoof",
            }],
            {
                "kind": "observation_only_actuator_feedback_spoof",
                "window": dict(feedback_window),
                "magnitude": "reported status opposite to the nominal command",
                "rationale": (
                    "spoof only the SCADA actuator-status feedback for one scan; the "
                    "physical command and plant truth are unchanged"),
            },
            f"observation-only status spoof on {feedback_window['link']}")

    coordinated_window = _first_window(windows, step=step, scans=1, duration=duration)
    if coordinated_window is not None:
        add(f"coordinated_{coordinated_window['link']}", "coordinated",
            [
                _sensor_event(coordinated_window, mode="freeze", scans=1, step=step,
                              attack_class="coordinated"),
                _guard_event(coordinated_window, scans=1, step=step,
                             attack_class="coordinated"),
            ],
            {
                "kind": "coordinated_sensor_and_actuator",
                "window": dict(coordinated_window),
                "magnitude": "single scan sensor freeze plus opposite guard action",
                "rationale": "both mechanisms share one nominal-crossing scan",
            },
            f"coordinated freeze and guard violation on {coordinated_window['link']}")

    if anchor is not None:
        add(f"adaptive_{sensor}", "adaptive",
            [_sensor_event(anchor, mode="offset", scans=sensor_window_scans, step=step,
                           value=adaptive, attack_class="adaptive")],
            {
                "kind": "bounded_adaptive_offset",
                "window": dict(anchor),
                "magnitude": adaptive,
                "magnitude_source": (
                    f"{adaptive_fraction:g} of the source low/high threshold gap"),
                "rationale": (
                    "pre-registered bounded offset; chosen without GFSM or STL feedback"),
            },
            f"bounded adaptive offset on {sensor}")

    return {
        "schema_version": 1,
        "topology": topology,
        "duration": duration,
        "scan_step": step,
        "seed": SEED,
        "pump_links": sorted(pumps),
        "guard_windows": windows,
        "selection_policy": selection_policy,
        "attack_taxonomy": _taxonomy(),
        "inherited_leak_labels": False,
        "detector_results_consulted": False,
        "scenarios": scenarios,
    }


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def _scenario_dir(topology_dir: Path, name: str) -> Path:
    return Path(topology_dir) / name


def write_plan(out_root: Path, topology: str, plan: Mapping[str, Any],
               nominal: Mapping[str, Any]) -> None:
    topology_dir = Path(out_root) / topology
    _write_json(topology_dir / "plan.json", {**plan, "nominal": dict(nominal)})


def execute_scenarios(topology: str, spec: TopologySpec, plan: Mapping[str, Any],
                      *, out_root: Path, inp_path: Path,
                      overwrite: bool = False,
                      names: Sequence[str] | None = None,
                      runner: Any | None = None,
                      writer: Any | None = None) -> list[Path]:
    """Simulate planned scenarios and publish each into its own directory."""
    from simulation import RunSpec
    from simulation.cyber import run_cyber, write_cyber_bundle

    run = runner or run_cyber
    write = writer or write_cyber_bundle
    selected = list(plan["scenarios"])
    if names is not None:
        wanted = set(names)
        missing = wanted - {scenario["name"] for scenario in selected}
        if missing:
            raise LogicAttackError(f"unknown scenarios: {sorted(missing)}")
        selected = [scenario for scenario in selected if scenario["name"] in wanted]
    topology_dir = Path(out_root) / topology
    outputs: list[Path] = []
    for scenario in selected:
        out_dir = _scenario_dir(topology_dir, scenario["name"])
        _check_output_dir(out_dir, overwrite)
        outputs.append(out_dir)
    for scenario, out_dir in zip(selected, outputs):
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
        try:
            run_id = f"{topology}-logic-{scenario['name']}"
            bundle = run(RunSpec(inp_path, run_id, SEED, "evaluation"),
                         list(scenario["events"]))
            write(bundle, staging)
            _write_json(staging / "scenario.json", {
                "schema_version": 1,
                "topology": topology,
                "scenario": scenario["name"],
                "description": scenario["description"],
                "attack_class": scenario["attack_class"],
                "run_id": run_id,
                "split": "evaluation",
                "seed": SEED,
                "duration": spec.duration,
                "scan_step": spec.scan_step,
                "events": list(scenario["events"]),
                "selection": dict(scenario["selection"]),
                "labels": {
                    "source": "injected_events_and_attack_free_physical_comparison",
                    "attack_class": "per-row from simulation.cyber",
                    "battledim_leak_labels_used": False,
                    "inherited_leak_labels": False,
                },
                "detector_results_consulted": False,
            })
            if out_dir.exists():
                if not overwrite:
                    raise LogicAttackExistsError(
                        f"refusing to overwrite existing run {out_dir}; pass --overwrite")
                shutil.rmtree(out_dir)
            os.replace(staging, out_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--topology", choices=["anytown", "ctown", "ltown", "all"],
                        default="all")
    parser.add_argument("--scenario", nargs="+", metavar="NAME",
                        help="optional scenario names to run (default: every planned scenario)")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--ltown-nominal", type=Path,
                        help="override the read-only 48 h Real/PDD L-Town nominal")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plan-only", action="store_true",
                        help="write nominals/plans but do not simulate attacks")
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    global SEED
    args = _parse_args(argv)
    SEED = int(args.seed)
    specs = topology_specs(args.out_root, ltown_nominal=args.ltown_nominal)
    topologies = sorted(specs) if args.topology == "all" else [args.topology]
    global_manifest: dict[str, Any] = {
        "schema_version": 1,
        "meta": {
            "selection": "nominal controlling-guard transitions only",
            "detector_results_consulted": False,
            "battledim_leak_labels_used": False,
            "no_detector_performance_claim": True,
        },
        "topologies": {},
    }
    try:
        for topology in topologies:
            spec = specs[topology]
            frame, nominal_manifest, nominal_dir = ensure_nominal(
                topology, spec, out_root=args.out_root, overwrite=args.overwrite)
            plan = build_plan(topology, frame, nominal_manifest, spec)
            write_plan(args.out_root, topology, plan, {
                "kind": ("generated_24h_300s_nominal"
                         if spec.nominal_kind == "generate"
                         else "existing_real_pdd_nominal"),
                "path": str(nominal_dir),
                "duration": spec.duration,
                "scan_step": spec.scan_step,
                "plant_truth_sha256": _sha256_file(nominal_dir / "plant_truth.csv"),
                "manifest_sha256": _sha256_file(nominal_dir / "manifest.json"),
            })
            global_manifest["topologies"][topology] = {
                "duration": spec.duration,
                "scan_step": spec.scan_step,
                "nominal": str(nominal_dir),
                "scenario_count": len(plan["scenarios"]),
                "attack_classes": sorted({
                    scenario["attack_class"] for scenario in plan["scenarios"]}),
            }
            _write_json(args.out_root / "manifest.json", global_manifest)
            if args.list_scenarios:
                for scenario in plan["scenarios"]:
                    print(f"{topology}\t{scenario['name']}\t{scenario['attack_class']}")
                continue
            if args.plan_only:
                print(f"planned {topology}: {len(plan['scenarios'])} scenarios -> "
                      f"{args.out_root / topology}")
                continue
            inp_path = (LTOWN_TOPOLOGY if topology == "ltown"
                        else args.out_root / topology / "topology.inp")
            outputs = execute_scenarios(
                topology, spec, plan, out_root=args.out_root, inp_path=inp_path,
                overwrite=args.overwrite, names=args.scenario)
            for output in outputs:
                print(output)
    except LogicAttackError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
