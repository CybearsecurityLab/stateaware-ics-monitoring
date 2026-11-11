#!/usr/bin/env python3
"""Reproducible supervised attack episodes for the hybrid ensemble.

The existing control-logic corpus under ``scripts/out/logic_attacks`` is the
held-out evaluation set.  This runner builds a disjoint training/validation
corpus from the *same* attack-free nominal traces, never from evaluation
outputs or detector scores.

Every class is anchored on a guard window observed in the existing nominal
trace.  Required actuator classes only use scans where the controlling guard is
active; sensor classes may use any observed scan so that small topologies still
provide enough distinct windows.  Windows that match the held-out plan are
excluded before selection.  Selection is deterministic (stable SHA-256 rank),
and the split seeds are fixed: 101 for training and 202 for validation.

Only existing nominals are read:

* ``scripts/out/logic_attacks/<town>/nominal`` for Anytown and C-Town;
* ``scripts/out/full_signal_pdd/ltown/nominal_two_day`` for the Real/PDD
  L-Town trace.

No nominal simulation is performed.  Plans and manifests are written before any
episode is simulated.  Each scenario is an independent ``run_cyber`` call with
the topology duration, split seed and split label.  Complete cyber bundles plus
the scenario plan are retained under
``scripts/out/supervised_episodes/<town>/<split>/``.

Usage
-----
    python scripts/run_supervised_episodes.py --topologies anytown ctown ltown --plan-only
    python scripts/run_supervised_episodes.py --topologies anytown
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
DEFAULT_OUT_ROOT = SCRIPTS_ROOT / "out" / "supervised_episodes"
LOGIC_ATTACKS_ROOT = SCRIPTS_ROOT / "out" / "logic_attacks"
LTOWN_NOMINAL_DIR = SCRIPTS_ROOT / "out" / "full_signal_pdd" / "ltown" / "nominal_two_day"
LTOWN_TOPOLOGY = SCRIPTS_ROOT / "out" / "full_signal_pdd" / "ltown" / "topology.inp"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SCAN_STEP = 300
TRAIN_SEED = 101
VALIDATION_SEED = 202
SPLIT_SEEDS = {"training": TRAIN_SEED, "validation": VALIDATION_SEED}
TRAIN_CLASSES = (
    "sensor_offset",
    "control_guard_violation",
    "actuator_feedback_spoof",
    "sensor_replay",
)
VALIDATION_CLASSES = (
    "sensor_freeze",
    "actuator_dos",
    "adaptive",
    "coordinated",
)
ALL_CLASSES = TRAIN_CLASSES + VALIDATION_CLASSES
CLASS_SPLIT = {
    **{name: "training" for name in TRAIN_CLASSES},
    **{name: "validation" for name in VALIDATION_CLASSES},
}
REQUIRE_GUARD_TRUE = {
    "control_guard_violation",
    "actuator_feedback_spoof",
    "actuator_dos",
    "coordinated",
}
REQUIRED_ORDER = (
    "actuator_dos",
    "control_guard_violation",
    "actuator_feedback_spoof",
    "coordinated",
)
SENSOR_CLASSES = {
    "sensor_offset",
    "sensor_freeze",
    "sensor_replay",
    "adaptive",
}
SENSOR_WINDOW_SCANS = 12
DOS_WINDOW_SCANS = 12
REPLAY_PERIOD_SCANS = {"anytown": 72, "ctown": 72, "ltown": 288}
ADAPTIVE_FRACTION = 0.01


class SupervisedEpisodeError(RuntimeError):
    """Raised for invalid plans, missing nominals or unsafe output paths."""


class SupervisedEpisodeExistsError(SupervisedEpisodeError):
    """Raised when an existing artifact would be overwritten without --overwrite."""


@dataclass(frozen=True)
class TopologySpec:
    name: str
    duration: int
    scan_step: int
    nominal_dir: Path
    inp_path: Path
    demand_model: str | None = None


def topology_specs(*, ltown_nominal: Path | None = None,
                   ltown_topology: Path | None = None) -> dict[str, TopologySpec]:
    """Return the existing-nominal specs for the three evaluated topologies."""
    return {
        "anytown": TopologySpec(
            "anytown", 86_400, SCAN_STEP,
            LOGIC_ATTACKS_ROOT / "anytown" / "nominal",
            LOGIC_ATTACKS_ROOT / "anytown" / "topology.inp",
        ),
        "ctown": TopologySpec(
            "ctown", 86_400, SCAN_STEP,
            LOGIC_ATTACKS_ROOT / "ctown" / "nominal",
            LOGIC_ATTACKS_ROOT / "ctown" / "topology.inp",
        ),
        "ltown": TopologySpec(
            "ltown", 172_800, SCAN_STEP,
            Path(ltown_nominal) if ltown_nominal is not None else LTOWN_NOMINAL_DIR,
            Path(ltown_topology) if ltown_topology is not None else LTOWN_TOPOLOGY,
            "PDA",
        ),
    }


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_nominal(nominal_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load an existing written nominal bundle without simulating anything."""
    nominal_dir = Path(nominal_dir)
    truth = nominal_dir / "plant_truth.csv"
    manifest_path = nominal_dir / "manifest.json"
    if not truth.is_file() or not manifest_path.is_file():
        raise SupervisedEpisodeError(
            f"nominal is incomplete (need plant_truth.csv and manifest.json): {nominal_dir}")
    frame = pd.read_csv(truth, float_precision="round_trip")
    return frame, _read_json(manifest_path)


def validate_nominal(topology: str, frame: pd.DataFrame,
                     manifest: Mapping[str, Any], spec: TopologySpec) -> None:
    duration = int(manifest.get("timesteps", {}).get("duration", -1))
    scan = int(manifest.get("effective_timing", {}).get("scan_timestep", -1))
    if duration != spec.duration or scan != spec.scan_step:
        raise SupervisedEpisodeError(
            f"{topology} nominal must be {spec.duration} s at {spec.scan_step} s "
            f"(got {duration} s at {scan} s)")
    if topology == "ltown":
        demand = str(manifest.get("hydraulic_options", {}).get("demand_model", "")).upper()
        if demand not in ("PDA", "PDD"):
            raise SupervisedEpisodeError("L-Town nominal must be the Real/PDD nominal")
    if frame.empty or "time_seconds" not in frame:
        raise SupervisedEpisodeError(f"{topology} nominal has no time grid")
    controls = manifest.get("controls")
    if not isinstance(controls, list) or not controls:
        raise SupervisedEpisodeError(f"{topology} nominal declares no controls")


def source_metadata(nominal_dir: Path, inp_path: Path,
                    spec: TopologySpec) -> dict[str, Any]:
    """Describe every input hash without rewriting any input."""
    nominal_dir = Path(nominal_dir)
    plant_truth = nominal_dir / "plant_truth.csv"
    manifest_path = nominal_dir / "manifest.json"
    metadata: dict[str, Any] = {
        "topology": spec.name,
        "nominal_dir": str(nominal_dir),
        "nominal_plant_truth_path": str(plant_truth),
        "nominal_plant_truth_sha256": _sha256_file(plant_truth),
        "nominal_manifest_path": str(manifest_path),
        "nominal_manifest_sha256": _sha256_file(manifest_path),
        "resolved_inp_path": str(inp_path),
        "resolved_inp_sha256": _sha256_file(inp_path),
        "source_inp_path": None,
        "source_inp_sha256": None,
        "source_sha256": None,
        "source_origin": None,
    }
    nominal_json = nominal_dir / "nominal.json"
    scenario_json = nominal_dir / "scenario.json"
    source_inp: str | None = None
    source_sha: str | None = None
    if nominal_json.is_file():
        payload = _read_json(nominal_json)
        source_inp = payload.get("source_inp")
        source_sha = payload.get("source_sha256")
        metadata["source_origin"] = str(nominal_json)
        metadata["source_metadata_sha256"] = _sha256_file(nominal_json)
    elif scenario_json.is_file():
        payload = _read_json(scenario_json)
        inputs = payload.get("input", {})
        source_inp = inputs.get("source") or inputs.get("resolved_from") or inputs.get("preferred")
        source_sha = inputs.get("source_sha256")
        metadata["source_origin"] = str(scenario_json)
        metadata["source_metadata_sha256"] = _sha256_file(scenario_json)
    if source_inp is not None:
        metadata["source_inp_path"] = str(source_inp)
        path = Path(source_inp)
        metadata["source_inp_sha256"] = _sha256_file(path) if path.is_file() else None
    metadata["source_sha256"] = (
        metadata["source_inp_sha256"] or source_sha or metadata["resolved_inp_sha256"])
    return metadata


# ---------------------------------------------------------------------------
# held-out exclusion
# ---------------------------------------------------------------------------

def held_out_windows(topology: str, *, logic_root: Path = LOGIC_ATTACKS_ROOT,
                     required: bool = True) -> dict[str, Any]:
    """Read only the held-out plan windows so selected windows cannot match."""
    plan_path = Path(logic_root) / topology / "plan.json"
    if not plan_path.is_file():
        if required:
            raise SupervisedEpisodeError(
                f"held-out plan is required for disjoint windows: {plan_path}")
        return {"available": False, "plan_path": str(plan_path),
                "plan_sha256": None, "excluded_window_count": 0}
    payload = _read_json(plan_path)
    excluded: set[tuple[str, int]] = set()
    excluded_times: set[int] = set()
    for window in payload.get("guard_windows", []):
        control = window.get("control_id")
        link = window.get("link")
        sensor = window.get("sensor")
        time = window.get("time_seconds")
        if time is None:
            continue
        excluded_times.add(int(time))
        for name in (control, link, sensor):
            if name:
                excluded.add((str(name), int(time)))
    for scenario in payload.get("scenarios", []):
        for event in scenario.get("events", []):
            target = event.get("target")
            start = event.get("start")
            if target is not None and start is not None:
                excluded_times.add(int(start))
                excluded.add((str(target), int(start)))
    return {
        "available": True,
        "plan_path": str(plan_path),
        "plan_sha256": _sha256_file(plan_path),
        "excluded_window_count": len(excluded),
        "excluded_time_count": len(excluded_times),
        "policy": "held-out window metadata used for exclusion only; labels and detector scores are never read",
        "_excluded": excluded,
        "_excluded_times": excluded_times,
    }


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def pump_links(manifest: Mapping[str, Any]) -> set[str]:
    """Return pump ids from the nominal manifest's ``speed_`` channels."""
    return {
        column[len("speed_"):]
        for column in manifest.get("expected_columns", [])
        if column.startswith("speed_")
    }


def candidate_windows(frame: pd.DataFrame, controls: Iterable[Mapping[str, Any]],
                      *, step: int, duration: int,
                      pumps: set[str]) -> list[dict[str, Any]]:
    """Enumerate source-derived guard windows for pump-controlling rules."""
    times = frame["time_seconds"].astype(int).tolist()
    windows: list[dict[str, Any]] = []
    for control in controls:
        link = str(control["link"])
        relation = str(control.get("relation"))
        if link not in pumps or relation not in ("lt", "gt"):
            continue
        sensor = str(control["sensor"])
        if sensor not in frame.columns:
            raise SupervisedEpisodeError(f"nominal has no channel for guard sensor {sensor!r}")
        values = frame[sensor].astype(float).to_numpy()
        threshold = float(control["threshold"])

        def holds(value: float) -> bool:
            return value < threshold if relation == "lt" else value > threshold

        states = [holds(float(value)) for value in values]
        transitions = [index for index in range(len(states))
                       if states[index] and (index == 0 or not states[index - 1])]
        nominal_status = "OPEN" if int(control["value"]) == 1 else "CLOSED"
        forced_status = "CLOSED" if nominal_status == "OPEN" else "OPEN"
        for index, time in enumerate(times):
            if time + step > duration:
                continue
            nearest = min(transitions, key=lambda item: (abs(item - index), item)) if transitions else None
            windows.append({
                "control_id": str(control["id"]),
                "link": link,
                "sensor": sensor,
                "relation": relation,
                "threshold": threshold,
                "direction": (("above_high" if relation == "gt" else "below_low")
                              if states[index] else "guard_inactive"),
                "scan_index": int(index),
                "time_seconds": int(time),
                "sensor_value": float(values[index]),
                "nominal_status": nominal_status,
                "forced_status": forced_status,
                "guard_state": bool(states[index]),
                "anchor_transition_time_seconds": (
                    int(times[nearest]) if nearest is not None else None),
                "anchor_transition_scan_index": (int(nearest) if nearest is not None else None),
            })
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
        raise SupervisedEpisodeError(f"no guard threshold for sensor {sensor!r}")
    return max(abs(min(bounds)) * 0.1, 1e-9)


def _required_scans(attack_class: str, *, sensor_window_scans: int,
                    dos_window_scans: int) -> int:
    if attack_class in SENSOR_CLASSES:
        return sensor_window_scans
    if attack_class == "actuator_dos":
        return dos_window_scans
    return 1


def _window_key(window: Mapping[str, Any]) -> tuple[str, int]:
    return str(window["control_id"]), int(window["time_seconds"])


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def _rank(topology: str, split: str, attack_class: str,
          window: Mapping[str, Any]) -> tuple[int, str]:
    material = (
        f"{topology}|{split}|{attack_class}|{window['control_id']}|"
        f"{window['time_seconds']}"
    ).encode("utf-8")
    return (0 if window["guard_state"] else 1,
            hashlib.sha256(material).hexdigest())


def _taxonomy() -> dict[str, str]:
    from simulation.cyber import ATTACK_CLASS_TAXONOMY

    return dict(ATTACK_CLASS_TAXONOMY)


def _event_for(attack_class: str, window: Mapping[str, Any], *, step: int,
               sensor_window_scans: int, dos_window_scans: int,
               replay_period_scans: int, offset: float,
               adaptive: float) -> list[dict[str, Any]]:
    start = int(window["time_seconds"])
    if attack_class == "control_guard_violation":
        return [{
            "kind": "actuator", "target": window["link"],
            "value": window["forced_status"], "start": start,
            "end": start + step, "attack_class": attack_class,
        }]
    if attack_class == "actuator_dos":
        return [{
            "kind": "actuator", "target": window["link"],
            "value": window["forced_status"], "start": start,
            "end": start + dos_window_scans * step, "attack_class": attack_class,
        }]
    if attack_class == "actuator_feedback_spoof":
        return [{
            "kind": "actuator_feedback", "target": window["link"],
            "value": window["forced_status"], "scope": "scada",
            "start": start, "end": start + step, "attack_class": attack_class,
        }]
    if attack_class == "coordinated":
        return [
            {
                "kind": "sensor", "target": window["sensor"], "mode": "freeze",
                "scope": "both", "start": start, "end": start + step,
                "attack_class": attack_class,
            },
            {
                "kind": "actuator", "target": window["link"],
                "value": window["forced_status"], "start": start,
                "end": start + step, "attack_class": attack_class,
            },
        ]
    event: dict[str, Any] = {
        "kind": "sensor", "target": window["sensor"],
        "mode": "freeze", "scope": "both", "start": start,
        "end": start + sensor_window_scans * step, "attack_class": attack_class,
    }
    if attack_class == "sensor_offset":
        event["mode"] = "offset"
        event["value"] = offset
    elif attack_class == "sensor_replay":
        event["mode"] = "replay"
        event["period"] = replay_period_scans
        event["noise"] = 0.0
    elif attack_class == "adaptive":
        event["mode"] = "offset"
        event["value"] = adaptive
    elif attack_class == "sensor_freeze":
        event["mode"] = "freeze"
    else:  # pragma: no cover - guarded by the class list
        raise SupervisedEpisodeError(f"unsupported attack class: {attack_class}")
    return [event]


def build_plan(topology: str, frame: pd.DataFrame,
               manifest: Mapping[str, Any], spec: TopologySpec, *,
               source: Mapping[str, Any] | None = None,
               logic_root: Path = LOGIC_ATTACKS_ROOT,
               sensor_window_scans: int = SENSOR_WINDOW_SCANS,
               dos_window_scans: int = DOS_WINDOW_SCANS,
               replay_period_scans: int | None = None,
               adaptive_fraction: float = ADAPTIVE_FRACTION,
               require_held_out: bool = True) -> dict[str, Any]:
    """Build the split plan without reading any evaluation result."""
    step = int(spec.scan_step)
    duration = int(spec.duration)
    controls = list(manifest.get("controls", []))
    pumps = pump_links(manifest)
    if not pumps:
        raise SupervisedEpisodeError(
            f"{topology} nominal manifest declares no pump channels")
    windows = candidate_windows(
        frame, controls, step=step, duration=duration, pumps=pumps)
    if not windows:
        raise SupervisedEpisodeError(
            f"{topology} nominal has no source-derived pump guard windows")
    held = held_out_windows(topology, logic_root=logic_root, required=require_held_out)
    excluded: set[tuple[str, int]] = held.pop("_excluded", set())
    excluded_times: set[int] = held.pop("_excluded_times", set())
    period = (REPLAY_PERIOD_SCANS[topology] if replay_period_scans is None
              else int(replay_period_scans))
    if period < 1:
        raise SupervisedEpisodeError("replay period must be positive")

    def fits(attack_class: str, window: Mapping[str, Any]) -> bool:
        scans = _required_scans(
            attack_class, sensor_window_scans=sensor_window_scans,
            dos_window_scans=dos_window_scans)
        if int(window["time_seconds"]) + scans * step > duration:
            return False
        if attack_class in REQUIRE_GUARD_TRUE and not window["guard_state"]:
            return False
        if attack_class == "actuator_dos" and window["forced_status"] != "CLOSED":
            return False
        if attack_class == "sensor_replay" and int(window["time_seconds"]) < period * step:
            return False
        return True

    selected: dict[str, dict[str, Any]] = {}
    used_keys: set[tuple[str, int]] = set()
    used_times: set[int] = set()

    def choose(attack_class: str) -> dict[str, Any]:
        split = CLASS_SPLIT[attack_class]
        candidates = [
            window for window in windows
            if int(window["time_seconds"]) not in excluded_times
            and window["anchor_transition_time_seconds"] is not None
            and _window_key(window) not in excluded
            and (window["link"], int(window["time_seconds"])) not in excluded
            and (window["sensor"], int(window["time_seconds"])) not in excluded
            and _window_key(window) not in used_keys
            and int(window["time_seconds"]) not in used_times
            and fits(attack_class, window)
        ]
        if not candidates:
            raise SupervisedEpisodeError(
                f"{topology} has no unused source-derived window for {attack_class} "
                f"({split}); add a distinct nominal observation or relax the class")
        candidates.sort(key=lambda window: _rank(topology, split, attack_class, window))
        window = candidates[0]
        used_keys.add(_window_key(window))
        used_times.add(int(window["time_seconds"]))
        return window

    order = [name for name in REQUIRED_ORDER if name in ALL_CLASSES]
    order.extend(name for name in ALL_CLASSES if name not in order)
    for attack_class in order:
        selected[attack_class] = choose(attack_class)

    source = dict(source) if source is not None else source_metadata(
        spec.nominal_dir, spec.inp_path, spec)
    scenarios: list[dict[str, Any]] = []
    for attack_class in ALL_CLASSES:
        window = selected[attack_class]
        split = CLASS_SPLIT[attack_class]
        seed = SPLIT_SEEDS[split]
        offset = offset_magnitude(window["sensor"], controls, fraction=0.5)
        adaptive = offset_magnitude(
            window["sensor"], controls, fraction=adaptive_fraction)
        events = _event_for(
            attack_class, window, step=step,
            sensor_window_scans=sensor_window_scans,
            dos_window_scans=dos_window_scans,
            replay_period_scans=period, offset=offset, adaptive=adaptive)
        target = window["sensor"] if attack_class in SENSOR_CLASSES else window["link"]
        name = f"{attack_class}_{_slug(target)}_{int(window['time_seconds'])}"
        guard_text = "active" if window["guard_state"] else "inactive"
        rationale = (
            f"{attack_class} anchored at nominal scan {window['time_seconds']} s "
            f"where {window['sensor']}={window['sensor_value']:.6g} and the "
            f"{window['control_id']} guard {window['relation']} "
            f"{window['threshold']:g} is {guard_text}; nearest source transition "
            f"at {window['anchor_transition_time_seconds']} s. Window selected "
            "deterministically from existing nominal observations after excluding "
            "held-out plan windows."
        )
        scenarios.append({
            "name": name,
            "split": split,
            "seed": seed,
            "attack_class": attack_class,
            "description": f"{attack_class} episode on {target} at {window['time_seconds']} s",
            "dir": f"{split}/{name}",
            "events": events,
            "labels": {
                "source": "injected_events_and_attack_free_physical_comparison",
                "attack_class": "per-row from simulation.cyber",
                "battledim_leak_labels_used": False,
                "inherited_leak_labels": False,
            },
            "selection": {
                "window": dict(window),
                "rationale": rationale,
                "magnitude": (
                    offset if attack_class == "sensor_offset"
                    else adaptive if attack_class == "adaptive"
                    else f"{_required_scans(attack_class, sensor_window_scans=sensor_window_scans, dos_window_scans=dos_window_scans)} scans"),
                "magnitude_source": (
                    "half the gap between the source low and high thresholds"
                    if attack_class == "sensor_offset"
                    else f"{adaptive_fraction:g} of the source low/high threshold gap"
                    if attack_class == "adaptive"
                    else "source guard window and class window length"),
                "detector_results_consulted": False,
            },
        })

    return {
        "schema_version": 1,
        "kind": "supervised_attack_episode_plan",
        "topology": topology,
        "duration": duration,
        "scan_step": step,
        "splits": {
            "training": {"seed": TRAIN_SEED, "attack_classes": list(TRAIN_CLASSES)},
            "validation": {"seed": VALIDATION_SEED, "attack_classes": list(VALIDATION_CLASSES)},
        },
        "nominal": source,
        "held_out": held,
        "provenance": {
            "generated_before_fitting": True,
            "fitting_performed": False,
            "evaluation_outputs_consulted": False,
            "detector_scores_consulted": False,
            "inherited_leak_labels": False,
            "new_nominal_simulation": False,
            "nominal_source_policy": "existing nominal traces only",
            "held_out_windows_used_for_exclusion_only": True,
        },
        "selection_policy": {
            "rule": (
                "deterministic SHA-256 rank over source-derived guard windows; "
                "actuator classes require an active guard; actuator DOS requires "
                "the CLOSED forced status; replay requires pre-attack history"
            ),
            "exclusion": "windows matching the held-out plan are removed before selection",
            "splits_disjoint": True,
            "training_seed": TRAIN_SEED,
            "validation_seed": VALIDATION_SEED,
        },
        "attack_taxonomy": _taxonomy(),
        "scenarios": scenarios,
    }


# ---------------------------------------------------------------------------
# writing and execution
# ---------------------------------------------------------------------------

def _write_json(path: Path, payload: Mapping[str, Any], *,
                overwrite: bool = False) -> None:
    path = Path(path)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return
        if not overwrite:
            raise SupervisedEpisodeExistsError(
                f"refusing to replace different artifact {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.",
        dir=path.parent, delete=False)
    try:
        with handle:
            handle.write(text)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _check_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and path.is_dir() and any(path.iterdir()) and not overwrite:
        raise SupervisedEpisodeExistsError(
            f"refusing to overwrite existing artifact {path}; pass --overwrite")


def build_manifest(plan: Mapping[str, Any], plan_sha256: str) -> dict[str, Any]:
    scenarios = []
    for scenario in plan["scenarios"]:
        window = scenario["selection"]["window"]
        scenarios.append({
            "name": scenario["name"],
            "split": scenario["split"],
            "seed": scenario["seed"],
            "attack_class": scenario["attack_class"],
            "dir": scenario["dir"],
            "window": {
                "control_id": window["control_id"],
                "link": window["link"],
                "sensor": window["sensor"],
                "time_seconds": window["time_seconds"],
                "guard_state": window["guard_state"],
                "forced_status": window["forced_status"],
            },
            "events": scenario["events"],
        })
    return {
        "schema_version": 1,
        "kind": "supervised_attack_episode_manifest",
        "topology": plan["topology"],
        "duration": plan["duration"],
        "scan_step": plan["scan_step"],
        "plan_file": "plan.json",
        "plan_sha256": plan_sha256,
        "splits": plan["splits"],
        "nominal": plan["nominal"],
        "held_out": plan["held_out"],
        "provenance": plan["provenance"],
        "scenarios": scenarios,
    }


def write_plan(out_root: Path, topology: str, plan: Mapping[str, Any], *,
               overwrite: bool = False) -> Path:
    """Write the plan and manifest before any episode is simulated."""
    topology_dir = Path(out_root) / topology
    plan_path = topology_dir / "plan.json"
    _write_json(plan_path, plan, overwrite=overwrite)
    manifest = build_manifest(plan, _sha256_file(plan_path))
    _write_json(topology_dir / "manifest.json", manifest, overwrite=overwrite)
    return topology_dir


def execute_scenarios(topology: str, spec: TopologySpec,
                      plan: Mapping[str, Any], *, out_root: Path,
                      overwrite: bool = False,
                      runner: Any | None = None,
                      writer: Any | None = None,
                      run_spec_factory: Any | None = None) -> list[Path]:
    """Run each planned scenario as an independent cyber bundle."""
    if runner is None:
        from simulation.cyber import run_cyber as runner
    if writer is None:
        from simulation.cyber import write_cyber_bundle as writer
    if run_spec_factory is None:
        from simulation import RunSpec as run_spec_factory

    topology_dir = Path(out_root) / topology
    if not (topology_dir / "plan.json").is_file() or not (
            topology_dir / "manifest.json").is_file():
        raise SupervisedEpisodeError(
            "plan and manifest must be written before episodes are executed")
    outputs: list[Path] = []
    for scenario in plan["scenarios"]:
        out_dir = topology_dir / scenario["dir"]
        _check_output_dir(out_dir, overwrite)
        outputs.append(out_dir)
    for scenario, out_dir in zip(plan["scenarios"], outputs):
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
        try:
            run_id = f"{topology}-supervised-{scenario['split']}-{scenario['name']}"
            run_spec = run_spec_factory(
                spec.inp_path, run_id, int(scenario["seed"]), scenario["split"])
            bundle = runner(run_spec, list(scenario["events"]))
            writer(bundle, staging)
            _write_json(staging / "scenario.json", {
                "schema_version": 1,
                "kind": "supervised_attack_episode",
                "topology": topology,
                "scenario": scenario["name"],
                "description": scenario["description"],
                "attack_class": scenario["attack_class"],
                "run_id": run_id,
                "split": scenario["split"],
                "seed": int(scenario["seed"]),
                "duration": spec.duration,
                "scan_step": spec.scan_step,
                "events": list(scenario["events"]),
                "selection": dict(scenario["selection"]),
                "labels": dict(scenario["labels"]),
                "provenance": {
                    "plan_file": "plan.json",
                    "plan_written_before_execution": True,
                    "evaluation_outputs_consulted": False,
                    "detector_scores_consulted": False,
                    "inherited_leak_labels": False,
                },
            })
            if out_dir.exists():
                if not overwrite:
                    raise SupervisedEpisodeExistsError(
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

def _parse_topologies(values: Sequence[str]) -> list[str]:
    names: list[str] = []
    for value in values:
        names.extend(part for part in value.split(",") if part)
    if "all" in names:
        return ["anytown", "ctown", "ltown"]
    unknown = sorted(set(names) - {"anytown", "ctown", "ltown"})
    if unknown:
        raise SupervisedEpisodeError(f"unknown topologies: {unknown}")
    return names


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--topologies", nargs="+", default=["all"],
                        metavar="NAME",
                        help="topologies to plan (anytown, ctown, ltown, all)")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plan-only", action="store_true",
                        help="write plans/manifests but do not simulate episodes")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        specs = topology_specs()
        topologies = _parse_topologies(args.topologies)
    except SupervisedEpisodeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for topology in topologies:
        spec = specs[topology]
        try:
            frame, manifest = load_nominal(spec.nominal_dir)
            validate_nominal(topology, frame, manifest, spec)
            source = source_metadata(spec.nominal_dir, spec.inp_path, spec)
            plan = build_plan(topology, frame, manifest, spec, source=source)
            write_plan(args.out_root, topology, plan, overwrite=args.overwrite)
            print(f"planned {topology}: {len(plan['scenarios'])} scenarios -> "
                  f"{args.out_root / topology}")
            if args.plan_only:
                continue
            outputs = execute_scenarios(
                topology, spec, plan, out_root=args.out_root,
                overwrite=args.overwrite)
            for output in outputs:
                print(output)
        except SupervisedEpisodeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
