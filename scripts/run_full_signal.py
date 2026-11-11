#!/usr/bin/env python3
"""Run reproducible full-signal cyber-physical scenarios.

The script wraps :func:`simulation.cyber.run_cyber` with the scenario
configurations used by the ``scripts/out/full_signal`` artifacts.  Run IDs,
splits and event windows are preserved, but every L-Town scenario now patches
the resolved INP to WNTR pressure-dependent demand (PDD): minimum pressure
0.0, required pressure 0.07 and pressure exponent 0.5.  L-Town traces are
therefore **not** byte-identical to the legacy demand-driven (DDA) artifacts.

For L-Town the fully resolved INP is published to
``<out-root>/ltown/topology.inp`` so export and tolerance scripts load the
same PDD network.  The script never writes into an existing scenario
directory, and never replaces a different resolved topology, unless
``--overwrite`` is supplied.  The raw benchmark INP is never modified.

Labels come from :mod:`simulation.cyber`: the ``attack_active`` flag is
derived from the injected event windows, ``delivery_or_command_modified``
from the scan-time delivery/command comparison, and ``physical_impact`` from
an attack-free closed-loop rerun.  BattLeDIM leak labels are never read,
imported or invented here.

Usage
-----
    python scripts/run_full_signal.py --topology ltown --scenario nominal_two_day
    python scripts/run_full_signal.py --topology ltown --out-root scripts/out/full_signal_pdd --scenario nominal_two_day t1_offset_48h t1_replay_48h pump1_dos_48h feedback_two_day
    python scripts/run_full_signal.py --list-scenarios
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_SIGNAL_ROOT = REPO_ROOT / "scripts" / "out" / "full_signal"
LTOWN_FALLBACK_INP = REPO_ROOT / "data" / "ltown" / "raw" / "L-TOWN_Real.inp"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

LABEL_POLICY = {
    "source": "injected_events_and_attack_free_physical_comparison",
    "battledim_leak_labels_used": False,
    "attack_active": "true while an injected event window contains the scan time",
    "delivery_or_command_modified": "scan-time sensor delivery or resolved command differs from the nominal value",
    "physical_impact": "plant truth differs from the attack-free closed-loop rerun (rtol=1e-9, atol=1e-10)",
}


class ScenarioError(RuntimeError):
    """Raised for unknown scenarios, unsafe paths or output collisions."""


class ScenarioExistsError(ScenarioError):
    """Raised for an existing run or a different resolved topology."""


@dataclass(frozen=True)
class DemandModel:
    """Hydraulic demand model written into the resolved INP ``[OPTIONS]``.

    WNTR calls pressure-dependent analysis ``PDA`` in EPANET INP files (and
    ``PDD`` in its Python API); both names select the same solver mode.
    """

    model: str = "PDA"
    minimum_pressure: float = 0.0
    required_pressure: float = 0.07
    pressure_exponent: float = 0.5

    def options(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "minimum_pressure": self.minimum_pressure,
            "required_pressure": self.required_pressure,
            "pressure_exponent": self.pressure_exponent,
        }


LTOWN_DEMAND_MODEL = DemandModel()


@dataclass(frozen=True)
class Scenario:
    name: str
    run_id: str
    split: str
    duration: int
    scan_step: int
    events: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    description: str = ""
    aliases: tuple[str, ...] = ()
    demand_model: DemandModel | None = None


T1_START = 86_400
TWO_DAY = 172_800
FEEDBACK_EVENTS = (
    {"kind": "sensor", "target": "T1", "mode": "constant", "value": 0.0,
     "scope": "both", "start": T1_START, "end": TWO_DAY},
)
T1_OFFSET_EVENTS = (
    {"kind": "sensor", "target": "T1", "mode": "offset", "value": 1.0,
     "scope": "both", "start": T1_START, "end": TWO_DAY},
)
# 86_400 / 300 = 288 scan steps, so every replay source lies strictly before
# the attack start: only pre-attack delivered history is ever replayed.
T1_REPLAY_EVENTS = (
    {"kind": "sensor", "target": "T1", "mode": "replay", "period": 288,
     "noise": 0.0, "scope": "both", "start": T1_START, "end": TWO_DAY},
)
PUMP1_DOS_EVENTS = (
    {"kind": "actuator", "target": "PUMP_1", "value": "CLOSED",
     "start": T1_START, "end": TWO_DAY},
)


SCENARIOS: dict[str, dict[str, Scenario]] = {
    "anytown": {
        "nominal": Scenario(
            "nominal", "anytown-nominal-full", "calibration", 3_000, 1,
            description="attack-free 3000 s full signal",
        ),
        "dos": Scenario(
            "dos", "anytown-dos-full", "evaluation", 1_000, 1,
            events=({"kind": "sensor", "target": "T41", "mode": "freeze",
                     "scope": "both", "start": 649, "end": 794},),
            description="T41 delivery freeze, both recipients",
        ),
        "mitm": Scenario(
            "mitm", "anytown-mitm-full", "evaluation", 1_000, 1,
            events=({"kind": "sensor", "target": "T41", "mode": "offset",
                     "value": 7.0, "scope": "both", "start": 649, "end": 794},),
            description="T41 7.0 m offset, both recipients",
        ),
        "adaptive_bounded_offset": Scenario(
            "adaptive_bounded_offset", "anytown-adaptive_bounded_offset",
            "evaluation", 1_000, 1,
            events=({"kind": "sensor", "target": "T41", "mode": "offset",
                     "value": 1e-06, "scope": "both", "start": 649, "end": 794},),
            description="T41 1e-06 m bounded offset, both recipients",
        ),
        "adaptive_coordinated_freeze": Scenario(
            "adaptive_coordinated_freeze",
            "anytown-adaptive_coordinated_freeze", "evaluation", 1_000, 1,
            events=(
                {"kind": "actuator", "target": "P78", "value": "OPEN",
                 "start": 649, "end": 794},
                {"kind": "sensor", "target": "T41", "mode": "freeze",
                 "scope": "both", "start": 649, "end": 794},
            ),
            description="P78 override plus T41 delivery freeze",
        ),
    },
    "ctown": {
        "nominal_short": Scenario(
            "nominal_short", "ctown-nominal-short", "calibration", 200, 1,
            description="attack-free 200 s full signal",
        ),
        "pump_activation": Scenario(
            "pump_activation", "ctown-pump_activation-short", "evaluation",
            200, 1,
            events=({"kind": "actuator", "target": "PU3", "value": "OPEN",
                     "start": 100, "end": 150},),
            description="PU3 actuator activation",
        ),
        "replay_pump": Scenario(
            "replay_pump", "ctown-replay_pump-short", "evaluation", 200, 1,
            events=(
                {"kind": "actuator", "target": "PU10", "value": "CLOSED",
                 "start": 100, "end": 150},
                {"kind": "actuator", "target": "PU11", "value": "CLOSED",
                 "start": 100, "end": 150},
                {"kind": "sensor", "target": "T7", "mode": "replay",
                 "period": 10, "noise": 0.0, "scope": "both",
                 "start": 100, "end": 150},
            ),
            description="pump override plus recipient-specific T7 replay",
        ),
    },
    "ltown": {
        "nominal_two_day": Scenario(
            "nominal_two_day", "ltown-nominal-two-day", "validation",
            TWO_DAY, 300, description="attack-free 48 h full signal",
            demand_model=LTOWN_DEMAND_MODEL,
        ),
        "feedback_two_day": Scenario(
            "feedback_two_day", "ltown-feedback-two-day", "evaluation",
            TWO_DAY, 300, events=FEEDBACK_EVENTS,
            description="T1 constant 0.0 feedback attack, both recipients",
            demand_model=LTOWN_DEMAND_MODEL,
        ),
        "t1_offset_48h": Scenario(
            "t1_offset_48h", "ltown-t1-offset-48h", "evaluation", TWO_DAY,
            300, events=T1_OFFSET_EVENTS,
            description="T1 1.0 m offset from second day, both recipients",
            aliases=("t1_offset",),
            demand_model=LTOWN_DEMAND_MODEL,
        ),
        "t1_replay_48h": Scenario(
            "t1_replay_48h", "ltown-t1-replay-48h", "evaluation", TWO_DAY,
            300, events=T1_REPLAY_EVENTS,
            description="recipient-specific T1 replay of pre-attack day",
            aliases=("t1_replay",),
            demand_model=LTOWN_DEMAND_MODEL,
        ),
        "pump1_dos_48h": Scenario(
            "pump1_dos_48h", "ltown-pump1-dos-48h", "evaluation", TWO_DAY,
            300, events=PUMP1_DOS_EVENTS,
            description="PUMP_1 actuator denial of service from second day",
            aliases=("pump1_dos",),
            demand_model=LTOWN_DEMAND_MODEL,
        ),
    },
}


def scenarios_for(topology: str) -> dict[str, Scenario]:
    try:
        return SCENARIOS[topology]
    except KeyError as exc:
        raise ScenarioError(
            f"unknown topology {topology!r}; choose from {sorted(SCENARIOS)}"
        ) from exc


def get_scenario(topology: str, name: str) -> Scenario:
    scenarios = scenarios_for(topology)
    if name in scenarios:
        return scenarios[name]
    for scenario in scenarios.values():
        if name in scenario.aliases:
            return scenario
    raise ScenarioError(
        f"unknown scenario {name!r} for {topology}; "
        f"choose from {sorted(scenarios)}"
    )


def build_events(scenario: Scenario) -> list[dict[str, Any]]:
    """Return a fresh, unannotated copy of the scenario's injected events."""
    return copy.deepcopy([dict(event) for event in scenario.events])


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_name(value: str, kind: str) -> str:
    if not _SAFE_NAME.match(value) or value in {".", ".."}:
        raise ScenarioError(f"unsafe {kind} name {value!r}")
    return value


def scenario_output_dir(topology: str, scenario: Scenario, out_root: Path) -> Path:
    return Path(out_root) / _validate_name(topology, "topology") / _validate_name(
        scenario.name, "scenario"
    )


def _dedupe(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def candidate_inputs(
    topology: str, input_root: Path, out_root: Path
) -> list[Path]:
    topology = _validate_name(topology, "topology")
    candidates = [
        Path(input_root) / topology / "topology.inp",
        Path(out_root) / topology / "topology.inp",
        DEFAULT_SIGNAL_ROOT / topology / "topology.inp",
    ]
    if topology == "ltown":
        candidates.append(LTOWN_FALLBACK_INP)
    else:
        candidates.append(REPO_ROOT / "data" / topology / f"{topology}.inp")
    return _dedupe(candidates)


def resolve_input(
    topology: str,
    *,
    input_root: Path = DEFAULT_SIGNAL_ROOT,
    out_root: Path = DEFAULT_SIGNAL_ROOT,
    explicit: str | Path | None = None,
) -> tuple[Path, bool]:
    """Return ``(input_path, used_fallback)``.

    The preferred topology artefact is ``<input_root>/<topology>/topology.inp``.
    L-Town falls back to the raw ``L-TOWN_Real.inp`` benchmark when that
    artefact is absent.
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ScenarioError(f"explicit input does not exist: {path}")
        return path, False
    candidates = candidate_inputs(topology, Path(input_root), Path(out_root))
    for index, path in enumerate(candidates):
        if path.is_file():
            return path, index > 0
    listed = ", ".join(str(path) for path in candidates)
    raise ScenarioError(f"no input INP for {topology}; tried: {listed}")


def check_output_dir(out_dir: Path, overwrite: bool) -> None:
    """Raise before simulating when an existing run would be overwritten."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return
    if not out_dir.is_dir():
        raise ScenarioError(f"output path is not a directory: {out_dir}")
    if not overwrite and any(out_dir.iterdir()):
        raise ScenarioExistsError(
            f"refusing to overwrite existing run {out_dir}; pass --overwrite"
        )


def _format_hms(seconds: int | float) -> str:
    total = int(seconds)
    if total < 0 or total != float(seconds):
        raise ScenarioError(f"timing value must be a nonnegative whole second: {seconds!r}")
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


_TIMING_FIELDS = (
    ("DURATION", "duration"),
    ("HYDRAULIC TIMESTEP", "scan_step"),
    ("REPORT TIMESTEP", "scan_step"),
    ("REPORT START", "report_start"),
)


def patch_timing_text(text: str, scenario: Scenario) -> str:
    """Patch only the [TIMES] entries that define the scenario window.

    The topology file is otherwise copied verbatim so controls, patterns and
    network data are preserved.
    """
    values = {
        "duration": scenario.duration,
        "scan_step": scenario.scan_step,
        "report_start": 0,
    }
    lines = text.splitlines()
    in_times = False
    seen: set[str] = set()
    insert_at: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper() == "[TIMES]":
            in_times = True
            insert_at = index + 1
            continue
        if in_times and stripped.startswith("["):
            in_times = False
        if not in_times and stripped.upper() != "[TIMES]":
            continue
        for key, value_key in _TIMING_FIELDS:
            if re.match(rf"^\s*{re.escape(key)}\b", line, re.IGNORECASE):
                lines[index] = f"{key:<20}\t{_format_hms(values[value_key])}"
                seen.add(key)
                break
    if insert_at is None:
        raise ScenarioError("input has no [TIMES] section")
    missing = [
        (key, value_key)
        for key, value_key in _TIMING_FIELDS
        if key not in seen
    ]
    for offset, (key, value_key) in enumerate(missing):
        lines.insert(insert_at + offset, f"{key:<20}\t{_format_hms(values[value_key])}")
    trailing = "\n" if text.endswith("\n") else ""
    return "\n".join(lines) + trailing


_OPTIONS_FIELDS = (
    ("DEMAND MODEL", "model"),
    ("MINIMUM PRESSURE", "minimum_pressure"),
    ("REQUIRED PRESSURE", "required_pressure"),
    ("PRESSURE EXPONENT", "pressure_exponent"),
)


def _format_option_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(float(value))


def patch_demand_text(text: str, demand_model: DemandModel | None) -> str:
    """Patch only demand-model entries in the ``[OPTIONS]`` section.

    ``None`` leaves the text untouched, so Anytown and C-Town keep the demand
    model declared by their own INP files.  Any existing demand keys are
    removed and the complete block is appended at the end of the section, so
    it always follows ``UNITS`` (WNTR needs the flow units to convert
    pressure).  The rest of the input is preserved verbatim.
    """
    if demand_model is None:
        return text
    values = demand_model.options()
    patterns = [
        re.compile(rf"^\s*{re.escape(key)}\b", re.IGNORECASE)
        for key, _ in _OPTIONS_FIELDS
    ]
    lines = text.splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip().upper() == "[OPTIONS]"
        ),
        None,
    )
    if start is None:
        raise ScenarioError("input has no [OPTIONS] section")
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].strip().startswith("[")
        ),
        len(lines),
    )
    section = [
        line
        for line in lines[start + 1:end]
        if not any(pattern.match(line) for pattern in patterns)
    ]
    trimmed = list(section)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    trailing = section[len(trimmed):]
    block = [
        f"{key:<20}\t{_format_option_value(values[value_key])}"
        for key, value_key in _OPTIONS_FIELDS
    ]
    resolved = (
        lines[:start + 1] + trimmed + block + trailing + lines[end:]
    )
    return "\n".join(resolved) + ("\n" if text.endswith("\n") else "")


def apply_scenario_patches(text: str, scenario: Scenario) -> str:
    """Return the resolved INP for ``scenario`` without touching the source."""
    return patch_demand_text(
        patch_timing_text(text, scenario), scenario.demand_model
    )


def _sha256_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def resolved_topology_path(topology: str, out_root: Path) -> Path:
    return Path(out_root) / _validate_name(topology, "topology") / "topology.inp"


def publish_resolved_topology(
    path: str | Path, text: str, *, overwrite: bool
) -> Path:
    """Publish the resolved INP atomically, reusing an identical file.

    A different existing resolved topology is treated like an existing
    scenario run: it is only replaced when ``overwrite`` is true.
    """
    path = Path(path)
    payload = text.encode("utf-8")
    if path.exists():
        if not path.is_file():
            raise ScenarioError(f"resolved topology path is not a file: {path}")
        if path.read_bytes() == payload:
            return path
        if not overwrite:
            raise ScenarioExistsError(
                f"refusing to replace different resolved topology {path}; "
                "pass --overwrite"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
    )
    try:
        with handle:
            handle.write(payload)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return path


def _default_runner() -> Callable[..., Any]:
    from simulation.cyber import run_cyber

    return run_cyber


def _default_writer() -> Callable[..., None]:
    from simulation.cyber import write_cyber_bundle

    return write_cyber_bundle


def _write_scenario_metadata(
    path: Path,
    *,
    topology: str,
    scenario: Scenario,
    input_path: Path,
    source_hash: str,
    resolved_hash: str,
    used_fallback: bool,
    seed: int,
    resolved_topology: Path | None = None,
) -> None:
    metadata = {
        "schema_version": 1,
        "topology": topology,
        "scenario": scenario.name,
        "description": scenario.description,
        "run_id": scenario.run_id,
        "seed": seed,
        "split": scenario.split,
        "duration": scenario.duration,
        "scan_step": scenario.scan_step,
        "events": build_events(scenario),
        "labels": LABEL_POLICY,
        "demand_model": (
            scenario.demand_model.options() if scenario.demand_model else None
        ),
        "reproduction": {
            "legacy_dda_byte_identical": (
                False if scenario.demand_model is not None else None
            ),
            "note": (
                "L-Town scenarios resolve the INP to WNTR PDD; traces are not "
                "byte-identical to the legacy DDA artifacts. Event windows and "
                "run IDs are preserved."
                if scenario.demand_model is not None
                else "Demand model unchanged from the source INP."
            ),
        },
        "input": {
            "preferred": str(DEFAULT_SIGNAL_ROOT / topology / "topology.inp"),
            "resolved_from": str(input_path),
            "source_sha256": source_hash,
            "resolved_sha256": resolved_hash,
            "used_fallback": used_fallback,
        },
        "resolved_topology": (
            {"path": str(resolved_topology), "sha256": resolved_hash}
            if resolved_topology is not None
            else None
        ),
        "timing_patch": {
            "duration": scenario.duration,
            "hydraulic_timestep": scenario.scan_step,
            "report_timestep": scenario.scan_step,
            "report_start": 0,
        },
    }
    (path / "scenario.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def execute_scenario(
    scenario: Scenario,
    *,
    topology: str,
    input_path: Path,
    out_dir: Path,
    overwrite: bool,
    plcs_path: Path | None = None,
    seed: int | None = None,
    used_fallback: bool = False,
    runner: Callable[..., Any] | None = None,
    writer: Callable[..., None] | None = None,
    resolved_text: str | None = None,
    source_sha256: str | None = None,
) -> Path:
    """Simulate one scenario and publish it atomically into ``out_dir``.

    ``resolved_text`` and ``source_sha256`` let a caller reuse a resolved INP
    that was already published, avoiding a second read of a large source.
    """
    from simulation import RunSpec

    run = runner or _default_runner()
    write = writer or _default_writer()
    out_dir = Path(out_dir)
    check_output_dir(out_dir, overwrite)
    scenario_seed = 42 if seed is None else int(seed)
    events = build_events(scenario)
    if resolved_text is None:
        input_text = Path(input_path).read_text(encoding="utf-8", errors="strict")
        patched = apply_scenario_patches(input_text, scenario)
    else:
        patched = resolved_text
    if source_sha256 is None:
        source_sha256 = _sha256_file(Path(input_path))
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent)
    )
    try:
        with tempfile.TemporaryDirectory(prefix="full_signal_inp_") as work:
            resolved = Path(work) / "topology.inp"
            resolved.write_text(patched, encoding="utf-8")
            resolved_hash = hashlib.sha256(resolved.read_bytes()).hexdigest()
            spec = RunSpec(resolved, scenario.run_id, scenario_seed, scenario.split)
            bundle = run(spec, events, plcs_path=plcs_path)
            write(bundle, staging)
            _write_scenario_metadata(
                staging,
                topology=topology,
                scenario=scenario,
                input_path=Path(input_path),
                source_hash=source_sha256,
                resolved_hash=resolved_hash,
                used_fallback=used_fallback,
                seed=scenario_seed,
                resolved_topology=(
                    out_dir.parent / "topology.inp"
                    if topology == "ltown"
                    else None
                ),
            )
        if out_dir.exists():
            if not overwrite:
                raise ScenarioExistsError(
                    f"refusing to overwrite existing run {out_dir}; pass --overwrite"
                )
            shutil.rmtree(out_dir)
        os.replace(staging, out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return out_dir


def run_scenarios(
    topology: str,
    names: Sequence[str],
    *,
    out_root: Path = DEFAULT_SIGNAL_ROOT,
    input_root: Path = DEFAULT_SIGNAL_ROOT,
    overwrite: bool = False,
    inp: str | Path | None = None,
    plcs_path: str | Path | None = None,
    seed: int | None = None,
    runner: Callable[..., Any] | None = None,
    writer: Callable[..., None] | None = None,
) -> list[Path]:
    selected_by_name: dict[str, Scenario] = {}
    for name in names:
        scenario = get_scenario(topology, name)
        selected_by_name[scenario.name] = scenario
    selected = list(selected_by_name.values())
    outputs = [
        scenario_output_dir(topology, scenario, Path(out_root))
        for scenario in selected
    ]
    for out_dir in outputs:
        check_output_dir(out_dir, overwrite)
    input_path, used_fallback = resolve_input(
        topology, input_root=Path(input_root), out_root=Path(out_root), explicit=inp
    )
    resolved_text: str | None = None
    source_sha256: str | None = None
    if topology == "ltown":
        raw_text = Path(input_path).read_text(encoding="utf-8", errors="strict")
        resolved_texts = {
            apply_scenario_patches(raw_text, scenario) for scenario in selected
        }
        if len(resolved_texts) != 1:
            raise ScenarioError(
                "L-Town scenarios must share one resolved topology; "
                "their timing or demand patches differ"
            )
        resolved_text = resolved_texts.pop()
        source_sha256 = _sha256_file(Path(input_path))
        publish_resolved_topology(
            resolved_topology_path(topology, Path(out_root)),
            resolved_text,
            overwrite=overwrite,
        )
        del raw_text
    return [
        execute_scenario(
            scenario,
            topology=topology,
            input_path=input_path,
            out_dir=out_dir,
            overwrite=overwrite,
            plcs_path=Path(plcs_path) if plcs_path is not None else None,
            seed=seed,
            used_fallback=used_fallback,
            runner=runner,
            writer=writer,
            resolved_text=resolved_text,
            source_sha256=source_sha256,
        )
        for scenario, out_dir in zip(selected, outputs)
    ]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--topology", choices=sorted(SCENARIOS))
    parser.add_argument(
        "--scenario", nargs="+", metavar="NAME",
        help="one or more scenario names; defaults to every scenario for the topology",
    )
    parser.add_argument(
        "--out-root", "--out", dest="out_root", type=Path,
        default=DEFAULT_SIGNAL_ROOT,
        help=(
            "output root (default: scripts/out/full_signal); L-Town also "
            "publishes <out-root>/ltown/topology.inp"
        ),
    )
    parser.add_argument(
        "--input-root", type=Path, default=DEFAULT_SIGNAL_ROOT,
        help=(
            "root holding <topology>/topology.inp (default: "
            "scripts/out/full_signal); L-Town falls back to "
            "data/ltown/raw/L-TOWN_Real.inp"
        ),
    )
    parser.add_argument("--inp", type=Path, help="explicit topology INP override")
    parser.add_argument("--plcs", type=Path, help="optional DHALSIM-style plcs.yaml")
    parser.add_argument("--seed", type=int, help="override the scenario seed (default 42)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help=(
            "replace existing scenario directories and a different resolved "
            "<out-root>/ltown/topology.inp (default: refuse)"
        ),
    )
    parser.add_argument(
        "--list-scenarios", action="store_true",
        help="print available topologies, scenarios and demand models, then exit",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "resolve configurations and report input, PDD demand options and "
            "output paths without writing or simulating"
        ),
    )
    return parser.parse_args(argv)


def _demand_label(scenario: Scenario) -> str:
    if scenario.demand_model is None:
        return "inp-default"
    model = scenario.demand_model
    return (
        f"PDD({model.model} min={model.minimum_pressure:g} "
        f"req={model.required_pressure:g} exp={model.pressure_exponent:g})"
    )


def _print_scenarios() -> None:
    for topology, scenarios in SCENARIOS.items():
        print(topology)
        for scenario in scenarios.values():
            aliases = f" (aliases: {', '.join(scenario.aliases)})" if scenario.aliases else ""
            print(
                f"  {scenario.name:28s} {scenario.duration:7d} s  "
                f"{scenario.split:11s} demand={_demand_label(scenario)}{aliases}"
            )


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[..., Any] | None = None,
    writer: Callable[..., None] | None = None,
) -> int:
    args = _parse_args(argv)
    if args.list_scenarios:
        _print_scenarios()
        return 0
    if not args.topology:
        print("error: --topology is required unless --list-scenarios is used", file=sys.stderr)
        return 2
    try:
        available = scenarios_for(args.topology)
        names = args.scenario or list(available)
        if args.dry_run:
            input_path, used_fallback = resolve_input(
                args.topology,
                input_root=args.input_root,
                out_root=args.out_root,
                explicit=args.inp,
            )
            print(f"topology: {args.topology}")
            print(f"input: {input_path}" + (" (fallback)" if used_fallback else ""))
            if args.topology == "ltown":
                print(
                    "resolved topology: "
                    f"{resolved_topology_path(args.topology, args.out_root)}"
                )
            for name in names:
                scenario = get_scenario(args.topology, name)
                out_dir = scenario_output_dir(args.topology, scenario, args.out_root)
                print(
                    f"scenario: {scenario.name} run_id={scenario.run_id} "
                    f"split={scenario.split} duration={scenario.duration} "
                    f"step={scenario.scan_step} events={len(scenario.events)} "
                    f"demand={_demand_label(scenario)} out={out_dir}"
                )
            return 0
        outputs = run_scenarios(
            args.topology,
            names,
            out_root=args.out_root,
            input_root=args.input_root,
            overwrite=args.overwrite,
            inp=args.inp,
            plcs_path=args.plcs,
            seed=args.seed,
            runner=runner,
            writer=writer,
        )
    except ScenarioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
