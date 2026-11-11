"""Full-signal dataset adapter and state-aware CUSUM configuration builder.

This module reads the complete cyber bundles emitted by
``scripts/run_full_signal.py`` / ``scripts/run_logic_attacks.py``:

* ``observed.csv`` -- the full-signal observation matrix (one row per scan),
* ``labels.csv`` -- per-scan ``attack_active`` plus optional ``attack_class``,
* ``manifest.json`` -- run provenance, controls and expected columns,
* ``events.json`` / ``scenario.json`` -- legacy event windows (optional).

It deliberately does **not** depend on the old profile/feeder heuristic that
dropped tanks without a pre-declared channel.  Instead it reads the resolved
INP directly:

* every pipe/pump/valve incident to a tank contributes its flow, signed so
  that positive means *into the tank* (a negative sign therefore drains),
* the tank cross-section is the cylinder area ``pi * diameter**2 / 4``,
* actuator status columns come from ``manifest["controls"]``, where each
  control's sensor is the tank and its link is the actuator.

The nominal run is split chronologically (64 % train, 16 % clean validation,
20 % nominal test).  Attack runs are never split and remain independent test
scenarios.  The module only builds configurations and frames; the caller runs
``StateAwareCusumDetector`` with its existing numerical semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from baselines.state_aware_cusum import StateAwareCusumConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_REPO_ROOT = Path("/Users/abanisenioluwaorojo/stateaware-ics-monitoring")
ACTIVE_REPO_ROOT = Path(os.environ.get("STATEAWARE_REPO_ROOT", REPO_ROOT))


def portable_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        try:
            return ACTIVE_REPO_ROOT / path.relative_to(LEGACY_REPO_ROOT)
        except ValueError:
            pass
    return path


DEFAULT_LOGIC_ROOT = REPO_ROOT / "scripts" / "out" / "logic_attacks"
DEFAULT_PDD_ROOT = REPO_ROOT / "scripts" / "out" / "full_signal_pdd"

SPLIT_SCHEMA = "dataio/full_signal_splits/v1"

# Mirrors ``simulation.cyber.ATTACK_CLASS_TAXONOMY`` so this adapter stays
# lightweight (no WNTR import) while using the same vocabulary.
ATTACK_CLASS_TAXONOMY: dict[str, str] = {
    "normal": "no injected attack window is active",
    "sensor_offset": "additive bias on a sensor value delivered to a recipient",
    "sensor_freeze": "recipient keeps the value delivered at attack activation",
    "sensor_replay": "recipient receives its own earlier delivered history",
    "sensor_constant": "sensor value replaced by an attacker constant",
    "sensor_series": "sensor value follows an attacker-supplied series",
    "actuator_dos": "physical actuator command overridden (denial of service)",
    "control_guard_violation": "actuator command that contradicts the controlling guard",
    "actuator_feedback_spoof": (
        "observation-only actuator feedback spoof; plant truth is unchanged"
    ),
    "coordinated": "more than one attack mechanism is active on the same scan",
    "adaptive": "pre-registered bounded/stealthy attack chosen without detector feedback",
}

_SENSOR_MODE_CLASSES = {
    "offset": "sensor_offset",
    "freeze": "sensor_freeze",
    "replay": "sensor_replay",
    "constant": "sensor_constant",
    "series": "sensor_series",
}

_INCOMPLETE_FILES = ("observed.csv", "labels.csv", "manifest.json")
_SECTION_RE = re.compile(r"^\s*\[([A-Za-z_]+)\]\s*$")
_US_FLOW_UNITS = frozenset({"CFS", "GPM", "MGD", "IMGD", "AFD"})


class FullSignalError(Exception):
    """Unrecoverable full-signal adapter error."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _sha256_frame(frame: pd.DataFrame, y_true: Sequence[int],
                  attack_class: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update(frame.to_csv(index=False, float_format="%.17g",
                          lineterminator="\n").encode("utf-8"))
    h.update(b"\n--labels--\n")
    h.update(",".join(str(int(v)) for v in y_true).encode("utf-8"))
    h.update(b"\n--attack_class--\n")
    h.update("\n".join(str(c) for c in attack_class).encode("utf-8"))
    return h.hexdigest()


def _coerce_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return (
        series.astype(str).str.strip().str.lower()
        .isin(("true", "1", "yes", "t", "y"))
        .to_numpy(dtype=bool)
    )


def _as_float(value: Any, *, what: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise FullSignalError(f"{what} is not numeric: {value!r}") from exc
    return out


def derive_attack_class(event: Mapping[str, Any]) -> str:
    """Class for one event, using an explicit label when present.

    Legacy bundles have no ``attack_class`` field; the class is then derived
    from ``kind``/``mode`` exactly as :mod:`simulation.cyber` does.  No leak
    label is invented: unsupported mechanisms fail explicitly.
    """
    explicit = event.get("attack_class")
    if explicit is not None:
        if explicit not in ATTACK_CLASS_TAXONOMY or explicit == "normal":
            raise FullSignalError(f"invalid explicit attack_class: {explicit!r}")
        return str(explicit)
    kind = event.get("kind")
    if kind == "sensor":
        mode = event.get("mode")
        try:
            return _SENSOR_MODE_CLASSES[mode]
        except KeyError as exc:
            raise FullSignalError(
                f"unsupported sensor mode for attack taxonomy: {mode!r}"
            ) from exc
    if kind == "actuator":
        return "actuator_dos"
    if kind == "actuator_feedback":
        return "actuator_feedback_spoof"
    if kind == "threshold":
        return "control_guard_violation"
    raise FullSignalError(
        f"unsupported attack mechanism for attack taxonomy: {kind!r}"
    )


def derive_attack_classes(events: Sequence[Mapping[str, Any]],
                          times: Sequence[float]) -> tuple[str, ...]:
    """Per-scan classes from event windows (``start <= t < end``)."""
    normalised: list[tuple[float, float, str]] = []
    for event in events:
        try:
            start = float(event["start"])
            end = float(event["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FullSignalError(f"event has invalid start/end: {event!r}") from exc
        normalised.append((start, end, derive_attack_class(event)))
    out: list[str] = []
    for raw_t in times:
        t = float(raw_t)
        active = {cls for start, end, cls in normalised if start <= t < end}
        if not active:
            out.append("normal")
        elif len(active) == 1:
            out.append(next(iter(active)))
        else:
            out.append("coordinated")
    return tuple(out)


def median_time_delta(frame: pd.DataFrame) -> float:
    """Median positive spacing of ``time_seconds`` (seconds)."""
    if "time_seconds" not in frame.columns:
        raise FullSignalError("frame has no 'time_seconds' column")
    times = pd.to_numeric(frame["time_seconds"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(times)):
        raise FullSignalError("time_seconds contains non-finite values")
    diffs = np.diff(times)
    diffs = diffs[diffs > 0]
    if not len(diffs):
        raise FullSignalError("cannot derive dt_s from time_seconds")
    return float(np.median(diffs))


# ---------------------------------------------------------------------------
# INP geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InpTank:
    id: str
    diameter: float
    diameter_m: float
    volume_curve: str | None


@dataclass(frozen=True)
class InpLink:
    id: str
    node1: str
    node2: str
    kind: str


@dataclass(frozen=True)
class InpTopology:
    path: Path
    sha256: str
    flow_units: str
    tanks: Mapping[str, InpTank]
    links: tuple[InpLink, ...]

    def incident_links(self, tank: str) -> tuple[InpLink, ...]:
        key = tank.casefold()
        return tuple(
            link for link in self.links
            if key in (link.node1.casefold(), link.node2.casefold())
        )


def _iter_inp_sections(path: Path):
    section: str | None = None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = _SECTION_RE.match(raw)
        if match:
            section = match.group(1).upper()
            continue
        if section is None:
            continue
        data = raw.split(";", 1)[0].strip()
        if data:
            yield section, data


def parse_inp_topology(path: str | Path) -> InpTopology:
    """Parse the INP sections the full-signal adapter needs."""
    path = Path(path)
    if not path.is_file():
        raise FullSignalError(f"resolved INP not found: {path}")
    units = ""
    tanks_raw: list[tuple[str, list[str]]] = []
    link_rows: list[tuple[str, str, list[str]]] = []
    for section, data in _iter_inp_sections(path):
        parts = data.split()
        if not parts:
            continue
        if section == "OPTIONS" and parts[0].upper() == "UNITS":
            if len(parts) >= 2:
                units = parts[1].upper()
        elif section == "TANKS":
            tanks_raw.append((parts[0], parts))
        elif section in ("PIPES", "PUMPS", "VALVES"):
            if len(parts) >= 3:
                link_rows.append((section, parts[0], parts))

    tanks: dict[str, InpTank] = {}
    for tank_id, parts in tanks_raw:
        if len(parts) < 6:
            raise FullSignalError(f"[TANKS] line for {tank_id!r} has too few fields")
        diameter = _as_float(parts[5], what=f"tank {tank_id} diameter")
        curve: str | None = None
        if len(parts) > 7:
            tail = parts[7]
            # EPANET fields after diameter are [min_vol, volume_curve,
            # overflow].  Whitespace splitting drops empty placeholders, so a
            # single extra token is either a curve id or an overflow flag.
            if not (len(parts) == 8 and tail.upper() in {"YES", "NO"}):
                curve = tail
        if curve in ("", "*"):
            curve = None
        if diameter <= 0:
            raise FullSignalError(
                f"tank {tank_id!r} has non-positive diameter {diameter!r}; "
                "cannot derive a constant cylinder area"
            )
        diameter_m = diameter * 0.3048 if units in _US_FLOW_UNITS else diameter
        tanks[tank_id.casefold()] = InpTank(
            id=tank_id, diameter=diameter, diameter_m=diameter_m,
            volume_curve=curve,
        )

    links: list[InpLink] = []
    seen: set[str] = set()
    for section, link_id, parts in link_rows:
        key = link_id.casefold()
        if key in seen:
            raise FullSignalError(f"duplicate INP link id {link_id!r}")
        seen.add(key)
        links.append(InpLink(
            id=link_id, node1=parts[1], node2=parts[2], kind=section.lower(),
        ))
    if not tanks:
        raise FullSignalError(f"resolved INP has no [TANKS]: {path}")
    return InpTopology(
        path=path, sha256=sha256_file(path), flow_units=units,
        tanks=tanks, links=tuple(links),
    )


def cylinder_area_m2(tank: InpTank) -> float:
    """Defensible constant area for a cylindrical EPANET tank."""
    if tank.volume_curve:
        raise FullSignalError(
            f"tank {tank.id!r} declares volume curve {tank.volume_curve!r}; "
            "it is not a cylinder and has no defensible constant area"
        )
    return math.pi * tank.diameter_m ** 2 / 4.0


# ---------------------------------------------------------------------------
# run loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FullSignalRun:
    topology: str
    name: str
    path: Path
    run_id: str
    split: str
    frame: pd.DataFrame
    labels: pd.DataFrame
    y_true: np.ndarray
    attack_class: tuple[str, ...]
    events: tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]
    scenario: Mapping[str, Any]
    inp_path: Path | None
    input_hashes: Mapping[str, str]

    @property
    def rows(self) -> int:
        return int(len(self.frame))


@dataclass(frozen=True)
class SkippedRun:
    path: Path
    topology: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "topology": self.topology, "reason": self.reason}


@dataclass(frozen=True)
class FullSignalDiscovery:
    topology: str
    nominal: FullSignalRun
    attacks: tuple[FullSignalRun, ...]
    skipped: tuple[SkippedRun, ...]
    plan: Mapping[str, Any] | None

    @property
    def runs(self) -> tuple[FullSignalRun, ...]:
        return (self.nominal,) + self.attacks


def _column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return {str(c).casefold(): str(c) for c in frame.columns}


def resolve_column(frame: pd.DataFrame, wanted: str) -> str | None:
    if wanted in frame.columns:
        return wanted
    return _column_lookup(frame).get(wanted.casefold())


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FullSignalError(f"cannot read JSON {path}: {exc}") from exc


def _resolve_inp_path(run_path: Path, scenario: Mapping[str, Any],
                      manifest: Mapping[str, Any]) -> Path | None:
    candidates: list[Path] = [run_path / "topology.inp", run_path.parent / "topology.inp"]
    resolved = scenario.get("resolved_topology")
    if isinstance(resolved, Mapping) and resolved.get("path"):
        candidates.append(Path(str(resolved["path"])))
    source = scenario.get("input")
    if isinstance(source, Mapping):
        for key in ("resolved_from", "preferred"):
            if source.get(key):
                candidates.append(Path(str(source[key])))
    candidate = manifest.get("inp_path")
    if candidate:
        candidates.append(Path(str(candidate)))
    for item in candidates:
        item = portable_path(item) if item.is_absolute() else REPO_ROOT / item
        if item.is_file():
            return item
    return None


def _run_input_hashes(run_path: Path, inp_path: Path | None) -> dict[str, str]:
    files = [run_path / name for name in _INCOMPLETE_FILES]
    files += [run_path / "events.json", run_path / "scenario.json",
              run_path / "nominal.json"]
    hashes: dict[str, str] = {}
    for path in files:
        if path.is_file():
            try:
                key = str(path.relative_to(run_path))
            except ValueError:
                key = str(path)
            hashes[key] = sha256_file(path)
    if inp_path is not None and inp_path.is_file():
        hashes[f"resolved_inp:{inp_path}"] = sha256_file(inp_path)
    return hashes


def load_full_signal_run(path: str | Path, topology: str, *,
                         name: str | None = None,
                         inp_path: str | Path | None = None) -> FullSignalRun:
    """Load one complete bundle; raise with an explicit reason otherwise."""
    path = Path(path)
    if not path.is_dir():
        raise FullSignalError(f"run directory not found: {path}")
    missing = [name for name in _INCOMPLETE_FILES if not (path / name).is_file()]
    if missing:
        raise FullSignalError(
            f"incomplete run {path}: missing {', '.join(missing)}"
        )

    manifest = _read_json(path / "manifest.json")
    if not isinstance(manifest, Mapping):
        raise FullSignalError(f"manifest for {path} is not a JSON object")
    scenario_path = path / "scenario.json"
    scenario = _read_json(scenario_path) if scenario_path.is_file() else {}
    if not isinstance(scenario, Mapping):
        raise FullSignalError(f"scenario for {path} is not a JSON object")
    events_path = path / "events.json"
    events_raw = _read_json(events_path) if events_path.is_file() else []
    if not isinstance(events_raw, list):
        raise FullSignalError(f"events for {path} is not a JSON list")
    events = tuple(dict(e) for e in events_raw if isinstance(e, Mapping))

    try:
        observed = pd.read_csv(path / "observed.csv")
        labels = pd.read_csv(path / "labels.csv")
    except (OSError, pd.errors.ParserError) as exc:
        raise FullSignalError(f"cannot read CSV bundle in {path}: {exc}") from exc
    if "time_seconds" not in observed.columns:
        raise FullSignalError(f"{path}/observed.csv has no time_seconds column")
    if "time_seconds" not in labels.columns:
        raise FullSignalError(f"{path}/labels.csv has no time_seconds column")
    if len(observed) != len(labels):
        raise FullSignalError(
            f"{path}: observed rows ({len(observed)}) != labels rows ({len(labels)})"
        )
    obs_t = pd.to_numeric(observed["time_seconds"], errors="coerce").to_numpy(dtype=float)
    lab_t = pd.to_numeric(labels["time_seconds"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(obs_t)) or not np.allclose(obs_t, lab_t, atol=1e-9):
        raise FullSignalError(f"{path}: observed and labels time_seconds do not align")

    if "attack_active" in labels.columns:
        active = _coerce_bool(labels["attack_active"])
    elif "attack_class" in labels.columns:
        active = labels["attack_class"].astype(str).ne("normal").to_numpy(dtype=bool)
    elif events:
        active = np.array(
            [cls != "normal" for cls in derive_attack_classes(events, obs_t)],
            dtype=bool,
        )
    else:
        active = np.zeros(len(labels), dtype=bool)

    if "attack_class" in labels.columns:
        classes = tuple(labels["attack_class"].astype(str))
        invalid = [c for c in classes if c not in ATTACK_CLASS_TAXONOMY]
        if invalid:
            raise FullSignalError(f"{path}: invalid attack_class labels {sorted(set(invalid))}")
        if np.any(active & (np.asarray(classes) == "normal")):
            raise FullSignalError(f"{path}: active rows labelled 'normal'")
        if np.any(~active & (np.asarray(classes) != "normal")):
            raise FullSignalError(f"{path}: inactive rows must carry 'normal'")
    else:
        classes = derive_attack_classes(events, obs_t)
        if not events and np.any(active):
            raise FullSignalError(
                f"{path}: active labels but no attack_class and no events to derive from"
            )

    labels = labels.copy()
    labels["time_seconds"] = obs_t
    labels["attack_class"] = list(classes)
    labels["attack_active"] = active

    if inp_path is None:
        resolved = _resolve_inp_path(path, scenario, manifest)
    else:
        resolved = Path(inp_path)
    chosen_name = name or path.name
    return FullSignalRun(
        topology=topology,
        name=chosen_name,
        path=path,
        run_id=str(manifest.get("run_id") or scenario.get("run_id") or chosen_name),
        split=str(manifest.get("split") or scenario.get("split") or ""),
        frame=observed.reset_index(drop=True),
        labels=labels.reset_index(drop=True),
        y_true=active.astype(int),
        attack_class=classes,
        events=events,
        manifest=manifest,
        scenario=scenario,
        inp_path=resolved,
        input_hashes=_run_input_hashes(path, resolved),
    )


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def _nominal_path(topology: str, topology_dir: Path,
                  plan: Mapping[str, Any] | None) -> Path:
    if topology == "ltown" and plan is not None:
        nominal = plan.get("nominal")
        if isinstance(nominal, Mapping) and nominal.get("path"):
            path = portable_path(str(nominal["path"]))
            return path if path.is_absolute() else REPO_ROOT / path
    return topology_dir / "nominal"


def _iter_attack_candidates(topology: str, root: Path,
                            skip_paths: set[Path],
                            skipped: list[SkippedRun]) -> Iterable[Path]:
    if not root.is_dir():
        return
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith("."):
            skipped.append(SkippedRun(
                path=path, topology=topology,
                reason="hidden/staging directory (name starts with '.')",
            ))
            continue
        if path.resolve() in skip_paths:
            continue
        yield path


def discover_full_signal_runs(
    topology: str,
    *,
    logic_root: str | Path = DEFAULT_LOGIC_ROOT,
    pdd_root: str | Path = DEFAULT_PDD_ROOT,
    include_full_signal_pdd: bool = False,
) -> FullSignalDiscovery:
    """Discover the nominal and attack bundles for one topology.

    Anytown/C-Town use ``logic_attacks/<topology>/nominal``.  L-Town uses the
    ``plan.json`` ``nominal.path`` (currently under ``full_signal_pdd``).
    Incomplete and staging directories are skipped with explicit reasons;
    a missing nominal is fatal.
    """
    logic_root = Path(logic_root)
    topology_dir = logic_root / topology
    plan: Mapping[str, Any] | None = None
    plan_path = topology_dir / "plan.json"
    if plan_path.is_file():
        plan = _read_json(plan_path)
        if not isinstance(plan, Mapping):
            raise FullSignalError(f"plan for {topology} is not a JSON object")

    nominal_path = _nominal_path(topology, topology_dir, plan)
    nominal = load_full_signal_run(nominal_path, topology, name="nominal")
    skip_paths = {nominal_path.resolve()}
    skipped: list[SkippedRun] = []

    attacks: list[FullSignalRun] = []
    seen_names: set[str] = set()
    roots = [topology_dir]
    if include_full_signal_pdd:
        roots.append(Path(pdd_root) / topology)
    for root in roots:
        for candidate in _iter_attack_candidates(topology, root, skip_paths, skipped):
            try:
                run = load_full_signal_run(candidate, topology)
            except FullSignalError as exc:
                skipped.append(SkippedRun(
                    path=candidate, topology=topology, reason=str(exc),
                ))
                continue
            if run.name in seen_names:
                skipped.append(SkippedRun(
                    path=candidate, topology=topology,
                    reason=f"duplicate scenario name {run.name!r}",
                ))
                continue
            if run.inp_path is None and nominal.inp_path is not None:
                run = replace(run, inp_path=nominal.inp_path)
            seen_names.add(run.name)
            attacks.append(run)

    return FullSignalDiscovery(
        topology=topology, nominal=nominal, attacks=tuple(attacks),
        skipped=tuple(skipped), plan=plan,
    )


# ---------------------------------------------------------------------------
# chronological nominal split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FullSignalSplit:
    name: str
    frame: pd.DataFrame
    y_true: np.ndarray
    attack_class: tuple[str, ...]
    start_time: float
    end_time: float
    digest: str

    @property
    def rows(self) -> int:
        return int(len(self.frame))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.rows,
            "start_time_seconds": float(self.start_time),
            "end_time_seconds": float(self.end_time),
            "sha256": self.digest,
        }


@dataclass(frozen=True)
class NominalSplit:
    train: FullSignalSplit
    validation: FullSignalSplit
    test: FullSignalSplit
    manifest: Mapping[str, Any]


def split_nominal_frame(
    frame: pd.DataFrame,
    y_true: Sequence[int],
    attack_class: Sequence[str],
    *,
    train_fraction: float = 0.64,
    validation_fraction: float = 0.16,
) -> NominalSplit:
    """Chronological 64 / 16 / 20 split with content hashes."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise FullSignalError("split fractions must be in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise FullSignalError("train + validation fraction must be < 1")
    if len(frame) != len(y_true) or len(frame) != len(attack_class):
        raise FullSignalError("frame, labels and attack classes have different lengths")
    if "time_seconds" not in frame.columns:
        raise FullSignalError("nominal frame has no time_seconds column")
    times = pd.to_numeric(frame["time_seconds"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(times)):
        raise FullSignalError("nominal time_seconds contains non-finite values")
    if len(times) > 1 and not np.all(np.diff(times) > 0):
        raise FullSignalError("nominal time_seconds must be strictly increasing")

    n = len(frame)
    cut1 = int(math.floor(n * train_fraction))
    cut2 = int(math.floor(n * (train_fraction + validation_fraction)))
    if cut1 < 1 or cut2 <= cut1 or cut2 >= n:
        raise FullSignalError(
            f"nominal run too short for 64/16/20 split: {n} rows"
        )

    y = np.asarray(y_true, dtype=int)
    cls = tuple(str(c) for c in attack_class)

    def _part(name: str, start: int, stop: int) -> FullSignalSplit:
        part_frame = frame.iloc[start:stop].reset_index(drop=True)
        part_y = y[start:stop]
        part_cls = cls[start:stop]
        return FullSignalSplit(
            name=name,
            frame=part_frame,
            y_true=part_y,
            attack_class=part_cls,
            start_time=float(times[start]),
            end_time=float(times[stop - 1]),
            digest=_sha256_frame(part_frame, part_y, part_cls),
        )

    train = _part("train", 0, cut1)
    validation = _part("clean_validation", cut1, cut2)
    test = _part("nominal_test", cut2, n)
    manifest = {
        "schema": SPLIT_SCHEMA,
        "chronological": True,
        "fractions": {
            "train": float(train_fraction),
            "clean_validation": float(validation_fraction),
            "nominal_test": float(1.0 - train_fraction - validation_fraction),
        },
        "total_rows": int(n),
        "segments": [part.to_dict() for part in (train, validation, test)],
        "boundaries": {
            "train_stop_index": int(cut1),
            "validation_stop_index": int(cut2),
        },
        "isolation": (
            "train, clean validation and nominal test are disjoint contiguous "
            "time ranges; attack runs are separate scenarios and are never split"
        ),
        "attack_scenarios_in_split": False,
    }
    return NominalSplit(train=train, validation=validation, test=test,
                        manifest=manifest)


def split_nominal_run(run: FullSignalRun) -> NominalSplit:
    return split_nominal_frame(run.frame, run.y_true, run.attack_class)


# ---------------------------------------------------------------------------
# CUSUM configuration builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeederSpec:
    tank: str
    link: str
    source_column: str
    signed_column: str
    sign: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tank": self.tank,
            "link": self.link,
            "source_column": self.source_column,
            "signed_column": self.signed_column,
            "sign": int(self.sign),
        }


@dataclass
class CusumBuild:
    topology: str
    config: StateAwareCusumConfig
    feeders: tuple[FeederSpec, ...]
    tanks: Mapping[str, Mapping[str, Any]]
    inp: InpTopology
    dt_seconds: float

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with the derived signed-flow columns attached."""
        out = frame.copy()
        for report in self.tanks.values():
            tank = str(report["tank"])
            if resolve_column(out, tank) is None:
                raise FullSignalError(f"frame has no level column for tank {tank!r}")
        for spec in self.feeders:
            source = resolve_column(out, spec.source_column)
            if source is None:
                raise FullSignalError(
                    f"frame has no flow column {spec.source_column!r} for "
                    f"{spec.tank!r}/{spec.link!r}"
                )
            out[spec.signed_column] = spec.sign * out[source].to_numpy(dtype=float)
        for tank, statuses in self.config.actuators.items():
            for status in statuses:
                if resolve_column(out, status) is None:
                    raise FullSignalError(
                        f"frame has no actuator status column {status!r} for tank {tank!r}"
                    )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "topology": self.topology,
            "inp_path": str(self.inp.path),
            "inp_sha256": self.inp.sha256,
            "inp_flow_units": self.inp.flow_units,
            "dt_seconds": float(self.dt_seconds),
            "dt_source": "nominal median time_seconds delta",
            "config_dt_source": self.config.dt_source,
            "flow_to_m3ps": float(self.config.flow_to_m3ps),
            "offset_correction": bool(self.config.offset_correction),
            "feeders": [spec.to_dict() for spec in self.feeders],
            "tanks": {tank: dict(report) for tank, report in sorted(self.tanks.items())},
            "actuators": {
                tank: list(statuses)
                for tank, statuses in sorted(self.config.actuators.items())
            },
        }


def _safe_token(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")


def build_cusum_config(
    topology: str,
    frame: pd.DataFrame,
    manifest: Mapping[str, Any],
    inp_path: str | Path | None,
    *,
    dt_s: float | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> CusumBuild:
    """Build a CUSUM config from the resolved INP and full-signal columns.

    Raises explicitly if a tank is not a cylinder, a level/flow/status column
    is missing, or the manifest declares no controls for an observed tank.
    """
    if inp_path is None:
        raise FullSignalError(
            f"{topology}: no resolved INP for the full-signal run"
        )
    inp = parse_inp_topology(inp_path)
    if dt_s is None:
        dt_s = median_time_delta(frame)
    dt_s = float(dt_s)
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise FullSignalError(f"invalid dt_s {dt_s!r}")

    controls = manifest.get("controls")
    if not isinstance(controls, Sequence) or not controls:
        raise FullSignalError(
            f"{topology}: manifest declares no controls; cannot map actuator "
            "statuses to tanks"
        )

    feeders: list[FeederSpec] = []
    tanks_report: dict[str, dict[str, Any]] = {}
    feeders_by_tank: dict[str, list[str]] = {}
    actuators_by_tank: dict[str, list[str]] = {}
    areas: dict[str, float] = {}

    for key, tank in sorted(inp.tanks.items()):
        level_column = resolve_column(frame, tank.id)
        if level_column is None:
            raise FullSignalError(
                f"{topology}: INP tank {tank.id!r} has no observed level column"
            )
        area = cylinder_area_m2(tank)
        incident = inp.incident_links(tank.id)
        if not incident:
            raise FullSignalError(
                f"{topology}: tank {tank.id!r} has no incident INP link; "
                "cannot build a mass-balance feeder"
            )
        tank_feeders: list[str] = []
        for link in incident:
            source_column = resolve_column(frame, f"f_{link.id}")
            if source_column is None:
                raise FullSignalError(
                    f"{topology}: tank {tank.id!r} link {link.id!r} has no flow "
                    f"column (looked for f_{link.id})"
                )
            if link.node2.casefold() == tank.id.casefold():
                sign = 1
            elif link.node1.casefold() == tank.id.casefold():
                sign = -1
            else:  # pragma: no cover - incident_links guarantees this
                raise FullSignalError(
                    f"{topology}: link {link.id!r} is not incident to {tank.id!r}"
                )
            signed_column = (
                f"signed_inflow__{_safe_token(tank.id)}__{_safe_token(link.id)}"
            )
            tank_feeders.append(signed_column)
            feeders.append(FeederSpec(
                tank=tank.id, link=link.id, source_column=source_column,
                signed_column=signed_column, sign=sign,
            ))

        tank_statuses: list[str] = []
        for control in controls:
            if not isinstance(control, Mapping):
                continue
            sensor = str(control.get("sensor", ""))
            if sensor.casefold() != tank.id.casefold():
                continue
            link_id = str(control.get("link", ""))
            if not link_id:
                continue
            status = resolve_column(frame, f"s_{link_id}")
            if status is None:
                raise FullSignalError(
                    f"{topology}: control for tank {tank.id!r} link {link_id!r} "
                    f"has no status column (looked for s_{link_id})"
                )
            if status not in tank_statuses:
                tank_statuses.append(status)

        feeders_by_tank[tank.id] = tank_feeders
        actuators_by_tank[tank.id] = tank_statuses
        areas[tank.id] = area
        tanks_report[tank.id] = {
            "tank": tank.id,
            "level_column": level_column,
            "diameter_inp": float(tank.diameter),
            "diameter_m": float(tank.diameter_m),
            "area_m2": float(area),
            "area_source": "cylinder_pi_d_squared_over_4",
            "inp_volume_curve": tank.volume_curve,
            "incident_links": [link.id for link in incident],
            "actuator_status_columns": tank_statuses,
        }

    tank_order = tuple(sorted(feeders_by_tank))
    config_kwargs: dict[str, Any] = {
        "topology": topology,
        "tanks": tank_order,
        "feeders": {tank: tuple(feeders_by_tank[tank]) for tank in tank_order},
        "outflows": {},
        "actuators": {tank: tuple(actuators_by_tank[tank]) for tank in tank_order},
        "area_m2": {tank: float(areas[tank]) for tank in tank_order},
        "dt_s": dt_s,
        "dt_source": "fixed",
        "flow_to_m3ps": 1.0,
        "offset_correction": False,
    }
    if overrides:
        config_kwargs.update(dict(overrides))
    config = StateAwareCusumConfig(**config_kwargs)
    return CusumBuild(
        topology=topology,
        config=config,
        feeders=tuple(feeders),
        tanks=tanks_report,
        inp=inp,
        dt_seconds=dt_s,
    )
