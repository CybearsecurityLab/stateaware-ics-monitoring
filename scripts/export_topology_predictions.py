#!/usr/bin/env python3
"""Export deterministic per-row topology predictions for the full-signal runs.

For every topology that has artifacts under ``scripts/out/full_signal`` this
script produces, per topology (a "run"):

* ``detector_predictions.csv`` — one row per (scenario, timestep) with the
  ground truth, the GFSM guard flag, the topology-STL flag, the fused flag,
  the fused anomaly score and the number of *unknown* (uncalibrated / missing
  / error) residuals seen at that timestep.
* ``metrics.json`` — per-arm metrics (gfsm / stl / combined) recomputed
  through :func:`stl.metrics.detection_metrics`, the nominal false-alarm
  counts and the unknown counts, plus the frozen STL tolerances and every
  input/output path used.

Arms
----
``gfsm``     the GFSM delta-conformance guard
             (:class:`monitor.gfsm_detector.GfsmAnomalyDetector`).
``stl``      the topology-driven physics residual detector
             (:class:`stl.topology_detector.TopologyStlDetector`).
``combined`` ``guard OR topology STL``; the fused score is
             ``max(guard_score, stl_score)``, so a raised flag always comes
             with a score of at least 1.

The topology STL is calibrated **once** per topology on the first 80 % of the
nominal observed signal (chronological), with the 99th percentile of the
absolute residual plus a per-unit additive margin, and with
``unknown_policy="report"``.  The tolerances are frozen and exported through
``to_artifact()``; no scenario is ever used for calibration.

The script is read-only with respect to the original artifacts: the
``data/`` submodule and ``scripts/out/full_signal`` are never modified.  The
L-Town surrogate PLC/ST/GFSM artifacts (which do not exist upstream) are
generated under the output root.

Usage
-----
    python scripts/export_topology_predictions.py
    python scripts/export_topology_predictions.py --topologies anytown ctown
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hydraulics import load_network  # noqa: E402
from invariants.model import InvariantsError  # noqa: E402
from invariants.state_label import (  # noqa: E402
    load_gfsm_components,
    resolve_fb_to_col_from_paths,
)
from monitor.gfsm_detector import GfsmAnomalyDetector  # noqa: E402
from stl.metrics import detection_metrics  # noqa: E402
from stl.topology_detector import TopologyStlDetector  # noqa: E402

# --------------------------------------------------------------------------
# Fixed configuration (see module docstring).
# --------------------------------------------------------------------------

#: STL calibration: percentile of |residual| plus a per-unit additive margin.
DEFAULT_PERCENTILE = 99.0
DEFAULT_ADDITIVE = {"m": 1e-4, "m^3": 1e-6, "m^3/s": 1e-9}
DEFAULT_UNKNOWN_POLICY = "report"
#: Fraction of the nominal observed signal (chronological) used to calibrate.
DEFAULT_CALIBRATION_FRACTION = 0.8

#: Per-topology wiring of the full-signal artifact tree.
TOPOLOGY_CONFIG: dict[str, dict[str, Any]] = {
    "anytown": {
        "nominal_scenario": "nominal",
        # INP next to the full-signal artifacts.
        "inp": "topology.inp",
    },
    "ctown": {
        "nominal_scenario": "nominal_short",
        "inp": "topology.inp",
    },
    "ltown": {
        "nominal_scenario": "nominal_two_day",
        # L-Town has no topology.inp next to the full-signal artifacts; the
        # raw benchmark network is used instead (data/ltown/raw/L-TOWN.inp).
        "inp": None,
        "inp_fallback": "ltown/raw/L-TOWN.inp",
    },
}

#: Explicit GFSM column mapping for the L-Town surrogate GFSM.  The
#: surrogate PLC1 owns the PUMP_1 actuator; its case variable PUMP_1_State is
#: observed through the pump status channel.
EXPLICIT_COLUMN_MAPPING: dict[str, dict[str, str]] = {
    "ltown": {"PLC1.PUMP_1_State": "s_PUMP_1"},
}

#: Surrogate PLC description for L-Town (DHALSIM-style plcs.yaml).
LTOWN_SURROGATE_PLCS = [
    {
        "name": "PLC1",
        "actuators": ["PUMP_1"],
        "sensors": ["PUMP_1F", "p227", "p235", "n54"],
    },
    {"name": "PLC2", "sensors": ["T1"]},
]

PREDICTION_COLUMNS = [
    "scenario",
    "row",
    "y_true",
    "y_pred_stl",
    "y_pred_gfsm",
    "y_pred",
    "anomaly_score",
    "uncalibrated_count",
]

ARM_NAMES = ("gfsm", "stl", "combined")


class ExportError(RuntimeError):
    """Raised when a topology cannot be exported at all."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _log(message: str) -> None:
    print(f"[export_topology_predictions] {message}", flush=True)


def _read_observed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "time_seconds" not in frame.columns:
        raise ExportError(f"{path} has no 'time_seconds' column")
    return frame


def _read_labels(path: Path) -> np.ndarray:
    labels = pd.read_csv(path)
    if "attack_active" not in labels.columns:
        raise ExportError(f"{path} has no 'attack_active' column")
    return labels["attack_active"].astype(int).to_numpy()


def _chronological(frame: pd.DataFrame) -> pd.DataFrame:
    """Return *frame* sorted by ``time_seconds`` (stable)."""
    return frame.sort_values("time_seconds", kind="stable").reset_index(drop=True)


def _unique_ci(columns: Iterable[str], candidates: Sequence[str]) -> list[str]:
    wanted = {c.lower() for c in candidates}
    return [c for c in columns if c.lower() in wanted]


def _resolve_frame_column(dataset_column: str, frame_columns: Sequence[str]) -> str | None:
    """Map a *dataset* column name onto the matching column of a signal frame.

    The committed dataset manifests (``data/<topology>/dataset``) use lower
    case names (``p78``, ``s_pu1``) while the full-signal exports use the
    upper case element ids (``S_P78``, ``S_PU1``).  The lookup therefore tries,
    in order: the exact name, a case-insensitive match, the ``S_``-prefixed
    variant and the ``S_``-stripped variant.  ``None`` is returned when
    nothing matches; an ambiguous match raises.
    """
    stripped = dataset_column[2:] if dataset_column.lower().startswith("s_") else None
    attempts = [dataset_column, f"s_{dataset_column}"]
    if stripped:
        attempts.append(stripped)
    for attempt in attempts:
        matches = _unique_ci(frame_columns, [attempt])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ExportError(
                f"ambiguous column resolution for dataset column "
                f"{dataset_column!r}: {matches}"
            )
    return None


# --------------------------------------------------------------------------
# Topology STL arm
# --------------------------------------------------------------------------


def _stl_predict(
    detector: TopologyStlDetector, frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(flags, scores, unknown_counts)`` for *frame*.

    This mirrors :meth:`TopologyStlDetector.predict` exactly (including its
    ``unknown_policy`` handling) but also returns the per-row count of
    *unknown* residuals (missing / error / uncalibrated), which the detector
    only exposes in aggregate through ``last_unknown``.
    """
    details = detector.predict_details(frame)
    n = len(details)
    flags = np.zeros(n, dtype=int)
    scores = np.zeros(n, dtype=float)
    unknown = np.zeros(n, dtype=int)
    for i, row in enumerate(details):
        unknown_rows = [
            d
            for d in row.values()
            if d["robustness"] is None
            and d["regime"] in ("missing", "error", "uncalibrated")
        ]
        unknown[i] = len(unknown_rows)
        if unknown_rows and (
            detector.unknown_policy == "raise"
            or any(d["regime"] in ("missing", "error") for d in unknown_rows)
        ):
            raise ValueError(
                "prediction contains unknown residuals: "
                + "; ".join(d["reason"] for d in unknown_rows)
            )
        violated = sum(
            1 for d in row.values() if d["robustness"] is not None and d["robustness"] < 0
        )
        flags[i] = 1 if violated else 0
        scores[i] = float(violated)
    return flags, scores, unknown


# --------------------------------------------------------------------------
# GFSM arm
# --------------------------------------------------------------------------


def _gfsm_components(gfsm_json: Mapping[str, Any]) -> list[tuple[str, str]]:
    return load_gfsm_components(dict(gfsm_json))


def _build_ltown_surrogate(
    *, out_root: Path, inp_path: Path
) -> tuple[Path, Path]:
    """Generate the L-Town surrogate PLC YAML / ST / GFSM artifacts.

    Returns ``(gfsm_dir, generated_dir)``.  Existing artifacts are reused
    (the generation is idempotent); nothing outside ``out_root`` is touched.
    """
    generated_dir = out_root / "ltown" / "generated"
    topo_dir = generated_dir / "ltown"
    gfsm_dir = topo_dir / "gfsm"
    gfsm_json = gfsm_dir / "ltown.gfsm.json"
    if gfsm_json.exists():
        _log(f"ltown: reusing existing surrogate GFSM at {gfsm_json}")
        return gfsm_dir, generated_dir

    import yaml

    generated_dir.mkdir(parents=True, exist_ok=True)
    plcs_path = generated_dir / "ltown_plcs.yaml"
    plcs_path.write_text(
        yaml.safe_dump(LTOWN_SURROGATE_PLCS, sort_keys=False), encoding="utf-8"
    )
    _log(f"ltown: wrote surrogate PLC definition {plcs_path}")

    env = dict(os.environ)
    # Propagate the interpreter's import path so the sub-processes find the
    # very same sources this script was started with (the checkout that owns
    # ``hydraulics`` / ``stl.topology_detector`` may differ from REPO_ROOT).
    import_path = [
        p for p in sys.path
        if p and Path(p).is_absolute() and Path(p).is_dir()
    ]
    env["PYTHONPATH"] = os.pathsep.join(
        import_path + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    python = sys.executable

    def _run(args: list[str]) -> None:
        proc = subprocess.run(
            [python, "-m", *args], env=env, capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise ExportError(
                f"command {' '.join(args)} failed ({proc.returncode}): "
                f"{proc.stderr.strip()[-800:]}"
            )

    _run([
        "st_gen",
        "--inp", str(inp_path),
        "--plcs", str(plcs_path),
        "--out", str(topo_dir),
        "--topology", "ltown",
    ])
    _log(f"ltown: generated surrogate ST in {topo_dir}")
    _run(["st_analyze", "--generated-dir", str(generated_dir), "--topology", "ltown"])
    _log(f"ltown: generated surrogate analysis in {topo_dir / 'analysis'}")
    _run(["gfsm", "--generated-dir", str(generated_dir), "--topology", "ltown"])
    _log(f"ltown: generated surrogate GFSM in {gfsm_dir}")
    if not gfsm_json.exists():
        raise ExportError(f"surrogate GFSM was not produced at {gfsm_json}")
    return gfsm_dir, generated_dir


def _resolve_gfsm_columns(
    *,
    topology: str,
    gfsm_dir: Path,
    data_root: Path,
    components: list[tuple[str, str]],
    frame_columns: Sequence[str],
) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """Resolve the GFSM components onto the columns of the signal frames.

    Anytown / C-Town use the *original* dataset manifests
    (``data/<topology>/dataset/dataset_manifest.yaml``) via
    :func:`invariants.state_label.resolve_fb_to_col_from_paths`; L-Town uses
    the explicit PUMP_1 mapping.  In both cases the resolved dataset column
    is finally mapped onto the matching column of the full-signal export.

    Returns ``(fb_to_col, mapping_report)``.
    """
    explicit = EXPLICIT_COLUMN_MAPPING.get(topology)
    if explicit is not None:
        fb_to_col: dict[tuple[str, str], str] = {}
        report: dict[str, str] = {}
        for component in components:
            key = f"{component[0]}.{component[1]}"
            dataset_column = explicit.get(key)
            if dataset_column is None:
                raise ExportError(
                    f"{topology}: no explicit column mapping for GFSM component {key}"
                )
            column = _resolve_frame_column(dataset_column, frame_columns)
            if column is None:
                raise ExportError(
                    f"{topology}: explicit column {dataset_column!r} for {key} "
                    f"is absent from the observed signal"
                )
            fb_to_col[component] = column
            report[key] = f"{dataset_column}->{column}"
        return fb_to_col, report

    try:
        fb_to_col, resolved_components, _gfsm = resolve_fb_to_col_from_paths(
            gfsm_dir, topology, data_root
        )
    except InvariantsError as exc:
        raise ExportError(f"{topology}: cannot resolve GFSM columns: {exc}") from exc
    if list(resolved_components) != list(components):
        raise ExportError(
            f"{topology}: GFSM components changed while resolving columns"
        )
    report = {}
    adapted: dict[tuple[str, str], str] = {}
    for component, dataset_column in fb_to_col.items():
        column = _resolve_frame_column(dataset_column, frame_columns)
        if column is None:
            raise ExportError(
                f"{topology}: dataset column {dataset_column!r} for component "
                f"{component} is absent from the observed signal"
            )
        adapted[component] = column
        report[f"{component[0]}.{component[1]}"] = f"{dataset_column}->{column}"
    return adapted, report


# --------------------------------------------------------------------------
# Per-topology export
# --------------------------------------------------------------------------


def _resolve_inp(
    topology: str, full_signal_root: Path, data_root: Path
) -> Path:
    config = TOPOLOGY_CONFIG[topology]
    candidates: list[Path] = []
    if config.get("inp"):
        candidates.append(full_signal_root / topology / config["inp"])
    if config.get("inp_fallback"):
        candidates.append(data_root / config["inp_fallback"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ExportError(
        f"{topology}: no INP found (looked at {[str(c) for c in candidates]})"
    )


def _scenario_dirs(full_signal_root: Path, topology: str) -> list[Path]:
    root = full_signal_root / topology
    if not root.is_dir():
        raise ExportError(f"{topology}: no full-signal directory at {root}")
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not dirs:
        raise ExportError(f"{topology}: no scenario directories under {root}")
    return dirs


def export_topology(
    *,
    topology: str,
    full_signal_root: Path,
    data_root: Path,
    out_root: Path,
    percentile: float,
    additive: Mapping[str, float],
    calibration_fraction: float,
    unknown_policy: str,
    verify_parity: bool = True,
) -> dict[str, Any]:
    """Export one topology; returns the metrics dictionary."""
    started = time.time()
    config = TOPOLOGY_CONFIG[topology]
    nominal_scenario = config["nominal_scenario"]
    inp_path = _resolve_inp(topology, full_signal_root, data_root)
    _log(f"{topology}: loading network {inp_path}")
    network = load_network(inp_path)

    scenario_dirs = _scenario_dirs(full_signal_root, topology)
    scenario_names = [p.name for p in scenario_dirs]
    if nominal_scenario not in scenario_names:
        raise ExportError(
            f"{topology}: nominal scenario {nominal_scenario!r} not found "
            f"(have {scenario_names})"
        )

    # ---- calibration: first 80 % of the nominal observed signal ----------
    nominal_dir = full_signal_root / topology / nominal_scenario
    nominal_frame = _chronological(_read_observed(nominal_dir / "observed.csv"))
    n_cal = int(len(nominal_frame) * calibration_fraction)
    if n_cal < 2:
        raise ExportError(f"{topology}: calibration split is too small ({n_cal} rows)")
    calibration_frame = nominal_frame.iloc[:n_cal]

    detector = TopologyStlDetector(
        network,
        {
            "percentile": percentile,
            "additive": dict(additive),
            "unknown_policy": unknown_policy,
        },
    )
    _log(f"{topology}: fitting topology STL on {n_cal} calibration rows")
    detector.fit([calibration_frame])
    stl_artifact = detector.to_artifact()
    _log(
        f"{topology}: frozen {len(stl_artifact['tolerances'])} tolerance keys"
    )

    # ---- GFSM arm --------------------------------------------------------
    gfsm_dir: Path
    if topology in EXPLICIT_COLUMN_MAPPING:
        gfsm_dir, _generated = _build_ltown_surrogate(
            out_root=out_root, inp_path=inp_path
        )
    else:
        gfsm_dir = data_root / "generated" / topology / "gfsm"
    gfsm_json_path = gfsm_dir / f"{topology}.gfsm.json"
    if not gfsm_json_path.exists():
        raise ExportError(f"{topology}: GFSM json not found at {gfsm_json_path}")
    gfsm_dict = json.loads(gfsm_json_path.read_text())
    components = _gfsm_components(gfsm_dict)

    nominal_columns = list(nominal_frame.columns)
    fb_to_col, column_report = _resolve_gfsm_columns(
        topology=topology,
        gfsm_dir=gfsm_dir,
        data_root=data_root,
        components=components,
        frame_columns=nominal_columns,
    )
    gfsm_detector = GfsmAnomalyDetector(
        gfsm_dir=gfsm_dir, topology=topology, fb_to_col=fb_to_col
    ).fit([])
    _log(f"{topology}: GFSM components {components} -> {fb_to_col}")

    # ---- evaluate every scenario ----------------------------------------
    rows: list[dict[str, Any]] = []
    per_scenario: dict[str, Any] = {}
    parity_report: dict[str, Any] = {}
    for scenario_dir in scenario_dirs:
        name = scenario_dir.name
        observed_path = scenario_dir / "observed.csv"
        labels_path = scenario_dir / "labels.csv"
        try:
            frame = _chronological(_read_observed(observed_path))
            y_true = _read_labels(labels_path)
            if len(y_true) != len(frame):
                raise ExportError(
                    f"{name}: labels ({len(y_true)}) and observed ({len(frame)}) "
                    "row counts differ"
                )
            if name == nominal_scenario:
                frame = frame.iloc[n_cal:].reset_index(drop=True)
                y_true = y_true[n_cal:]
            missing_cols = [
                c for c in fb_to_col.values() if c not in frame.columns
            ]
            if missing_cols:
                raise ExportError(
                    f"{name}: GFSM columns {sorted(set(missing_cols))} missing"
                )
            stl_flags, stl_scores, unknown = _stl_predict(detector, frame)
            if name == nominal_scenario and verify_parity:
                reference = np.asarray(detector.predict(frame).flags, dtype=int)
                parity = bool(np.array_equal(reference, stl_flags))
                parity_report[name] = {
                    "matches_detector_predict": parity,
                    "reference_flags": int(reference.sum()),
                    "exported_flags": int(stl_flags.sum()),
                }
                if not parity:
                    raise ExportError(
                        f"{name}: exported STL flags differ from "
                        f"TopologyStlDetector.predict"
                    )
            gfsm_flags = np.asarray(gfsm_detector.predict(frame).flags, dtype=int)
            gfsm_scores = gfsm_flags.astype(float)
            combined_flags = np.logical_or(stl_flags.astype(bool), gfsm_flags.astype(bool))
            combined_flags = combined_flags.astype(int)
            combined_scores = np.maximum(stl_scores, gfsm_scores)

            for i in range(len(frame)):
                rows.append(
                    {
                        "scenario": name,
                        "row": i,
                        "y_true": int(y_true[i]),
                        "y_pred_stl": int(stl_flags[i]),
                        "y_pred_gfsm": int(gfsm_flags[i]),
                        "y_pred": int(combined_flags[i]),
                        "anomaly_score": float(combined_scores[i]),
                        "uncalibrated_count": int(unknown[i]),
                    }
                )

            scenario_metrics = {
                "status": "ok",
                "rows": int(len(frame)),
                "attack_rows": int(y_true.sum()),
                "unknown_rows": int((unknown > 0).sum()),
                "unknown_residuals": int(unknown.sum()),
                "arms": {
                    "gfsm": detection_metrics(y_true, gfsm_flags, gfsm_scores),
                    "stl": detection_metrics(y_true, stl_flags, stl_scores),
                    "combined": detection_metrics(
                        y_true, combined_flags, combined_scores
                    ),
                },
            }
            per_scenario[name] = scenario_metrics
            _log(
                f"{topology}/{name}: rows={len(frame)} attack={int(y_true.sum())} "
                f"stl_flags={int(stl_flags.sum())} gfsm_flags={int(gfsm_flags.sum())}"
            )
        except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
            per_scenario[name] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "rows": 0,
                "attack_rows": 0,
            }
            _log(f"{topology}/{name}: FAILED {type(exc).__name__}: {exc}")

    if not rows:
        raise ExportError(f"{topology}: no scenario could be evaluated")

    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    y_all = predictions["y_true"].to_numpy()
    stl_all = predictions["y_pred_stl"].to_numpy()
    gfsm_all = predictions["y_pred_gfsm"].to_numpy()
    combined_all = predictions["y_pred"].to_numpy()
    score_all = predictions["anomaly_score"].to_numpy()

    arms = {
        "gfsm": detection_metrics(y_all, gfsm_all, gfsm_all.astype(float)),
        "stl": detection_metrics(y_all, stl_all, score_all),
        "combined": detection_metrics(y_all, combined_all, score_all),
    }

    nominal_mask = predictions["scenario"] == nominal_scenario
    nominal_false_alarms = {
        "gfsm": int(gfsm_all[nominal_mask.to_numpy()].sum()),
        "stl": int(stl_all[nominal_mask.to_numpy()].sum()),
        "combined": int(combined_all[nominal_mask.to_numpy()].sum()),
    }
    unknown_counts = predictions["uncalibrated_count"].to_numpy()
    unknown_summary = {
        "total_unknown_residuals": int(unknown_counts.sum()),
        "rows_with_unknown": int((unknown_counts > 0).sum()),
        "rows_total": int(len(predictions)),
    }

    # ---- write outputs ---------------------------------------------------
    out_dir = out_root / topology
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "detector_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    artifact_path = out_dir / "topology_stl_artifact.json"
    artifact_path.write_text(
        json.dumps(stl_artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    metrics = {
        "schema": "topology-predictions/v1",
        "topology": topology,
        "generated_at_unix": int(started),
        "runtime_seconds": round(time.time() - started, 3),
        "config": {
            "percentile": percentile,
            "additive": dict(sorted(additive.items())),
            "unknown_policy": unknown_policy,
            "calibration_fraction": calibration_fraction,
            "calibration_rows": int(n_cal),
            "nominal_scenario": nominal_scenario,
            "scenarios": scenario_names,
        },
        "inputs": {
            "full_signal_root": str(full_signal_root),
            "data_root": str(data_root),
            "inp": str(inp_path),
            "nominal_observed": str(nominal_dir / "observed.csv"),
            "gfsm_json": str(gfsm_json_path),
        },
        "outputs": {
            "detector_predictions_csv": str(predictions_path),
            "metrics_json": str(out_dir / "metrics.json"),
            "stl_artifact_json": str(artifact_path),
        },
        "stl": {
            "tolerances": stl_artifact["tolerances"],
            "formulas": stl_artifact["formulas"],
            "coverage": stl_artifact["coverage"],
        },
        "gfsm": {
            "components": [f"{plc}.{case_var}" for plc, case_var in components],
            "column_mapping": column_report,
        },
        "arms": arms,
        "per_scenario": per_scenario,
        "nominal_false_alarms": nominal_false_alarms,
        "unknown_count": unknown_summary,
        "stl_parity_check": parity_report,
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _log(
        f"{topology}: wrote {predictions_path} ({len(predictions)} rows) and "
        f"{metrics_path}"
    )
    return metrics


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="export_topology_predictions",
        description=(
            "Export deterministic per-row topology predictions (GFSM guard, "
            "topology STL, combined) for the full-signal artifacts."
        ),
    )
    p.add_argument(
        "--full-signal-root",
        type=Path,
        default=REPO_ROOT / "scripts" / "out" / "full_signal",
        help="directory holding <topology>/<scenario>/observed.csv",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data",
        help="data root (dataset manifests, generated GFSM, raw L-Town INP)",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "scripts" / "out" / "topology_predictions",
        help="output root for detector_predictions.csv / metrics.json",
    )
    p.add_argument(
        "--topologies",
        nargs="+",
        default=list(TOPOLOGY_CONFIG),
        help="topologies to export (default: all known)",
    )
    p.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PERCENTILE,
        help="STL tolerance percentile (default: 99)",
    )
    p.add_argument(
        "--additive",
        type=str,
        default=",".join(f"{k}={v}" for k, v in DEFAULT_ADDITIVE.items()),
        help="per-unit additive margins, e.g. 'm=1e-4,m^3=1e-6,m^3/s=1e-9'",
    )
    p.add_argument(
        "--calibration-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_FRACTION,
        help="fraction of the nominal signal used for calibration (default: 0.8)",
    )
    p.add_argument(
        "--no-verify-parity",
        action="store_true",
        help="skip the parity check against TopologyStlDetector.predict",
    )
    p.add_argument(
        "--unknown-policy",
        choices=("raise", "report"),
        default=DEFAULT_UNKNOWN_POLICY,
        help="topology STL unknown-residual policy (default: report)",
    )
    return p


def _parse_additive(raw: str) -> dict[str, float]:
    additive: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"invalid --additive entry {item!r}")
        additive[key.strip()] = float(value)
    if not additive:
        raise SystemExit("--additive must define at least one unit margin")
    return additive


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    additive = _parse_additive(args.additive)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    failures: list[str] = []
    for topology in args.topologies:
        if topology not in TOPOLOGY_CONFIG:
            _log(f"{topology}: unknown topology, skipping")
            failures.append(topology)
            continue
        try:
            metrics = export_topology(
                topology=topology,
                full_signal_root=Path(args.full_signal_root),
                data_root=Path(args.data_root),
                out_root=out_root,
                percentile=args.percentile,
                additive=additive,
                calibration_fraction=args.calibration_fraction,
                unknown_policy=args.unknown_policy,
                verify_parity=not args.no_verify_parity,
            )
            summary[topology] = {
                "rows": int(metrics["unknown_count"]["rows_total"]),
                "arms": {
                    arm: {
                        "mcc": metrics["arms"][arm]["mcc"],
                        "balanced_accuracy": metrics["arms"][arm]["balanced_accuracy"],
                        "f1": metrics["arms"][arm]["f1"],
                    }
                    for arm in ARM_NAMES
                },
                "nominal_false_alarms": metrics["nominal_false_alarms"],
                "unknown_count": metrics["unknown_count"],
            }
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            _log(f"{topology}: FAILED {type(exc).__name__}: {exc}")
            failures.append(topology)

    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps({"topologies": summary, "failed": failures}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _log(f"wrote {summary_path}")
    if failures and len(failures) == len(args.topologies):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
