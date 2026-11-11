#!/usr/bin/env python3
"""Export per-row GFSM, global-Phi and topology-STL predictions.

For every topology with a ``scripts/out/logic_attacks`` corpus this script
writes, under ``scripts/out/logic_attack_predictions/<topology>/``:

* ``detector_predictions.csv`` — one row per (scenario, timestep) with labels,
  all three detector flags and scores, the GFSM OR Phi OR STL fused result,
  and the per-row count of unknown residuals.
* ``metrics.json`` — per-arm MCC, balanced accuracy and AUPRC for every
  scenario and the pooled corpus, calibration provenance, frozen tolerances,
  input hashes, and held-out nominal false-alarm counts.

Calibration
-----------
The topology STL is calibrated on the first 80 % of the nominal signal
(chronological), using the repository default 99th percentile of ``|residual|``
plus additive margins scaled by 1.0. Global NiaARM Phi uses the first 80 % of
that calibration interval for mining and the remaining 20 % for its violation
threshold, giving the nominal 64/16/20 train/validation/test partition. Attack
scenarios never contribute to mining or calibration.

Nominals
--------
Anytown and C-Town use ``logic_attacks/<topology>/nominal`` with the corpus
``topology.inp``.  L-Town reuses the read-only 48 h Real/PDD nominal at
``scripts/out/full_signal_pdd/ltown/nominal_two_day`` and its PDD
``topology.inp``; only L-Town attack scenarios are read from
``logic_attacks/ltown``.  ``plan.json`` is hashed and recorded but no nominal
or PDD artifact is copied or modified.  Hidden staging directories and
scenario directories missing ``observed.csv`` / ``labels.csv`` are excluded.

Usage
-----
    python scripts/export_logic_attack_predictions.py
    python scripts/export_logic_attack_predictions.py --topologies anytown ctown
    python scripts/export_logic_attack_predictions.py --topologies ltown
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn

try:
    from scripts import export_topology_predictions as export
    from scripts import sweep_topology_tolerances as sweep
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import export_topology_predictions as export  # type: ignore
    import sweep_topology_tolerances as sweep  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Frozen calibration: repository default percentile and additive scale 1.
DEFAULT_PERCENTILE = 99.0
DEFAULT_ADDITIVE_SCALE = 1.0

DEFAULT_LOGIC_ROOT = REPO_ROOT / "scripts" / "out" / "logic_attacks"
DEFAULT_OUT_ROOT = REPO_ROOT / "scripts" / "out" / "logic_attack_predictions"
LTOWN_NOMINAL_DIR = REPO_ROOT / "scripts" / "out" / "full_signal_pdd" / "ltown" / "nominal_two_day"
LTOWN_INP = REPO_ROOT / "scripts" / "out" / "full_signal_pdd" / "ltown" / "topology.inp"

PREDICTION_COLUMNS = [
    "scenario",
    "row",
    "time_seconds",
    "attack_class",
    "attack_active",
    "observable_attack",
    "y_true",
    "y_true_observation_aware",
    "y_pred_gfsm",
    "y_pred_gfsm_timing",
    "y_pred_gfsm_obligations",
    "y_pred_phi",
    "y_pred_stl",
    "y_pred",
    "y_pred_combined_obligations",
    "phi_score",
    "gfsm_obligation_score",
    "stl_score",
    "combined_score",
    "combined_obligations_score",
    "uncalibrated_count",
]

ARM_NAMES = (
    "gfsm", "gfsm_timing", "gfsm_obligations", "phi", "stl", "combined",
    "combined_obligations",
)


class LogicAttackExportError(RuntimeError):
    """Raised when a logic-attack corpus cannot be exported."""


def _log(message: str) -> None:
    print(f"[export_logic_attack_predictions] {message}", flush=True)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value: Any) -> Any:
    return sweep._clean(value)


def _topology_paths(topology: str, logic_root: Path,
                    ltown_nominal_dir: Path | None,
                    ltown_inp: Path | None) -> tuple[Path, str, Path]:
    """Return ``(nominal_dir, nominal_name, inp_path)`` for *topology*."""
    if topology == "ltown":
        nominal_dir = Path(ltown_nominal_dir or LTOWN_NOMINAL_DIR)
        inp_path = Path(ltown_inp or LTOWN_INP)
        return nominal_dir, nominal_dir.name, inp_path
    root = Path(logic_root) / topology
    return root / "nominal", "nominal", root / "topology.inp"


def _read_attack_class(labels_path: Path) -> np.ndarray:
    """Return per-row attack classes; ``normal`` when the column is absent."""
    labels = pd.read_csv(labels_path)
    if "attack_class" in labels.columns:
        return labels["attack_class"].fillna("normal").astype(str).to_numpy()
    return np.full(len(labels), "normal", dtype=object)


def observable_attack_mask(frame: pd.DataFrame, reference: pd.DataFrame) -> np.ndarray:
    if set(frame.columns) != set(reference.columns):
        raise LogicAttackExportError("scenario and nominal observation columns differ")
    indexed = reference.set_index("time_seconds")
    times = frame["time_seconds"].to_numpy()
    if indexed.index.has_duplicates or not np.isin(times, indexed.index).all():
        raise LogicAttackExportError("scenario times do not align with nominal observations")
    columns = [column for column in frame.columns if column != "time_seconds"]
    actual = frame[columns].to_numpy(dtype=float)
    expected = indexed.loc[times, columns].to_numpy(dtype=float)
    return np.any(~np.isclose(actual, expected, rtol=1e-9, atol=1e-10), axis=1)


def _scenario_path(topology: str, name: str, nominal: str,
                   logic_root: Path, nominal_dir: Path) -> Path:
    if name == nominal:
        return Path(nominal_dir)
    return Path(logic_root) / topology / name


def _excluded_scenarios(topology: str, logic_root: Path) -> list[dict[str, str]]:
    """List hidden/staging or incomplete scenario directories (informational)."""
    root = Path(logic_root) / topology
    if not root.is_dir():
        return []
    excluded: list[dict[str, str]] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        if directory.name.startswith("."):
            excluded.append({"name": directory.name, "reason": "staging/hidden"})
            continue
        if not sweep._is_complete_scenario(directory):
            excluded.append({"name": directory.name, "reason": "incomplete"})
    return excluded


def _plan_metadata(plan_path: Path, nominal_dir: Path) -> dict[str, Any] | None:
    """Hash the corpus plan and record (never copy) its nominal reference."""
    plan_path = Path(plan_path)
    if not plan_path.is_file():
        return None
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    nominal = payload.get("nominal") if isinstance(payload, Mapping) else None
    recorded = None
    if isinstance(nominal, Mapping) and nominal.get("path"):
        recorded = str(Path(nominal["path"]).resolve())
    return {
        "path": str(plan_path),
        "sha256": _hash(plan_path),
        "nominal": nominal,
        "nominal_path_matches": recorded == str(Path(nominal_dir).resolve()),
        "scenario_count": len(payload.get("scenarios", []))
        if isinstance(payload, Mapping) else None,
    }


def metrics(frame: pd.DataFrame, nominal_mask: pd.Series,
            truth_column: str = "y_true") -> dict[str, Any]:
    """Per-arm metrics for *frame*; GFSM and STL use their own scores."""
    result: dict[str, Any] = {
        "rows": int(len(frame)),
        "attack_rows": int(frame[truth_column].sum()),
        "unknown_rows": int((frame["uncalibrated_count"] > 0).sum()),
        "unknown_residuals": int(frame["uncalibrated_count"].sum()),
    }
    arms = {
        "gfsm": (frame["y_pred_gfsm"].to_numpy(), frame["y_pred_gfsm"].to_numpy(float)),
        "gfsm_timing": (frame["y_pred_gfsm_timing"].to_numpy(), frame["y_pred_gfsm_timing"].to_numpy(float)),
        "gfsm_obligations": (frame["y_pred_gfsm_obligations"].to_numpy(), frame["gfsm_obligation_score"].to_numpy()),
        "phi": (frame["y_pred_phi"].to_numpy(), frame["phi_score"].to_numpy()),
        "stl": (frame["y_pred_stl"].to_numpy(), frame["stl_score"].to_numpy()),
        "combined": (frame["y_pred"].to_numpy(), frame["combined_score"].to_numpy()),
        "combined_obligations": (
            frame["y_pred_combined_obligations"].to_numpy(),
            frame["combined_obligations_score"].to_numpy(),
        ),
    }
    for arm, (prediction, score) in arms.items():
        values = export.detection_metrics(
            frame[truth_column].to_numpy(), prediction, score)
        result[arm] = {
            "mcc": values["mcc"],
            "balanced_accuracy": values["balanced_accuracy"],
            "auprc": values["aucpr"],
            "precision": values["precision"],
            "recall": values["recall"],
             "f1": values["f1"],
            "tp": values["tp"],
            "fp": values["fp"],
            "tn": values["tn"],
            "fn": values["fn"],
            "nominal_false_alarms": int(
                ((prediction == 1) & (frame[truth_column].to_numpy() == 0)
                 & nominal_mask.to_numpy()).sum()),
        }
    return result


def export_topology(
    *,
    topology: str,
    logic_root: Path = DEFAULT_LOGIC_ROOT,
    data_root: Path = export.REPO_ROOT / "data",
    out_root: Path = DEFAULT_OUT_ROOT,
    ltown_nominal_dir: Path | None = None,
    ltown_inp: Path | None = None,
    percentile: float = DEFAULT_PERCENTILE,
    additive_scale: float = DEFAULT_ADDITIVE_SCALE,
    prepared: dict[str, Any] | None = None,
    output_namespace: str | None = None,
) -> dict[str, Any]:
    """Export one topology's logic-attack predictions and metrics."""
    started = time.time()
    logic_root = Path(logic_root)
    nominal_dir, nominal_name, inp_path = _topology_paths(
        topology, logic_root, ltown_nominal_dir, ltown_inp)
    if not nominal_dir.is_dir():
        raise LogicAttackExportError(
            f"{topology}: nominal directory not found at {nominal_dir}")
    if not inp_path.is_file():
        raise LogicAttackExportError(f"{topology}: INP not found at {inp_path}")

    _log(f"{topology}: preparing scenarios (nominal={nominal_name})")
    if prepared is None:
        prepared = sweep.prepare(
            topology, logic_root, Path(data_root), Path(out_root),
            nominal_dir=nominal_dir, nominal_name=nominal_name, inp_path=inp_path,
            skip_incomplete=True,
        )
    plan = _plan_metadata(logic_root / topology / "plan.json", nominal_dir)
    excluded = _excluded_scenarios(topology, logic_root)

    additive = {
        unit: margin * additive_scale
        for unit, margin in export.DEFAULT_ADDITIVE.items()
    }
    _log(
        f"{topology}: calibrating STL on {len(prepared['calibration'])} nominal "
        f"rows (percentile={percentile}, additive_scale={additive_scale})"
    )
    evaluated, artifact = sweep.evaluate(prepared, percentile, additive_scale)
    n_cal = int(len(prepared["calibration"]))
    nominal = str(prepared["nominal"])
    nominal_reference = export._chronological(
        export._read_observed(nominal_dir / "observed.csv"))

    rows: list[dict[str, Any]] = []
    per_scenario: dict[str, Any] = {}
    for name, (frame, truth, _gfsm_flags) in prepared["scenarios"].items():
        group = evaluated[evaluated["scenario"] == name]
        if len(group) != len(frame):
            raise LogicAttackExportError(
                f"{topology}/{name}: evaluated {len(group)} rows, expected {len(frame)}")
        labels = prepared["label_frames"][name]
        offset = n_cal if name == nominal else 0
        label_frame = labels.iloc[offset:].reset_index(drop=True)
        if "attack_class" in label_frame.columns:
            attack_class = label_frame["attack_class"].fillna("normal").astype(str).to_numpy()
        else:
            attack_class = np.full(len(label_frame), "normal", dtype=object)
        if "observable_attack" in label_frame:
            observable = label_frame["observable_attack"].astype(str).str.lower().isin(
                ["true", "1"]).to_numpy()
        else:
            observable = observable_attack_mask(frame, nominal_reference)
        observation_aware = np.logical_or(truth, observable).astype(int)
        if len(attack_class) != len(group):
            raise LogicAttackExportError(
                f"{topology}/{name}: attack_class length mismatch")

        stl_flags = group["stl"].to_numpy()
        gfsm_flags = group["gfsm"].to_numpy()
        timing_flags = group["gfsm_timing"].to_numpy()
        obligation_flags = group["gfsm_obligations"].to_numpy()
        obligation_scores = group["gfsm_obligation_score"].to_numpy()
        combined_obligation_flags = group["combined_obligations"].to_numpy()
        combined_obligation_scores = group["combined_obligations_score"].to_numpy()
        phi_flags = group["phi"].to_numpy()
        combined_flags = group["combined"].to_numpy()
        stl_scores = group["stl_score"].to_numpy()
        phi_scores = group["phi_score"].to_numpy()
        combined_scores = group["combined_score"].to_numpy()
        expected_flags = np.logical_or.reduce((stl_flags, gfsm_flags, phi_flags)).astype(int)
        if not np.array_equal(combined_flags, expected_flags):
            raise LogicAttackExportError(
                f"{topology}/{name}: combined flags are not GFSM OR Phi OR STL")
        if not np.allclose(
            combined_scores, np.maximum.reduce((stl_scores, gfsm_flags, phi_scores))
        ):
            raise LogicAttackExportError(
                f"{topology}/{name}: combined score is not max(STL, GFSM, Phi)")
        expected_obligations = np.logical_or.reduce(
            (stl_flags, obligation_flags, phi_flags)
        ).astype(int)
        if not np.array_equal(combined_obligation_flags, expected_obligations):
            raise LogicAttackExportError(
                f"{topology}/{name}: obligation fusion is not GFSM_obligations OR Phi OR STL")
        if not np.allclose(
            combined_obligation_scores,
            np.maximum.reduce((stl_scores, obligation_scores, phi_scores)),
        ):
            raise LogicAttackExportError(
                f"{topology}/{name}: obligation fused score is not max(STL, obligations, Phi)")

        times = frame["time_seconds"].to_numpy()
        start = len(rows)
        for index in range(len(group)):
            rows.append({
                "scenario": name,
                "row": index,
                "time_seconds": times[index],
                "attack_class": attack_class[index],
                "attack_active": int(truth[index]),
                "observable_attack": int(observable[index]),
                "y_true": int(group["y_true"].iloc[index]),
                "y_true_observation_aware": int(observation_aware[index]),
                "y_pred_gfsm": int(gfsm_flags[index]),
                "y_pred_gfsm_timing": int(timing_flags[index]),
                "y_pred_gfsm_obligations": int(obligation_flags[index]),
                "y_pred_phi": int(phi_flags[index]),
                "y_pred_stl": int(stl_flags[index]),
                "y_pred": int(combined_flags[index]),
                "y_pred_combined_obligations": int(combined_obligation_flags[index]),
                "phi_score": float(phi_scores[index]),
                "gfsm_obligation_score": float(obligation_scores[index]),
                "stl_score": float(stl_scores[index]),
                "combined_score": float(combined_scores[index]),
                "combined_obligations_score": float(combined_obligation_scores[index]),
                "uncalibrated_count": int(group["unknown"].iloc[index]),
            })

        scenario_frame = pd.DataFrame(rows[start:])
        scenario_mask = scenario_frame["scenario"] == nominal
        per_scenario[name] = metrics(scenario_frame, scenario_mask)
        per_scenario[name]["observation_aware_metrics"] = metrics(
            scenario_frame, scenario_mask, "y_true_observation_aware")
        _log(
            f"{topology}/{name}: rows={len(group)} attack={int(truth.sum())} "
            f"stl_flags={int(stl_flags.sum())} gfsm_flags={int(gfsm_flags.sum())} "
            f"phi_flags={int(phi_flags.sum())} "
            f"timing_flags={int(timing_flags.sum())} "
            f"obligation_flags={int(obligation_flags.sum())}"
        )

    if not rows:
        raise LogicAttackExportError(f"{topology}: no complete scenario to export")

    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    nominal_mask = predictions["scenario"] == nominal
    aggregate = metrics(predictions, nominal_mask)
    nominal_false_alarms = {
        arm: aggregate[arm]["nominal_false_alarms"] for arm in ARM_NAMES
    }
    unknown_summary = {
        "total_unknown_residuals": int(predictions["uncalibrated_count"].sum()),
        "rows_with_unknown": int((predictions["uncalibrated_count"] > 0).sum()),
        "rows_total": int(len(predictions)),
    }

    if output_namespace is not None and (
        output_namespace in ("", ".", "..")
        or Path(output_namespace).name != output_namespace
    ):
        raise LogicAttackExportError("output_namespace must be one safe directory name")
    out_dir = Path(out_root) / topology
    if output_namespace is not None:
        out_dir = out_dir / output_namespace
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "detector_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    artifact_path = out_dir / "topology_stl_artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    inputs = dict(prepared["inputs"])
    if plan is not None:
        inputs[plan["path"]] = plan["sha256"]

    result = {
        "schema": "logic-attack-predictions/v3",
        "topology": topology,
        "generated_at_unix": int(started),
        "runtime_seconds": round(time.time() - started, 3),
        "config": {
            "percentile": percentile,
            "additive": dict(sorted(additive.items())),
            "additive_scale": additive_scale,
            "calibration_fraction": export.DEFAULT_CALIBRATION_FRACTION,
            "calibration_rows": n_cal,
            "nominal_scenario": nominal,
            "nominal_dir": str(nominal_dir),
            "output_namespace": output_namespace,
            "tuned_on_attacks": False,
            "fusion": "GFSM OR Phi OR STL",
            "obligation_fusion": "GFSM_obligations OR Phi OR STL",
            "gfsm_obligation_timing": "previous observation supplies guards; next observation supplies actuator feedback",
            "unknown_policy": export.DEFAULT_UNKNOWN_POLICY,
            "scenarios": list(prepared["scenarios"]),
        },
        "inputs": inputs,
        "outputs": {
            "detector_predictions_csv": str(predictions_path),
            "metrics_json": str(out_dir / "metrics.json"),
            "stl_artifact_json": str(artifact_path),
        },
        "stl": {
            "tolerances": artifact.get("tolerances"),
            "formulas": artifact.get("formulas"),
            "coverage": artifact.get("coverage"),
        },
        "gfsm": {"column_mapping": prepared["mapping"]},
        "gfsm_obligations": {
            "semantics": "previous-input transition guards plus next-observation action realization",
            "first_observation_policy": "no prior command history; no new obligation",
            "reason_counts": {
                name: len(reasons)
                for name, reasons in prepared["obligation_reasons"].items()
            },
        },
        "phi": {
            key: prepared["phi"].model[key]
            for key in (
                "schema", "granularity", "observable_policy", "calibration",
                "niaarm", "violation_fp_budget", "violation_threshold",
                "violation_threshold_note", "all_ok",
            )
        },
        "arms": aggregate,
        "observation_aware_arms": metrics(
            predictions, nominal_mask, "y_true_observation_aware"),
        "label_policy": {
            "primary": "y_true = attack_active (injection window; unchanged)",
            "secondary": "y_true_observation_aware = attack_active OR observable_attack",
            "observable_attack": "observations differ from matched nominal rows; rtol=1e-9, atol=1e-10",
            "attack_class": "injection-window mechanism; may be normal during post-injection effects",
        },
        "per_scenario": per_scenario,
        "nominal_false_alarms": {
            **nominal_false_alarms,
            "rows": int(nominal_mask.sum()),
            "held_out": True,
        },
        "unknown_count": unknown_summary,
        "excluded": excluded,
        "plan": plan,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "code_sha256": {
                str(path): _hash(path)
                for path in (
                    Path(__file__), Path(sweep.__file__),
                    REPO_ROOT / "src/invariants/full_signal_phi.py",
                    REPO_ROOT / "src/monitor/gfsm_detector.py",
                    REPO_ROOT / "src/monitor/gfsm_obligations.py",
                    REPO_ROOT / "src/stl/topology_detector.py",
                )
            },
        },
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(_clean(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    _log(
        f"{topology}: wrote {predictions_path} ({len(predictions)} rows) and "
        f"{metrics_path}")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_logic_attack_predictions",
        description=(
            "Export per-row GFSM-only / topology-STL-only / combined predictions "
            "and metrics for the logic-attack corpus."),
    )
    parser.add_argument(
        "--topologies", nargs="+", choices=list(export.TOPOLOGY_CONFIG),
        default=list(export.TOPOLOGY_CONFIG))
    parser.add_argument(
        "--logic-root", type=Path, default=DEFAULT_LOGIC_ROOT,
        help="logic_attacks corpus root (default: scripts/out/logic_attacks)")
    parser.add_argument(
        "--data-root", type=Path, default=export.REPO_ROOT / "data",
        help="data root (dataset manifests and generated GFSM)")
    parser.add_argument(
        "--out-root", type=Path, default=DEFAULT_OUT_ROOT,
        help="output root (default: scripts/out/logic_attack_predictions)")
    parser.add_argument(
        "--ltown-nominal-dir", type=Path, default=LTOWN_NOMINAL_DIR,
        help="read-only L-Town 48 h Real/PDD nominal")
    parser.add_argument(
        "--ltown-inp", type=Path, default=LTOWN_INP,
        help="L-Town PDD topology.inp")
    parser.add_argument(
        "--percentile", type=float, default=DEFAULT_PERCENTILE,
        help="STL tolerance percentile (default: 99)")
    parser.add_argument(
        "--additive-scale", type=float, default=DEFAULT_ADDITIVE_SCALE,
        help="scale applied to the repository additive margins (default: 1)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if len(set(args.topologies)) != len(args.topologies):
        raise SystemExit("duplicate topologies")
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    failures: list[str] = []
    for topology in args.topologies:
        try:
            result = export_topology(
                topology=topology,
                logic_root=args.logic_root,
                data_root=args.data_root,
                out_root=args.out_root,
                ltown_nominal_dir=args.ltown_nominal_dir,
                ltown_inp=args.ltown_inp,
                percentile=args.percentile,
                additive_scale=args.additive_scale,
            )
            summary[topology] = {
                "rows": result["unknown_count"]["rows_total"],
                "arms": {
                    arm: {
                        "mcc": result["arms"][arm]["mcc"],
                        "balanced_accuracy": result["arms"][arm]["balanced_accuracy"],
                        "auprc": result["arms"][arm]["auprc"],
                    }
                    for arm in ARM_NAMES
                },
                "nominal_false_alarms": result["nominal_false_alarms"],
                "unknown_count": result["unknown_count"],
                "excluded": result["excluded"],
            }
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            _log(f"{topology}: FAILED {type(exc).__name__}: {exc}")
            failures.append(topology)
    summary_path = Path(args.out_root) / "summary.json"
    summary_path.write_text(
        json.dumps(_clean({"topologies": summary, "failed": failures}),
                   indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    _log(f"wrote {summary_path}")
    if failures and len(failures) == len(args.topologies):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
