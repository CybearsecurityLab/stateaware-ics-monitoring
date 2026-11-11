#!/usr/bin/env python3
"""Run the exact maximal, thesis-inspired IM-DISCO adaptation.

Actuator roles come from the topology/GFSM mapping; every other observable
numeric channel is a sensor.  No features or episodes are trimmed and attack
labels are used only for post-prediction metrics.  Results are separate from
the preserved uncapped-unavailable artifacts.

``--provenance-only`` retains its legacy role for the original uncapped output
root and does not fit or invoke FP-Growth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from baselines.imdisco import IMDISCO, IMDISCOConfig
from dataio.full_signal import (
    discover_full_signal_runs,
    load_full_signal_run,
    portable_path,
    split_nominal_run,
)
from invariants.state_label import label_frame
from stl.metrics import detection_metrics

OUT_ROOT = REPO_ROOT / "scripts/out/full_signal_imdisco_maximal"
UNCAPPED_OUT_ROOT = REPO_ROOT / "scripts/out/full_signal_imdisco"
EXCLUDED_PREFIXES = ("leak_", "demand_", "emitter_")
EXCLUDED_COLUMNS = {
    "time_seconds", "timestamp", "attack_active", "attack_class", "label",
    "physical_impact", "delivery_or_command_modified", "observable_attack",
}


def observable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Exactly mirror run_full_signal_hybrid.observable_frame."""
    columns = []
    for column in frame.columns:
        if column in EXCLUDED_COLUMNS or column.startswith(EXCLUDED_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return frame.loc[:, columns].copy()


def frame_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, float_format="%.17g", lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def observable_policy(
    source: pd.DataFrame,
    observable: pd.DataFrame,
    *,
    sensors: tuple[str, ...],
    actuators: tuple[str, ...],
) -> dict[str, Any]:
    removed = [column for column in source.columns if column not in observable.columns]
    excluded_by_policy = [
        column for column in source.columns
        if column in EXCLUDED_COLUMNS or column.startswith(EXCLUDED_PREFIXES)
    ]
    dropped_non_numeric = [
        column for column in removed if column not in excluded_by_policy
    ]
    return {
        "schema": "imdisco-observable-features/v1",
        "selection": "numeric observed columns only; timestamp, labels and simulator truth excluded",
        "excluded_columns": sorted(EXCLUDED_COLUMNS),
        "excluded_prefixes": list(EXCLUDED_PREFIXES),
        "source_column_count": len(source.columns),
        "removed_column_count": len(removed),
        "removed_columns": removed,
        "excluded_by_name_or_prefix": excluded_by_policy,
        "dropped_non_numeric_columns": dropped_non_numeric,
        "remaining_numeric_feature_count": len(observable.columns),
        "remaining_numeric_features": list(observable.columns),
        "role_assignment": (
            "actuators are exact fb_to_col.values() from the topology/GFSM mapping; "
            "sensors are every other observable numeric channel; no dtype/cardinality inference"
        ),
        "feature_roles": {
            "sensors": list(sensors),
            "actuators": list(actuators),
        },
        "remaining_feature_counts_by_topology_role": {
            "sensors": len(sensors),
            "actuators": len(actuators),
            "total": len(sensors) + len(actuators),
        },
        "feature_trimming": False,
        "episode_trimming": False,
        "attack_labels_used_as_features_or_modes": False,
        "observable_frame_sha256": frame_sha256(observable),
    }


def topology_feature_roles(
    train: pd.DataFrame,
    fb_to_col: dict,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Exact topology roles, with no dtype/cardinality inference or trimming."""
    actuators = tuple(dict.fromkeys(str(column) for column in fb_to_col.values()))
    missing_actuators = [column for column in actuators if column not in train.columns]
    if missing_actuators:
        raise ValueError(f"topology-mapped actuator columns are not observable: {missing_actuators}")
    actuator_set = set(actuators)
    sensors = tuple(
        column for column in train.columns if column != "mode" and column not in actuator_set
    )
    return sensors, actuators


def _script_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_nominal(topology: str):
    if topology == "ltown":
        plan_path = REPO_ROOT / "scripts/out/logic_attacks/ltown/plan.json"
        plan = json.loads(plan_path.read_text())
        path = portable_path(plan["nominal"]["path"])
    else:
        path = REPO_ROOT / f"scripts/out/logic_attacks/{topology}/nominal"
    return load_full_signal_run(path, topology, name="nominal")


def _prepare_training(topology: str, split) -> tuple[pd.DataFrame, dict, dict, dict]:
    # Imported lazily so the structural audit can reuse observable_frame
    # without importing the unrelated hydraulic/export stack.
    from scripts import export_topology_predictions as export

    if topology == "ltown":
        gfsm_dir = REPO_ROOT / "scripts/out/logic_attack_predictions/ltown/generated/ltown/gfsm"
    else:
        gfsm_dir = REPO_ROOT / "data/generated" / topology / "gfsm"
    gfsm_path = gfsm_dir / f"{topology}.gfsm.json"
    gfsm_json = json.loads(gfsm_path.read_text())
    components = export._gfsm_components(gfsm_json)
    fb_to_col, mapping = export._resolve_gfsm_columns(
        topology=topology, gfsm_dir=gfsm_dir, data_root=REPO_ROOT / "data",
        components=components, frame_columns=list(split.train.frame.columns),
    )
    source = split.train.frame
    train = observable_frame(source)
    train["mode"] = label_frame(source, components, fb_to_col).to_numpy()
    sensors, actuators = topology_feature_roles(train, fb_to_col)
    policy = observable_policy(
        source, train.drop(columns=["mode"]), sensors=sensors, actuators=actuators
    )
    inputs = {
        "gfsm_path": str(gfsm_path),
        "gfsm_sha256": _script_sha256(gfsm_path),
    }
    # The nominal bundle's complete input hashes are added by run_topology.
    return train, mapping, policy, inputs


def run_topology(topology: str, *, provenance_only: bool = False) -> dict:
    nominal = _load_nominal(topology) if provenance_only else None
    discovery = None if provenance_only else discover_full_signal_runs(topology)
    if discovery is not None:
        nominal = discovery.nominal
    split = split_nominal_run(nominal)
    train, mapping, policy, input_hashes = _prepare_training(topology, split)
    input_hashes = {**nominal.input_hashes, **input_hashes}
    roles = policy["feature_roles"]
    config = IMDISCOConfig(
        mode_column="mode",
        sensor_columns=tuple(roles["sensors"]),
        actuator_columns=tuple(roles["actuators"]),
        prune_subsumed_rules=not provenance_only,
    )
    feature_provenance = {
        "runner_path": "scripts/run_full_signal_imdisco.py",
        "runner_sha256": _script_sha256(Path(__file__)),
        "analysis_path": "scripts/analyze_imdisco_tractability.py",
        "input_hashes": input_hashes,
        "mining_invoked_for_feature_policy_computation": False,
        "note": (
            "Legacy uncapped status/reason is preserved in provenance-only mode. "
            "Normal execution writes only the separate maximal adaptation artifacts."
        ),
        "feature_trimming": False,
        "episode_trimming": False,
        "attack_labels_used_for_training": False,
    }
    out_root = UNCAPPED_OUT_ROOT if provenance_only else OUT_ROOT
    out_dir = out_root / topology
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"

    if provenance_only:
        result = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {
            "schema": "full-signal-imdisco/v1",
            "topology": topology,
            "status": "unavailable",
            "reason": "NotRun: no IM-DISCO evaluation result is available",
        }
        # Deliberately do not change status, reason, metrics, or predictions.
        result["config"] = config.to_dict()
        result["observable_policy"] = policy
        result["feature_policy_provenance"] = feature_provenance
        metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
        return result

    try:
        detector = IMDISCO(config).fit(train)
    except Exception as exc:
        result = {
            "schema": "full-signal-imdisco-maximal/v1",
            "adaptation": "thesis-inspired exact maximal closed-pattern rules; not official code reproduction",
            "topology": topology,
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "config": config.to_dict(),
            "observable_policy": policy,
            "feature_policy_provenance": feature_provenance,
            "mapping": mapping,
            "nominal_split": split.manifest,
            "mode_count": int(train["mode"].nunique()),
        }
        metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
        return result

    metrics, rows = {}, []
    for run in [split.test, *discovery.attacks]:
        original = run.frame
        frame = observable_frame(original)
        prediction = detector.predict(frame)
        metrics[run.name] = detection_metrics(run.y_true, prediction.anomaly_flags, prediction.anomaly_scores)
        rows.append(pd.DataFrame({
            "scenario": run.name, "row": range(len(frame)),
            "time_seconds": original["time_seconds"].to_numpy(),
            "attack_class": run.attack_class, "y_true": run.y_true,
            "y_pred": prediction.anomaly_flags, "score": prediction.anomaly_scores,
            "matched_rule": prediction.matched_rule_ids,
        }))
    pd.concat(rows, ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)
    result = {
        "schema": "full-signal-imdisco-maximal/v1",
        "adaptation": "thesis-inspired exact maximal closed-pattern rules; not official code reproduction",
        "topology": topology,
        "status": "ok", "metrics": metrics, "mapping": mapping,
        "config": config.to_dict(),
        "observable_policy": policy,
        "feature_policy_provenance": feature_provenance,
        "nominal_split": split.manifest,
        "model": detector.provenance(),
        "mining": detector.artifacts()["mining"],
        "mode_count": int(train["mode"].nunique()),
        "evaluation_runs": [split.test.name, *[run.name for run in discovery.attacks]],
        "evaluation_episode_trimming": False,
    }
    (out_dir / "model.json").write_text(
        json.dumps(detector.artifacts(), indent=2, sort_keys=True, default=str) + "\n"
    )
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=["anytown", "ctown", "ltown"], required=True)
    parser.add_argument(
        "--provenance-only", action="store_true",
        help="update config/observable-feature metadata without fitting or mining",
    )
    args = parser.parse_args()
    result = run_topology(args.topology, provenance_only=args.provenance_only)
    suffix = " (provenance only; no mining)" if args.provenance_only else ""
    root = UNCAPPED_OUT_ROOT if args.provenance_only else OUT_ROOT
    print(f"{args.topology}: {result['status']}{suffix} -> {root / args.topology}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
