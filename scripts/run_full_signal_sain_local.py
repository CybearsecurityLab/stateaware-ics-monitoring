#!/usr/bin/env python3
"""Evaluate local per-CASE-state SAIN on the full-signal scenarios."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from baselines.sain import (  # noqa: E402
    QUANTMINER_CODE_COMMIT,
    SAIN_QUANTMINER_CONFIG,
    SainLocalDetector,
    mine_local_invariants,
)
from dataio.full_signal import (  # noqa: E402
    discover_full_signal_runs,
    split_nominal_run,
)
from scripts import export_topology_predictions as export  # noqa: E402
from scripts.run_full_signal_hybrid import (  # noqa: E402
    EXCLUDED_COLUMNS,
    EXCLUDED_PREFIXES,
    observable_frame,
)
from stl.metrics import detection_metrics  # noqa: E402

OUT_ROOT = REPO_ROOT / "scripts/out/full_signal_sain_local"
QUANTMINER_OUT_ROOT = REPO_ROOT / "scripts/out/full_signal_sain_local_quantminer"
OUT_ROOTS = {
    "niaarm": OUT_ROOT,
    "quantminer": QUANTMINER_OUT_ROOT,
}
TOPOLOGIES = ("anytown", "ctown", "ltown")
MIN_OBSERVATIONS = 50
MAX_EVALS = 5000
SEED = 42
FP_BUDGET = 0.01


def _json_payload(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"


def _json_sha256(value: Any) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def mining_feature_columns(
    frame: pd.DataFrame,
    state_columns: Sequence[str],
) -> list[str]:
    """Select observable numeric features, excluding every CASE selector."""
    states = set(state_columns)
    observed = observable_frame(frame)
    missing = sorted(states.difference(observed.columns))
    if missing:
        raise RuntimeError(
            f"component state columns are not observable numeric columns: {missing}"
        )
    return [column for column in observed.columns if column not in states]


def _feature_schema(
    train: pd.DataFrame,
    feature_cols: Sequence[str],
    state_columns: Sequence[str],
) -> dict[str, Any]:
    observed = observable_frame(train)
    mining = observed.loc[:, list(feature_cols)]
    schema = {
        "observable_columns": list(observed.columns),
        "observable_dtypes": {
            column: str(observed[column].dtype) for column in observed.columns
        },
        "mining_columns": list(mining.columns),
        "mining_dtypes": {
            column: str(mining[column].dtype) for column in mining.columns
        },
        "component_state_columns": sorted(set(state_columns)),
        "observable_policy": {
            "selection": "numeric observed columns only; simulator truth and labels excluded",
            "excluded_columns": sorted(EXCLUDED_COLUMNS),
            "excluded_prefixes": list(EXCLUDED_PREFIXES),
            "component_state_columns_excluded_from_mining": True,
        },
    }
    schema["observable_sha256"] = _json_sha256({
        "columns": schema["observable_columns"],
        "dtypes": schema["observable_dtypes"],
    })
    schema["mining_sha256"] = _json_sha256({
        "columns": schema["mining_columns"],
        "dtypes": schema["mining_dtypes"],
    })
    return schema


def _resolve_components(topology: str, frame_columns: list[str]):
    if topology == "ltown":
        gfsm_dir = (
            REPO_ROOT
            / "scripts/out/logic_attack_predictions/ltown/generated/ltown/gfsm"
        )
    else:
        gfsm_dir = REPO_ROOT / "data/generated" / topology / "gfsm"
    gfsm_json = json.loads((gfsm_dir / f"{topology}.gfsm.json").read_text())
    components = export._gfsm_components(gfsm_json)
    fb_to_col, mapping_report = export._resolve_gfsm_columns(
        topology=topology,
        gfsm_dir=gfsm_dir,
        data_root=REPO_ROOT / "data",
        components=components,
        frame_columns=frame_columns,
    )
    return components, fb_to_col, mapping_report


def _input_hashes(discovery) -> dict[str, Any]:
    return {
        "nominal": dict(sorted(discovery.nominal.input_hashes.items())),
        "attacks": {
            run.name: dict(sorted(run.input_hashes.items()))
            for run in discovery.attacks
        },
    }


def _mining_errors(model: dict[str, Any]) -> list[dict[str, Any]]:
    errors = []
    for component, entry in model.get("components", {}).items():
        for state_id, state in entry.get("local_states", {}).items():
            if state.get("status") == "mining_error":
                errors.append({
                    "component": component,
                    "state_id": state_id,
                    "observations": state.get("observations"),
                    "errors": list(state.get("errors") or []),
                })
    return errors


def _local_state_report(model: dict[str, Any]) -> dict[str, Any]:
    return {
        component: {
            "column": entry["column"],
            "state_ids": sorted(entry.get("local_states", {})),
            "state_count": len(entry.get("local_states", {})),
        }
        for component, entry in model.get("components", {}).items()
    }


def _rule_counts(model: dict[str, Any]) -> dict[str, Any]:
    by_local_state = {
        component: {
            state_id: len(state.get("rules") or [])
            for state_id, state in entry.get("local_states", {}).items()
        }
        for component, entry in model.get("components", {}).items()
    }
    by_component = {
        component: sum(states.values())
        for component, states in by_local_state.items()
    }
    return {
        "total": sum(by_component.values()),
        "by_component": by_component,
        "by_local_state": by_local_state,
    }


def _resolve_out_root(out_root: Path | None, engine: str) -> Path:
    if engine not in OUT_ROOTS:
        raise ValueError(f"unsupported local SAIN mining engine: {engine!r}")
    return Path(out_root) if out_root is not None else OUT_ROOTS[engine]


def pooled_detection_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    """Recompute pooled metrics from the renderer-compatible prediction columns."""
    required = {"y_true", "y_pred", "score"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise RuntimeError(f"predictions are missing pooled metric columns: {missing}")
    return detection_metrics(
        predictions["y_true"].to_numpy(dtype=int),
        predictions["y_pred"].to_numpy(dtype=int),
        predictions["score"].to_numpy(dtype=float),
    )


def _base_result(
    topology: str,
    discovery,
    nominal_split,
    mapping_report,
    feature_schema: dict[str, Any],
    n_jobs: int,
    engine: str,
) -> dict[str, Any]:
    result = {
        "schema": "full-signal-sain-local/v1",
        "engine": engine,
        "topology": topology,
        "granularity": "local_per_case_state",
        "config": {
            "min_observations": MIN_OBSERVATIONS,
            "max_evals": MAX_EVALS,
            "seed": SEED,
            "fp_budget": FP_BUDGET,
            "n_jobs": n_jobs,
        },
        "feature_schema": feature_schema,
        "mapping": mapping_report,
        "nominal_split": nominal_split.manifest,
        "split_hashes": {
            part.name: part.digest
            for part in (
                nominal_split.train,
                nominal_split.validation,
                nominal_split.test,
            )
        },
        "input_hashes": _input_hashes(discovery),
        "evaluation_protocol": {
            "nominal_train_fraction": 0.64,
            "clean_validation_fraction": 0.16,
            "held_out_nominal_fraction": 0.20,
            "attack_runs": "all held out; never split or used for fitting/calibration",
            "evaluation_scenarios": [
                nominal_split.test.name,
                *(run.name for run in discovery.attacks),
            ],
        },
        "skipped_runs": [item.to_dict() for item in discovery.skipped],
    }
    if engine == "quantminer":
        result["official_quantminer_commit"] = QUANTMINER_CODE_COMMIT
        result["config"]["quantminer"] = dict(SAIN_QUANTMINER_CONFIG.__dict__)
    return result


def _remove_stale_predictions(out_dir: Path) -> None:
    (out_dir / "predictions.csv").unlink(missing_ok=True)


def run_topology(
    topology: str,
    out_root: Path | None = None,
    *,
    n_jobs: int = 1,
    engine: str = "niaarm",
    max_attributes: int | None = None,
) -> dict[str, Any]:
    out_root = _resolve_out_root(out_root, engine)
    discovery = discover_full_signal_runs(topology)
    nominal_split = split_nominal_run(discovery.nominal)
    components, fb_to_col, mapping_report = _resolve_components(
        topology, list(discovery.nominal.frame.columns),
    )
    state_columns = list(dict.fromkeys(fb_to_col.values()))
    feature_cols = mining_feature_columns(nominal_split.train.frame, state_columns)
    feature_schema = _feature_schema(
        nominal_split.train.frame, feature_cols, state_columns,
    )
    out_dir = Path(out_root) / topology
    out_dir.mkdir(parents=True, exist_ok=True)
    result = _base_result(
        topology, discovery, nominal_split, mapping_report, feature_schema,
        n_jobs, engine,
    )

    train = observable_frame(nominal_split.train.frame)
    validation = observable_frame(nominal_split.validation.frame)
    expected_columns = list(train.columns)
    for run in [nominal_split.validation, nominal_split.test, *discovery.attacks]:
        columns = list(observable_frame(run.frame).columns)
        if columns != expected_columns:
            raise RuntimeError(f"{topology}: observable feature schemas differ in {run.name}")

    try:
        model = mine_local_invariants(
            topology=topology,
            components=components,
            fb_to_col=fb_to_col,
            training_frames=[train],
            validation_frames=[validation],
            feature_cols=feature_cols,
            min_observations=MIN_OBSERVATIONS,
            max_evals=MAX_EVALS,
            seed=SEED,
            fp_budget=FP_BUDGET,
            keep_going=True,
            n_jobs=n_jobs,
            engine=engine,
            max_attributes=max_attributes,
            quantminer_dir=(
                out_dir / "quantminer" if engine == "quantminer" else None
            ),
        )
    except Exception as exc:
        _remove_stale_predictions(out_dir)
        (out_dir / "model.json").unlink(missing_ok=True)
        result.update({
            "status": "unavailable",
            "reason": f"local invariant mining failed: {type(exc).__name__}: {exc}",
        })
        (out_dir / "metrics.json").write_text(_json_payload(result))
        return result

    model["feature_schema"] = feature_schema
    model["nominal_split"] = nominal_split.manifest
    model_path = out_dir / "model.json"
    model_payload = _json_payload(model)
    model_path.write_text(model_payload)
    model_sha256 = hashlib.sha256(model_payload.encode("utf-8")).hexdigest()
    mining_errors = _mining_errors(model)
    result.update({
        "status": "ok" if not mining_errors else "mining_errors",
        "model_sha256": model_sha256,
        "adaptations": model["adaptations"],
        "rule_counts": _rule_counts(model),
        "local_states": _local_state_report(model),
        "mining_errors": mining_errors,
        "model_all_ok": model["all_ok"],
    })
    result[engine] = model[engine]
    if engine == "quantminer":
        result["config"]["quantminer"] = dict(model["quantminer"]["config"])

    detector = SainLocalDetector(model)
    metrics: dict[str, Any] = {}
    rows = []
    try:
        for run in [nominal_split.test, *discovery.attacks]:
            frame = observable_frame(run.frame)
            prediction = detector.predict(frame)
            metrics[run.name] = detection_metrics(
                run.y_true, prediction.flags, prediction.scores,
            )
            rows.append(pd.DataFrame({
                "scenario": run.name,
                "row": np.arange(len(run.frame)),
                "time_seconds": run.frame["time_seconds"].to_numpy(),
                "attack_class": run.attack_class,
                "y_true": run.y_true,
                "y_pred": prediction.flags,
                "score": prediction.scores,
            }))
    except Exception as exc:
        _remove_stale_predictions(out_dir)
        result.update({
            "status": "unavailable",
            "reason": f"local detector evaluation failed: {type(exc).__name__}: {exc}",
            "metrics": metrics,
        })
        (out_dir / "metrics.json").write_text(_json_payload(result))
        return result

    predictions = pd.concat(rows, ignore_index=True)
    predictions_path = out_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    written = pd.read_csv(predictions_path)
    result["metrics"] = metrics
    result["pooled"] = pooled_detection_metrics(written)
    result["pooled_source"] = {
        "path": "predictions.csv",
        "columns": ["y_true", "y_pred", "score"],
        "rows": int(len(written)),
        "sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
    }
    (out_dir / "metrics.json").write_text(_json_payload(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=TOPOLOGIES, required=True)
    parser.add_argument("--engine", choices=tuple(OUT_ROOTS), default="niaarm")
    parser.add_argument(
        "--out-root", type=Path, default=None,
        help="output root (default depends on --engine)",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--max-attributes", type=int, choices=[2, 3], default=None,
        help="QuantMiner numeric-attribute cap; default uses the project adaptation",
    )
    args = parser.parse_args()
    out_root = _resolve_out_root(args.out_root, args.engine)
    result = run_topology(
        args.topology, out_root, n_jobs=args.jobs, engine=args.engine,
        max_attributes=args.max_attributes,
    )
    print(
        f"{args.topology} ({args.engine}): {result['status']}; "
        f"{len(result.get('metrics', {}))} scenarios -> "
        f"{out_root / args.topology}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
