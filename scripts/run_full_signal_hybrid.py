#!/usr/bin/env python3
"""Train and evaluate the hybrid ensemble on manifested full-signal episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from baselines.hybrid_ensemble import (
    HybridEnsemble,
    HybridEnsembleConfig,
    select_top_features,
)
from dataio.full_signal import (
    discover_full_signal_runs,
    load_full_signal_run,
    split_nominal_run,
)
from stl.metrics import detection_metrics

EPISODE_ROOT = REPO_ROOT / "scripts/out/supervised_episodes"
OUT_ROOT = REPO_ROOT / "scripts/out/full_signal_hybrid"
EXCLUDED_PREFIXES = ("leak_", "demand_", "emitter_")
EXCLUDED_COLUMNS = {
    "time_seconds", "timestamp", "attack_active", "attack_class", "label",
    "physical_impact", "delivery_or_command_modified", "observable_attack",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = []
    for column in frame.columns:
        if column in EXCLUDED_COLUMNS or column.startswith(EXCLUDED_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return frame.loc[:, columns].copy()


def load_episodes(topology: str, split: str) -> list:
    root = EPISODE_ROOT / topology / split
    if not root.is_dir():
        raise RuntimeError(f"missing manifested episode split: {root}")
    episodes = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "observed.csv").is_file():
            episodes.append(load_full_signal_run(path, topology))
    if not episodes:
        raise RuntimeError(f"no complete episodes in {root}")
    return episodes


def combine(runs, observable_only: bool = True):
    frames, labels, segments = [], [], []
    for index, run in enumerate(runs):
        frame = observable_frame(run.frame) if observable_only else run.frame.copy()
        frames.append(frame)
        labels.append(np.asarray(run.y_true, dtype=int))
        segments.append(np.full(len(frame), index, dtype=int))
    if len({tuple(frame.columns) for frame in frames}) != 1:
        raise RuntimeError("episode feature schemas differ")
    return pd.concat(frames, ignore_index=True), np.concatenate(labels), np.concatenate(segments)


def score(model, run, features: tuple[str, ...], training_frame: pd.DataFrame,
          segment: int = 0):
    """Score every evaluation row using only nominal-training history."""
    warmup = model.config.min_scored
    if len(training_frame) < warmup:
        raise RuntimeError(
            f"nominal training split has {len(training_frame)} rows; "
            f"Hybrid warm-up needs {warmup}"
        )
    prefix = observable_frame(training_frame.iloc[-warmup:]).loc[:, list(features)]
    evaluation = observable_frame(run.frame).loc[:, list(features)]
    frame = pd.concat([prefix, evaluation], ignore_index=True)
    result = model.predict_proba(frame, segments=np.full(len(frame), segment, dtype=int))

    expected_index = np.arange(warmup, warmup + len(run.frame), dtype=int)
    if len(result.proba) != len(run.frame) or not np.array_equal(
            np.asarray(result.index, dtype=int), expected_index):
        raise RuntimeError(
            f"{run.name}: Hybrid did not score every evaluation row "
            f"(expected {len(run.frame)}, got {len(result.proba)})"
        )
    predictions = pd.DataFrame({
        "row": np.arange(len(run.frame), dtype=int),
        "time_seconds": run.frame["time_seconds"].to_numpy(dtype=float),
        "attack_class": np.asarray(run.attack_class, dtype=str),
        "y_true": np.asarray(run.y_true, dtype=int),
        "score": np.asarray(result.proba, dtype=float),
    })
    y = predictions["y_true"].to_numpy()
    probabilities = predictions["score"].to_numpy()
    metrics = detection_metrics(y, (probabilities >= model.config.threshold).astype(int), probabilities)
    metrics.update({
        "dropped_rows": 0,
        "nominal_warmup_rows": int(warmup),
        "row_population": {
            "evaluation_rows": int(len(run.frame)),
            "scored_evaluation_rows": int(len(predictions)),
            "dropped_evaluation_rows": 0,
            "warmup_rows_prepended": int(warmup),
            "model_input_rows": int(len(frame)),
        },
    })
    return predictions, metrics


def json_digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def warmup_provenance(nominal_split, model) -> dict:
    train = nominal_split.train
    rows = int(model.config.min_scored)
    prefix = train.frame.iloc[-rows:]
    return {
        "source": "nominal_split.train.frame",
        "source_partition": "train",
        "source_partition_fraction": 0.64,
        "selection": "final model.config.min_scored rows",
        "rows": rows,
        "training_rows": int(len(train.frame)),
        "training_start_row": int(len(train.frame) - rows),
        "training_stop_row_exclusive": int(len(train.frame)),
        "start_time_seconds": float(prefix["time_seconds"].iloc[0]),
        "end_time_seconds": float(prefix["time_seconds"].iloc[-1]),
        "training_split_sha256": train.digest,
        "nominal_split_manifest_sha256": json_digest(nominal_split.manifest),
        "contains_validation_rows": False,
        "contains_nominal_test_rows": False,
        "applied_to": "every evaluation run",
    }


def default_observable_policy(selection) -> dict:
    return {
        "excluded_prefixes": list(EXCLUDED_PREFIXES),
        "excluded_columns": sorted(EXCLUDED_COLUMNS),
        "selection": "numeric observed columns only; simulator truth and labels excluded",
        "feature_block": "all observable raw features plus temporal features for the training-only top-10 (paper-like)",
        "selection_provenance": selection.to_dict(),
    }


def run_topology(topology: str, out_root: Path = OUT_ROOT, *,
                 reuse_model: bool = False):
    discovery = discover_full_signal_runs(topology)
    nominal_split = split_nominal_run(discovery.nominal)
    out_dir = Path(out_root) / topology
    out_dir.mkdir(parents=True, exist_ok=True)
    previous = None
    train_runs = None
    validation_runs = None

    if reuse_model:
        metrics_path = out_dir / "metrics.json"
        validation_path = out_dir / "validation_predictions.csv"
        if not metrics_path.is_file() or not validation_path.is_file():
            raise RuntimeError(
                f"--reuse-model requires existing metrics and validation predictions in {out_dir}"
            )
        previous = json.loads(metrics_path.read_text())
        # Load XGBoost before torch/pickle, matching the ensemble's documented
        # macOS OpenMP import-order workaround during fitting.
        import xgboost  # noqa: F401
        import torch
        torch.set_num_threads(1)
        model = HybridEnsemble.load(out_dir / "model")
        features = tuple(model.raw_columns_)
        if not features:
            features = tuple(previous["model_provenance"]["raw_columns"])
        observable_policy = previous["observable_policy"]
        episodes = previous["episodes"]
        all_metrics = {"validation": previous["metrics"]["validation"]}
        validation = None
        validation_probabilities = None
        validation_labels = None
        validation_positions = None
    else:
        train_runs = [nominal_split.train, *load_episodes(topology, "training")]
        validation_runs = [nominal_split.validation, *load_episodes(topology, "validation")]
        train_x, train_y, train_segments = combine(train_runs)
        validation_x, _, validation_segments = combine(validation_runs)
        selection = select_top_features(train_x, train_y)
        features = tuple(train_x.columns)
        config = HybridEnsembleConfig()
        model = HybridEnsemble(config)
        model.fit(train_x, train_y, segments=train_segments)
        model.save(out_dir / "model")
        observable_policy = default_observable_policy(selection)
        episodes = {
            "root": str(EPISODE_ROOT / topology),
            "plan_sha256": digest(EPISODE_ROOT / topology / "plan.json"),
            "manifest_sha256": digest(EPISODE_ROOT / topology / "manifest.json"),
            "training_scenarios": [run.name for run in train_runs],
            "validation_scenarios": [run.name for run in validation_runs],
            "held_out_root": str(REPO_ROOT / "scripts/out/logic_attacks" / topology),
        }

        validation_frame = observable_frame(validation_x)
        validation = model.predict_proba(validation_frame, segments=validation_segments)
        validation_probabilities = validation.proba
        validation_labels = np.concatenate([run.y_true for run in validation_runs])
        validation_positions = validation.index
        all_metrics = {
            "validation": detection_metrics(
                validation_labels[validation_positions],
                (validation_probabilities >= model.config.threshold).astype(int),
                validation_probabilities,
            )
        }

    config = model.config
    warmup = warmup_provenance(nominal_split, model)
    evaluation_runs = [nominal_split.test, *discovery.attacks]
    episodes["evaluation_scenarios"] = [run.name for run in evaluation_runs]
    predictions = []
    for run in evaluation_runs:
        prediction, metrics = score(model, run, features, nominal_split.train.frame)
        metrics["warmup_provenance"] = warmup
        name = run.name
        prediction.insert(0, "scenario", name)
        predictions.append(prediction)
        all_metrics[name] = metrics

    pd.concat(predictions, ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)
    if not reuse_model:
        validation_output = pd.DataFrame({
            "row": validation.index,
            "score": validation_probabilities,
            "y_true": validation_labels[validation_positions],
        })
        validation_output.to_csv(out_dir / "validation_predictions.csv", index=False)

    fit_audit = model.provenance_.get("fit_audit", {})
    selection_provenance = model.provenance_.get("feature_selection", {})
    metadata = {
        "schema": "full-signal-hybrid/v2",
        "topology": topology,
        "config": config.to_dict(),
        "model_reused_without_retraining": bool(reuse_model),
        "model_path": str(out_dir / "model"),
        "observable_policy": observable_policy,
        "episodes": episodes,
        "episode_manifests_preserved": bool(reuse_model),
        "validation_predictions": {
            "path": str(out_dir / "validation_predictions.csv"),
            "preserved_from_existing_run": bool(reuse_model),
            "sha256": digest(out_dir / "validation_predictions.csv"),
        },
        "nominal_split": nominal_split.manifest,
        "warmup": warmup,
        "training_split_hashes": {
            "nominal_train_sha256": nominal_split.train.digest,
            "nominal_split_manifest_sha256": json_digest(nominal_split.manifest),
            "observable_training_frame_sha256": fit_audit.get("train_frame_digest"),
            "feature_selection_training_sha256": selection_provenance.get("train_digest"),
            "episode_plan_sha256": episodes.get("plan_sha256"),
            "episode_manifest_sha256": episodes.get("manifest_sha256"),
        },
        "prediction_row_population": {
            "evaluation_rows": int(sum(len(run.frame) for run in evaluation_runs)),
            "prediction_rows": int(sum(len(frame) for frame in predictions)),
            "dropped_evaluation_rows": 0,
        },
        "metrics": all_metrics,
        "model_provenance": model.provenance_,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=["anytown", "ctown", "ltown"], required=True)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument(
        "--reuse-model", action="store_true",
        help="load the existing topology model and re-score without retraining",
    )
    args = parser.parse_args()
    result = run_topology(args.topology, args.out_root, reuse_model=args.reuse_model)
    print(f"{args.topology}: wrote {Path(args.out_root) / args.topology}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
