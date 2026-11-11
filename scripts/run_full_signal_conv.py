#!/usr/bin/env python3
"""Train ConvBiLSTM-AE on normal data and evaluate held-out full-signal runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from baselines.conv_bilstm_ae import ConvBiLSTMAE, ConvBiLSTMAEConfig
from dataio.full_signal import discover_full_signal_runs, split_nominal_run
from scripts.run_full_signal_hybrid import observable_frame
from stl.metrics import detection_metrics

OUT_ROOT = REPO_ROOT / "scripts/out/full_signal_conv"


def run_topology(topology: str, device: str, *, reuse_model: bool = False) -> dict:
    discovery = discover_full_signal_runs(topology)
    split = split_nominal_run(discovery.nominal)
    out_dir = OUT_ROOT / topology
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold_before_rescore = None
    if reuse_model:
        detector = ConvBiLSTMAE.load(out_dir / "model")
        threshold_before_rescore = float(detector.threshold)
        validation_errors = detector.reconstruction_errors(
            detector.windowize(observable_frame(split.validation.frame))
        )
        detector.threshold = float(np.quantile(
            validation_errors, detector.config.threshold_quantile
        ))
        print(
            f"{topology}: reused {out_dir / 'model'} without retraining; "
            f"validation threshold {threshold_before_rescore} -> {detector.threshold}"
        )
    else:
        config = ConvBiLSTMAEConfig.faithful(device=device)
        detector = ConvBiLSTMAE(config)
        detector.fit(observable_frame(split.train.frame), observable_frame(split.validation.frame))
        print(f"{topology}: training losses={detector.history}")
    detector.save(out_dir / "model")
    metrics, oracle_metrics, observation_aware_metrics, rows = {}, {}, {}, []
    for run in [split.test, *discovery.attacks]:
        frame = observable_frame(run.frame)
        errors = detector.predict(frame)
        finite = errors.row_errors[np.isfinite(errors.row_errors)]
        print(f"{topology}/{run.name}: error_range={finite.min()}..{finite.max()}")
        try:
            primary = detector.score(frame, run.y_true)
            oracle = detector.score_oracle(frame, run.y_true)
        except FloatingPointError as exc:
            primary = {"status": "partial_error", "reason": str(exc)}
            oracle = {"status": "partial_error", "reason": str(exc)}
        metrics[run.name] = primary
        oracle_metrics[run.name] = oracle
        errors = detector.predict(frame)
        run_labels = getattr(run, "labels", None)
        observable = (
            run_labels["observable_attack"].to_numpy(dtype=bool)
            if run_labels is not None and "observable_attack" in run_labels.columns
            else np.zeros(len(run.frame), dtype=bool)
        )
        observation_labels = np.asarray(run.y_true, dtype=bool) | observable
        observation_aware_metrics[run.name] = detection_metrics(
            observation_labels, errors.row_predictions, errors.row_errors
        )
        rows.append(pd.DataFrame({
            "scenario": run.name,
            "row": np.arange(len(frame)),
            "time_seconds": run.frame["time_seconds"].to_numpy(),
            "attack_class": run.attack_class,
            "y_true": run.y_true,
            "observable_attack": observable,
            "y_true_observation_aware": observation_labels,
            "reconstruction_error": errors.row_errors,
            "y_pred": errors.row_predictions,
        }))
    out_dir = OUT_ROOT / topology
    out_dir.mkdir(parents=True, exist_ok=True)
    detector.save(out_dir / "model")
    pd.concat(rows, ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)
    result = {
        "schema": "full-signal-conv-bilstm/v2",
        "topology": topology,
        "metrics": metrics,
        "observation_aware_secondary": observation_aware_metrics,
        "test_oracle_secondary": oracle_metrics,
        "primary_threshold": "normal-validation quantile only",
        "oracle_warning": "secondary test-optimal analysis; not used by primary model",
        "config": detector.config.to_dict(),
        "reconstruction_error_precision": (
            "float32 model inference; float64 residual subtraction, squaring, "
            "and mean reduction"
        ),
        "model_reused_without_retraining": reuse_model,
        "threshold_before_float64_rescore": threshold_before_rescore,
        "threshold_recomputed_on_normal_validation": reuse_model,
        "nominal_split": split.manifest,
        "feature_count": len(detector.feature_columns),
        "feature_columns": detector.feature_columns,
        "observable_policy": "numeric observed columns; simulator truth and labels excluded",
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=["anytown", "ctown", "ltown"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reuse-model", action="store_true",
        help="load the existing topology model and regenerate scores without retraining",
    )
    args = parser.parse_args()
    result = run_topology(args.topology, args.device, reuse_model=args.reuse_model)
    print(f"{args.topology}: wrote {OUT_ROOT / args.topology}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
