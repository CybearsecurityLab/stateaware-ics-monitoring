#!/usr/bin/env python3
"""Evaluate full-signal SICAMS with explicit covariance and later-stage gaps."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from baselines.sicams import SicamsConfig
from baselines.sicams_stages import SicamsFullPipeline, select_observed_columns
from baselines.sicams_topology import build_sicams_topology_inputs
from dataio.full_signal import discover_full_signal_runs, split_nominal_run
from scripts.run_full_signal_hybrid import observable_frame
from stl.metrics import detection_metrics

OUT_ROOT = REPO_ROOT / "scripts/out/full_signal_sicams"


def run_topology(topology: str) -> dict:
    discovery = discover_full_signal_runs(topology)
    split = split_nominal_run(discovery.nominal)
    config = SicamsConfig()
    pipeline = SicamsFullPipeline(config)
    try:
        calibration_frame = observable_frame(split.train.frame)
        selected, _ = select_observed_columns(calibration_frame)
        pipeline.fit(
            calibration_frame,
            feature_names=selected,
            spatial_clusters={column: [column] for column in selected},
        )
    except Exception as exc:
        result = {
            "schema": "full-signal-sicams/v1",
            "topology": topology,
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "config": config.to_dict(),
            "nominal_split": split.manifest,
            "observable_policy": "all numeric observed channels; no feature trimming",
        }
        out_dir = OUT_ROOT / topology
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
        return result

    inp_paths = [discovery.nominal.inp_path, *(run.inp_path for run in discovery.attacks)]
    if any(path is None for path in inp_paths):
        raise RuntimeError(f"{topology}: every SICAMS run must resolve its INP")
    inp_hashes = {
        hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in inp_paths
    }
    if len(inp_hashes) != 1:
        raise RuntimeError(f"{topology}: evaluated runs resolve different INP hashes")
    support_nodes = None
    policy = "all_observed_node_channels"
    if topology == "ltown":
        configuration = yaml.safe_load(
            (REPO_ROOT / "data/ltown/raw/dataset_configuration.yaml").read_text()
        )
        support_nodes = set(configuration["pressure_sensors"])
        if len(support_nodes) != 33:
            raise RuntimeError("L-Town official pressure inventory must contain 33 nodes")
        policy = "official_pressure_33"
    topology_inputs = build_sicams_topology_inputs(
        discovery.nominal.inp_path,
        selected,
        support_nodes=support_nodes,
        policy=policy,
    )

    metrics, rows, change_points = {}, [], {}
    discrimination, prelocalization, stage_artifacts = {}, {}, {}
    out_dir = OUT_ROOT / topology
    stage_dir = out_dir / "stages"
    stage_dir.mkdir(parents=True, exist_ok=True)
    for run in [split.test, *discovery.attacks]:
        frame = observable_frame(run.frame)
        result = pipeline.run_full(
            frame,
            sensor_nodes=topology_inputs.sensor_nodes,
            adjacency=topology_inputs.adjacency,
        )
        detection = result.detection
        metrics[run.name] = detection_metrics(run.y_true, detection.flags, detection.moving_average)
        change_points[run.name] = {name: value.to_dict() for name, value in result.change_points.items()}
        discrimination[run.name] = {
            key: value for key, value in result.discrimination.items()
            if key != "per_row"
        }
        prelocalization_summary = {
            key: value for key, value in result.prelocalization.items()
            if key != "result"
        }
        if "result" in result.prelocalization:
            located = result.prelocalization["result"]
            prelocalization_summary.update({
                "top_node": located["top_node"],
                "top_value": located["top_value"],
                "support_node_count": len(located["support_nodes"]),
            })
        prelocalization[run.name] = prelocalization_summary
        stage_path = stage_dir / f"{run.name}.json.gz"
        with gzip.open(stage_path, "wt", encoding="utf-8") as handle:
            json.dump({
                "discrimination": result.discrimination,
                "prelocalization": result.prelocalization,
                "leak_volume": result.leak_volume,
                "latency": result.latency,
                "provenance": result.provenance,
            }, handle, sort_keys=True)
        stage_artifacts[run.name] = {
            "path": str(stage_path.relative_to(out_dir)),
            "sha256": hashlib.sha256(stage_path.read_bytes()).hexdigest(),
        }
        rows.append(pd.DataFrame({
            "scenario": run.name,
            "row": np.arange(len(frame)),
            "time_seconds": run.frame["time_seconds"].to_numpy(),
            "attack_class": run.attack_class,
            "y_true": run.y_true,
            "t2": detection.t2,
            "tf2": detection.tf2,
            "moving_average": detection.moving_average,
            "y_pred": detection.flags,
        }))
    out_dir = OUT_ROOT / topology
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "topology.json").write_text(json.dumps(
        topology_inputs.to_dict(), indent=2, sort_keys=True
    ) + "\n")
    result = {
        "schema": "full-signal-sicams/v2",
        "topology": topology,
        "status": "ok",
        "metrics": metrics,
        "change_points": change_points,
        "discrimination": discrimination,
        "prelocalization": prelocalization,
        "stage_artifacts": stage_artifacts,
        "topology": topology_inputs.to_dict(include_graph=False),
        "config": config.to_dict(),
        "nominal_split": split.manifest,
        "feature_columns": pipeline.feature_names_,
        "feature_count": len(pipeline.feature_names_ or []),
        "calibration": pipeline.calibration_,
        "observable_policy": "all numeric observed channels; no feature trimming",
        "additional_stages": {
            "discrimination": (
                "per-row z^2 contribution shares saved; exact M_a classification "
                "unavailable without complete externally sourced operational ranges"
            ),
            "prelocalization": (
                "INP-backed Eq. 17 field saved at the retrospective peak row; "
                "no leak-localization accuracy is claimed for the cyber-attack corpus"
            ),
            "leak_volume": (
                "unavailable: current runs contain no nonzero aligned independent "
                "leak-volume training targets"
            ),
            "spatial_max_envelope": "adaptation; per-cluster results retained in change_points",
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=["anytown", "ctown", "ltown"], required=True)
    args = parser.parse_args()
    result = run_topology(args.topology)
    print(f"{args.topology}: {result['status']} -> {OUT_ROOT / args.topology}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
