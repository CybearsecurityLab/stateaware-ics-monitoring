#!/usr/bin/env python3
"""Build the expanded experiment evidence pack from stored artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stl.metrics import detection_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scripts/out/expansion_pack"
TOPOLOGIES = ("anytown", "ctown", "ltown")


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def ablation(topology: str) -> dict:
    path = ROOT / f"scripts/out/logic_attack_predictions/{topology}/detector_predictions.csv"
    frame = pd.read_csv(path)
    result = {}
    for scope, truth in (("injection_window", "y_true"),
                         ("observation_aware", "y_true_observation_aware")):
        arms = {}
        for arm, flag, score in (("gfsm", "y_pred_gfsm", "y_pred_gfsm"),
                                 ("gfsm_timing", "y_pred_gfsm_timing", "y_pred_gfsm_timing"),
                                 ("gfsm_obligations", "y_pred_gfsm_obligations", "gfsm_obligation_score"),
                                 ("phi", "y_pred_phi", "phi_score"),
                                 ("stl", "y_pred_stl", "stl_score"),
                                 ("combined", "y_pred", "combined_score"),
                                 ("combined_obligations", "y_pred_combined_obligations", "combined_obligations_score")):
            arms[arm] = detection_metrics(frame[truth], frame[flag], frame[score])
        result[scope] = arms
    classes = {}
    for attack_class in sorted(frame["attack_class"].unique()):
        subset = frame[frame["attack_class"] == attack_class]
        if attack_class == "normal":
            continue
        classes[attack_class] = {
            arm: detection_metrics(
                subset["y_true"], subset[flag], subset[score])
            for arm, flag, score in (("gfsm", "y_pred_gfsm", "y_pred_gfsm"),
                                     ("gfsm_timing", "y_pred_gfsm_timing", "y_pred_gfsm_timing"),
                                     ("gfsm_obligations", "y_pred_gfsm_obligations", "gfsm_obligation_score"),
                                     ("phi", "y_pred_phi", "phi_score"),
                                     ("stl", "y_pred_stl", "stl_score"),
                                     ("combined", "y_pred", "combined_score"),
                                     ("combined_obligations", "y_pred_combined_obligations", "combined_obligations_score"))
        }
    result["per_attack_class"] = classes
    result["prediction_file"] = str(path)
    result["rows"] = len(frame)
    return result


def baselines(topology: str) -> dict:
    result = {}
    for name in ("cusum", "sain", "sain_local", "sicams", "imdisco", "imdisco_maximal", "hybrid", "conv"):
        method_root = {
            "sain_local": "full_signal_sain_local_quantminer",
            "imdisco_maximal": "full_signal_imdisco_maximal",
        }.get(name, f"full_signal_{name}")
        path = ROOT / f"scripts/out/{method_root}/{topology}/metrics.json"
        result[name] = load(path) or {"status": "pending", "path": str(path)}
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tolerance = load(ROOT / "scripts/out/logic_tolerance_sweep/summary.json")
    pack = {
        "schema": "expansion-evidence/v2",
        "ablation": {topology: ablation(topology) for topology in TOPOLOGIES},
        "baselines": {topology: baselines(topology) for topology in TOPOLOGIES},
        "tolerance": tolerance,
        "notes": {
            "primary_protocol": "injection_window",
            "secondary_protocol": "observation_aware",
            "oracle": "test-oracle ConvBiLSTM thresholds are secondary only",
            "unavailable": "unavailable is not zero and is never used as a numeric score",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(pack, indent=2, sort_keys=True, default=str) + "\n")
    lines = ["# Expanded evidence pack", ""]
    for topology in TOPOLOGIES:
        arms = pack["ablation"][topology]["injection_window"]
        lines.append(f"## {topology}")
        for arm, values in arms.items():
            lines.append(f"- {arm}: MCC={values['mcc']:.4f}, BA={values['balanced_accuracy']:.4f}, TP={values['tp']}, FP={values['fp']}, FN={values['fn']}")
        for name, result in pack["baselines"][topology].items():
            status = result.get("status", "ok")
            lines.append(f"- {name}: {status}" + (f" — {result['reason']}" if result.get("reason") else ""))
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'summary.json'} and {OUT / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
