#!/usr/bin/env python3
"""Render LNCS result tables from expanded experiment artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.sain import SAIN_QUANTMINER_CONFIG
from stl.metrics import detection_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT_ABLATION = ROOT / "paper/State-Aware-ICS-Security/results_ablation.tex"
OUT_TOLERANCE = ROOT / "paper/State-Aware-ICS-Security/results_tolerance.tex"
OUT_BASELINES = ROOT / "paper/State-Aware-ICS-Security/results_baselines.tex"
TOPOLOGIES = ("anytown", "ctown", "ltown")
ARMS = (("gfsm_obligations", "y_pred_gfsm_obligations", "gfsm_obligation_score"),
        ("phi", "y_pred_phi", "phi_score"),
        ("stl", "y_pred_stl", "stl_score"),
        ("combined_obligations", "y_pred_combined_obligations", "combined_obligations_score"))


def fmt(value: float, *, best: bool = False) -> str:
    text = f"{value:.4f}"
    return f"\\textbf{{{text}}}" if best else text


def fmt_baseline(value: float, *, best: bool = False) -> str:
    text = f"{value:.3f}"
    return f"\\textbf{{{text}}}" if best else text


def ablation() -> str:
    lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
             r"Topology & Arm & MCC & BA & AUPRC & TP & FP & FN \\"]
    for topology in TOPOLOGIES:
        frame = pd.read_csv(ROOT / f"scripts/out/logic_attack_predictions/{topology}/detector_predictions.csv")
        metrics = [detection_metrics(frame.y_true, frame[flag], frame[score])
                   for _, flag, score in ARMS]
        best = {key: max(values[key] for values in metrics)
                for key in ("mcc", "balanced_accuracy", "aucpr")}
        for index, ((arm, _, _), values) in enumerate(zip(ARMS, metrics)):
            prefix = {"anytown": "Anytown", "ctown": "C-Town", "ltown": "L-Town"}[topology] if index == 0 else " "
            arm_label = {
                "gfsm_obligations": "G",
                "phi": r"$\Phi$",
                "stl": "STL",
                "combined_obligations": "C",
            }[arm]
            lines.append(
                f"{prefix} & {arm_label} & "
                f"{fmt(values['mcc'], best=values['mcc'] == best['mcc'])} & "
                f"{fmt(values['balanced_accuracy'], best=values['balanced_accuracy'] == best['balanced_accuracy'])} & "
                f"{fmt(values['aucpr'], best=values['aucpr'] == best['aucpr'])} & "
                rf"{values['tp']} & {values['fp']} & {values['fn']} \\")
        positives = int(frame.y_true.sum())
        lines.append(
            rf" & Ref & 1.0000 & 1.0000 & 1.0000 & {positives} & 0 & 0 \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def baseline_one(method: str, topology: str) -> dict | None:
    if method in {"framework", "framework_obligations"}:
        frame = pd.read_csv(
            ROOT / f"scripts/out/logic_attack_predictions/{topology}/detector_predictions.csv"
        )
        flag = "y_pred_combined_obligations"
        score = "combined_obligations_score"
        return {
            "status": "ok",
            "metrics": detection_metrics(frame["y_true"], frame[flag], frame[score]),
        }
    method_root = {
        "sain_local": "full_signal_sain_local_quantminer_bitset",
        "imdisco_maximal": "full_signal_imdisco_maximal",
    }.get(method, f"full_signal_{method}")
    if method == "sain_local":
        metrics_path = ROOT / f"scripts/out/{method_root}/{topology}/metrics.json"
        if metrics_path.is_file():
            document = json.loads(metrics_path.read_text())
            status = document.get("status", "ok")
            if status != "ok":
                configured_width = (
                    document.get("config", {}).get("quantminer", {})
                    .get("max_attributes")
                )
                if (
                    configured_width is not None
                    and int(configured_width)
                    != SAIN_QUANTMINER_CONFIG.max_attributes
                ):
                    return {"status": "pending"}
                return {"status": status}
    path = ROOT / f"scripts/out/{method_root}/{topology}/predictions.csv"
    if not path.is_file():
        unavailable = ROOT / f"scripts/out/full_signal_{method}/{topology}/metrics.json"
        if unavailable.is_file():
            return {"status": "unavailable"}
        return {"status": "pending"}
    frame = pd.read_csv(path)
    if method == "hybrid":
        scores = frame["score"]
        predictions = (scores >= 0.5).astype(int)
    elif method == "conv":
        if not np.isfinite(frame["reconstruction_error"]).all():
            raise ValueError(f"non-finite ConvBiLSTM scores in {path}")
        scores = frame["reconstruction_error"]
        predictions = frame["y_pred"]
    else:
        score_column = {"cusum": "cusum_score", "sain": "score",
                        "sain_local": "score", "imdisco_maximal": "score",
                        "sicams": "moving_average"}[method]
        scores = frame[score_column]
        predictions = frame["y_pred"]
    values = detection_metrics(frame["y_true"], predictions, scores)
    return {"status": "ok", "metrics": values}


def baselines() -> str:
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Method & Anytown & C-Town & L-Town \\", r"\midrule"]
    methods = ("framework", "cusum", "sain", "sain_local", "sicams", "imdisco_maximal", "hybrid", "conv")
    results = {
        (method, topology): baseline_one(method, topology)
        for method in methods for topology in TOPOLOGIES
    }
    best = {}
    for topology in TOPOLOGIES:
        available = [
            entry["metrics"] for (_, entry_topology), entry in results.items()
            if entry_topology == topology and entry and entry["status"] == "ok"
        ]
        best[topology] = {
            key: max(values[key] for values in available)
            for key in ("mcc", "balanced_accuracy", "aucpr")
        }
    for method in methods:
        label = {
            "framework": "Ours",
            "framework_obligations": "O+O",
            "cusum": "CUSUM",
            "sain": "SAIN-G",
            "sain_local": "SAIN-L",
            "sicams": "SICAMS",
            "imdisco": "IM-DISCO",
            "imdisco_maximal": "IMD-max",
            "hybrid": "HE",
            "conv": "CAE",
        }[method]
        citation = {
            "framework": None,
            "framework_obligations": None,
            "cusum": "ghaeini2018state",
            "sain": "abbas2024sain",
            "sain_local": "abbas2024sain",
            "sicams": "sicams2025",
            "imdisco": "imdisco2024",
            "imdisco_maximal": "saraiva2024imdisco",
            "hybrid": "ahmed2025hybridensemblemethoddetecting",
            "conv": "lee2024advancing",
        }[method]
        cells = []
        for topology in TOPOLOGIES:
            entry = results[(method, topology)]
            if entry["status"] != "ok":
                cells.append(entry["status"].replace("_", r"\_"))
                continue
            values = entry["metrics"]
            cells.append(
                f"{fmt_baseline(values['mcc'], best=values['mcc'] == best[topology]['mcc'])}/"
                f"{fmt_baseline(values['balanced_accuracy'], best=values['balanced_accuracy'] == best[topology]['balanced_accuracy'])}/"
                f"{fmt_baseline(values['aucpr'], best=values['aucpr'] == best[topology]['aucpr'])}"
            )
        citation_text = "" if citation is None else f"~\\cite{{{citation}}}"
        lines.append(f"{label}{citation_text} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def tolerance() -> str:
    frame = pd.read_csv(ROOT / "scripts/out/logic_tolerance_sweep/metrics.csv")
    lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{STL score range} & \multicolumn{4}{c}{Nominal false alarms (min--max)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}",
        r"Topology & MCC & BA & G & $\Phi$ & S & C \\",
        r"\midrule",
    ]
    for topology in TOPOLOGIES:
        subset = frame[(frame.topology == topology) & (frame.scenario == "ALL")
                       & (frame.scope == "injection_window")]
        stl = subset[subset.arm == "stl"]
        alarms = {}
        for arm in ("gfsm", "phi", "stl", "combined"):
            values = subset[subset.arm == arm].nominal_false_alarms
            alarms[arm] = f"{int(values.min())}--{int(values.max())}"
        label = {"anytown": "Anytown", "ctown": "C-Town", "ltown": "L-Town"}[topology]
        lines.append(
            f"{label} & {fmt(stl.mcc.min())}--{fmt(stl.mcc.max())} & "
            f"{fmt(stl.balanced_accuracy.min())}--{fmt(stl.balanced_accuracy.max())} & "
            f"{alarms['gfsm']} & {alarms['phi']} & {alarms['stl']} & {alarms['combined']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main() -> int:
    header = "% Generated by scripts/render_expansion_results.py; do not edit metrics manually.\n"
    documents = {
        OUT_ABLATION: (
            "\\begin{table}[t]\n\\centering\n"
            "\\caption{Expanded attack-corpus ablation. Counts include all held-out rows; injection windows define positive labels. Arms: GFSM (G), $\\Phi$, STL, and fusion (C). Ref is an oracle reference, not a method.}\n"
            "\\label{tab:expanded_ablation}\n"
            f"{ablation()}\n\\end{{table}}\n"
        ),
        OUT_TOLERANCE: (
            "\\begin{table}[tb]\n\\centering\n"
            "\\caption{Tolerance sensitivity over nine configurations. Nominal alarms use held-out attack-free rows only. Alarm ranges are GFSM (G), $\\Phi$, STL (S), and combined (C).}\n"
            "\\label{tab:tolerance_sweep}\n"
            f"{tolerance()}\n\\end{{table}}\n"
        ),
        OUT_BASELINES: (
            "\\begin{table}[t]\n\\centering\n"
            "\\caption{Full-signal baseline results (MCC/BA/AUPRC). Best values are bolded per topology, including ties. Ours is the fused detector with previous-input guards and action obligations. SAIN-G/-L are global/local SAIN; ConvAE is ConvBiLSTM-AE; HE is the Hybrid Ensemble. Unavailable methods and pending runs are explicit.}\n"
            "\\label{tab:expanded_baselines}\n"
            f"{baselines()}\n\\end{{table}}\n"
        ),
    }
    for path, body in documents.items():
        path.write_text(header + body)
        print(f"wrote {path}")
    (ROOT / "paper/State-Aware-ICS-Security/results_expansion.tex").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
