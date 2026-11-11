#!/usr/bin/env python3
"""Render LNCS result tables from expanded experiment artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stl.metrics import detection_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/State-Aware-ICS-Security/results_expansion.tex"
TOPOLOGIES = ("anytown", "ctown", "ltown")
ARMS = (("gfsm", "y_pred_gfsm", "y_pred_gfsm"),
        ("phi", "y_pred_phi", "phi_score"),
        ("stl", "y_pred_stl", "stl_score"),
        ("combined", "y_pred", "combined_score"))


def fmt(value: float, *, best: bool = False) -> str:
    text = f"{value:.4f}"
    return f"\\textbf{{{text}}}" if best else text


def ablation() -> str:
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Topology & Arm & MCC & BA & AUPRC & TP & FP & FN \\"]
    for topology in TOPOLOGIES:
        frame = pd.read_csv(ROOT / f"scripts/out/logic_attack_predictions/{topology}/detector_predictions.csv")
        metrics = [detection_metrics(frame.y_true, frame[flag], frame[score])
                   for _, flag, score in ARMS]
        best = {key: max(values[key] for values in metrics)
                for key in ("mcc", "balanced_accuracy", "aucpr")}
        for index, ((arm, _, _), values) in enumerate(zip(ARMS, metrics)):
            prefix = {"anytown": "Anytown", "ctown": "C-Town", "ltown": "L-Town"}[topology] if index == 0 else " "
            arm_label = {"gfsm": "GFSM", "phi": r"$\Phi$", "stl": "STL", "combined": "Combined"}[arm]
            lines.append(
                f"{prefix} & {arm_label} & "
                f"{fmt(values['mcc'], best=values['mcc'] == best['mcc'])} & "
                f"{fmt(values['balanced_accuracy'], best=values['balanced_accuracy'] == best['balanced_accuracy'])} & "
                f"{fmt(values['aucpr'], best=values['aucpr'] == best['aucpr'])} & "
                rf"{values['tp']} & {values['fp']} & {values['fn']} \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def baseline_one(method: str, topology: str) -> dict | None:
    path = ROOT / f"scripts/out/full_signal_{method}/{topology}/predictions.csv"
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
                        "sain_local": "score", "sicams": "moving_average"}[method]
        scores = frame[score_column]
        predictions = frame["y_pred"]
    values = detection_metrics(frame["y_true"], predictions, scores)
    return {"status": "ok", "metrics": values}


def baselines() -> str:
    lines = [r"\begin{tabular}{llll}", r"\toprule",
             r"Method & Anytown & C-Town & L-Town \\", r"\midrule"]
    methods = ("cusum", "sain", "sain_local", "sicams", "imdisco", "hybrid", "conv")
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
            "cusum": "State-aware CUSUM",
            "sain": "SAIN global (source-derived)",
            "sain_local": "SAIN local (NiaARM)",
            "sicams": "SICAMS",
            "imdisco": "IM-DISCO",
            "hybrid": "Hybrid ensemble",
            "conv": "ConvBiLSTM-AE",
        }[method]
        cells = []
        for topology in TOPOLOGIES:
            entry = results[(method, topology)]
            if entry["status"] != "ok":
                cells.append(entry["status"])
                continue
            values = entry["metrics"]
            cells.append(
                "\\shortstack[l]{"
                f"MCC {fmt(values['mcc'], best=values['mcc'] == best[topology]['mcc'])}\\\\"
                f"BA {fmt(values['balanced_accuracy'], best=values['balanced_accuracy'] == best[topology]['balanced_accuracy'])}\\\\"
                f"AUPRC {fmt(values['aucpr'], best=values['aucpr'] == best[topology]['aucpr'])}"
                "}"
            )
        lines.append(f"{label} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def tolerance() -> str:
    frame = pd.read_csv(ROOT / "scripts/out/logic_tolerance_sweep/metrics.csv")
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Topology & STL MCC range & STL BA range & Nominal false alarms (min--max) \\", r"\midrule"]
    for topology in TOPOLOGIES:
        subset = frame[(frame.topology == topology) & (frame.scenario == "ALL")
                       & (frame.scope == "injection_window")]
        stl = subset[subset.arm == "stl"]
        alarms = {}
        for arm in ("gfsm", "phi", "stl", "combined"):
            values = subset[subset.arm == arm].nominal_false_alarms
            alarms[arm] = f"{int(values.min())}--{int(values.max())}"
        alarm_cell = (
            f"G {alarms['gfsm']}; $\\Phi$ {alarms['phi']}\\\\"
            f"S {alarms['stl']}; C {alarms['combined']}"
        )
        lines.append(
            f"{topology} & {fmt(stl.mcc.min())}--{fmt(stl.mcc.max())} & "
            f"{fmt(stl.balanced_accuracy.min())}--{fmt(stl.balanced_accuracy.max())} & "
            f"\\shortstack[l]{{{alarm_cell}}}"
            f"\\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main() -> int:
    result = (
        "% Generated by scripts/render_expansion_results.py; do not edit metrics manually.\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Expanded attack-corpus ablation. Counts are pooled over injection-window rows.}\n"
        "\\label{tab:expanded_ablation}\n"
        f"{ablation()}\n\\end{{table}}\n\n"
        "\\begin{table}[tb]\n\\centering\n"
        "\\caption{Tolerance sensitivity over nine configurations. Nominal alarms use held-out attack-free rows only. Alarm ranges are GFSM (G), $\\Phi$, STL (S), and combined (C).}\n"
        "\\label{tab:tolerance_sweep}\n"
        f"{tolerance()}\n\\end{{table}}\n\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Full-signal baseline results. The best MCC, balanced accuracy (BA), and AUPRC are bolded per topology, including ties. Unavailable methods and pending runs are reported explicitly.}\n"
        "\\label{tab:expanded_baselines}\n"
        f"{baselines()}\n\\end{{table}}\n"
    )
    OUT.write_text(result)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
