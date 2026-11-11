#!/usr/bin/env python3
"""Evaluate source-derived partial SAIN on full-signal scenarios."""
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

from baselines.sain_native import (
    SainNativeDetector,
    load_fsm_documents,
    load_invariant_documents,
    mine_native_sain,
)
from dataio.full_signal import discover_full_signal_runs, split_nominal_run
from scripts import export_topology_predictions as export
from stl.metrics import detection_metrics

OUT_ROOT = REPO_ROOT / "scripts/out/full_signal_sain"


def run_topology(topology: str) -> dict:
    discovery = discover_full_signal_runs(topology)
    nominal_split = split_nominal_run(discovery.nominal)
    if topology == "ltown":
        gfsm_dir = REPO_ROOT / "scripts/out/logic_attack_predictions/ltown/generated/ltown/gfsm"
        analysis_dir = gfsm_dir.parent / "analysis"
        inp = REPO_ROOT / "scripts/out/full_signal_pdd/ltown/topology.inp"
        components = export._gfsm_components(json.loads((gfsm_dir / "ltown.gfsm.json").read_text()))
        fb_to_col, mapping_report = export._resolve_gfsm_columns(
            topology=topology, gfsm_dir=gfsm_dir, data_root=REPO_ROOT / "data",
            components=components, frame_columns=list(discovery.nominal.frame.columns),
        )
        column_map = {"T1": "T1", "PUMP_1": "s_PUMP_1", "PUMP_1F": "f_PUMP_1",
                      "p227": "p227", "p235": "p235", "n54": "n54"}
    else:
        gfsm_dir = REPO_ROOT / "data/generated" / topology / "gfsm"
        analysis_dir = gfsm_dir.parent / "analysis"
        gfsm_json = json.loads((gfsm_dir / f"{topology}.gfsm.json").read_text())
        components = export._gfsm_components(gfsm_json)
        fb_to_col, mapping_report = export._resolve_gfsm_columns(
            topology=topology, gfsm_dir=gfsm_dir, data_root=REPO_ROOT / "data",
            components=components, frame_columns=list(discovery.nominal.frame.columns),
        )
        if topology == "anytown":
            column_map = {"T41": "T41", "T42": "T42", "P78": "s_P78", "P79": "s_P79"}
        else:
            column_map = {name: name for name in discovery.nominal.frame.columns}
            column_map.update({f"L_T{i}": f"T{i}" for i in range(1, 8)})
    gfsm_json = json.loads((gfsm_dir / f"{topology}.gfsm.json").read_text())
    fsm_documents = load_fsm_documents(gfsm_dir, topology)
    invariant_documents = load_invariant_documents(analysis_dir, topology, fsm_documents)
    model = mine_native_sain(
        topology=topology, gfsm_json=gfsm_json,
        fsm_documents=fsm_documents, components=components, fb_to_col=fb_to_col,
        column_map=column_map, training_frames=[nominal_split.train.frame],
        invariant_documents=invariant_documents,
        quantminer_dir=OUT_ROOT / topology / "quantminer",
    )
    detector = SainNativeDetector(model)
    metrics, rows = {}, []
    for run in [nominal_split.test, *discovery.attacks]:
        prediction = detector.predict(run.frame)
        metrics[run.name] = detection_metrics(run.y_true, prediction.flags, prediction.scores)
        rows.append(pd.DataFrame({
            "scenario": run.name,
            "row": np.arange(len(run.frame)),
            "time_seconds": run.frame["time_seconds"].to_numpy(),
            "attack_class": run.attack_class,
            "y_true": run.y_true,
            "y_pred": prediction.flags,
            "score": prediction.scores,
        }))
    out_dir = OUT_ROOT / topology
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)
    result = {
        "schema": "full-signal-sain/v1",
        "topology": topology,
        "metrics": metrics,
        "model": model,
        "mapping": mapping_report,
        "nominal_split": nominal_split.manifest,
        "invariant_document_count": len(invariant_documents),
        "source_gaps": (
            "Official QuantMiner rule artifacts are generated separately; the "
            "original SAIN QuantMiner configuration and active SMONITOR runtime "
            "are not reproduced. Detection uses the documented offline empirical "
            "envelope adaptation."
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    (out_dir / "model.json").write_text(json.dumps(model, indent=2, sort_keys=True, default=str) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=["anytown", "ctown", "ltown"], required=True)
    args = parser.parse_args()
    result = run_topology(args.topology)
    print(f"{args.topology}: {len(result['metrics'])} scenarios -> {OUT_ROOT / args.topology}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
