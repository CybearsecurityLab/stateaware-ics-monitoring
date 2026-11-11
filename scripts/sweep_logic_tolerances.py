#!/usr/bin/env python3
"""Tolerance sensitivity over the complete logic-attack corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts import export_logic_attack_predictions as logic_export

OUT_ROOT = logic_export.REPO_ROOT / "scripts/out/logic_tolerance_sweep"
PERCENTILES = (95.0, 99.0, 99.9)
SCALES = (0.0, 1.0, 10.0)


def flatten(result: dict) -> list[dict]:
    topology = result["topology"]
    config = result["config"]
    rows = []
    for scope, metrics in (("injection_window", result["arms"]),
                           ("observation_aware", result["observation_aware_arms"])):
        for arm in logic_export.ARM_NAMES:
            row = {"topology": topology, "scenario": "ALL", "scope": scope,
                   "arm": arm, **metrics[arm]}
            rows.append(row)
    for scenario, values in result["per_scenario"].items():
        for scope, key in (("injection_window", None),
                           ("observation_aware", "observation_aware_metrics")):
            selected = values if key is None else values[key]
            for arm in logic_export.ARM_NAMES:
                rows.append({"topology": topology, "scenario": scenario,
                             "scope": scope, "arm": arm, **selected[arm]})
    for row in rows:
        row.update({"percentile": config["percentile"],
                    "additive_scale": config["additive_scale"]})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies", nargs="+",
                        choices=["anytown", "ctown", "ltown"],
                        default=["anytown", "ctown", "ltown"])
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    results, csv_rows = [], []
    prepared = {}
    for topology in args.topologies:
        nominal_dir, nominal_name, inp_path = logic_export._topology_paths(
            topology, logic_export.DEFAULT_LOGIC_ROOT, None, None
        )
        prepared[topology] = logic_export.sweep.prepare(
            topology,
            logic_export.DEFAULT_LOGIC_ROOT,
            logic_export.REPO_ROOT / "data",
            args.out_root,
            nominal_dir=nominal_dir,
            nominal_name=nominal_name,
            inp_path=inp_path,
            skip_incomplete=True,
        )
    for percentile in PERCENTILES:
        for scale in SCALES:
            for topology in args.topologies:
                result = logic_export.export_topology(
                    topology=topology, out_root=args.out_root,
                    percentile=percentile, additive_scale=scale,
                    prepared=prepared[topology],
                    output_namespace=f"p{percentile}_s{scale}",
                )
                results.append(result)
                csv_rows.extend(flatten(result))
    pd.DataFrame(csv_rows).to_csv(args.out_root / "metrics.csv", index=False)
    summary = {
        "schema": "logic-tolerance-sweep/v2",
        "percentiles": PERCENTILES,
        "additive_scales": SCALES,
        "calibration": "STL first 80% nominal; Phi nominal 64/16 train/validation; attacks independent",
        "fusion": "GFSM OR Phi OR STL",
        "primary_label": "injection_window",
        "secondary_label": "observation_aware (reported separately)",
        "closed_flow_epsilon_m3_per_s": 1e-6,
        "results": results,
    }
    (args.out_root / "summary.json").write_text(
        json.dumps(logic_export._clean(summary), indent=2, sort_keys=True,
                   allow_nan=False) + "\n")
    print(f"wrote {args.out_root / 'summary.json'} and {args.out_root / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
