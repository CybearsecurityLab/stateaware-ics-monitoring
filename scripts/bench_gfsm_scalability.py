#!/usr/bin/env python3
"""Measure offline GFSM construction cost without changing evaluation artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "scripts/out/gfsm_scalability"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(topology: str, generated_dir: Path, jobs: int, max_states: int) -> dict:
    generated_dir = generated_dir.resolve()
    with tempfile.TemporaryDirectory(prefix=f"gfsm-scalability-{topology}-") as tmp:
        out_dir = Path(tmp) / "gfsm"
        command = [
            sys.executable, "-m", "gfsm",
            "--generated-dir", str(generated_dir),
            "--topology", topology,
            "--out-dir", str(out_dir),
            "--jobs", str(jobs),
            "--max-states", str(max_states),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([
            str(ROOT / "src"), env.get("PYTHONPATH", "")
        ])
        started = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        elapsed = time.perf_counter() - started
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        document = json.loads((out_dir / f"{topology}.gfsm.json").read_text())
        manifest = json.loads((out_dir / f"{topology}_gfsm_manifest.json").read_text())
        local_states = []
        for plc in manifest.get("plcs", []):
            for block in plc.get("stage2_fsms", []):
                local_states.append({
                    "plc": plc["name"],
                    "case_variable": f"{block['actuator']}_State",
                    "states": int(block["states"]),
                })
    theoretical = 1
    for block in local_states:
        theoretical *= block["states"]
    reachable = len(document.get("states") or {})
    pruning_ratio = 1.0 - reachable / theoretical if theoretical else None
    rss_divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return {
        "schema": "gfsm-scalability/v1",
        "topology": topology,
        "command": command,
        "jobs": jobs,
        "max_states": max_states,
        "elapsed_seconds": elapsed,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / rss_divisor,
        "local_blocks": local_states,
        "theoretical_product_states": theoretical,
        "reachable_states": reachable,
        "pruned_states": theoretical - reachable,
        "pruning_ratio": pruning_ratio,
        "transitions": len(document.get("transitions") or []),
        "input_manifest": {
            "path": str(generated_dir / topology / "analysis" / f"{topology}_analysis_manifest.json"),
            "sha256": sha256(generated_dir / topology / "analysis" / f"{topology}_analysis_manifest.json"),
        },
        "output_manifest": manifest,
        "software": {
            "python": sys.version,
            "platform": sys.platform,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=["anytown", "ctown", "ltown"], required=True)
    parser.add_argument("--generated-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=100000)
    args = parser.parse_args()
    generated = args.generated_dir or (
        ROOT / "scripts/out/logic_attack_predictions/ltown/generated"
        if args.topology == "ltown" else ROOT / "data/generated"
    )
    result = run(args.topology, generated, args.jobs, args.max_states)
    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.topology}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({k: result[k] for k in (
        "topology", "elapsed_seconds", "peak_rss_mib", "theoretical_product_states",
        "reachable_states", "pruning_ratio", "transitions",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
