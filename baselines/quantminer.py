"""Headless Python wrapper for the vendored QuantMiner Java implementation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO_ROOT / "vendor/quantminer"
UTIL_JAR = VENDOR_ROOT / "dist/ostermillerutils_1_07_00.jar"


@dataclass(frozen=True)
class QuantMinerConfig:
    min_support: float = 0.10
    min_confidence: float = 0.60
    population: int = 250
    generations: int = 100
    max_attributes: int = 3


def compile_cli(work_dir: Path) -> Path:
    work_dir = Path(work_dir)
    classes = work_dir / "classes"
    marker = classes / "QuantMinerCli.class"
    sources = sorted((VENDOR_ROOT / "upstream/src").rglob("*.java"))
    cli = VENDOR_ROOT / "QuantMinerCli.java"
    if marker.is_file() and marker.stat().st_mtime >= max(path.stat().st_mtime for path in [*sources, cli]):
        return classes
    classes.mkdir(parents=True, exist_ok=True)
    command = ["javac", "-cp", str(UTIL_JAR), "-d", str(classes),
               *map(str, sources), str(cli)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"QuantMiner compilation failed: {result.stderr}")
    return classes


def run_quantminer(frame: pd.DataFrame, out_dir: Path,
                   config: QuantMinerConfig = QuantMinerConfig()) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "training.csv"
    output_path = out_dir / "rules.csv"
    frame.to_csv(input_path, index=False)
    classes = compile_cli(out_dir / "build")
    command = [
        "java", "-cp", f"{classes}:{UTIL_JAR}", "QuantMinerCli",
        str(input_path), str(output_path), str(config.min_support),
        str(config.min_confidence), str(config.population),
        str(config.generations), str(config.max_attributes),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode or not output_path.is_file():
        raise RuntimeError(f"QuantMiner failed: {result.stderr or result.stdout}")
    provenance = {
        "implementation": "official_vendored_quantminer_java",
        "upstream_commit": "587eea87c3603a2e0ef0e4b032422ffdedf5b645",
        "command": command,
        "config": config.__dict__,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "rules_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "rule_count": max(0, len(output_path.read_text().splitlines()) - 1),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return {"rules_path": output_path, "provenance": provenance}
