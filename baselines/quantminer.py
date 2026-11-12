"""Headless Python wrapper for the vendored QuantMiner Java implementation."""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
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


def _source_records() -> list[dict[str, str]]:
    paths = sorted((VENDOR_ROOT / "upstream/src").rglob("*.java"))
    paths.append(VENDOR_ROOT / "QuantMinerCli.java")
    return [
        {
            "path": str(path.relative_to(VENDOR_ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]


def _source_tree_sha256(records: list[dict[str, str]]) -> str:
    tree = hashlib.sha256()
    for record in records:
        tree.update(record["path"].encode("utf-8"))
        tree.update(record["sha256"].encode("ascii"))
    return tree.hexdigest()


def compile_cli(work_dir: Path) -> Path:
    """Compile once per exact source tree and share classes across workers."""
    del work_dir  # Retained for compatibility with the original per-state API.
    sources = sorted((VENDOR_ROOT / "upstream/src").rglob("*.java"))
    cli = VENDOR_ROOT / "QuantMinerCli.java"
    digest = _source_tree_sha256(_source_records())
    cache_root = Path(os.environ.get(
        "QUANTMINER_CLASS_CACHE", REPO_ROOT / "scripts/out/quantminer_classes",
    ))
    cache_dir = cache_root / digest
    classes = cache_dir / "classes"
    marker = classes / "QuantMinerCli.class"
    if marker.is_file():
        return classes
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / "compile.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if marker.is_file():
            return classes
        temporary = cache_dir / f"tmp-{os.getpid()}"
        temporary_classes = temporary / "classes"
        shutil.rmtree(temporary, ignore_errors=True)
        temporary_classes.mkdir(parents=True)
        command = ["javac", "-cp", str(UTIL_JAR), "-d", str(temporary_classes),
                   *map(str, sources), str(cli)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            shutil.rmtree(temporary, ignore_errors=True)
            raise RuntimeError(f"QuantMiner compilation failed: {result.stderr}")
        if marker.is_file():
            shutil.rmtree(temporary, ignore_errors=True)
        else:
            os.replace(temporary_classes, classes)
            shutil.rmtree(temporary, ignore_errors=True)
    return classes


def _source_provenance() -> dict:
    records = _source_records()
    version = subprocess.run(["java", "-version"], capture_output=True, text=True)
    return {
        "files": records,
        "source_tree_sha256": _source_tree_sha256(records),
        "java_version": (version.stderr or version.stdout).strip(),
    }


def _gpu_provenance() -> dict:
    requested = os.environ.get("QUANTMINER_GPU", "").lower() in {
        "1", "true", "yes", "on",
    }
    library = os.environ.get("QUANTMINER_GPU_LIBRARY")
    record = {
        "requested": requested,
        "profile_requested": os.environ.get(
            "QUANTMINER_GPU_PROFILE", "",
        ).lower() in {"1", "true", "yes", "on"},
        "library": library,
        "library_sha256": None,
    }
    if library and Path(library).is_file():
        record["library_sha256"] = hashlib.sha256(
            Path(library).read_bytes(),
        ).hexdigest()
    return record


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
        "pristine_upstream": False,
        "source_patch_applied": (
            "exact cached BigInteger combinatorics and 64-bit combination index; "
            "optional HIP/JNI and packed-CPU support-count paths"
        ),
        "source_provenance": _source_provenance(),
        "source_provenance_timing": "recorded_by_wrapper_for_this_execution",
        "gpu": _gpu_provenance(),
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
