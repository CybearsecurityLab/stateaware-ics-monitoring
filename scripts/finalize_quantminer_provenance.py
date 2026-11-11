#!/usr/bin/env python3
"""Separate legacy QuantMiner annotations from execution-time provenance.

The listed artifacts predate per-execution source hashing.  An earlier version
of this script incorrectly presented hashes of the current checkout as evidence
of the code that produced them.  This migration preserves those hashes only as
retroactive annotations.  It never visits a new artifact unless that artifact
is explicitly added to ``LEGACY_ARTIFACTS``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_KEYS = {
    "pristine_upstream",
    "source_patch_applied",
    "source_provenance",
}
LEGACY_ARTIFACTS = [
    (root, topology)
    for root in (
        "full_signal_sain_local_quantminer",
        "full_signal_sain_local_quantminer_three_attribute",
        "full_signal_sain_local_quantminer_unfiltered",
        "full_signal_sain_local_quantminer_two_attribute_unfiltered",
    )
    for topology in ("anytown", "ctown", "ltown")
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def migrate_legacy_fields(payload: Any) -> None:
    """Migrate legacy fields recursively, including model-embedded provenance."""
    if isinstance(payload, list):
        for item in payload:
            migrate_legacy_fields(item)
        return
    if not isinstance(payload, dict):
        return
    for key, value in list(payload.items()):
        if key != "current_source_annotation":
            migrate_legacy_fields(value)
    if not SOURCE_KEYS.intersection(payload):
        return
    if payload.get("source_provenance_timing") == "recorded_by_wrapper_for_this_execution":
        return
    annotation = {
        "timing": "retroactive_annotation_not_execution_evidence",
        "pristine_upstream": payload.pop("pristine_upstream", None),
        "source_patch_applied": payload.pop("source_patch_applied", None),
        "source_provenance": payload.pop("source_provenance", None),
    }
    payload["execution_provenance"] = {
        "status": "not_recorded",
        "note": (
            "The original execution did not record source and Java-runtime "
            "hashes. Preserved outputs, configurations and errors do not "
            "establish which source revision produced them."
        ),
    }
    payload["current_source_annotation"] = annotation


def migrate_file(path: Path) -> None:
    payload = json.loads(path.read_text())
    migrate_legacy_fields(payload)
    write_json(path, payload)


def main() -> int:
    for root, topology in LEGACY_ARTIFACTS:
        topology_dir = ROOT / "scripts/out" / root / topology
        if not topology_dir.is_dir():
            continue
        for path in topology_dir.rglob("provenance.json"):
            migrate_file(path)
        for name in ("metrics.json", "model.json"):
            path = topology_dir / name
            if path.is_file():
                migrate_file(path)
        metrics_path = topology_dir / "metrics.json"
        model_path = topology_dir / "model.json"
        if metrics_path.is_file() and model_path.is_file():
            metrics = json.loads(metrics_path.read_text())
            metrics["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
            write_json(metrics_path, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
