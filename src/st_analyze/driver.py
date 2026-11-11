"""Stage-2 driver: orchestrate the analyzer across one topology."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adapter import analyze_source
from .manifest import build_manifest
from .model import StAnalyzeError


def _count_invariants_by_type(invariants: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for inv in invariants:
        counts[inv["type"]] = counts.get(inv["type"], 0) + 1
    return counts


def analyze_topology(
    *,
    generated_dir: Path,
    topology: str,
    out_dir: Path | None,
    keep_going: bool = False,
) -> dict[str, Any]:
    """Analyze every PLC .st listed in <topology>_manifest.json."""
    topo_dir = Path(generated_dir) / topology
    src_manifest_path = topo_dir / f"{topology}_manifest.json"
    if not src_manifest_path.exists():
        raise StAnalyzeError(
            f"st_gen manifest not found: {src_manifest_path}"
        )

    src_manifest_text = src_manifest_path.read_text()
    try:
        src_manifest = json.loads(src_manifest_text)
    except json.JSONDecodeError as exc:
        raise StAnalyzeError(
            f"invalid st_gen manifest {src_manifest_path}: {exc}"
        ) from exc

    out = Path(out_dir) if out_dir is not None else topo_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    plc_entries: list[dict[str, Any]] = []
    all_ok = True

    for plc in src_manifest.get("plcs", []):
        name = plc["name"]
        st_path = topo_dir / plc["file"]
        stem = st_path.stem  # e.g. anytown_plc1
        fsms = plc.get("fsms", [])

        if not st_path.exists():
            all_ok = False
            plc_entries.append({
                "name": name, "st_file": plc["file"], "status": "missing",
                "errors": [f"st file not found: {plc['file']}"],
                "artifacts": {}, "counts": {}, "st_gen_fsms": fsms,
            })
            if not keep_going:
                break
            continue

        result = analyze_source(st_path.read_text())

        ast_xml_f = f"{stem}.ast.xml"
        inv_f = f"{stem}.invariants.json"
        dot_f = f"{stem}.pdg.dot"
        pdgj_f = f"{stem}.pdg.json"

        (out / ast_xml_f).write_text(result.ast_xml)
        (out / inv_f).write_text(
            json.dumps(result.invariants, indent=2, sort_keys=True) + "\n"
        )
        (out / dot_f).write_text(result.pdg_dot)
        (out / pdgj_f).write_text(
            json.dumps(result.pdg_structured, indent=2, sort_keys=True) + "\n"
        )

        components = result.pdg_structured.get("components")
        if components is not None:
            pdg_nodes = sum(
                len(state.get("nodes", []))
                for component in components
                for state in component.get("states", [])
            )
        else:
            pdg_nodes = sum(
                len(graph.get("nodes", []))
                for graph in result.pdg_structured.values()
            )
        status = "ok"
        errors = list(result.errors)
        if fsms and pdg_nodes == 0 and result.ok:
            status = "error"
            errors.append("st_gen manifest lists FSMs but PDG is empty")
        elif not result.ok:
            status = "error"

        if status != "ok":
            all_ok = False

        plc_entries.append({
            "name": name,
            "st_file": plc["file"],
            "status": status,
            "errors": errors,
            "artifacts": {
                "ast_xml": ast_xml_f,
                "invariants_json": inv_f,
                "pdg_dot": dot_f,
                "pdg_json": pdgj_f,
            },
            "counts": {
                "pdg_nodes": pdg_nodes,
                "invariants_by_type": _count_invariants_by_type(
                    result.invariants
                ),
            },
            "st_gen_fsms": fsms,
        })

        if status != "ok" and not keep_going:
            break

    manifest = build_manifest(
        topology=topology,
        source_manifest_name=src_manifest_path.name,
        source_manifest_text=src_manifest_text,
        plc_entries=plc_entries,
    )
    manifest["all_ok"] = all_ok
    (out / f"{topology}_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest
