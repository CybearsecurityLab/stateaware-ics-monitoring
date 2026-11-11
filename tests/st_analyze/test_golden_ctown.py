import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from st_analyze.driver import analyze_topology

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "data" / "generated"
TOPOLOGY = "ctown"


@pytest.mark.skipif(
    not (GEN / TOPOLOGY / f"{TOPOLOGY}_manifest.json").exists(),
    reason="generated data missing",
)
def test_every_plc_analyzed_and_artifacts_valid(tmp_path: Path):
    out = tmp_path / "analysis"
    manifest = analyze_topology(
        generated_dir=GEN, topology=TOPOLOGY, out_dir=out
    )

    src = json.loads(
        (GEN / TOPOLOGY / f"{TOPOLOGY}_manifest.json").read_text()
    )
    assert len(manifest["plcs"]) == len(src["plcs"])

    for entry in manifest["plcs"]:
        assert entry["status"] == "ok", f"{entry['name']}: {entry.get('errors')}"
        stem = Path(entry["st_file"]).stem
        ET.fromstring((out / f"{stem}.ast.xml").read_text())  # well-formed
        invs = json.loads((out / f"{stem}.invariants.json").read_text())
        assert isinstance(invs, list)
        # PLCs with FSMs must yield a non-empty PDG (paper Definition 1).
        if entry["st_gen_fsms"]:
            assert entry["counts"]["pdg_nodes"] >= 1


@pytest.mark.skipif(
    not (GEN / TOPOLOGY / f"{TOPOLOGY}_manifest.json").exists(),
    reason="generated data missing",
)
def test_analysis_is_deterministic(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    analyze_topology(generated_dir=GEN, topology=TOPOLOGY, out_dir=a)
    analyze_topology(generated_dir=GEN, topology=TOPOLOGY, out_dir=b)

    non_manifest = sorted(
        p.name for p in a.iterdir()
        if not p.name.endswith("_analysis_manifest.json")
    )
    assert non_manifest, "analyze_topology wrote no artifact files"
    for name in non_manifest:
        assert (a / name).read_text() == (b / name).read_text(), name


@pytest.mark.skipif(
    not (GEN / TOPOLOGY / f"{TOPOLOGY}_manifest.json").exists(),
    reason="generated data missing",
)
def test_every_ctown_case_selector_has_complete_namespaced_pdgs(tmp_path: Path):
    out = tmp_path / "analysis"
    manifest = analyze_topology(
        generated_dir=GEN, topology=TOPOLOGY, out_dir=out
    )

    for entry in manifest["plcs"]:
        source = (GEN / TOPOLOGY / entry["st_file"]).read_text()
        selectors = re.findall(r"CASE\s+(\w+_State)\s+OF", source)
        stem = Path(entry["st_file"]).stem
        pdg = json.loads((out / f"{stem}.pdg.json").read_text())
        components = pdg.get("components", [])
        assert [c["case_var"] for c in components] == selectors
        assert {c["plc"] for c in components} <= {entry["name"]}

        identities = [
            (c["plc"], c["case_var"], state["state_id"])
            for c in components
            for state in c["states"]
        ]
        assert len(identities) == len(set(identities))
        assert entry["counts"]["pdg_nodes"] == sum(
            len(state["nodes"]) for c in components for state in c["states"]
        )

        invariants = json.loads(
            (out / f"{stem}.invariants.json").read_text()
        )
        if len(selectors) > 1:
            assert {inv["case_var"] for inv in invariants} == set(selectors)

        dot = (out / f"{stem}.pdg.dot").read_text()
        assert dot.count("subgraph cluster_") == 2 * len(selectors)
        for component in components:
            selector = component["case_var"]
            actuator = selector.removesuffix("_State")
            assert [s["state_id"] for s in component["states"]] == ["0", "1"]
            assert component["states"][0]["nodes"][0]["statement"] == (
                f"{actuator} := FALSE"
            )
            assert any(
                node["type"] == "condition"
                for state in component["states"]
                for node in state["nodes"]
            )
            for state_id in ("0", "1"):
                assert f"{entry['name']} / {selector} / State {state_id}" in dot
