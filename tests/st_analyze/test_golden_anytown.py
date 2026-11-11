import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from st_analyze.driver import analyze_topology

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "data" / "generated"
TOPOLOGY = "anytown"


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
def test_p78_and_p79_pdgs_and_dot_are_complete(tmp_path: Path):
    out = tmp_path / "analysis"
    manifest = analyze_topology(
        generated_dir=GEN, topology=TOPOLOGY, out_dir=out
    )
    pdg = json.loads((out / "anytown_plc1.pdg.json").read_text())
    components = pdg["components"]

    assert [c["case_var"] for c in components] == ["P78_State", "P79_State"]
    assert {c["plc"] for c in components} == {"PLC1"}
    identities = [
        (c["plc"], c["case_var"], state["state_id"])
        for c in components
        for state in c["states"]
    ]
    assert len(identities) == len(set(identities)) == 4
    for component in components:
        actuator = component["case_var"].removesuffix("_State")
        sensor = "T41" if actuator == "P78" else "T42"
        assert [s["state_id"] for s in component["states"]] == ["0", "1"]
        assert [
            node["statement"] for node in component["states"][0]["nodes"]
        ] == [
            f"{actuator} := FALSE",
            f"IF {sensor} < 5",
            f"{actuator}_State := 1",
        ]

    invariants = json.loads(
        (out / "anytown_plc1.invariants.json").read_text()
    )
    assert {inv["case_var"] for inv in invariants} == {
        "P78_State", "P79_State"
    }
    assert pdg["0"]["nodes"][0]["statement"] == "P79 := FALSE"

    dot = (out / "anytown_plc1.pdg.dot").read_text()
    assert dot.count("subgraph cluster_") == 4
    for selector in ("P78_State", "P79_State"):
        for state_id in ("0", "1"):
            assert f"PLC1 / {selector} / State {state_id}" in dot
    for statement in (
        "P78 := FALSE", "P78 := TRUE", "P79 := FALSE", "P79 := TRUE",
        "IF T41 < 5", "IF T42 > 8",
    ):
        assert statement in dot

    plc1 = next(p for p in manifest["plcs"] if p["name"] == "PLC1")
    assert plc1["counts"]["pdg_nodes"] == sum(
        len(state["nodes"]) for c in components for state in c["states"]
    )


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
