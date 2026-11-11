import hashlib
import json

from scripts import finalize_quantminer_provenance as finalize


def test_legacy_annotations_are_not_execution_provenance(tmp_path, monkeypatch):
    topology = tmp_path / "scripts/out/legacy/anytown"
    topology.mkdir(parents=True)
    legacy_source = {"source_tree_sha256": "current-at-annotation-time"}
    model = {
        "components": {
            "PLC1.S": {
                "quantminer_provenance": {
                    "pristine_upstream": False,
                    "source_patch_applied": "patch",
                    "source_provenance": legacy_source,
                }
            }
        }
    }
    metrics = {"status": "mining_errors", "source_provenance": legacy_source}
    (topology / "model.json").write_text(json.dumps(model))
    (topology / "metrics.json").write_text(json.dumps(metrics))
    monkeypatch.setattr(finalize, "ROOT", tmp_path)
    monkeypatch.setattr(finalize, "LEGACY_ARTIFACTS", [("legacy", "anytown")])

    assert finalize.main() == 0

    migrated_model = json.loads((topology / "model.json").read_text())
    provenance = migrated_model["components"]["PLC1.S"]["quantminer_provenance"]
    assert provenance["execution_provenance"]["status"] == "not_recorded"
    annotation = provenance["current_source_annotation"]
    assert annotation["timing"] == "retroactive_annotation_not_execution_evidence"
    assert annotation["source_provenance"] == legacy_source
    assert "source_provenance" not in provenance
    migrated_metrics = json.loads((topology / "metrics.json").read_text())
    assert migrated_metrics["execution_provenance"]["status"] == "not_recorded"
    assert migrated_metrics["model_sha256"] == hashlib.sha256(
        (topology / "model.json").read_bytes()
    ).hexdigest()


def test_execution_captured_provenance_is_preserved():
    payload = {
        "source_provenance": {"source_tree_sha256": "execution-tree"},
        "source_provenance_timing": "recorded_by_wrapper_for_this_execution",
    }
    finalize.migrate_legacy_fields(payload)
    assert payload["source_provenance"] == {"source_tree_sha256": "execution-tree"}
    assert "current_source_annotation" not in payload
    assert "execution_provenance" not in payload
