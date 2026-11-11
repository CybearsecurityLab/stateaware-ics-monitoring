from scripts import create_reproducibility_snapshot as snapshot


def test_snapshot_selection_covers_evidence_and_excludes_worktrees():
    selected = {
        path.relative_to(snapshot.ROOT).as_posix()
        for path in snapshot.selected_files()
    }
    assert "scripts/out/expansion_pack/summary.json" in selected
    assert "scripts/out/full_signal_imdisco/tractability.json" in selected
    assert "scripts/out/full_signal_conv/ltown/model/model.pt" in selected
    assert "scripts/create_reproducibility_snapshot.py" in selected
    assert "paper/State-Aware-ICS-Security/dimva2027.tex" in selected
    assert not any(".worktrees" in path.split("/") for path in selected)
    assert not any(".git" in path.split("/") for path in selected)


def test_snapshot_hash_reads_large_files_in_chunks(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"evidence")
    assert snapshot.sha256_file(path) == (
        "ee8250fb76e094b34b471f13a73dbbe51d1ae142e9df59d7c0d31ec20f0a0a8e"
    )
