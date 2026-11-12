from concurrent.futures import ThreadPoolExecutor
import hashlib

import pandas as pd

from baselines.quantminer import (
    QuantMinerConfig,
    _gpu_provenance,
    _source_provenance,
    compile_cli,
    run_quantminer,
)


def test_official_quantminer_cli_generates_quantitative_rules(tmp_path):
    frame = pd.DataFrame({
        "x": range(1, 21),
        "y": range(2, 42, 2),
        "mode": ["A"] * 20,
    })
    result = run_quantminer(
        frame, tmp_path,
        QuantMinerConfig(min_support=0.2, min_confidence=0.6,
                         population=20, generations=5),
    )
    rules = result["rules_path"].read_text()
    assert "x in [" in rules
    assert "y in [" in rules
    assert result["provenance"]["rule_count"] > 0
    assert result["provenance"]["implementation"] == "official_vendored_quantminer_java"


def test_compile_cli_shares_source_addressed_classes(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTMINER_CLASS_CACHE", str(tmp_path / "cache"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        classes = list(pool.map(compile_cli, [tmp_path / "a", tmp_path / "b"]))
    assert classes[0] == classes[1]
    assert (classes[0] / "QuantMinerCli.class").is_file()
    assert classes[0].parent.name == _source_provenance()["source_tree_sha256"]


def test_gpu_provenance_records_opt_in_and_library_hash(tmp_path, monkeypatch):
    library = tmp_path / "libquantminer_gpu.so"
    library.write_bytes(b"gpu-library")
    monkeypatch.setenv("QUANTMINER_GPU", "1")
    monkeypatch.setenv("QUANTMINER_GPU_LIBRARY", str(library))
    monkeypatch.setenv("QUANTMINER_GPU_PROFILE", "0")
    record = _gpu_provenance()
    assert record["requested"] is True
    assert record["profile_requested"] is False
    assert record["library_sha256"] == hashlib.sha256(b"gpu-library").hexdigest()
