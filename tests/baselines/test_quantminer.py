import pandas as pd

from baselines.quantminer import QuantMinerConfig, run_quantminer


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
