"""Tests for the official-QuantMiner local SAIN mining engine."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from baselines.sain import (
    QUANTMINER_CODE_COMMIT,
    SAIN_QUANTMINER_CONFIG,
    QuantMinerRuleError,
    SainLocalDetector,
    mine_local_invariants,
    parse_quantminer_rules,
)


def _write_rules(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "Left",
        "Right",
        "Frequency[L and R]",
        "Support[L and R]",
        "Confidence[L->R]",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _row(left: str, right: str) -> dict[str, str]:
    return {
        "Left": left,
        "Right": right,
        "Frequency[L and R]": "5",
        "Support[L and R]": "25.0 %",
        "Confidence[L->R]": "83.33 %",
    }


def test_compound_official_csv_converts_to_invariant_schema(tmp_path):
    path = tmp_path / "rules.csv"
    _write_rules(path, [_row(
        "x in [1.0; 4.5]  AND  y in [-2; 3e1]",
        "z in [0; 10] AND w in [2.5; 7.5]",
    )])

    rules = parse_quantminer_rules(path)

    assert len(rules) == 1
    rule = rules[0]
    assert rule.support == 0.25
    assert rule.confidence == pytest.approx(0.8333)
    assert rule.lift == 0.0
    assert [atom.to_dict() for atom in rule.antecedent] == [
        {"col": "x", "op": "in", "val": [1.0, 4.5]},
        {"col": "y", "op": "in", "val": [-2.0, 30.0]},
    ]
    assert [atom.to_dict() for atom in rule.consequent] == [
        {"col": "z", "op": "in", "val": [0.0, 10.0]},
        {"col": "w", "op": "in", "val": [2.5, 7.5]},
    ]


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        ("mode = A", "x in [0; 1]", "unsupported Left atom"),
        ("x >= 0", "x in [0; 1]", "unsupported Left atom"),
        ("x in [0; 1]", "", "empty Right side"),
        ("x in [0; 1] OR y in [0; 1]", "z in [0; 1]", "unsupported Left atom"),
    ],
)
def test_unsupported_official_rule_forms_are_explicit(
    tmp_path, left, right, message,
):
    path = tmp_path / "rules.csv"
    _write_rules(path, [_row(left, right)])

    with pytest.raises(QuantMinerRuleError, match=message):
        parse_quantminer_rules(path)


def test_quantminer_model_detects_with_parsed_official_rules(
    tmp_path, monkeypatch,
):
    def fake_quantminer(frame, out_dir, config):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        training_path = out_dir / "training.csv"
        rules_path = out_dir / "rules.csv"
        frame.to_csv(training_path, index=False)
        assert config == SAIN_QUANTMINER_CONFIG
        _write_rules(rules_path, [_row(
            "x in [0; 10] AND y in [0; 10]", "z in [0; 5]",
        )])
        provenance = {
            "implementation": "official_vendored_quantminer_java",
            "upstream_commit": QUANTMINER_CODE_COMMIT,
            "config": config.__dict__,
            "input_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
            "rules_sha256": hashlib.sha256(rules_path.read_bytes()).hexdigest(),
            "rule_count": 1,
        }
        (out_dir / "provenance.json").write_text(json.dumps(provenance) + "\n")
        return {"rules_path": rules_path, "provenance": provenance}

    monkeypatch.setattr("baselines.sain.run_quantminer", fake_quantminer)
    train = pd.DataFrame({
        "p1": [0] * 4,
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [1.0] * 4,
        "z": [1.0] * 4,
    })
    model = mine_local_invariants(
        topology="synthetic",
        components=[("PLC1", "S")],
        fb_to_col={("PLC1", "S"): "p1"},
        training_frames=[train],
        validation_frames=[train],
        feature_cols=["x", "y", "z"],
        min_observations=2,
        engine="quantminer",
        quantminer_dir=tmp_path / "official",
        fp_budget=0.01,
        seed=42,
    )

    assert model["engine"] == "quantminer"
    assert model["granularity"] == "local_per_case_state"
    assert model["quantminer"]["official_commit"] == QUANTMINER_CODE_COMMIT
    assert model["quantminer"]["config"] == {
        "min_support": 0.1,
        "min_confidence": 0.7,
        "population": 250,
        "generations": 100,
        "max_attributes": 2,
    }
    assert model["quantminer"]["rule_count"] == 3
    state = model["components"]["PLC1.S"]["local_states"]["0"]
    assert state["rule_count"] == 3
    assert state["official_rule_count"] == 1
    assert state["constant_invariant_count"] == 2
    assert Path(state["rules_path"]).is_file()
    training_csv = pd.read_csv(Path(state["rules_path"]).with_name("training.csv"))
    assert list(training_csv.columns) == ["x"]

    evaluation = pd.DataFrame({
        "p1": [0, 0, 0, 0, 9],
        "x": [1.0, 20.0, 1.0, 1.0, 1.0],
        "y": [1.0, 1.0, 20.0, 1.0, 1.0],
        "z": [6.0, 1.0, 1.0, 1.0, 1.0],
    })
    prediction = SainLocalDetector(model).predict(evaluation)
    # Violation, inactive antecedent, violated constant point invariant,
    # satisfied rule, and unknown local state, respectively.
    assert prediction.flags.tolist() == [1, 0, 1, 0, 1]
    assert prediction.scores.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0]


def test_state_local_constants_are_excluded_with_provenance(tmp_path, monkeypatch):
    seen_columns = []

    def fake_quantminer(frame, out_dir, config):
        seen_columns.append(list(frame.columns))
        out_dir.mkdir(parents=True, exist_ok=True)
        rules_path = out_dir / "rules.csv"
        _write_rules(rules_path, [])
        provenance = {
            "implementation": "official_vendored_quantminer_java",
            "upstream_commit": QUANTMINER_CODE_COMMIT,
            "config": config.__dict__,
            "input_sha256": hashlib.sha256(b"training").hexdigest(),
            "rules_sha256": hashlib.sha256(rules_path.read_bytes()).hexdigest(),
            "rule_count": 0,
        }
        (out_dir / "provenance.json").write_text(json.dumps(provenance) + "\n")
        return {"rules_path": rules_path, "provenance": provenance}

    monkeypatch.setattr("baselines.sain.run_quantminer", fake_quantminer)
    train = pd.DataFrame({
        "p1": [0] * 4,
        "variable": [1.0, 2.0, 3.0, 4.0],
        "static": [7.0] * 4,
    })
    model = mine_local_invariants(
        topology="synthetic",
        components=[("PLC1", "S")],
        fb_to_col={("PLC1", "S"): "p1"},
        training_frames=[train],
        validation_frames=[train],
        feature_cols=["variable", "static"],
        min_observations=2,
        engine="quantminer",
        quantminer_dir=tmp_path / "official",
    )
    state = model["components"]["PLC1.S"]["local_states"]["0"]
    assert seen_columns == [["variable"]]
    assert state["mining_feature_columns"] == ["variable"]
    assert state["removed_constant_columns"] == ["static"]
    assert state["constant_invariant_count"] == 1
    assert state["rule_count"] == 1
    rule = state["rules"][0]
    assert rule["antecedent"] == []
    assert rule["consequent"] == [{"col": "static", "op": "in", "val": [7.0, 7.0]}]


def test_quantminer_engine_requires_independent_artifact_directory():
    frame = pd.DataFrame({"p1": [0], "x": [1.0]})
    with pytest.raises(ValueError, match="quantminer_dir"):
        mine_local_invariants(
            topology="synthetic",
            components=[("PLC1", "S")],
            fb_to_col={("PLC1", "S"): "p1"},
            training_frames=[frame],
            validation_frames=[frame],
            feature_cols=["x"],
            min_observations=1,
            engine="quantminer",
        )
