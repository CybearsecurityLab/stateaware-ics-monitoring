"""Focused tests for the analysis-only IM-DISCO tractability audit."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from baselines.imdisco import (
    IMDISCO,
    IMDISCOConfig,
    _prune_subsumed,
    derive_predicates,
)
from scripts import analyze_imdisco_tractability as audit
from scripts import run_full_signal_imdisco as runner


def _decimal(count: dict) -> int:
    return int(count["decimal"])


def test_fixed_pattern_counts_complete_uncapped_family():
    # Four predicate items in every one of ten transactions.  With the mode
    # item there are 2^5-1 frequent itemsets and 2^4-1 nonempty antecedents.
    result = audit.count_uncapped_pattern_family([(0b1111, 10)], 10)
    assert result["status"] == "exact"
    assert result["power_terms"] == [{"coefficient": 1, "exponent": 4}]
    assert _decimal(result["predicate_itemsets_including_empty"]) == 16
    assert _decimal(result["retained_mode_consequent_rules"]) == 15
    assert _decimal(result["frequent_itemsets_including_optional_mode"]) == 31
    assert result["retained_mode_consequent_rules"]["formula"] == "2^4 - 1"


def test_alternative_state_and_steadytime_items_are_not_atomic_pairs():
    # Shared core bits 0,1; pattern A adds bits 2,3 and pattern B adds 4,5.
    # The union of subset lattices is 2^4 + 2^4 - 2^2, not a count obtained
    # by treating either State/SteadyTime pair as one indivisible item.
    result = audit.count_uncapped_pattern_family([(0b001111, 10), (0b110011, 10)], 10)
    assert result["power_terms"] == [
        {"coefficient": 2, "exponent": 4},
        {"coefficient": -1, "exponent": 2},
    ]
    assert _decimal(result["predicate_itemsets_including_empty"]) == 28
    assert _decimal(result["retained_mode_consequent_rules"]) == 27
    assert _decimal(result["frequent_itemsets_including_optional_mode"]) == 55


def test_infrequent_minority_patterns_do_not_truncate_frequent_family():
    # Anytown-like structure: the 178-row pattern contains every frequent
    # item.  Alternative items in the two minority patterns are infrequent.
    majority = 0b0011111
    minority_a = 0b0101101
    minority_b = 0b1001011
    result = audit.count_uncapped_pattern_family(
        [(majority, 178), (minority_a, 2), (minority_b, 4)], 10
    )
    assert result["power_terms"] == [{"coefficient": 1, "exponent": 5}]
    assert _decimal(result["retained_mode_consequent_rules"]) == 31
    assert _decimal(result["frequent_itemsets_including_optional_mode"]) == 63


def test_support_can_be_accumulated_across_otherwise_infrequent_patterns():
    # Neither pattern is frequent alone, but their two shared core items are.
    result = audit.count_uncapped_pattern_family([(0b001111, 6), (0b110011, 6)], 10)
    assert result["power_terms"] == [{"coefficient": 1, "exponent": 2}]
    assert _decimal(result["retained_mode_consequent_rules"]) == 3
    assert _decimal(result["frequent_itemsets_including_optional_mode"]) == 7


def test_wide_uncapped_count_uses_formulas_not_eager_itemsets():
    width = 40
    result = audit.count_uncapped_pattern_family([((1 << width) - 1, 1)], 1)
    assert _decimal(result["retained_mode_consequent_rules"]) == 2**width - 1
    assert _decimal(result["frequent_itemsets_including_optional_mode"]) == 2**(width + 1) - 1
    assert result["retained_mode_consequent_rules"]["approximate"]["decimal_exponent"] == 12


def test_closed_pattern_count_matches_brute_force_on_small_mixed_support_case():
    patterns = [(0b0111, 4), (0b1011, 4), (0b1101, 4)]
    min_support = 6
    frequent = []
    for itemset in range(1, 1 << 4):
        support = sum(count for bits, count in patterns if itemset & ~bits == 0)
        if support >= min_support:
            frequent.append(itemset)
    result = audit.count_uncapped_pattern_family(patterns, min_support)
    # The public formula also includes empty; brute force above is nonempty.
    assert _decimal(result["predicate_itemsets_including_empty"]) == len(frequent) + 1
    assert _decimal(result["retained_mode_consequent_rules"]) == len(frequent)
    assert _decimal(result["frequent_itemsets_including_optional_mode"]) == 2 * (len(frequent) + 1) - 1


def test_analysis_rejects_itemset_length_cap():
    frame = pd.DataFrame({"x": [1.0], "a": [1], "mode": ["m"]})
    with pytest.raises(ValueError, match="uncapped"):
        audit.analyze_training_frame(
            frame,
            IMDISCOConfig(mode_column="mode", max_itemset_len=2),
        )


def test_small_frame_parity_with_real_baseline_miner():
    pytest.importorskip("mlxtend")
    frame = pd.DataFrame({
        "velocity": [10.0, 12.0, 0.0, 1.0, 20.0, 22.0],
        "accel": [1.0, 2.0, 0.0, 0.5, 3.0, 4.0],
        "door": [0, 0, 1, 1, 0, 0],
        "mode": ["riding", "riding", "onstation", "onstation", "riding", "riding"],
    })
    config = IMDISCOConfig(
        mode_column="mode",
        sensor_columns=("velocity", "accel"),
        actuator_columns=("door",),
        prune_subsumed_rules=False,
    )
    report, _ = audit.analyze_training_frame(frame, config)
    model = IMDISCO(config).fit(frame)

    assert report["training_rows"] == 6
    assert report["min_support_count"] == math.ceil(0.05 * 6)
    assert report["existing_guard"]["would_raise_mining_too_large"] is False
    assert _decimal(report["uncapped_totals"]["frequent_itemsets"]) == model._mining["n_frequent_itemsets"]
    assert _decimal(report["uncapped_totals"]["final_maximal_rules"]) == len(
        _prune_subsumed(model.rules)
    )
    assert report["uncapped_totals"]["retained_mode_consequent_rules"]["status"] == "not_derived"
    assert report["item_semantics"]["state_items_shared_across_modes"] is True


def test_existing_guard_uses_predicate_powerset_estimate():
    frame = pd.DataFrame({
        "x": [float(i) for i in range(10)],
        "a": [1] * 10,
        "mode": ["m"] * 10,
    })
    config = IMDISCOConfig(mode_column="mode", max_frequent_itemsets=100)
    report, _ = audit.analyze_training_frame(frame, config)
    # MinMax + within-mode Gradient + shared State; final SteadyTime is ongoing.
    # The frame-start pattern is a subset of the four-item later-row pattern.
    assert _decimal(report["existing_guard"]["estimated_frequent_itemsets"]) == 2**4 - 1
    assert report["existing_guard"]["would_raise_mining_too_large"] is False
    assert report["feature_roles"]["sensor_count"] == 1
    assert report["feature_roles"]["actuator_count"] == 1
    assert report["unique_transaction_patterns"] == 2
    assert report["predicate_count"] == 3
    assert report["shared_state_predicate_count"] == 1


def test_observable_policy_excludes_every_hidden_channel_before_predicates():
    rows = 10
    frame = pd.DataFrame({
        "time_seconds": [300 * i for i in range(rows)],
        "timestamp": [1000 + i for i in range(rows)],
        "attack_active": [0] * rows,
        "attack_class": [0] * rows,
        "label": [0] * rows,
        "physical_impact": [0.0] * rows,
        "delivery_or_command_modified": [0] * rows,
        "observable_attack": [0] * rows,
        "leak_hidden": [0.0] * rows,
        "demand_hidden": [0.0] * rows,
        "emitter_hidden": [0.0] * rows,
        "observable_sensor": [float(i) for i in range(rows)],
        "s_observable": [1] * rows,
        "nonnumeric_observable": ["not-retained-by-hybrid-policy"] * rows,
        "mode": ["m"] * rows,
    })
    source = frame.drop(columns=["mode"])
    observable = runner.observable_frame(source)
    assert list(observable.columns) == ["observable_sensor", "s_observable"]
    assert audit.observable_frame is runner.observable_frame

    filtered = observable.copy()
    filtered["mode"] = frame["mode"]
    sensors, actuators = ("observable_sensor",), ("s_observable",)
    predicates = derive_predicates(
        filtered, filtered["mode"], sensors, actuators
    )
    assert {predicate.variable for predicate in predicates} == {
        "observable_sensor", "s_observable"
    }
    assert {predicate.kind for predicate in predicates} == {
        "minmax", "gradient", "state"
    }

    report, _ = audit.analyze_training_frame(
        frame,
        IMDISCOConfig(
            mode_column="mode", sensor_columns=sensors, actuator_columns=actuators
        ),
    )
    policy = report["observable_policy"]
    assert policy["excluded_columns"] == sorted({
        "time_seconds", "timestamp", "attack_active", "attack_class", "label",
        "physical_impact", "delivery_or_command_modified", "observable_attack",
    })
    assert policy["excluded_prefixes"] == ["leak_", "demand_", "emitter_"]
    assert policy["remaining_numeric_feature_count"] == 2
    assert policy["remaining_feature_counts_by_topology_role"] == {
        "sensors": 1, "actuators": 1, "total": 2
    }
    assert policy["feature_roles"] == {
        "sensors": ["observable_sensor"], "actuators": ["s_observable"]
    }
    assert policy["feature_trimming"] is False
    assert policy["episode_trimming"] is False
    assert set(policy["removed_columns"]) == set(source.columns) - set(observable.columns)
    assert "time_seconds" not in report["feature_roles"]["sensors"]
    assert report["feature_roles"]["time_seconds_is_sensor"] is False
    assert len(policy["observable_frame_sha256"]) == 64


def test_runner_roles_come_from_topology_not_binary_cardinality():
    train = pd.DataFrame({
        "s_mapped_float": [0.0, 1.0],
        "s_binary_but_sensor": [0, 1],
        "s_continuous": [1.5, 2.5],
        "mode": ["a", "b"],
    })
    sensors, actuators = runner.topology_feature_roles(
        train, {("PLC1", "P1_State"): "s_mapped_float"}
    )
    assert actuators == ("s_mapped_float",)
    assert sensors == ("s_binary_but_sensor", "s_continuous")
    assert runner.OUT_ROOT.name == "full_signal_imdisco_maximal"
    assert runner.UNCAPPED_OUT_ROOT.name == "full_signal_imdisco"
    assert runner.OUT_ROOT != runner.UNCAPPED_OUT_ROOT
