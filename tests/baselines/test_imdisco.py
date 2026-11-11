"""Tests for the IM-DISCO predicate/runtime baseline.

Uncapped mining tests use a tiny spy in place of mlxtend so that arm is
exercised even when mlxtend is not installed.  The exact maximal arm has no
mlxtend dependency; parity tests use the real package when available.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, balanced_accuracy_score, matthews_corrcoef

from baselines import imdisco


# ---------------------------------------------------------------------------
# Helpers


SENSORS = ("velocity", "accel")
ACTUATORS = ("door",)


def train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "velocity": [10.0, 12.0, 0.0, 1.0, 20.0, 22.0],
            "accel": [1.0, 2.0, 0.0, 0.5, 3.0, 4.0],
            "door": ["closed", "closed", "open", "open", "closed", "closed"],
            "mode": ["riding", "riding", "onstation", "onstation", "riding", "riding"],
        }
    )


def config(**overrides) -> imdisco.IMDISCOConfig:
    # Most legacy pipeline tests spy on the explicitly uncapped mlxtend arm.
    base = dict(
        sensor_columns=SENSORS,
        actuator_columns=ACTUATORS,
        prune_subsumed_rules=False,
    )
    base.update(overrides)
    return imdisco.IMDISCOConfig(**base)


class FakeMiner:
    """Spy/test double; never imported by the production module."""

    def __init__(self, frequent=None, rules=None):
        self.frequent = frequent
        self.rules = rules
        self.fpgrowth_kwargs = None
        self.association_rules_kwargs = None
        self.transaction_columns = None

    def fpgrowth(self, transactions, **kwargs):
        self.fpgrowth_kwargs = dict(kwargs)
        self.transaction_columns = list(transactions.columns)
        if self.frequent is not None:
            return self.frequent
        if transactions.shape[1]:
            item = transactions.columns[0]
            return pd.DataFrame({"support": [1.0], "itemsets": [frozenset({item})]})
        return pd.DataFrame(columns=["support", "itemsets"])

    def association_rules(self, itemsets, **kwargs):
        self.association_rules_kwargs = dict(kwargs)
        if self.rules is not None:
            return self.rules
        return pd.DataFrame(
            columns=["antecedents", "consequents", "support", "confidence", "lift"]
        )


def install_fake(monkeypatch, fake: FakeMiner | None = None) -> FakeMiner:
    fake = fake or FakeMiner()
    monkeypatch.setattr(
        imdisco,
        "_load_mlxtend",
        lambda: SimpleNamespace(
            fpgrowth=fake.fpgrowth, association_rules=fake.association_rules
        ),
    )
    return fake


def fit_model(monkeypatch, frame=None, mode_values=None, cfg=None, fake=None):
    frame = train_frame() if frame is None else frame
    mode_values = frame["mode"] if mode_values is None else mode_values
    fake = install_fake(monkeypatch, fake)
    model = imdisco.IMDISCO(cfg or config())
    model.fit(frame, mode_values)
    return model, fake


def one_predicate(predicates, kind, variable, mode):
    matches = [
        p
        for p in predicates
        if p.kind == kind and p.variable == variable and p.mode == mode
    ]
    assert len(matches) == 1, (kind, variable, mode, matches)
    return matches[0]


def inject_rules(model, mode_to_predicates):
    """Install crafted rules so runtime first-match semantics can be tested."""
    rules = []
    for mode, predicates in mode_to_predicates.items():
        rules.append(
            imdisco.InvariantRule(
                mode=mode,
                mode_key=imdisco._value_key(mode),
                antecedent=tuple(predicates),
                support=1.0,
                confidence=1.0,
                lift=1.0,
            )
        )
    order = {imdisco._value_key(mode): i for i, mode in enumerate(model.mode_order)}
    rules.sort(key=lambda r: (order[r.mode_key], -len(r.antecedent_items), r.signature))
    model._rules = tuple(rules)
    grouped = {}
    for rule in rules:
        grouped.setdefault(rule.mode_key, []).append(rule)
    model._rules_by_mode_key = {key: tuple(value) for key, value in grouped.items()}
    return rules


# ---------------------------------------------------------------------------
# Predicate generation


def test_sensor_minmax_bounds_are_per_mode_and_inclusive():
    frame = train_frame()
    predicates = imdisco.derive_predicates(
        frame, frame["mode"], SENSORS, ACTUATORS
    )
    riding = one_predicate(predicates, "minmax", "velocity", "riding")
    onstation = one_predicate(predicates, "minmax", "velocity", "onstation")

    assert (riding.lo, riding.hi) == (10.0, 22.0)
    assert (onstation.lo, onstation.hi) == (0.0, 1.0)
    assert riding.inclusive is True

    assert riding.evaluate(
        values={"velocity": 10.0}, gradients={"velocity": 9.0}, run_lengths={"door": 1.0}
    )
    assert riding.evaluate(
        values={"velocity": 22.0}, gradients={"velocity": 1.0}, run_lengths={"door": 1.0}
    )
    assert not riding.evaluate(
        values={"velocity": 22.1}, gradients={"velocity": 1.0}, run_lengths={"door": 1.0}
    )


def test_gradient_excludes_mode_entries_and_transition_is_entry_only():
    frame = train_frame()
    predicates = imdisco.derive_predicates(
        frame, frame["mode"], SENSORS, ACTUATORS
    )
    riding = one_predicate(predicates, "gradient", "velocity", "riding")

    # Velocity diffs are [0, 2, -12, 1, 19, 2].  Rows 1, 3 and 5
    # are within-mode; rows 2 and 4 are transitions and row 0 starts the frame.
    assert (riding.lo, riding.hi) == (2.0, 2.0)
    onstation = one_predicate(predicates, "gradient", "velocity", "onstation")
    assert (onstation.lo, onstation.hi) == (1.0, 1.0)
    transitions = [p for p in predicates if p.kind == "transitiongradient"]
    assert {(p.mode, p.previous_mode) for p in transitions} == {
        ("onstation", "riding"), ("riding", "onstation")
    }
    arriving = next(
        p for p in transitions if p.mode == "onstation" and p.variable == "velocity"
    )
    assert (arriving.lo, arriving.hi) == (-12.0, -12.0)

    gradient = imdisco._gradient_frame(frame, ("velocity", "accel"))
    runs = {"door": imdisco._run_lengths(frame["door"])}
    transactions, _ = imdisco._transactions(
        frame,
        frame["mode"].to_numpy(),
        [imdisco._value_key(v) for v in frame["mode"]],
        predicates,
        gradient,
        runs,
    )
    # The onstation TransitionGradient item occurs only at row 2, not row 3.
    assert transactions[arriving.item].tolist() == [False, False, True, False, False, False]


def test_steadytime_bounds_use_consecutive_runs_per_state():
    frame = pd.DataFrame(
        {
            "velocity": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "door": ["ON", "ON", "ON", "OFF", "OFF", "ON"],
            "mode": ["m1", "m1", "m1", "m2", "m2", "m2"],
        }
    )
    predicates = imdisco.derive_predicates(
        frame, frame["mode"], ("velocity",), ("door",)
    )
    # Completed totals are retrospective on every run row; final ON is ongoing.
    assert np.isnan(imdisco._run_lengths(frame["door"])[-1])
    assert imdisco._run_lengths(frame["door"])[:5].tolist() == [3, 3, 3, 2, 2]
    on_m1 = [
        p
        for p in predicates
        if p.kind == "steadytime" and p.mode == "m1" and p.state_key == "str:ON"
    ]
    off_m2 = [
        p
        for p in predicates
        if p.kind == "steadytime" and p.mode == "m2" and p.state_key == "str:OFF"
    ]
    on_m2 = [
        p
        for p in predicates
        if p.kind == "steadytime" and p.mode == "m2" and p.state_key == "str:ON"
    ]

    assert (on_m1[0].lo, on_m1[0].hi) == (3.0, 3.0)
    assert (off_m2[0].lo, off_m2[0].hi) == (2.0, 2.0)
    assert on_m2 == []

    assert on_m1[0].evaluate(
        values={"door": "ON"}, gradients={}, run_lengths={"door": 3.0}
    )
    assert not on_m1[0].evaluate(
        values={"door": "ON"}, gradients={}, run_lengths={"door": 2.0}
    )
    assert not on_m1[0].evaluate(
        values={"door": "OFF"}, gradients={}, run_lengths={"door": 2.0}
    )


def test_steadytime_excludes_mode_crossing_run_with_provenance():
    frame = pd.DataFrame({
        "door": ["ON", "ON", "ON", "OFF"],
        "mode": ["m1", "m2", "m2", "m2"],
    })
    predicates = imdisco.derive_predicates(frame, frame["mode"], (), ("door",))
    assert not [p for p in predicates if p.kind == "steadytime"]
    durations, records = imdisco._completed_run_records(
        frame["door"], [imdisco._value_key(v) for v in frame["mode"]]
    )
    assert np.isnan(durations).all()
    assert records[0]["exclusion_reason"] == "crosses_mode_boundary"
    assert records[1]["exclusion_reason"] == "final_ongoing_run"

    report = imdisco._steadytime_provenance(
        frame, ("door",), [imdisco._value_key(v) for v in frame["mode"]]
    )
    actuator = report["actuators"]["door"]
    assert actuator["excluded_mode_crossing_runs"] == 1
    assert actuator["excluded_final_ongoing_runs"] == 1
    assert report["causal"] is False


def test_strict_bounds_option_restores_printed_formulas():
    frame = train_frame()
    predicates = imdisco.derive_predicates(
        frame, frame["mode"], SENSORS, ACTUATORS, inclusive_bounds=False
    )
    riding = one_predicate(predicates, "minmax", "velocity", "riding")
    assert not riding.evaluate(
        values={"velocity": 10.0}, gradients={"velocity": 1.0}, run_lengths={"door": 1.0}
    )
    assert riding.evaluate(
        values={"velocity": 11.0}, gradients={"velocity": 1.0}, run_lengths={"door": 1.0}
    )


def test_bounds_artifact_is_sorted_and_json_safe():
    frame = train_frame()
    predicates = imdisco.derive_predicates(
        frame, frame["mode"], SENSORS, ACTUATORS
    )
    bounds = imdisco.bounds_artifact(predicates)
    assert bounds["minmax"]["riding"]["velocity"]["lo"] == 10.0
    assert bounds["actuator_states"]["*"]["door"] == ["str:closed", "str:open"]
    json.dumps(bounds)


# ---------------------------------------------------------------------------
# Mining configuration


def test_fp_growth_and_association_rule_thresholds_are_forwarded(monkeypatch):
    _, fake = fit_model(monkeypatch)
    assert fake.fpgrowth_kwargs["min_support"] == 0.05
    assert fake.fpgrowth_kwargs["use_colnames"] is True
    # Paper Section 4: all frequent itemsets, no maximum length.
    assert "max_len" not in fake.fpgrowth_kwargs
    assert fake.association_rules_kwargs["metric"] == "confidence"
    assert fake.association_rules_kwargs["min_threshold"] == 1.0


def test_custom_thresholds_and_itemset_cap_are_forwarded(monkeypatch):
    cfg = config(minsup=0.2, minconf=0.9, max_itemset_len=3)
    _, fake = fit_model(monkeypatch, cfg=cfg)
    assert fake.fpgrowth_kwargs["min_support"] == 0.2
    assert fake.fpgrowth_kwargs["max_len"] == 3
    assert fake.association_rules_kwargs["min_threshold"] == 0.9


def test_invalid_thresholds_are_rejected():
    with pytest.raises(ValueError):
        imdisco.IMDISCOConfig(minsup=0.0)
    with pytest.raises(ValueError):
        imdisco.IMDISCOConfig(minconf=1.1)
    with pytest.raises(ValueError):
        imdisco.IMDISCOConfig(max_itemset_len=0)


def test_default_has_no_itemset_length_cap():
    cfg = imdisco.IMDISCOConfig()
    assert cfg.max_itemset_len is None
    assert cfg.prune_subsumed_rules is True


def test_tractability_limit_fails_explicitly(monkeypatch):
    install_fake(monkeypatch)
    cfg = config(max_frequent_itemsets=1)
    model = imdisco.IMDISCO(cfg)
    with pytest.raises(imdisco.MiningTooLargeError, match="max_frequent_itemsets|cap"):
        model.fit(train_frame(), train_frame()["mode"])


# ---------------------------------------------------------------------------
# Runtime semantics


def test_first_matching_conflict_uses_explicit_mode_order(monkeypatch):
    # Both modes have identical observed sensor ranges, so both rules match.
    frame = pd.DataFrame(
        {
            "velocity": [0.0, 1.0, 0.0, 1.0],
            "door": ["closed", "closed", "closed", "closed"],
            "mode": ["alpha", "alpha", "beta", "beta"],
        }
    )
    model, _ = fit_model(
        monkeypatch,
        frame=frame,
        cfg=config(sensor_columns=("velocity",), mode_order=("beta", "alpha")),
    )
    predicates = imdisco.derive_predicates(frame, frame["mode"], ("velocity",), ("door",))
    inject_rules(
        model,
        {
            "alpha": [one_predicate(predicates, "minmax", "velocity", "alpha")],
            "beta": [one_predicate(predicates, "minmax", "velocity", "beta")],
        },
    )

    result = model.predict(pd.DataFrame({"velocity": [0.5], "door": ["closed"]}))
    assert result.predicted_modes == ("beta",)
    assert result.anomaly_flags.tolist() == [0]


def test_default_mode_order_is_lexicographic(monkeypatch):
    frame = pd.DataFrame(
        {
            "velocity": [0.0, 1.0, 0.0, 1.0],
            "door": ["closed"] * 4,
            "mode": ["zeta", "zeta", "alpha", "alpha"],
        }
    )
    model, _ = fit_model(
        monkeypatch, frame=frame, cfg=config(sensor_columns=("velocity",))
    )
    predicates = imdisco.derive_predicates(frame, frame["mode"], ("velocity",), ("door",))
    inject_rules(
        model,
        {
            "zeta": [one_predicate(predicates, "minmax", "velocity", "zeta")],
            "alpha": [one_predicate(predicates, "minmax", "velocity", "alpha")],
        },
    )
    result = model.predict(pd.DataFrame({"velocity": [0.5], "door": ["closed"]}))
    assert model.mode_order == ("alpha", "zeta")
    assert result.predicted_modes == ("alpha",)


def test_unmatched_and_out_of_training_rows_are_anomalous(monkeypatch):
    model, _ = fit_model(monkeypatch)
    predicates = imdisco.derive_predicates(
        train_frame(), train_frame()["mode"], SENSORS, ACTUATORS
    )
    inject_rules(
        model,
        {
            "riding": [
                one_predicate(predicates, "minmax", "velocity", "riding"),
                next(
                    p for p in predicates
                    if p.kind == "state" and p.variable == "door"
                    and p.state_key == "str:closed"
                ),
            ]
        },
    )

    in_training = model.predict(pd.DataFrame({"velocity": [11.0], "accel": [1.5], "door": ["closed"]}))
    out_of_training = model.predict(pd.DataFrame({"velocity": [999.0], "accel": [1.5], "door": ["closed"]}))
    unknown_state = model.predict(pd.DataFrame({"velocity": [11.0], "accel": [1.5], "door": ["ajar"]}))

    assert in_training.predicted_modes == ("riding",)
    assert in_training.anomaly_flags.tolist() == [0]
    assert out_of_training.predicted_modes == (None,)
    assert out_of_training.anomaly_flags.tolist() == [1]
    assert out_of_training.anomaly_scores.tolist() == [1.0]
    assert unknown_state.anomaly_flags.tolist() == [1]


def test_unknown_modes_can_never_be_predicted(monkeypatch):
    model, _ = fit_model(monkeypatch)
    assert model.first_matching_mode(["never-seen"]) is None
    assert model.first_matching_mode(["never-seen", "riding"]) == "riding"


# ---------------------------------------------------------------------------
# Determinism / no test refit


def test_predict_does_not_refit_or_mutate_bounds(monkeypatch):
    model, _ = fit_model(monkeypatch)
    before = model.artifacts_json()
    predicates_before = tuple(p.item for p in model.predicates)
    counts_before = dict(model.artifacts()["training"]["mode_counts"])

    # The mode column is irrelevant at predict time; including it must not leak.
    test = pd.DataFrame(
        {
            "velocity": [11.0, 999.0],
            "accel": [1.5, 2.0],
            "door": ["closed", "closed"],
            "mode": ["onstation", "onstation"],
        }
    )
    model.predict(test)

    assert model.artifacts_json() == before
    assert tuple(p.item for p in model.predicates) == predicates_before
    assert model.artifacts()["training"]["mode_counts"] == counts_before


def test_artifacts_and_predictions_are_deterministic(monkeypatch):
    model_a, _ = fit_model(monkeypatch)
    model_b, _ = fit_model(monkeypatch)
    assert model_a.artifacts_json() == model_b.artifacts_json()

    test = pd.DataFrame({"velocity": [11.0, 0.5], "accel": [1.5, 0.2], "door": ["closed", "open"]})
    assert model_a.predict(test).to_dict() == model_b.predict(test).to_dict()


def test_artifacts_record_provenance_and_missing_originals(monkeypatch):
    model, _ = fit_model(monkeypatch)
    artifacts = model.artifacts()
    parsed = json.loads(model.artifacts_json())
    assert parsed == artifacts

    provenance = artifacts["provenance"]
    assert provenance["original_rule_ordering_available"] is False
    assert provenance["missing_original_rule_ordering"] is True
    assert provenance["original_artifacts_available"] is False
    assert provenance["artifact_availability"]["paper_rule_files"] is False
    assert "reproduction_gaps" in provenance and provenance["reproduction_gaps"]
    assert isinstance(provenance["mlxtend_available"], bool)
    assert artifacts["mining"]["minsup"] == 0.05
    assert artifacts["mining"]["minconf"] == 1.0


# ---------------------------------------------------------------------------
# Expert labels and miner availability


def test_attack_label_is_not_used_as_expert_mode(monkeypatch):
    install_fake(monkeypatch)
    frame = train_frame()
    frame["label"] = [0, 0, 1, 1, 0, 0]
    model = imdisco.IMDISCO(config())
    with pytest.raises(ValueError, match="expert mode labels"):
        model.fit(frame)


def test_explicit_mode_column_is_accepted(monkeypatch):
    install_fake(monkeypatch)
    frame = train_frame()
    model = imdisco.IMDISCO(config(mode_column="mode"))
    model.fit(frame)
    assert model.mode_order == ("onstation", "riding")


def test_missing_mlxtend_is_reported_not_replaced(monkeypatch):
    def raise_import_error():
        raise ImportError("mlxtend is missing")

    monkeypatch.setattr(imdisco, "_import_frequent_patterns", raise_import_error)
    with pytest.raises(imdisco.MissingMinerError, match="mlxtend"):
        imdisco._load_mlxtend()


def test_explicit_feature_leakage_is_rejected():
    frame = train_frame()
    frame["label"] = [0, 0, 1, 1, 0, 0]
    with pytest.raises(ValueError, match="must not be used as features"):
        imdisco._resolve_columns(
            frame,
            imdisco.IMDISCOConfig(sensor_columns=("label",), actuator_columns=("door",)),
        )


# ---------------------------------------------------------------------------
# Metrics


def test_anomaly_metrics_schema_and_values():
    truth = np.array([0, 0, 1, 1])
    flags = np.array([0, 1, 1, 0])
    metrics = imdisco.anomaly_metrics(truth, flags)

    required = {
        "n",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1",
        "mcc",
        "balanced_accuracy",
        "unadjusted_balanced_accuracy",
        "average_precision",
        "average_precision_kind",
        "positive_rate",
        "predicted_positive_rate",
    }
    assert required <= set(metrics)
    assert (metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]) == (1, 1, 1, 1)
    assert metrics["mcc"] == pytest.approx(matthews_corrcoef(truth, flags))
    assert metrics["balanced_accuracy"] == pytest.approx(
        balanced_accuracy_score(truth, flags)
    )
    assert metrics["unadjusted_balanced_accuracy"] == metrics["balanced_accuracy"]
    assert metrics["average_precision"] == pytest.approx(
        average_precision_score(truth, flags)
    )
    assert metrics["average_precision_kind"] == "coarse_binary_rule_match"


def test_anomaly_metrics_single_class_returns_nan_ap():
    metrics = imdisco.anomaly_metrics([0, 0, 0], [0, 0, 1])
    assert np.isnan(metrics["average_precision"])
    assert metrics["n"] == 3


def test_anomaly_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        imdisco.anomaly_metrics([0, 1], [0])


# ---------------------------------------------------------------------------
# Real miner (skipped when mlxtend is unavailable, which is reported)


def test_real_mlxtend_end_to_end_if_available():
    pytest.importorskip("mlxtend")
    frame = train_frame()
    model = imdisco.IMDISCO(config())
    model.fit(frame, frame["mode"])
    result = model.predict(frame.drop(columns=["mode"]))
    assert len(result) == len(frame)
    assert model.rules
    assert set(result.predicted_modes) <= set(model.mode_order)


def test_rules_use_mode_consequent_and_subsumed_antecedents_are_pruned(monkeypatch):
    frame = train_frame()
    first, _ = fit_model(monkeypatch, frame=frame)

    by_mode = {}
    for predicate in first.predicates:
        if predicate.mode_key is not None:
            by_mode.setdefault(predicate.mode_key, []).append(predicate.item)

    rows = []
    for mode_key, items in by_mode.items():
        mode_item = imdisco._mode_item(mode_key)
        for item in items:
            rows.append(
                {
                    "antecedents": frozenset({item}),
                    "consequents": frozenset({mode_item}),
                    "support": 0.5,
                    "confidence": 1.0,
                    "lift": 1.0,
                }
            )
        rows.append(
            {
                "antecedents": frozenset(items),
                "consequents": frozenset({mode_item}),
                "support": 0.5,
                "confidence": 1.0,
                "lift": 1.0,
            }
        )
        # A rule whose consequent is a predicate, not a mode, must be dropped.
        if len(items) >= 2:
            rows.append(
                {
                    "antecedents": frozenset({items[0]}),
                    "consequents": frozenset({items[1]}),
                    "support": 0.5,
                    "confidence": 1.0,
                    "lift": 1.0,
                }
            )

    fake = FakeMiner(rules=pd.DataFrame(rows))
    install_fake(monkeypatch, fake)
    model = imdisco.IMDISCO(config()).fit(frame, frame["mode"])
    pruned = imdisco._prune_subsumed(model.rules)

    # One synthetic maximal antecedent per mode remains after post-ARG pruning.
    assert len(pruned) == len(by_mode)
    assert {rule.mode for rule in pruned} == {"onstation", "riding"}
    assert all(rule.mode_key in by_mode for rule in pruned)


# ---------------------------------------------------------------------------
# Exact maximal adaptation and corrected item semantics


@pytest.mark.parametrize("minconf", [1.0, 0.5])
def test_maximal_closed_rules_parity_with_uncapped_mlxtend(minconf):
    pytest.importorskip("mlxtend")
    frame = pd.DataFrame({
        "x": [1.0, 2.0, 4.0, 5.0, 7.0, 8.0],
        "door": [1, 0, 1, 0, 1, 0],
        "mode": ["a", "a", "b", "b", "c", "c"],
    })
    common = dict(
        sensor_columns=("x",), actuator_columns=("door",),
        mode_column="mode", minconf=minconf, max_frequent_itemsets=None,
    )
    uncapped = imdisco.IMDISCO(
        imdisco.IMDISCOConfig(**common, prune_subsumed_rules=False)
    ).fit(frame)
    maximal = imdisco.IMDISCO(
        imdisco.IMDISCOConfig(**common, prune_subsumed_rules=True)
    ).fit(frame)
    expected = imdisco._prune_subsumed(uncapped.rules)

    assert {rule.rule_id for rule in maximal.rules} == {rule.rule_id for rule in expected}
    expected_by_id = {rule.rule_id: rule for rule in expected}
    for rule in maximal.rules:
        other = expected_by_id[rule.rule_id]
        assert rule.support == pytest.approx(other.support)
        assert rule.confidence == pytest.approx(other.confidence)
        assert rule.lift == pytest.approx(other.lift)
    assert maximal._mining["n_frequent_itemsets"] == uncapped._mining["n_frequent_itemsets"]
    assert maximal._mining["n_rules_before_pruning"] is None
    assert maximal._mining["n_qualified_closed_rules"] >= len(maximal.rules)


def test_shared_state_items_have_cross_mode_confidence():
    pytest.importorskip("mlxtend")
    frame = pd.DataFrame({
        "door": [1, 0, 1, 0],
        "mode": ["a", "a", "b", "b"],
    })
    cfg = imdisco.IMDISCOConfig(
        sensor_columns=(), actuator_columns=("door",), mode_column="mode",
        minconf=0.5, max_frequent_itemsets=None, prune_subsumed_rules=False,
    )
    model = imdisco.IMDISCO(cfg).fit(frame)
    state_predicates = [p for p in model.predicates if p.kind == "state"]
    assert len(state_predicates) == 2
    assert all(p.mode_key is None for p in state_predicates)
    on_item = next(p.item for p in state_predicates if p.state_key == "int:1")
    rule = next(
        rule for rule in model.rules
        if rule.mode == "a" and rule.antecedent_items == (on_item,)
    )
    # Shared ON occurs twice overall and once in mode a.
    assert rule.support == pytest.approx(0.25)
    assert rule.confidence == pytest.approx(0.5)
    assert rule.lift == pytest.approx(1.0)


def test_huge_dense_maximal_fit_never_invokes_fpgrowth(monkeypatch):
    def fail_if_loaded():
        raise AssertionError("maximal adaptation must not load or invoke mlxtend")

    monkeypatch.setattr(imdisco, "_load_mlxtend", fail_if_loaded)
    rows = 8
    width = 40
    frame = pd.DataFrame({
        **{f"s{index}": [float(index)] * rows for index in range(width)},
        "mode": ["m"] * rows,
    })
    model = imdisco.IMDISCO(imdisco.IMDISCOConfig(
        sensor_columns=tuple(f"s{index}" for index in range(width)),
        actuator_columns=(), mode_column="mode",
    )).fit(frame)
    assert len(model.rules) == 1
    assert model._mining["n_frequent_itemsets"] > 2_000_000
    assert model._mining["frequent_itemsets_materialized"] is False


def test_prediction_assigns_completed_total_to_all_run_rows(monkeypatch):
    model, _ = fit_model(monkeypatch)
    steady = imdisco.Predicate(
        kind="steadytime", variable="door", mode="riding",
        mode_key=imdisco._value_key("riding"), state_key="str:closed",
        lo=2.0, hi=2.0, n=1,
    )
    inject_rules(model, {"riding": [steady]})
    result = model.predict(pd.DataFrame({
        "velocity": [11.0, 11.0, 11.0],
        "accel": [1.5, 1.5, 1.5],
        "door": ["closed", "closed", "open"],
    }))
    assert result.predicted_modes == ("riding", "riding", None)
    assert result.anomaly_flags.tolist() == [0, 0, 1]
