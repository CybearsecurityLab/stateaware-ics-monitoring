#!/usr/bin/env python3
"""Analysis-only corrected IM-DISCO tractability report; never calls mlxtend.

Run with ``uv run --no-sync python scripts/analyze_imdisco_tractability.py``.

The audit constructs the thesis-corrected transactions (shared State items,
entry-only TransitionGradient and completed-run SteadyTime), then counts the
complete uncapped frequent-itemset family through frequent closed patterns and
inclusion--exclusion.  It also derives exact final maximal rules without
materializing unpruned ARG output.  Historical mode-local exponents are not
reused.  Unpruned ARG counts are marked unavailable because shared-State
confidence is non-monotone under antecedent subsets.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from baselines.imdisco import (  # noqa: E402
    IMDISCOConfig,
    _completed_run_records,
    _gradient_frame,
    _maximal_rules_from_transactions,
    _mode_item,
    _resolve_columns,
    _resolve_modes,
    _row_bitsets,
    _transactions,
    derive_predicates,
)
from dataio.full_signal import (  # noqa: E402
    load_full_signal_run,
    portable_path,
    split_nominal_run,
)
from invariants.state_label import (  # noqa: E402
    label_frame,
    load_gfsm_components,
    resolve_fb_to_col_from_paths,
)  # noqa: E402
from scripts.run_full_signal_imdisco import (  # noqa: E402
    observable_frame,
    observable_policy,
    topology_feature_roles,
)

OUT_PATH = REPO_ROOT / "scripts/out/full_signal_imdisco_maximal/tractability.json"
TOPOLOGIES = ("anytown", "ctown", "ltown")
# These are audit-algorithm guards, not itemset-length/output caps.  If one is
# reached, exact counts are reported as unavailable rather than truncated.
MAX_CLOSED_PATTERNS = 4096
MAX_MAXIMAL_FREQUENT_PATTERNS = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _approximation(number: int) -> dict[str, Any]:
    if number < 0:
        raise ValueError("approximation expects a nonnegative integer")
    if number == 0:
        return {"decimal_mantissa": 0.0, "decimal_exponent": 0, "log2": None}
    log10_value = math.log10(number)
    exponent = int(math.floor(log10_value))
    return {
        "decimal_mantissa": 10.0 ** (log10_value - exponent),
        "decimal_exponent": exponent,
        "log2": math.log2(number),
    }


def _count_json(number: int) -> dict[str, Any]:
    return {"decimal": str(number), "approximate": _approximation(number)}


def _power_formula(terms: Mapping[int, int], *, multiplier: int = 1, subtract: int = 0) -> str:
    pieces: list[str] = []
    for exponent in sorted(terms, reverse=True):
        coefficient = terms[exponent] * multiplier
        if not coefficient:
            continue
        magnitude = abs(coefficient)
        product = f"{magnitude}*2^{exponent}" if magnitude != 1 else f"2^{exponent}"
        if not pieces:
            pieces.append(f"-{product}" if coefficient < 0 else product)
        else:
            pieces.append(f"{'+' if coefficient > 0 else '-'} {product}")
    if subtract or not pieces:
        if not pieces:
            pieces.append(str(-subtract))
        elif subtract > 0:
            pieces.append(f"- {subtract}")
        else:
            pieces.append(f"+ {-subtract}")
    return " ".join(pieces)


def _evaluate_power_formula(terms: Mapping[int, int], *, multiplier: int = 1, subtract: int = 0) -> int:
    return multiplier * sum(coefficient * (1 << exponent) for exponent, coefficient in terms.items()) - subtract


def count_uncapped_pattern_family(
    patterns: Sequence[tuple[int, int]],
    min_support_count: int,
) -> dict[str, Any]:
    """Count all frequent subsets without enumerating that exponential family.

    ``patterns`` contains ``(transaction_bitset, multiplicity)`` pairs for one
    mode.  Closures of frequent itemsets are intersections of transaction
    patterns.  Maximal frequent closed patterns cover the complete frequent
    family; inclusion--exclusion counts the union of their subset lattices.
    Empty is included in the returned predicate-family count.  No itemset
    length cap is accepted or applied.
    """
    if min_support_count < 1:
        raise ValueError("min_support_count must be positive")
    if not patterns:
        raise ValueError("at least one transaction pattern is required")
    if any(count < 1 for _, count in patterns):
        raise ValueError("pattern multiplicities must be positive")

    closed = {bitset for bitset, _ in patterns}
    for bitset, _ in patterns:
        closed |= {candidate & bitset for candidate in tuple(closed)}
        if len(closed) > MAX_CLOSED_PATTERNS:
            return {
                "status": "not_derived",
                "reason": f"more than {MAX_CLOSED_PATTERNS} closed transaction patterns",
            }

    def support(candidate: int) -> int:
        return sum(count for bitset, count in patterns if candidate & ~bitset == 0)

    frequent_closed = {candidate for candidate in closed if support(candidate) >= min_support_count}
    maximal = [
        candidate
        for candidate in frequent_closed
        if not any(
            candidate != other and candidate & ~other == 0
            for other in frequent_closed
        )
    ]
    if len(maximal) > MAX_MAXIMAL_FREQUENT_PATTERNS:
        return {
            "status": "not_derived",
            "reason": (
                f"more than {MAX_MAXIMAL_FREQUENT_PATTERNS} maximal frequent "
                "closed patterns"
            ),
        }

    maximal.sort(key=int)
    terms: Counter[int] = Counter()
    for size in range(1, len(maximal) + 1):
        coefficient = 1 if size % 2 else -1
        for combination in itertools.combinations(maximal, size):
            intersection = combination[0]
            for candidate in combination[1:]:
                intersection &= candidate
            terms[intersection.bit_count()] += coefficient
    terms = Counter({exponent: coefficient for exponent, coefficient in terms.items() if coefficient})
    predicate_count = _evaluate_power_formula(terms)
    if predicate_count < 1:
        raise AssertionError("a supported mode must at least contain the empty itemset")
    return {
        "status": "exact",
        "closed_pattern_count": len(closed),
        "maximal_frequent_closed_pattern_count": len(maximal),
        "power_terms": [
            {"coefficient": terms[exponent], "exponent": exponent}
            for exponent in sorted(terms, reverse=True)
        ],
        "predicate_itemsets_including_empty": {
            "formula": _power_formula(terms),
            **_count_json(predicate_count),
        },
        "retained_mode_consequent_rules": {
            "formula": _power_formula(terms, subtract=1),
            **_count_json(predicate_count - 1),
        },
        "frequent_itemsets_including_optional_mode": {
            "formula": _power_formula(terms, multiplier=2, subtract=1),
            **_count_json(2 * predicate_count - 1),
        },
    }


def _pattern_digest(signature: Any) -> str:
    return hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()


def analyze_training_frame(
    frame: pd.DataFrame,
    config: IMDISCOConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Audit the corrected shared-State/entry-gradient transactions exactly.

    No FP-Growth itemsets or unpruned association rules are materialized.  The
    frequent-itemset and final maximal-rule counts are exact.  An uncapped ARG
    rule count is not inferred from subset lattices because shared State items
    make confidence non-monotone under antecedent subsets.
    """
    if config.max_itemset_len is not None:
        raise ValueError("this audit implements only the uncapped max_itemset_len=None case")
    if not config.inclusive_bounds:
        raise ValueError("this structural audit currently requires inclusive_bounds=True")
    if float(config.minconf) != 1.0:
        raise ValueError("the tractability report requires minconf=1.0")

    frame = frame.reset_index(drop=True)
    if config.mode_column is None or config.mode_column not in frame.columns:
        raise ValueError("the structural audit requires an explicit mode column")
    mode_labels = frame[config.mode_column].copy()
    source_features = frame.drop(columns=[config.mode_column])
    observable = observable_frame(source_features)
    frame = observable.copy()
    frame[config.mode_column] = mode_labels.to_numpy()
    mode_values, mode_keys = _resolve_modes(frame, config.mode_column, config.mode_column)
    sensors, actuators = _resolve_columns(frame, config)
    feature_policy = observable_policy(
        source_features, observable, sensors=sensors, actuators=actuators
    )
    predicates = derive_predicates(
        frame, mode_values, sensors, actuators,
        inclusive_bounds=config.inclusive_bounds,
    )
    gradient = _gradient_frame(frame, sensors)
    runs = {
        column: _completed_run_records(frame[column].tolist(), mode_keys)[0]
        for column in actuators
    }
    transactions, columns = _transactions(
        frame, mode_values, mode_keys, predicates, gradient, runs
    )
    predicate_by_item = {predicate.item: predicate for predicate in predicates}
    mode_by_item = {_mode_item(key): value for value, key in zip(mode_values, mode_keys)}
    mode_key_by_item = {_mode_item(key): key for key in mode_keys}
    rules, mining = _maximal_rules_from_transactions(
        transactions,
        predicate_by_item,
        mode_by_item,
        mode_key_by_item=mode_key_by_item,
        minsup=config.minsup,
        minconf=config.minconf,
    )
    frequent_count = mining["n_frequent_itemsets"]
    min_support_count = mining["min_support_count"]
    mode_counts = Counter(mode_keys)
    public_modes = [
        {
            "mode_key": key,
            "rows": count,
            "support": count / len(frame),
            "supported_for_mining": count >= min_support_count,
            "mode_local_predicate_count": sum(
                predicate.mode_key == key for predicate in predicates
            ),
        }
        for key, count in sorted(mode_counts.items())
    ]
    cap = config.max_frequent_itemsets
    guard = {
        "max_frequent_itemsets": cap,
        "estimated_frequent_itemsets": _count_json(frequent_count),
        "count_kind": "exact corrected-transaction count, not the superseded mode-local powerset estimate",
        "would_raise_mining_too_large": cap is not None and frequent_count > cap,
        "stage_in_existing_fit": "before mlxtend import and FP-Growth in the uncapped arm",
    }
    aggregate = {
        "status": "exact_with_unpruned_arg_count_unavailable",
        "method": (
            "frequent closed transaction patterns and inclusion-exclusion for itemsets; "
            "maximal qualifying closed antecedents for final rules"
        ),
        "frequent_itemsets": _count_json(frequent_count),
        "retained_mode_consequent_rules": {
            "status": "not_derived",
            "reason": (
                "shared State items make confidence non-monotone; the old per-mode "
                "powerset formula is invalid and no unpruned ARG count is claimed"
            ),
        },
        "qualified_closed_rules_intermediate": _count_json(
            mining["n_qualified_closed_rules"]
        ),
        "final_maximal_rules": _count_json(len(rules)),
    }
    public = {
        "training_rows": len(frame),
        "observable_policy": feature_policy,
        "feature_roles": {
            "sensor_count": len(sensors),
            "actuator_count": len(actuators),
            "sensors": list(sensors),
            "actuators": list(actuators),
            "time_seconds_is_sensor": "time_seconds" in sensors,
            "continuous_discretization": (
                "none; MinMax, ordinary Gradient and entry-only TransitionGradient bounds"
            ),
        },
        "item_semantics": {
            "state_items_shared_across_modes": True,
            "mode_items_are_separate_consequents": True,
            "transitiongradient_entry_only": True,
            "steadytime_completed_run_total_on_every_run_row": True,
            "final_ongoing_and_mode_crossing_training_runs_excluded": True,
        },
        "minsup": float(config.minsup),
        "minconf": float(config.minconf),
        "min_support_count": min_support_count,
        "mode_count": len(public_modes),
        "supported_mode_count": sum(mode["supported_for_mining"] for mode in public_modes),
        "unique_transaction_patterns": len(set(_row_bitsets(transactions, columns))),
        "predicate_count": len(predicates),
        "shared_state_predicate_count": sum(
            predicate.kind == "state" and predicate.mode_key is None
            for predicate in predicates
        ),
        "modes": public_modes,
        "uncapped_totals": aggregate,
        "existing_guard": guard,
        "closed_pattern_mining": mining,
    }
    return public, {}


def _resolve_frame_column(dataset_column: str, frame_columns: Sequence[str]) -> str:
    """Match the runner's case-insensitive full-signal column adaptation."""
    stripped = dataset_column[2:] if dataset_column.lower().startswith("s_") else None
    attempts = [dataset_column, f"s_{dataset_column}"]
    if stripped:
        attempts.append(stripped)
    for attempt in attempts:
        matches = [column for column in frame_columns if column.lower() == attempt.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"ambiguous full-signal column for {dataset_column!r}")
    raise ValueError(f"cannot resolve {dataset_column!r} in the full-signal frame")


def _topology_inputs(topology: str, frame_columns: Sequence[str]) -> tuple[
    Path, Path, dict[tuple[str, str], str], dict[str, str], list[Path]
]:
    if topology == "ltown":
        plan_path = REPO_ROOT / "scripts/out/logic_attacks/ltown/plan.json"
        plan = json.loads(plan_path.read_text())
        nominal_path = portable_path(plan["nominal"]["path"])
        gfsm_dir = REPO_ROOT / "scripts/out/logic_attack_predictions/ltown/generated/ltown/gfsm"
        extra_inputs = [plan_path]
    else:
        nominal_path = REPO_ROOT / f"scripts/out/logic_attacks/{topology}/nominal"
        gfsm_dir = REPO_ROOT / f"data/generated/{topology}/gfsm"
        extra_inputs = []
    gfsm_path = gfsm_dir / f"{topology}.gfsm.json"
    components = load_gfsm_components(json.loads(gfsm_path.read_text()))
    if topology == "ltown":
        dataset_columns = {"PLC1.PUMP_1_State": "s_PUMP_1"}
        fb_to_col = {}
        mapping = {}
        for component in components:
            key = f"{component[0]}.{component[1]}"
            dataset_column = dataset_columns[key]
            column = _resolve_frame_column(dataset_column, frame_columns)
            fb_to_col[component] = column
            mapping[key] = f"{dataset_column}->{column}"
    else:
        fb_dataset, resolved_components, _ = resolve_fb_to_col_from_paths(
            gfsm_dir, topology, REPO_ROOT / "data"
        )
        if list(resolved_components) != list(components):
            raise ValueError(f"{topology}: GFSM components changed during column resolution")
        fb_to_col = {}
        mapping = {}
        for component, dataset_column in fb_dataset.items():
            column = _resolve_frame_column(dataset_column, frame_columns)
            fb_to_col[component] = column
            mapping[f"{component[0]}.{component[1]}"] = f"{dataset_column}->{column}"
        extra_inputs.append(REPO_ROOT / f"data/{topology}/dataset/dataset_manifest.yaml")
    return nominal_path, gfsm_path, fb_to_col, mapping, extra_inputs


def analyze_topology(topology: str, config: IMDISCOConfig) -> dict[str, Any]:
    started = time.perf_counter()
    # Load only the nominal bundle, not every attack episode.
    nominal_path = REPO_ROOT / f"scripts/out/logic_attacks/{topology}/nominal"
    if topology == "ltown":
        plan = json.loads((REPO_ROOT / "scripts/out/logic_attacks/ltown/plan.json").read_text())
        nominal_path = portable_path(plan["nominal"]["path"])
    nominal = load_full_signal_run(nominal_path, topology, name="nominal")
    split = split_nominal_run(nominal)
    load_seconds = time.perf_counter() - started

    nominal_path, gfsm_path, fb_to_col, mapping, extra_inputs = _topology_inputs(
        topology, split.train.frame.columns
    )
    train = split.train.frame.copy()
    components = load_gfsm_components(json.loads(gfsm_path.read_text()))
    train["mode"] = label_frame(train, components, fb_to_col)
    role_frame = observable_frame(train.drop(columns=["mode"]))
    role_frame["mode"] = train["mode"].to_numpy()
    sensors, actuators = topology_feature_roles(role_frame, fb_to_col)
    config = replace(
        config,
        sensor_columns=sensors,
        actuator_columns=actuators,
        prune_subsumed_rules=False,
    )
    audit_started = time.perf_counter()
    structural, _ = analyze_training_frame(train, config)
    audit_seconds = time.perf_counter() - audit_started

    metrics_path = REPO_ROOT / f"scripts/out/full_signal_imdisco/{topology}/metrics.json"
    historical = None
    if metrics_path.is_file():
        historical_metrics = json.loads(metrics_path.read_text())
        reason = historical_metrics.get("reason")
        guard_decimal = structural["existing_guard"]["estimated_frequent_itemsets"]["decimal"]
        historical = {
            "path": _relative(metrics_path),
            "sha256": sha256_file(metrics_path),
            "status": historical_metrics.get("status"),
            "reason": reason,
            "guard_estimate_matches_historical_reason": (
                guard_decimal in reason
                if isinstance(reason, str) and "MiningTooLargeError" in reason
                else None
            ),
            "status_and_reason_preserved": True,
        }
    input_hashes = dict(nominal.input_hashes)
    input_hashes["gfsm"] = sha256_file(gfsm_path)
    for path in extra_inputs:
        input_hashes[_relative(path)] = sha256_file(path)
    preprocessing = {
        "nominal_load_and_split_seconds": load_seconds,
        "vectorized_structural_audit_seconds": audit_seconds,
        "baseline_predicate_derivation_invoked": True,
        "baseline_transaction_dataframe_constructed": True,
        "corrected_shared_state_and_transition_semantics": True,
    }
    mining = {
        "invoked": False,
        "fpgrowth_invoked": False,
        "association_rules_invoked": False,
        "seconds": 0.0,
        "note": "analysis-only; no uncapped itemsets or unpruned ARG rules were materialized; final maximal closed-pattern rules were derived",
    }
    if topology == "ltown":
        mining["historical_timeout_interpretation"] = {
            "scope": "historical whole-fit/whole-run artifact; stage was not instrumented",
            "mlxtend_fpgrowth_seconds": None,
            "attribute_600_seconds_to_mlxtend": False,
        }
    return {
        "topology": topology,
        "nominal": {
            "path": _relative(nominal_path),
            "run_id": nominal.run_id,
            "input_hashes": input_hashes,
            "split_manifest": split.manifest,
        },
        "gfsm": {"path": _relative(gfsm_path), "column_mapping": mapping},
        "preprocessing": preprocessing,
        "mining": mining,
        "historical_metrics": historical,
        "structural_audit": structural,
    }


def build_report(topologies: Iterable[str] = TOPOLOGIES) -> dict[str, Any]:
    config = IMDISCOConfig(mode_column="mode", prune_subsumed_rules=False)
    results = {}
    for topology in topologies:
        results[topology] = analyze_topology(topology, config)
    return {
        "schema": "full-signal-imdisco-tractability/v1",
        "analysis_only": True,
        "script": {
            "path": _relative(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "config": config.to_dict(),
        "observable_policy_source": {
            "path": "scripts/run_full_signal_imdisco.py",
            "sha256": sha256_file(REPO_ROOT / "scripts/run_full_signal_imdisco.py"),
            "applied_before_topology_role_assignment_and_structural_counting": True,
            "retains_every_observable_numeric_feature_episode_and_mode": True,
        },
        "no_cap_assurance": {
            "max_itemset_len": None,
            "feature_or_episode_trimming_beyond_approved_observable_policy": False,
            "frequent_itemsets_materialized": False,
            "rule_cap_applied": False,
        },
        "counting_method": (
            "Distinct corrected transaction patterns are integer bitsets. Closed patterns "
            "are intersections; maximal frequent closed patterns cover every frequent "
            "itemset, whose union is counted exactly by inclusion-exclusion. Final maximal "
            "rules use qualifying closed antecedents and strict subsumption. Shared State "
            "confidence is non-monotone, so no unpruned ARG count is inferred from lattices."
        ),
        "paper_discrepancy_notes": [
            "Paper Section 4 specifies mlxtend FP-Growth with minsup=0.05 and association rules with minconf=1; original uncapped FP-Growth remains potentially exponential.",
            "Paper Section 5.4 reports one maximal conjunction per mode but does not specify the reduction; thesis Section 4.4.2 and Listing A.3 specify redundancy removal after ARG.",
            "Thesis Table 4.3 shares actuator State items across modes; Table 4.2 motivates a separate TransitionGradient. This implementation makes TransitionGradient entry-only and records that adaptation.",
            "The thesis Appendix A provides partial source snippets, not complete code, training rides, rule files, mlxtend version, or original rule ordering; this is not an official-code reproduction.",
            "Historical exponents from mode-keyed State and running-counter SteadyTime transactions are superseded and are not final method claims.",
        ],
        "topologies": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topology", choices=TOPOLOGIES, action="append", dest="topologies",
        help="audit only this topology (repeatable; default: all)",
    )
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    report = build_report(args.topologies or TOPOLOGIES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for topology, result in report["topologies"].items():
        totals = result["structural_audit"]["uncapped_totals"]
        count = totals["frequent_itemsets"]["decimal"]
        maximal = totals["final_maximal_rules"]["decimal"]
        print(f"{topology}: analysis-only; frequent itemsets = {count}; final maximal rules = {maximal}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
