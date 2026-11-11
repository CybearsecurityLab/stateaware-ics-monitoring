#!/usr/bin/env python3
"""Analysis-only IM-DISCO tractability report; never invokes rule mining.

Run with ``uv run --no-sync python scripts/analyze_imdisco_tractability.py``.

The audit constructs only the small set of distinct training transaction
patterns, represented as integer bitsets.  It counts the complete uncapped
frequent-itemset family through frequent closed patterns and
inclusion--exclusion.  It does not call the baseline's expensive predicate
derivation, construct its one-hot DataFrame, import mlxtend, or run detection.
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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from baselines.imdisco import (  # noqa: E402
    IMDISCOConfig,
    _resolve_columns,
    _resolve_modes,
    _value_key,
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
)

OUT_PATH = REPO_ROOT / "scripts/out/full_signal_imdisco/tractability.json"
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
    """Structurally audit a training frame without deriving Predicate objects."""
    if config.max_itemset_len is not None:
        raise ValueError("this audit implements only the uncapped max_itemset_len=None case")
    if not config.inclusive_bounds:
        raise ValueError("this structural audit currently requires inclusive_bounds=True")
    if float(config.minconf) != 1.0:
        raise ValueError("retained-rule formulas require minconf=1.0")

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
    numeric = (
        frame.loc[:, list(sensors)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if sensors
        else np.empty((len(frame), 0), dtype=float)
    )
    gradient = np.empty_like(numeric)
    if len(frame):
        gradient[0] = 0.0
    if len(frame) > 1:
        gradient[1:] = np.diff(numeric, axis=0)
    actuator_keys = {
        column: tuple(_value_key(value) for value in frame[column].tolist())
        for column in actuators
    }

    indices_by_mode: dict[str, list[int]] = {}
    for index, key in enumerate(mode_keys):
        indices_by_mode.setdefault(key, []).append(index)
    min_support_count = int(math.ceil(float(config.minsup) * len(frame)))
    public_modes: list[dict[str, Any]] = []
    internal_modes: dict[str, Any] = {}
    aggregate_terms: Counter[int] = Counter()
    supported_modes = 0
    exact = True

    for mode_key in sorted(indices_by_mode):
        indices = indices_by_mode[mode_key]
        rows = len(indices)
        mode_numeric = numeric[indices]
        mode_gradient = gradient[indices]
        minmax_exists = ~np.isnan(mode_numeric).all(axis=0)
        gradient_exists = ~np.isnan(mode_gradient).all(axis=0)
        grouped: dict[Any, list[int]] = {}
        for index in indices:
            sensor_signature = (
                np.packbits(~np.isnan(numeric[index])).tobytes(),
                np.packbits(~np.isnan(gradient[index])).tobytes(),
            )
            actuator_signature = tuple(actuator_keys[column][index] for column in actuators)
            signature = (sensor_signature, actuator_signature)
            grouped.setdefault(signature, []).append(index)

        item_indices: dict[tuple[str, str, str | None], int] = {}
        kind_counts: Counter[str] = Counter()
        patterns: list[tuple[int, int]] = []
        pattern_reports: list[dict[str, Any]] = []
        for signature in sorted(grouped, key=repr):
            members = grouped[signature]
            representative = members[0]
            bitset = 0
            for column_index, column in enumerate(sensors):
                if minmax_exists[column_index] and not np.isnan(numeric[representative, column_index]):
                    key = ("minmax", column, None)
                    if key not in item_indices:
                        item_indices[key] = len(item_indices)
                        kind_counts["minmax"] += 1
                    bitset |= 1 << item_indices[key]
                if gradient_exists[column_index] and not np.isnan(gradient[representative, column_index]):
                    key = ("gradient", column, None)
                    if key not in item_indices:
                        item_indices[key] = len(item_indices)
                        kind_counts["gradient"] += 1
                    bitset |= 1 << item_indices[key]
            for column in actuators:
                state_key = actuator_keys[column][representative]
                if state_key is None:
                    continue
                for kind in ("state", "steadytime"):
                    key = (kind, column, state_key)
                    if key not in item_indices:
                        item_indices[key] = len(item_indices)
                        kind_counts[kind] += 1
                    bitset |= 1 << item_indices[key]
            patterns.append((bitset, len(members)))
            pattern_reports.append({
                "sha256": _pattern_digest(signature),
                "rows": len(members),
                "width_including_mode": bitset.bit_count() + 1,
            })

        predicate_count = len(item_indices)
        supported_for_mining = rows >= min_support_count
        supported_for_guard = rows / len(frame) >= float(config.minsup)
        mode_report: dict[str, Any] = {
            "mode_key": mode_key,
            "rows": rows,
            "support": rows / len(frame),
            "supported_for_mining": supported_for_mining,
            "supported_for_existing_guard": supported_for_guard,
            "predicate_count": predicate_count,
            "predicate_kinds": dict(sorted(kind_counts.items())),
            "unique_transaction_patterns": len(patterns),
            "fixed_transaction_pattern": len(patterns) == 1,
            "transaction_widths_including_mode": sorted({
                pattern["width_including_mode"] for pattern in pattern_reports
            }),
            "patterns": sorted(
                pattern_reports,
                key=lambda pattern: (-pattern["rows"], pattern["sha256"]),
            ),
            "existing_guard_estimated_frequent_itemsets": (
                _count_json((1 << (predicate_count + 1)) - 1)
                if supported_for_guard else _count_json(0)
            ),
        }
        internal_modes[mode_key] = {
            "patterns": patterns,
            "predicate_count": predicate_count,
            "supported_for_guard": supported_for_guard,
        }
        if supported_for_mining:
            supported_modes += 1
            family = count_uncapped_pattern_family(patterns, min_support_count)
            mode_report["uncapped_family"] = family
            if family["status"] == "exact":
                for term in family["power_terms"]:
                    aggregate_terms[term["exponent"]] += term["coefficient"]
            else:
                exact = False
        else:
            mode_report["uncapped_family"] = {
                "status": "exact",
                "reason": "no item from this mode meets minimum support",
                "retained_mode_consequent_rules": _count_json(0),
                "frequent_itemsets_including_optional_mode": _count_json(0),
            }
        public_modes.append(mode_report)

    guard_estimate = sum(
        (1 << (internal_modes[mode["mode_key"]]["predicate_count"] + 1)) - 1
        for mode in public_modes
        if internal_modes[mode["mode_key"]]["supported_for_guard"]
    )
    cap = config.max_frequent_itemsets
    guard = {
        "max_frequent_itemsets": cap,
        "estimated_frequent_itemsets": _count_json(guard_estimate),
        "would_raise_mining_too_large": cap is not None and guard_estimate > cap,
        "stage_in_existing_fit": (
            "after predicate derivation and transaction DataFrame construction; "
            "before mlxtend import and FP-Growth"
        ),
    }
    aggregate: dict[str, Any]
    if exact:
        aggregate_terms = Counter({
            exponent: coefficient
            for exponent, coefficient in aggregate_terms.items() if coefficient
        })
        rule_count = _evaluate_power_formula(aggregate_terms, subtract=supported_modes)
        itemset_count = _evaluate_power_formula(
            aggregate_terms, multiplier=2, subtract=supported_modes
        )
        aggregate = {
            "status": "exact",
            "method": (
                "frequent closed transaction patterns plus inclusion-exclusion; "
                "mode-consequent antecedents are nonempty frequent predicate subsets"
            ),
            "power_terms": [
                {"coefficient": aggregate_terms[exponent], "exponent": exponent}
                for exponent in sorted(aggregate_terms, reverse=True)
            ],
            "retained_mode_consequent_rules": {
                "formula": _power_formula(aggregate_terms, subtract=supported_modes),
                **_count_json(rule_count),
            },
            "frequent_itemsets": {
                "formula": _power_formula(
                    aggregate_terms, multiplier=2, subtract=supported_modes
                ),
                **_count_json(itemset_count),
            },
        }
    else:
        aggregate = {
            "status": "not_derived",
            "reason": "at least one supported mode exceeded an analysis-algorithm guard",
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
                "none; each mode/sensor contributes MinMax and first-Gradient "
                "bound items, not histogram bins"
            ),
        },
        "minsup": float(config.minsup),
        "minconf": float(config.minconf),
        "min_support_count": min_support_count,
        "mode_count": len(public_modes),
        "supported_mode_count": supported_modes,
        "unique_transaction_patterns": sum(
            mode["unique_transaction_patterns"] for mode in public_modes
        ),
        "modes": public_modes,
        "uncapped_totals": aggregate,
        "existing_guard": guard,
    }
    internal = {"modes": internal_modes}
    return public, internal


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
        "baseline_predicate_derivation_invoked": False,
        "baseline_transaction_dataframe_constructed": False,
    }
    mining = {
        "invoked": False,
        "fpgrowth_invoked": False,
        "association_rules_invoked": False,
        "seconds": 0.0,
        "note": "analysis-only; no uncapped itemsets or rules were materialized",
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
    config = IMDISCOConfig(mode_column="mode")
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
            "applied_before_role_inference_and_structural_counting": True,
            "retains_every_observable_numeric_feature_episode_and_mode": True,
        },
        "no_cap_assurance": {
            "max_itemset_len": None,
            "feature_or_episode_trimming_beyond_approved_observable_policy": False,
            "frequent_itemsets_materialized": False,
            "rule_cap_applied": False,
        },
        "counting_method": (
            "Distinct transaction patterns are integer bitsets. All closed patterns "
            "are intersections; maximal frequent closed patterns cover every frequent "
            "itemset. Inclusion-exclusion counts their subset-lattice union exactly. "
            "State and SteadyTime items remain independent in every subset."
        ),
        "paper_discrepancy_notes": [
            "Paper Section 4 specifies mlxtend FP-Growth with minsup=0.05 and association rules with minconf=1; it reports no itemset-length cap or subsumption pruning.",
            "Paper Section 5.4 reports 76 predicates but only four final invariant rules, one per mode. The supplied paper does not explain how the stated all-frequent-itemsets procedure reduces to four rules.",
            "The supplied primary paper provides no official code, training rides, rule files, mlxtend version, or original rule ordering; no official-artifact pruning or cap behavior is inferred.",
            "Paper Section 5.5 runtimes used two sensors, three actuators and 52 predicates; they are not comparable to the full-signal widths audited here.",
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
        formula = totals.get("frequent_itemsets", {}).get("formula", totals["status"])
        print(f"{topology}: analysis-only; frequent itemsets = {formula}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
