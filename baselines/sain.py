"""SAIN-style *local* per-CASE-state invariant baseline.

This module is the separately named **local-vs-global adaptation**
(:class:`SainLocalDetector`), not the paper-faithful native SAIN path.  The
source-derived, per-global-state implementation is in
:mod:`baselines.sain_native`.  The local adaptation has separate NiaARM and
official QuantMiner mining engines; neither engine is native SAIN.

Upstream reference (official, cited here so the baseline is traceable)
---------------------------------------------------------------------
* Paper: "SAIN: Improving ICS Attack Detection Sensitivity via State-Aware
  Invariants", USENIX Security 2024 (Abbas et al.).
  https://www.usenix.org/conference/usenixsecurity24/presentation/abbas
* Official code/artifacts: https://github.com/purseclab/SAIN
  pinned commit ``0a55724c7a5a352635d235d5f1a2e741045c4bed``
  (2024-07-18, "Add files via upload").

What this module is — and is not
--------------------------------
SAIN derives *state-aware* invariants: the plant traces are partitioned into
state-specific sub-traces and the invariant bounds are quantified per state,
so the bounds are tighter than state-agnostic ones.  The official
implementation does that for the *global* ICS state of its two testbeds and
enforces the invariants inside a PLC runtime monitor.

This module implements a **SAIN-style local** variant of that idea on top of
the repository's existing machinery:

* the partition is **local per CASE state**: each GFSM component (one PLC
  CASE selector) gets its own state partition and its own rule set, mined
  separately, instead of the cross-product composite state used by the
  ``invariants`` stage;
* the legacy NiaARM engine uses :func:`invariants.mine.mine_state`; the
  QuantMiner engine uses the official vendored Java implementation through
  :func:`baselines.quantminer.run_quantminer` and evaluates the rules parsed
  from its actual ``rules.csv`` output;
* both engines use confidence >= 0.7 and support >= 0.1, and the violation
  threshold is selected per component on a **validation** split, so no
  threshold is calibrated on the rows its rules were fitted on.

This is **not** a byte-exact reproduction of the official SAIN artifact and
must not be described as one.  The adaptations are:

1. **Passive per-row monitor.**  SAIN's SMONITOR is an active PLC-runtime
   agent that intercepts external variable-write requests and *denies* them
   when an invariant would be violated.  Here the monitor is passive: it
   scores recorded trace rows after the fact.  Nothing is blocked, and no
   request-level causality is modelled.
2. **Association-rule mining adaptation.**  The NiaARM engine retains the
   repository's earlier NiaARM + Niapy PSO adaptation.  The QuantMiner engine
   instead runs the official QuantMiner Java code at the commit recorded in
   the model, with an explicitly reported experimental configuration.  That
   configuration is not claimed to be SAIN's original configuration, and the
   resulting association rules are not the paper's source-derived invariant
   templates.
3. **Surrogate PLC ST.**  The official artifact ships the real Structured
   Text of its Fischertechnik and Tennessee Eastman testbeds.  Here the PLC
   programs are the surrogate/generated ST of this repository, so the state
   partition comes from the GFSM CASE selectors rather than from SAIN's own
   PDG/FSM extraction of a real program.
4. **Local granularity.**  SAIN quantifies invariants per global ICS state.
   This baseline quantifies them per local CASE state of each component, so
   that the local and the global partition can be compared on identical data.

Runtime semantics
-----------------
Per component the detector keeps

* the rules mined for each *local* state of that component, and
* a per-component violation threshold selected on the validation split.

At prediction time a row is flagged when **any** component flags it (the
OR of the local detectors).  A row whose local state was never mined for a
component is flagged as an *unknown local state*.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd

from baselines.quantminer import QuantMinerConfig, run_quantminer
from invariants.mine import MinerConfig, mine_state
from invariants.model import Atom, InvariantsError, MinedRule
from invariants.rule_eval import frame_violation_counts
from invariants.threshold import select_violation_threshold

from monitor.protocol import StepFlags

__all__ = [
    "QUANTMINER_CODE_COMMIT",
    "SAIN_CODE_COMMIT",
    "SAIN_CODE_URL",
    "SAIN_PAPER_URL",
    "SAIN_ADAPTATIONS",
    "SAIN_QUANTMINER_ADAPTATIONS",
    "SAIN_QUANTMINER_CONFIG",
    "SAIN_SCHEMA",
    "ComparisonScenario",
    "QuantMinerRuleError",
    "SainLocalDetector",
    "compare_local_vs_global",
    "local_state_ids",
    "mine_local_invariants",
    "parse_quantminer_rules",
    "run_sain_local_comparison",
]

#: Official upstream code/artifacts for the SAIN paper.
SAIN_CODE_URL = "https://github.com/purseclab/SAIN"
SAIN_CODE_COMMIT = "0a55724c7a5a352635d235d5f1a2e741045c4bed"
SAIN_PAPER_URL = (
    "https://www.usenix.org/conference/usenixsecurity24/presentation/abbas"
)

#: The adaptations listed in the module docstring, kept machine-readable so
#: the artifact and the docs cannot drift apart.
SAIN_ADAPTATIONS: tuple[str, ...] = (
    "this class is the local per-CASE-state adaptation, not native SAIN; "
    "the source-derived per-global-state path is baselines.sain_native",
    "passive per-row monitor (upstream SMONITOR is an active PLC-runtime "
    "write-request interceptor; here violations are only scored on recorded "
    "rows, nothing is denied)",
    "NiaARM association rules replace the upstream Quantminer invariant-"
    "template quantification (Quantminer configuration not reproducible "
    "here); confidence>=0.7 / support>=0.1 filters are kept",
    "surrogate/generated PLC Structured Text replaces the upstream testbed "
    "programs; the state partition comes from the GFSM CASE selectors, not "
    "from SAIN's PDG-based FSM extraction",
    "local per-CASE-state granularity: each component is mined and "
    "calibrated separately instead of per global ICS state",
)

#: Official QuantMiner commit vendored by :mod:`baselines.quantminer`.
QUANTMINER_CODE_COMMIT = "587eea87c3603a2e0ef0e4b032422ffdedf5b645"
SAIN_QUANTMINER_CONFIG = QuantMinerConfig(
    min_support=0.1,
    min_confidence=0.7,
    population=250,
    generations=100,
    max_attributes=2,
)

#: Adaptations for the official-QuantMiner local engine.  In particular, the
#: numeric settings used here are experiment settings, not a claim to recover
#: an unavailable original SAIN configuration.
SAIN_QUANTMINER_ADAPTATIONS: tuple[str, ...] = (
    "this class is the local per-CASE-state adaptation, not native SAIN; "
    "the source-derived per-global-state path is baselines.sain_native",
    "passive per-row monitor (upstream SMONITOR is an active PLC-runtime "
    "write-request interceptor; here violations are only scored on recorded "
    "rows, nothing is denied and SMONITOR is not reproduced)",
    "official vendored QuantMiner Java association-rule mining with the "
    "reported experimental configuration; this is not claimed to be SAIN's "
    "original QuantMiner configuration or source-template quantification",
    "two-attribute local rule cap approved for tractability; the failed "
    "three-attribute full-width run is preserved separately and is not "
    "presented as the same mining configuration",
    "exact state-local constant channels are excluded only from QuantMiner "
    "input and retained as separately recorded point-interval invariants; "
    "every nonconstant observable feature is retained and all constants are named",
    "surrogate/generated PLC Structured Text replaces the upstream testbed "
    "programs; the state partition comes from the GFSM CASE selectors, not "
    "from SAIN's PDG-based FSM extraction",
    "local per-CASE-state granularity: each component is mined and "
    "calibrated separately instead of per global ICS state",
)

SAIN_SCHEMA = "baselines/sain_local/v1"


class QuantMinerRuleError(ValueError):
    """An official QuantMiner CSV row cannot be represented losslessly."""


def local_state_ids(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Per-row local state id of one component, as strings.

    The id is the integer value of the component's CASE-selector column,
    exactly the id the GFSM composite label uses for that component
    (``<plc>.<case_var>:<sid>``), so local and global state ids are
    directly comparable.
    """
    if column not in frame.columns:
        raise InvariantsError(
            f"local state column {column!r} missing from frame "
            f"(columns={list(frame.columns)[:8]})"
        )
    values = frame[column].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise InvariantsError(
            f"non-finite value in local state column {column!r}; cannot "
            "derive the local state"
        )
    return values.astype(int).astype(str)


_QUANTMINER_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
_QUANTMINER_INTERVAL = re.compile(
    rf"^(?P<column>.+?)\s+in\s+\[\s*(?P<lo>{_QUANTMINER_NUMBER})\s*;"
    rf"\s*(?P<hi>{_QUANTMINER_NUMBER})\s*\]$"
)
_QUANTMINER_PERCENT = re.compile(
    rf"^(?P<value>{_QUANTMINER_NUMBER})\s*%$"
)
_QUANTMINER_REQUIRED_COLUMNS = {
    "Left", "Right", "Support[L and R]", "Confidence[L->R]",
}


def _quantminer_percentage(value: Any, *, column: str, row_number: int) -> float:
    text = "" if value is None else str(value).strip()
    match = _QUANTMINER_PERCENT.fullmatch(text)
    if match is None:
        raise QuantMinerRuleError(
            f"QuantMiner rules.csv row {row_number}: {column} must be a "
            f"numeric percentage, got {value!r}"
        )
    return float(match.group("value")) / 100.0


def _quantminer_atoms(text: Any, *, side: str, row_number: int) -> list[Atom]:
    if text is None or not str(text).strip():
        raise QuantMinerRuleError(
            f"QuantMiner rules.csv row {row_number}: empty {side} side"
        )
    atoms = []
    # QuantMiner writes two spaces around AND.  Whitespace-flexible splitting
    # accepts that exact output as well as single-spaced CSV fixtures.
    for raw in re.split(r"\s+AND\s+", str(text).strip()):
        value = raw.strip()
        match = _QUANTMINER_INTERVAL.fullmatch(value)
        if re.search(r"\s+(?:OR|NOT)\s+", value):
            match = None
        if match is None:
            raise QuantMinerRuleError(
                f"QuantMiner rules.csv row {row_number}: unsupported {side} "
                f"atom {value!r}; only numeric '<column> in [<lo>; <hi>]' "
                "atoms and compound ' AND ' are supported"
            )
        column = match.group("column").strip()
        if not column:
            raise QuantMinerRuleError(
                f"QuantMiner rules.csv row {row_number}: empty column in "
                f"{side} atom {value!r}"
            )
        atoms.append(Atom(
            col=column,
            op="in",
            val=[float(match.group("lo")), float(match.group("hi"))],
        ))
    return atoms


def parse_quantminer_rules(path: Path | str) -> list[MinedRule]:
    """Parse official QuantMiner ``rules.csv`` into the invariant rule schema.

    ``Left`` is the antecedent and ``Right`` the consequent.  QuantMiner's
    inclusive numeric intervals map directly to inclusive ``in`` atoms.  The
    local arm has numeric-only inputs, so categorical atoms and every other
    operator are rejected instead of being silently discarded or approximated.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(_QUANTMINER_REQUIRED_COLUMNS.difference(fields))
        if missing:
            raise QuantMinerRuleError(
                f"QuantMiner rules.csv is missing required columns: {missing}"
            )
        rules = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise QuantMinerRuleError(
                    f"QuantMiner rules.csv row {row_number}: too many CSV fields"
                )
            rules.append(MinedRule(
                id=f"quantminer-{row_number - 1:06d}",
                antecedent=_quantminer_atoms(
                    row.get("Left"), side="Left", row_number=row_number,
                ),
                consequent=_quantminer_atoms(
                    row.get("Right"), side="Right", row_number=row_number,
                ),
                support=_quantminer_percentage(
                    row.get("Support[L and R]"),
                    column="Support[L and R]", row_number=row_number,
                ),
                confidence=_quantminer_percentage(
                    row.get("Confidence[L->R]"),
                    column="Confidence[L->R]", row_number=row_number,
                ),
            ))
    return rules


@dataclass
class _ComponentModel:
    """Per-component local invariant model."""

    plc: str
    case_var: str
    column: str
    states: dict[str, dict[str, Any]] = field(default_factory=dict)
    threshold: int = 1
    threshold_note: str | None = None

    @property
    def key(self) -> str:
        return f"{self.plc}.{self.case_var}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "plc": self.plc,
            "case_var": self.case_var,
            "column": self.column,
            "violation_threshold": int(self.threshold),
            "violation_threshold_note": self.threshold_note,
            "local_states": self.states,
        }


def _mine_component(
    *,
    plc: str,
    case_var: str,
    column: str,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: MinerConfig,
    min_observations: int,
    keep_going: bool,
) -> tuple[_ComponentModel, bool]:
    """Mine the rules of one component, per local state."""
    model = _ComponentModel(plc=plc, case_var=case_var, column=column)
    ids = local_state_ids(train_df, column)
    all_ok = True
    for state_id in sorted(set(ids.tolist())):
        slice_df = train_df.loc[ids == state_id]
        nobs = int(len(slice_df))
        if nobs < min_observations:
            model.states[state_id] = {
                "observations": nobs,
                "status": "insufficient_data",
                "rules": [],
            }
            continue
        try:
            rules: list[MinedRule] = mine_state(slice_df, feature_cols, cfg)
            model.states[state_id] = {
                "observations": nobs,
                "status": "ok",
                "rules": [r.to_dict() for r in rules],
            }
        except Exception as exc:  # noqa: BLE001 - mining is best-effort
            all_ok = False
            model.states[state_id] = {
                "observations": nobs,
                "status": "mining_error",
                "errors": [repr(exc)],
                "rules": [],
            }
            if not keep_going:
                break
    return model, all_ok


def _quantminer_state(
    *,
    plc: str,
    case_var: str,
    column: str,
    state_id: str,
    state_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: QuantMinerConfig,
    min_observations: int,
    artifact_dir: Path,
) -> tuple[dict[str, Any], bool]:
    """Mine one independent component state without changing Java semantics."""
    component_key = f"{plc}.{case_var}"
    nobs = int(len(state_df))
    if nobs < min_observations:
        return {
            "observations": nobs,
            "status": "insufficient_data",
            "rule_count": 0,
            "rules": [],
        }, True
    mining_columns = [
        column for column in feature_cols
        if state_df[column].max() > state_df[column].min()
    ]
    removed_constant = [
        column for column in feature_cols if column not in mining_columns
    ]
    constant_rules = []
    for column in removed_constant:
        value = float(state_df[column].iloc[0])
        if not np.isfinite(value):
            raise InvariantsError(
                f"cannot build a point invariant for {component_key} state "
                f"{state_id} column {column}: missing or nonfinite value"
            )
        constant_rules.append(MinedRule(
            id=f"state-local-constant-{component_key}-{state_id}-{column}",
            antecedent=[],
            consequent=[Atom(col=column, op="in", val=[value, value])],
            support=1.0,
            confidence=1.0,
        ))
    safe_state = re.sub(r"[^A-Za-z0-9_.-]+", "_", state_id)
    state_digest = hashlib.sha256(state_id.encode("utf-8")).hexdigest()[:12]
    state_dir = Path(artifact_dir) / f"state_{safe_state}_{state_digest}"
    if not mining_columns:
        return {
            "observations": nobs,
            "status": "constant_only",
            "mining_feature_columns": [],
            "removed_constant_columns": removed_constant,
            "constant_invariant_count": len(constant_rules),
            "rule_count": len(constant_rules),
            "official_rule_count": 0,
            "rules": [rule.to_dict() for rule in constant_rules],
        }, True
    try:
        result = run_quantminer(
            state_df.loc[:, mining_columns], state_dir, cfg,
        )
        official_rules = parse_quantminer_rules(result["rules_path"])
        official_count = int(result["provenance"]["rule_count"])
        if official_count != len(official_rules):
            raise QuantMinerRuleError(
                f"QuantMiner rule count mismatch for {component_key} state "
                f"{state_id}: provenance={official_count}, parsed={len(official_rules)}"
            )
        rules = [*constant_rules, *official_rules]
        return {
            "observations": nobs,
            "status": "ok",
            "mining_feature_columns": mining_columns,
            "removed_constant_columns": removed_constant,
            "constant_invariant_count": len(constant_rules),
            "rule_count": len(rules),
            "official_rule_count": official_count,
            "rules_path": str(result["rules_path"]),
            "quantminer_provenance": result["provenance"],
            "rules": [rule.to_dict() for rule in rules],
        }, True
    except Exception as exc:  # noqa: BLE001 - mining is best-effort
        return {
            "observations": nobs,
            "status": "mining_error",
            "rule_count": 0,
            "errors": [repr(exc)],
            "artifact_dir": str(state_dir),
            "rules": [],
        }, False


def _quantminer_component(
    *,
    plc: str,
    case_var: str,
    column: str,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: QuantMinerConfig,
    min_observations: int,
    keep_going: bool,
    artifact_dir: Path,
) -> tuple[_ComponentModel, bool]:
    """Mine and parse actual official QuantMiner rules for one component."""
    model = _ComponentModel(plc=plc, case_var=case_var, column=column)
    ids = local_state_ids(train_df, column)
    all_ok = True
    for state_id in sorted(set(ids.tolist())):
        entry, state_ok = _quantminer_state(
            plc=plc,
            case_var=case_var,
            column=column,
            state_id=state_id,
            state_df=train_df.loc[ids == state_id],
            feature_cols=feature_cols,
            cfg=cfg,
            min_observations=min_observations,
            artifact_dir=artifact_dir,
        )
        model.states[state_id] = entry
        all_ok = all_ok and state_ok
        if not state_ok and not keep_going:
            break
    return model, all_ok


def _calibrate_component(
    model: _ComponentModel,
    val_df: pd.DataFrame,
    fp_budget: float,
) -> None:
    """Select the component's violation threshold on the validation rows.

    The threshold is chosen on validation rows only — never on the rows the
    rules were fitted on.  A validation row whose local state was never
    mined counts as *state-absent*, mirroring the runtime signal the monitor
    raises for an unknown state.
    """
    ids = local_state_ids(val_df, model.column)
    counts = np.zeros(len(val_df), dtype=int)
    known = np.zeros(len(val_df), dtype=bool)
    for state_id, entry in model.states.items():
        sel = ids == state_id
        if not sel.any():
            continue
        known |= sel
        rules = entry.get("rules") or []
        if not rules:
            continue
        counts[sel] = frame_violation_counts(val_df.loc[sel], rules)
    absent = ~known
    threshold, note = select_violation_threshold(counts, absent, fp_budget)
    model.threshold = int(threshold)
    model.threshold_note = note


def mine_local_invariants(
    *,
    topology: str,
    components: Sequence[tuple[str, str]],
    fb_to_col: dict[tuple[str, str], str],
    training_frames: Sequence[pd.DataFrame],
    validation_frames: Sequence[pd.DataFrame],
    feature_cols: Sequence[str],
    min_observations: int = 50,
    max_evals: int = 5000,
    seed: int = 42,
    fp_budget: float = 0.01,
    keep_going: bool = True,
    n_jobs: int = 1,
    engine: str = "niaarm",
    quantminer_dir: Path | str | None = None,
    max_attributes: int | None = None,
) -> dict[str, Any]:
    """Mine the local (per-CASE-state) SAIN-style invariant model.

    Rules are mined on ``training_frames`` only; the per-component violation
    thresholds are selected on ``validation_frames`` only.  The two frame
    lists must be disjoint in practice — nothing here can enforce that, but
    the caller is expected to pass the output of
    :func:`dataio.splits.split_calibration_frames`.

    ``feature_cols`` is used verbatim for every component, so the local
    model and the global ``invariants`` stage can be compared on identical
    feature columns.  ``engine='niaarm'`` preserves the legacy miner.
    ``engine='quantminer'`` runs official QuantMiner and requires a separate
    ``quantminer_dir`` for its raw per-component/per-state artifacts.
    """
    if not training_frames:
        raise InvariantsError("mine_local_invariants needs training frames")
    if not validation_frames:
        raise InvariantsError(
            "mine_local_invariants needs validation frames: the violation "
            "threshold must be selected on rows the rules were not fitted on"
        )
    if engine not in {"niaarm", "quantminer"}:
        raise ValueError(f"unsupported local SAIN mining engine: {engine!r}")
    if engine == "niaarm" and max_attributes is not None:
        raise ValueError("max_attributes applies only to the QuantMiner engine")
    if engine == "quantminer" and quantminer_dir is None:
        raise ValueError("quantminer_dir is required for the QuantMiner engine")
    train_df = pd.concat(list(training_frames), ignore_index=True)
    val_df = pd.concat(list(validation_frames), ignore_index=True)
    feature_cols = list(feature_cols)

    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1")
    if engine == "niaarm":
        cfg: MinerConfig | QuantMinerConfig = MinerConfig(
            max_evals=max_evals, seed=seed,
        )
        worker = _mine_component
    else:
        if max_attributes is not None and max_attributes < 1:
            raise ValueError("max_attributes must be >= 1")
        cfg = (
            SAIN_QUANTMINER_CONFIG if max_attributes is None
            else QuantMinerConfig(
                min_support=SAIN_QUANTMINER_CONFIG.min_support,
                min_confidence=SAIN_QUANTMINER_CONFIG.min_confidence,
                population=SAIN_QUANTMINER_CONFIG.population,
                generations=SAIN_QUANTMINER_CONFIG.generations,
                max_attributes=int(max_attributes),
            )
        )
        worker = _quantminer_component
    models: dict[str, _ComponentModel] = {}
    all_ok = True

    jobs = []
    for index, (plc, case_var) in enumerate(components):
        column = fb_to_col.get((plc, case_var))
        if column is None:
            raise InvariantsError(
                f"no dataset column resolved for component "
                f"(plc={plc}, case_var={case_var!r})"
            )
        job = dict(
            plc=plc, case_var=case_var, column=column,
            train_df=train_df, feature_cols=feature_cols, cfg=cfg,
            min_observations=min_observations, keep_going=keep_going,
        )
        if engine == "quantminer":
            key = f"{plc}.{case_var}"
            safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
            key_digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
            job["artifact_dir"] = (
                Path(quantminer_dir)
                / f"{index:03d}_{safe_key}_{key_digest}"
            )
        jobs.append(job)
    state_parallel = engine == "quantminer" and n_jobs > 1 and keep_going
    if state_parallel:
        for job in jobs:
            model = _ComponentModel(
                plc=job["plc"], case_var=job["case_var"], column=job["column"],
            )
            models[model.key] = model
        state_jobs = []
        for job in jobs:
            ids = local_state_ids(job["train_df"], job["column"])
            for state_id in sorted(set(ids.tolist())):
                state_jobs.append((
                    f"{job['plc']}.{job['case_var']}",
                    dict(
                        plc=job["plc"],
                        case_var=job["case_var"],
                        column=job["column"],
                        state_id=state_id,
                        state_df=job["train_df"].loc[ids == state_id],
                        feature_cols=job["feature_cols"],
                        cfg=job["cfg"],
                        min_observations=job["min_observations"],
                        artifact_dir=job["artifact_dir"],
                    ),
                ))
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [
                (key, state_job["state_id"], pool.submit(
                    _quantminer_state, **state_job,
                ))
                for key, state_job in state_jobs
            ]
            for key, state_id, future in futures:
                entry, state_ok = future.result()
                models[key].states[state_id] = entry
                all_ok = all_ok and state_ok
        results = []
    elif n_jobs == 1:
        results = [worker(**job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(worker, **job) for job in jobs]
            results = [future.result() for future in futures]
    for model, ok in results:
        models[model.key] = model
        all_ok = all_ok and ok

    for model in models.values():
        _calibrate_component(model, val_df, fp_budget)

    result = {
        "schema": SAIN_SCHEMA,
        "engine": engine,
        "topology": topology,
        "upstream": {
            "paper_url": SAIN_PAPER_URL,
            "code_url": SAIN_CODE_URL,
            "code_commit": SAIN_CODE_COMMIT,
            "reproduction": "not byte-exact; see adaptations",
        },
        "adaptations": list(
            SAIN_ADAPTATIONS if engine == "niaarm"
            else SAIN_QUANTMINER_ADAPTATIONS
        ),
        "granularity": "local_per_case_state",
        "violation_fp_budget": float(fp_budget),
        "feature_cols": feature_cols,
        "calibration": {
            "mode": "train_validation_split",
            "threshold_source": "validation",
            "fit_rows": int(len(train_df)),
            "threshold_rows": int(len(val_df)),
        },
        "components": {k: m.to_dict() for k, m in models.items()},
        "all_ok": all_ok,
    }
    if engine == "niaarm":
        result["niaarm"] = {
            "max_evals": int(max_evals),
            "confidence_min": 0.7,
            "support_min": 0.1,
            "min_observations": int(min_observations),
            "seed": int(seed),
            "algorithm": "ParticleSwarmAlgorithm",
            "n_jobs": int(n_jobs),
        }
    else:
        quant_cfg = cfg
        assert isinstance(quant_cfg, QuantMinerConfig)
        result["quantminer"] = {
            "implementation": "official_vendored_quantminer_java",
            "official_commit": QUANTMINER_CODE_COMMIT,
            "config": dict(quant_cfg.__dict__),
            "min_observations": int(min_observations),
            "seed": int(seed),
            "seed_note": (
                "experiment/split seed; the official QuantMiner CLI exposes "
                "no random-seed control"
            ),
            "n_jobs": int(n_jobs),
            "parallel_granularity": "state" if state_parallel else "component",
            "artifact_root": str(quantminer_dir),
            "rule_count": sum(
                len(state.get("rules") or [])
                for model in models.values()
                for state in model.states.values()
            ),
        }
    return result


class SainLocalDetector:
    """Runtime detector for the local SAIN-style model.

    Flags a row when *any* component flags it (OR of the local detectors).
    A component flags a row when the row's local state for that component
    was never mined (*unknown local state*), or when the number of violated
    rules of that local state reaches the component's validation-selected
    threshold.  ``scores`` is the number of flagging components, aligned
    row-by-row with ``flags`` (``flags > 0`` iff ``scores > 0``).
    """

    name = "sain_local"

    def __init__(self, model: dict[str, Any]):
        self._model = model
        self._components = model.get("components") or {}

    @classmethod
    def from_artifact(cls, path: Path | str) -> "SainLocalDetector":
        return cls(json.loads(Path(path).read_text()))

    @property
    def model(self) -> dict[str, Any]:
        return self._model

    @property
    def feature_cols(self) -> list[str]:
        return list(self._model.get("feature_cols") or [])

    def fit(self, calibration_frames: list[pd.DataFrame]
            ) -> "SainLocalDetector":
        """No-op fit: the model is already mined. Present for the monitor
        detector protocol."""
        return self

    def predict(self, frame: pd.DataFrame) -> StepFlags:
        n = len(frame)
        flags = np.zeros(n, dtype=int)
        scores = np.zeros(n, dtype=float)
        for entry in self._components.values():
            column = entry["column"]
            ids = local_state_ids(frame, column)
            states = entry.get("local_states") or {}
            threshold = max(1, int(entry.get("violation_threshold", 1)))
            for state_id in sorted(set(ids.tolist())):
                sel = np.where(ids == state_id)[0]
                mined = states.get(state_id)
                if mined is None:
                    # Unknown local state for this component.
                    flags[sel] = 1
                    scores[sel] += 1.0
                    continue
                rules = mined.get("rules") or []
                if not rules:
                    continue
                sub = frame.iloc[sel]
                counts = frame_violation_counts(sub, rules)
                fired = counts >= threshold
                if fired.any():
                    flags[sel[fired]] = 1
                    scores[sel[fired]] += 1.0
        return StepFlags(flags=flags, scores=scores)


@dataclass
class ComparisonScenario:
    """One evaluation scenario: a frame plus its ground-truth labels."""

    name: str
    frame: pd.DataFrame
    labels: np.ndarray


class _DetectorLike(Protocol):
    name: str

    def predict(self, frame: pd.DataFrame) -> StepFlags: ...


def compare_local_vs_global(
    *,
    scenarios: Sequence[ComparisonScenario],
    local_detector: _DetectorLike,
    global_detector: _DetectorLike,
    feature_cols: Sequence[str] | None = None,
    out_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Reusable comparison entry point: local SAIN vs. global invariants.

    Both detectors are evaluated on *identical* frames and, when
    ``feature_cols`` is given, the comparison records the feature columns
    both models were built on so a mismatch is visible in the artifact.

    The metric schema is the one used by the hybrid monitor
    (:func:`stl.metrics.detection_metrics`), so the comparison output can be
    consumed by the same tooling as the monitor manifest.
    """
    from stl.metrics import detection_metrics

    rows: list[dict[str, Any]] = []
    ys: list[np.ndarray] = []
    local_p: list[np.ndarray] = []
    local_s: list[np.ndarray] = []
    global_p: list[np.ndarray] = []
    global_s: list[np.ndarray] = []
    for sc in scenarios:
        y = np.asarray(sc.labels).astype(int)
        local = local_detector.predict(sc.frame)
        glob = global_detector.predict(sc.frame)
        local_scores = (local.scores if local.scores is not None
                        else local.flags.astype(float))
        glob_scores = (glob.scores if glob.scores is not None
                       else glob.flags.astype(float))
        ys.append(y)
        local_p.append(local.flags)
        local_s.append(local_scores)
        global_p.append(glob.flags)
        global_s.append(glob_scores)
        rows.append({
            "name": sc.name,
            "n": int(len(y)),
            "attack_rows": int(y.sum()),
            "local": {
                "flags": int(local.flags.sum()),
                "metrics": detection_metrics(y, local.flags, local_scores),
            },
            "global": {
                "flags": int(glob.flags.sum()),
                "metrics": detection_metrics(y, glob.flags, glob_scores),
            },
        })

    def _agg(ps: list[np.ndarray], ss: list[np.ndarray]) -> dict[str, Any]:
        if not ys:
            return {}
        return detection_metrics(
            np.concatenate(ys), np.concatenate(ps), np.concatenate(ss))

    result: dict[str, Any] = {
        "schema": "baselines/compare/v1",
        "scenarios": rows,
        "summary": {
            "local": _agg(local_p, local_s),
            "global": _agg(global_p, global_s),
        },
    }
    if feature_cols is not None:
        result["feature_cols"] = list(feature_cols)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "sain_local_vs_global.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
        )
    return result


def run_sain_local_comparison(
    *,
    topology: str,
    data_root: Path | str = Path("data"),
    gfsm_dir: Path | str | None = None,
    invariants_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    seed: int = 42,
    fraction: float = 0.8,
    min_observations: int = 50,
    max_evals: int = 5000,
    fp_budget: float = 0.01,
) -> dict[str, Any]:
    """End-to-end local-vs-global comparison for one topology.

    This is the reusable entry point: it loads the dataset, splits the
    calibration frames with :func:`dataio.splits.split_calibration_frames`,
    mines the global invariants and the local SAIN model on the *same*
    train/validation split with the *same* feature columns, and evaluates
    both on the untouched evaluation scenarios.

    Nothing here rewrites the driver wiring — the global side goes through
    :func:`invariants.driver.mine_topology` and the existing runtime
    detector, the local side through this module.
    """
    from dataio.loader import load_topology
    from dataio.model import DataIoError
    from dataio.splits import split_dataset
    from invariants.driver import mine_topology
    from invariants.state_label import resolve_fb_to_col_from_paths
    from monitor.invariants_detector import InvariantsAnomalyDetector

    data_root = Path(data_root)
    gfsm_dir = Path(gfsm_dir) if gfsm_dir is not None else (
        data_root / "generated" / topology / "gfsm")
    invariants_dir = Path(invariants_dir) if invariants_dir is not None else (
        data_root / "generated" / topology / "invariants")

    try:
        dataset = load_topology(topology, data_root=data_root)
    except DataIoError as exc:
        raise InvariantsError(str(exc)) from exc

    split = split_dataset(dataset, seed=seed, fraction=fraction)

    try:
        fb_to_col, components, _gfsm = resolve_fb_to_col_from_paths(
            gfsm_dir, topology, data_root)
    except InvariantsError as exc:
        raise InvariantsError(str(exc)) from exc

    # Identical feature columns for both sides.  The global driver derives
    # them from the fitting rows when none are given; derive them the same
    # way here and hand the very same list to both miners.
    skip = set(fb_to_col.values())
    fit_df = pd.concat(split.train_frames, ignore_index=True)
    numeric = fit_df.select_dtypes(include="number").columns
    feature_cols = [c for c in numeric if c not in skip]

    mine_topology(
        topology=topology, data_root=data_root, gfsm_dir=gfsm_dir,
        out_dir=invariants_dir, fb_to_col=fb_to_col,
        min_observations=min_observations, max_evals=max_evals, seed=seed,
        feature_cols=feature_cols, fp_budget=fp_budget,
        training_frames=split.train_frames,
        validation_frames=split.validation_frames,
        split_manifest=split.manifest,
    )

    local_model = mine_local_invariants(
        topology=topology, components=components, fb_to_col=fb_to_col,
        training_frames=split.train_frames,
        validation_frames=split.validation_frames,
        feature_cols=feature_cols,
        min_observations=min_observations, max_evals=max_evals, seed=seed,
        fp_budget=fp_budget,
    )

    global_det = InvariantsAnomalyDetector(
        invariants_dir=invariants_dir, gfsm_dir=gfsm_dir,
        data_root=data_root, topology=topology, components=components,
        fb_to_col=fb_to_col,
    ).fit([])
    local_det = SainLocalDetector(local_model)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "sain_local_model.json").write_text(
            json.dumps(local_model, indent=2, sort_keys=True, default=str) + "\n"
        )
        (out / "split_manifest.json").write_text(
            json.dumps(split.manifest, indent=2, sort_keys=True, default=str) + "\n"
        )

    scenarios = [
        ComparisonScenario(name=sc.name, frame=sc.frame,
                           labels=np.asarray(sc.labels))
        for sc in dataset.eval_scenarios
    ]
    return compare_local_vs_global(
        scenarios=scenarios,
        local_detector=local_det,
        global_detector=global_det,
        feature_cols=feature_cols,
        out_dir=out_dir,
    )
