"""Partial, explicitly adapted SAIN baseline over source-derived templates.

This module is deliberately separate from :mod:`baselines.sain`.  The latter
keeps :class:`baselines.sain.SainLocalDetector`, a NiaARM-based *local*
per-CASE-state adaptation.  This module implements the part of SAIN
(USENIX Security 2024, Abbas et al., Sections 5.2-5.3) that the published
paper, the official artifact, and this repository's artifacts actually
support, and records the rest as gaps.

What is implemented
-------------------
* **State identification** uses the source CASE-selector signature (the GFSM
  composite state), not a learned anomaly model.
* **Intra-state invariants** are conditional actuation implications taken
  from this repository's PDG invariant metadata
  (:mod:`iec_st_compiler.invariants`).  A template has the paper's shape
  ``IF <source variable ranges> THEN <actuation variable> := <source action>``.
  The antecedent ranges are quantified on the state sub-trace.  The
  implication is violated only when the antecedent holds and the source
  action does not; a false antecedent never produces a violation.
* **Inter-state invariants** are transition guards taken from the FSM/PDG
  transition tables.  They are checked only when a state change is actually
  observed, so remaining in a state while its exit guard is false is legal.
* **Unsupported source expressions** (disjunctions, negations, arithmetic,
  expression right-hand sides, unsupported actions) raise
  :class:`SainTemplateError` instead of being weakened or approximated.

What is *not* reproduced
------------------------
The official SAIN artifact at commit ``0a55724`` (``Application/SAIN.zip``)
contains an AST/FSM extractor (``Form1.cs``), not QuantMiner or SMONITOR.
The separate official QuantMiner Java project is vendored and can produce
per-state rule artifacts through :func:`run_quantminer`; the passive detector's
empirical state-sub-trace envelope remains an adaptation, not a complete SAIN
reproduction.  Active PLC write interception, denial, and use-def
instrumentation of SMONITOR are not implemented.  See :data:`SAIN_NATIVE_GAPS`
and ``docs/baseline_fidelity_logic.md``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from invariants.state_label import label_frame
from monitor.protocol import StepFlags

__all__ = [
    "SAIN_NATIVE_ADAPTATIONS",
    "SAIN_NATIVE_GAPS",
    "SAIN_NATIVE_PAPER_URL",
    "SAIN_NATIVE_SCHEMA",
    "SAIN_NATIVE_STATUS",
    "SAIN_NATIVE_UPSTREAM_COMMIT",
    "SAIN_NATIVE_UPSTREAM_URL",
    "SainNativeDetector",
    "SainNativeError",
    "SainTemplateError",
    "TemplateSpec",
    "derive_source_templates",
    "load_fsm_documents",
    "load_invariant_documents",
    "mine_native_sain",
    "parse_transition_condition",
    "run_native_sain",
]

SAIN_NATIVE_SCHEMA = "baselines/sain_native/v2"
SAIN_NATIVE_STATUS = "partial_adapted"
SAIN_NATIVE_UPSTREAM_URL = "https://github.com/purseclab/SAIN"
SAIN_NATIVE_UPSTREAM_COMMIT = "0a55724c7a5a352635d235d5f1a2e741045c4bed"
SAIN_NATIVE_PAPER_URL = (
    "https://www.usenix.org/conference/usenixsecurity24/presentation/abbas"
)

SAIN_NATIVE_ADAPTATIONS: tuple[str, ...] = (
    "offline passive detection only: upstream SMONITOR intercepts and denies "
    "external PLC variable writes at runtime; no write interception, denial, "
    "or use-def instrumentation is implemented or claimed here",
    "intra-state templates come from this repository's PDG invariant "
    "metadata over surrogate PLC Structured Text, not from SAIN's original "
    "FT or Tennessee Eastman programs and not from FSM transition guards",
    "inter-state guards come from the FSM/PDG transition tables and are only "
    "checked when a state change is observed; staying in a state does not "
    "require its exit guard to hold",
    "unresolved antecedent/guard bounds are filled with an empirical "
    "state-sub-trace envelope; this is a project adaptation, not QuantMiner, "
    "and is not a fully faithful SAIN reproduction",
    "guards are parsed as conjunctions of variable-to-literal comparisons; "
    "disjunctions, negations, arithmetic, and expression right-hand sides "
    "fail explicitly instead of being approximated",
)

SAIN_NATIVE_GAPS: tuple[str, ...] = (
    "the official SAIN package does not bundle QuantMiner; the separately "
    "vendored official QuantMiner can be run per state, but is not the "
    "original SAIN execution configuration and does not supply SMONITOR",
    "SMONITOR active runtime is absent from the official artifact and this "
    "repository; no write interception, denial, or use-def instrumentation",
    "the official artifact ships an AST/FSM extractor, not the PDG invariant "
    "extractor used here (src/iec_st_compiler/invariants.py)",
    "the PLC Structured Text is surrogate code generated from EPANET/DHALSIM "
    "controls, not the original vendor or benchmark PLC programs",
    "multi-CASE PDG invariant aggregation is a project adapter around the "
    "vendored analyzer, not an official SAIN extractor capability",
    "no numeric paper results are reproduced or claimed",
)


class SainNativeError(Exception):
    """Base error for the partial native SAIN baseline."""


class SainTemplateError(SainNativeError):
    """Raised when a source expression cannot be represented exactly."""


@dataclass(frozen=True)
class SourceComparison:
    """One source comparison; ``literal=None`` means an unresolved bound."""

    variable: str
    operator: str
    literal: float | int | bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "operator": self.operator,
            "literal": self.literal,
        }


@dataclass(frozen=True)
class TemplateSpec:
    """A source-derived template.

    ``kind`` is ``"intra_actuation"`` for a conditional actuation implication
    (antecedent comparisons plus a source action) or ``"inter"`` for a
    transition guard.  ``from_local`` is the source local state for both.
    """

    kind: str
    template_id: str
    plc: str
    case_var: str
    from_local: str
    to_local: str | None
    condition: str
    comparisons: tuple[SourceComparison, ...]
    actuation_var: str | None = None
    actuation_value: float | int | bool | None = None
    source_invariant_type: str | None = None
    unconditional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "template_id": self.template_id,
            "plc": self.plc,
            "case_var": self.case_var,
            "from_local": self.from_local,
            "to_local": self.to_local,
            "condition": self.condition,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "actuation_var": self.actuation_var,
            "actuation_value": self.actuation_value,
            "source_invariant_type": self.source_invariant_type,
            "unconditional": self.unconditional,
        }


_TOKEN_RE = re.compile(
    r"\s*(?:(?P<op><=|>=|<>|=|<|>)"
    r"|(?P<and>\bAND\b)"
    r"|(?P<or>\bOR\b)"
    r"|(?P<not>\bNOT\b)"
    r"|(?P<number>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"|(?P<bool>\bTRUE\b|\bFALSE\b)"
    r"|(?P<ident>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
    r"|(?P<lparen>\()"
    r"|(?P<rparen>\)))"
)

_SUPPORTED_OPERATORS = {"<", "<=", ">", ">=", "=", "<>"}
_ACTION_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:=\s*(.+)$")


def _tokenize(condition: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(condition):
        if condition[pos:].strip() == "":
            break
        match = _TOKEN_RE.match(condition, pos)
        if match is None:
            raise SainTemplateError(
                f"unsupported token at {condition[pos:pos + 24]!r} in "
                f"condition {condition!r}"
            )
        kind = match.lastgroup
        if kind is None:
            raise SainTemplateError(f"cannot tokenize condition {condition!r}")
        tokens.append((kind, match.group(kind)))
        pos = match.end()
    return tokens


def parse_transition_condition(condition: str) -> tuple[SourceComparison, ...]:
    """Parse a source condition into exact variable-to-literal comparisons.

    Only conjunctions of simple comparisons are accepted.  Arithmetic,
    disjunctions, negations, and anything else raise :class:`SainTemplateError`
    rather than being silently weakened or approximated.
    """
    tokens = _tokenize(condition)
    if not tokens:
        raise SainTemplateError("empty transition condition")

    index = 0

    def peek() -> tuple[str, str] | None:
        return tokens[index] if index < len(tokens) else None

    def consume() -> tuple[str, str]:
        nonlocal index
        if index >= len(tokens):
            raise SainTemplateError(f"unexpected end of condition {condition!r}")
        token = tokens[index]
        index += 1
        return token

    def comparison() -> SourceComparison:
        kind, variable = consume()
        if kind != "ident":
            raise SainTemplateError(
                f"expected a variable, got {variable!r} in {condition!r}"
            )
        op_kind, operator = consume()
        if op_kind != "op" or operator not in _SUPPORTED_OPERATORS:
            raise SainTemplateError(
                f"expected a comparison operator after {variable!r} in "
                f"{condition!r}"
            )
        lit_kind, literal_text = consume()
        if lit_kind == "number":
            literal: float | int | bool
            literal = float(literal_text)
        elif lit_kind == "bool":
            literal = literal_text == "TRUE"
        else:
            raise SainTemplateError(
                f"unsupported right-hand side {literal_text!r} after "
                f"{variable!r}; only numeric/boolean literals are accepted"
            )
        return SourceComparison(variable=variable, operator=operator,
                                literal=literal)

    comparisons: list[SourceComparison] = []
    while True:
        token = peek()
        if token is None:
            break
        kind, text = token
        if kind in ("lparen", "rparen"):
            consume()
            continue
        if kind == "and":
            consume()
            continue
        if kind in ("or", "not"):
            raise SainTemplateError(
                f"unsupported {text} operator in condition {condition!r}; "
                "SAIN source conjunctions are kept exact"
            )
        comparisons.append(comparison())
        token = peek()
        if token is None:
            break
        kind, text = token
        if kind in ("or", "not"):
            raise SainTemplateError(
                f"unsupported {text} operator in condition {condition!r}; "
                "SAIN source conjunctions are kept exact"
            )
        if kind not in ("and", "rparen", "lparen"):
            raise SainTemplateError(
                f"unexpected token {text!r} in condition {condition!r}"
            )
    if not comparisons:
        raise SainTemplateError(f"no comparisons in condition {condition!r}")
    return tuple(comparisons)


def _parse_action_value(value_text: Any) -> float | int | bool:
    """Parse a source action value; reject expressions explicitly."""
    if value_text is None:
        raise SainTemplateError("actuation action has no value")
    text = str(value_text).strip()
    if not text:
        raise SainTemplateError("actuation action has an empty value")
    upper = text.upper()
    if upper == "TRUE":
        return True
    if upper == "FALSE":
        return False
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    try:
        numeric = float(text)
    except ValueError:
        raise SainTemplateError(
            f"unsupported actuation value {value_text!r}; only boolean and "
            "numeric literals are accepted"
        ) from None
    if not np.isfinite(numeric):
        raise SainTemplateError(
            f"unsupported actuation value {value_text!r}; finite numeric "
            "literals are required"
        )
    return numeric


def _invariant_action(invariant: dict[str, Any]) -> tuple[str, float | int | bool]:
    """Recover the source ``(variable, value)`` action from a PDG invariant."""
    act_var = invariant.get("actuation_var")
    act_value = invariant.get("actuation_value")
    if act_var and act_value is not None:
        return str(act_var), _parse_action_value(act_value)

    action = invariant.get("action")
    if isinstance(action, str):
        match = _ACTION_ASSIGN_RE.match(action.strip())
        if match:
            return match.group(1), _parse_action_value(match.group(2))
    raise SainTemplateError(
        f"cannot recover actuation action from invariant {invariant.get('id')!r}"
    )


def _intra_specs_from_invariant(
    plc: str,
    case_var: str,
    invariant: dict[str, Any],
) -> TemplateSpec | None:
    """Convert one PDG invariant dict into an intra-state template."""
    inv_type = invariant.get("type")
    state_id = invariant.get("state_id")
    if state_id is None:
        raise SainTemplateError(
            f"invariant {invariant.get('id')!r} has no state_id"
        )
    local = str(state_id)
    template_id = f"intra:{plc}.{case_var}:{local}:{invariant.get('id')}"

    if inv_type == "single":
        sensing = invariant.get("sensing_var")
        operator = invariant.get("operator")
        act_var, act_value = _invariant_action(invariant)
        if not sensing:
            raise SainTemplateError(
                f"single-variable invariant {invariant.get('id')!r} has no "
                "sensing_var"
            )
        if operator not in _SUPPORTED_OPERATORS:
            raise SainTemplateError(
                f"unsupported operator {operator!r} in invariant "
                f"{invariant.get('id')!r}"
            )
        if sensing == "STATE":
            return TemplateSpec(
                kind="intra_actuation",
                template_id=template_id,
                plc=plc,
                case_var=case_var,
                from_local=local,
                to_local=None,
                condition=str(invariant.get("structure") or "TRUE"),
                comparisons=(),
                actuation_var=act_var,
                actuation_value=act_value,
                source_invariant_type="single",
                unconditional=True,
            )
        comparison = SourceComparison(
            variable=str(sensing), operator=str(operator), literal=None
        )
        return TemplateSpec(
            kind="intra_actuation",
            template_id=template_id,
            plc=plc,
            case_var=case_var,
            from_local=local,
            to_local=None,
            condition=str(invariant.get("structure") or f"{sensing} {operator} [#]"),
            comparisons=(comparison,),
            actuation_var=act_var,
            actuation_value=act_value,
            source_invariant_type="single",
        )

    if inv_type == "multi":
        condition = invariant.get("condition")
        act_var, act_value = _invariant_action(invariant)
        if condition is None:
            raise SainTemplateError(
                f"multi-variable invariant {invariant.get('id')!r} has no "
                "condition"
            )
        text = str(condition).strip()
        if text == "" or text.upper() == "TRUE":
            comparisons: tuple[SourceComparison, ...] = ()
            unconditional = True
        else:
            comparisons = parse_transition_condition(text)
            unconditional = False
        return TemplateSpec(
            kind="intra_actuation",
            template_id=template_id,
            plc=plc,
            case_var=case_var,
            from_local=local,
            to_local=None,
            condition=text or "TRUE",
            comparisons=comparisons,
            actuation_var=act_var,
            actuation_value=act_value,
            source_invariant_type="multi",
            unconditional=unconditional,
        )

    return None


def derive_source_templates(
    fsm_documents: Sequence[tuple[str, dict[str, Any]]],
    components: Sequence[tuple[str, str]],
    invariant_documents: Sequence[tuple[str, str, Sequence[dict[str, Any]]]] = (),
) -> tuple[list[TemplateSpec], list[dict[str, Any]]]:
    """Derive inter-state guards and intra-state actuation implications.

    Inter-state templates come from FSM transition tables (and PDG
    ``inter`` invariants when supplied).  Intra-state templates come *only*
    from PDG invariant metadata: outgoing FSM guards are transition
    conditions and are not relabelled as intra-state invariants.
    """
    known = set(components)
    specs: list[TemplateSpec] = []
    warnings: list[dict[str, Any]] = []

    for plc, document in fsm_documents:
        for fb in document.get("function_blocks", []):
            case_var = fb.get("case_variable") or fb.get("name")
            if (plc, case_var) not in known:
                continue
            for transition in fb.get("transitions", []):
                condition = transition.get("condition")
                from_local = str(transition["from_state"])
                to_local = str(transition["to_state"])
                base = (
                    f"{plc}.{case_var}:{from_local}->{to_local}:"
                    f"{transition.get('id')}"
                )
                if not condition or condition == "No Check":
                    warnings.append({
                        "plc": plc,
                        "case_var": case_var,
                        "transition": transition.get("id"),
                        "reason": "no source guard; transition is unconditional",
                    })
                    specs.append(TemplateSpec(
                        kind="inter",
                        template_id=f"inter:{base}",
                        plc=plc,
                        case_var=case_var,
                        from_local=from_local,
                        to_local=to_local,
                        condition="No Check",
                        comparisons=(),
                        unconditional=True,
                    ))
                    continue
                comparisons = parse_transition_condition(str(condition))
                specs.append(TemplateSpec(
                    kind="inter",
                    template_id=f"inter:{base}",
                    plc=plc,
                    case_var=case_var,
                    from_local=from_local,
                    to_local=to_local,
                    condition=str(condition),
                    comparisons=comparisons,
                ))

    for plc, case_var, invariants in invariant_documents:
        if (plc, case_var) not in known:
            warnings.append({
                "plc": plc,
                "case_var": case_var,
                "reason": "invariant document does not match a GFSM component",
            })
            continue
        for invariant in invariants:
            inv_type = invariant.get("type")
            if inv_type == "inter":
                condition = invariant.get("transition_condition")
                if condition is None:
                    raise SainTemplateError(
                        f"inter-state invariant {invariant.get('id')!r} has no "
                        "transition_condition"
                    )
                text = str(condition).strip()
                source_state = invariant.get("source_state")
                dest_state = invariant.get("dest_state")
                if source_state is None or dest_state is None:
                    raise SainTemplateError(
                        f"inter-state invariant {invariant.get('id')!r} must "
                        "name both source_state and dest_state"
                    )
                source = str(source_state)
                target = str(dest_state)
                if text == "" or text.upper() == "TRUE":
                    comparisons: tuple[SourceComparison, ...] = ()
                    unconditional = True
                else:
                    comparisons = parse_transition_condition(text)
                    unconditional = False
                specs.append(TemplateSpec(
                    kind="inter",
                    template_id=(
                        f"inter:{plc}.{case_var}:{source}->{target}:"
                        f"{invariant.get('id')}"
                    ),
                    plc=plc,
                    case_var=case_var,
                    from_local=source,
                    to_local=target,
                    condition=text or "TRUE",
                    comparisons=comparisons,
                    unconditional=unconditional,
                ))
                continue
            spec = _intra_specs_from_invariant(plc, case_var, invariant)
            if spec is not None:
                specs.append(spec)
                continue
            warnings.append({
                "plc": plc,
                "case_var": case_var,
                "invariant": invariant.get("id"),
                "reason": f"unsupported invariant type {inv_type!r}",
            })

    specs.sort(key=lambda s: s.template_id)
    return specs, warnings


def _local_ids(state: str, components: Sequence[tuple[str, str]]) -> dict[str, str]:
    parts = state.split("|")
    if len(parts) != len(components):
        raise SainNativeError(
            f"state {state!r} has {len(parts)} components, expected "
            f"{len(components)}"
        )
    out: dict[str, str] = {}
    for part, (plc, case_var) in zip(parts, components):
        if ":" not in part:
            raise SainNativeError(f"malformed state segment {part!r} in {state!r}")
        left, sid = part.rsplit(":", 1)
        expected = f"{plc}.{case_var}"
        if left != expected:
            raise SainNativeError(
                f"state segment {part!r} does not match component {expected!r}"
            )
        out[expected] = sid
    return out


def _resolve_column(variable: str, column_map: dict[str, str]) -> str | None:
    if variable in column_map:
        return column_map[variable]
    upper = {str(k).upper(): v for k, v in column_map.items()}
    if variable.upper() in upper:
        return upper[variable.upper()]
    return None


def _numeric_values(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float).to_numpy()
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _quantify_comparison(
    comparison: SourceComparison,
    frame: pd.DataFrame,
    column_map: dict[str, str],
) -> dict[str, Any]:
    column = _resolve_column(comparison.variable, column_map)
    base = {
        "source": comparison.to_dict(),
        "column": column,
        "operator": None,
        "bound": None,
        "bound_origin": None,
        "observed_min": None,
        "observed_max": None,
        "observations": 0,
        "status": "unquantified",
    }
    if column is None or column not in frame.columns:
        base["status"] = "missing_column"
        return base
    if len(frame) == 0:
        base["status"] = "insufficient_data"
        return base

    if comparison.literal is not None:
        base["operator"] = "==" if comparison.operator == "=" else (
            "!=" if comparison.operator == "<>" else comparison.operator
        )
        base["bound"] = comparison.literal
        base["bound_origin"] = "source_literal"
        base["observations"] = int(len(frame))
        base["status"] = "ok"
        return base

    if comparison.operator in ("=", "<>"):
        values = frame[column].dropna()
        if len(values) == 0:
            base["status"] = "insufficient_data"
            return base
        if comparison.operator == "<>":
            base["status"] = "unsupported_operator"
            return base
        unique = sorted({_jsonable_scalar(v) for v in values.tolist()})
        base["observations"] = int(len(values))
        base["operator"] = "==" if len(unique) == 1 else "in"
        base["bound"] = unique[0] if len(unique) == 1 else unique
        base["bound_origin"] = "empirical_state_subtrace"
        base["status"] = "ok"
        return base

    values = _numeric_values(frame[column])
    if len(values) == 0:
        base["status"] = "insufficient_data"
        return base
    lo = float(np.min(values))
    hi = float(np.max(values))
    base["observed_min"] = lo
    base["observed_max"] = hi
    base["observations"] = int(len(values))
    if comparison.operator in ("<", "<="):
        base["operator"] = "<="
        base["bound"] = hi
    elif comparison.operator in (">", ">="):
        base["operator"] = ">="
        base["bound"] = lo
    else:
        base["status"] = "unsupported_operator"
        return base
    base["bound_origin"] = "empirical_state_subtrace"
    base["status"] = "ok"
    return base


def _quantify_spec(
    spec: TemplateSpec,
    frame: pd.DataFrame,
    column_map: dict[str, str],
    state: str,
    observations: int,
) -> dict[str, Any]:
    atoms = [_quantify_comparison(c, frame, column_map) for c in spec.comparisons]
    ok = all(atom["status"] == "ok" for atom in atoms)
    actuation = None
    if spec.actuation_var is not None:
        act_column = _resolve_column(spec.actuation_var, column_map)
        act_status = "ok" if act_column in frame.columns else "missing_column"
        actuation = {
            "variable": spec.actuation_var,
            "value": spec.actuation_value,
            "column": act_column,
            "status": act_status,
        }
        ok = ok and act_status == "ok"
    return {
        "template_id": spec.template_id,
        "kind": spec.kind,
        "source_invariant_type": spec.source_invariant_type,
        "state": state,
        "plc": spec.plc,
        "case_var": spec.case_var,
        "from_local": spec.from_local,
        "to_local": spec.to_local,
        "condition": spec.condition,
        "observations": int(observations),
        "status": "ok" if ok else "unquantified",
        "atoms": atoms,
        "actuation": actuation,
        "unconditional": spec.unconditional,
    }


def _component_records(
    components: Sequence[tuple[str, str]],
    fb_to_col: dict[tuple[str, str], str],
) -> list[dict[str, str]]:
    return [
        {"plc": plc, "case_var": case_var, "column": fb_to_col[(plc, case_var)]}
        for plc, case_var in components
    ]


def _official_quantminer_artifacts(
    specs: Sequence[TemplateSpec],
    states: Mapping[str, Any],
    train_df: pd.DataFrame,
    state_ids: pd.Series,
    column_map: Mapping[str, str],
    output_root: Path,
) -> list[dict[str, Any]]:
    from baselines.quantminer import QuantMinerConfig, run_quantminer

    variables = sorted({
        comparison.variable
        for spec in specs
        for comparison in spec.comparisons
        if comparison.variable in column_map
    })
    artifacts = []
    for state, entry in states.items():
        count = int(entry["observations"])
        if count < 2:
            artifacts.append({
                "state": state,
                "observations": count,
                "variables": variables,
                "status": "unavailable",
                "reason": "QuantMiner CSV analysis requires at least two observations",
            })
            continue
        selected = train_df.loc[(state_ids == state).to_numpy()]
        columns = [column_map[name] for name in variables]
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", state)
        try:
            result = run_quantminer(
                selected.loc[:, columns],
                Path(output_root) / safe,
                QuantMinerConfig(),
            )
        except Exception as exc:
            artifacts.append({
                "state": state,
                "observations": count,
                "variables": variables,
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            continue
        artifacts.append({
            "state": state,
            "observations": count,
            "variables": variables,
            "status": "ok",
            "rules_path": str(result["rules_path"]),
            "rule_count": result["provenance"]["rule_count"],
            "provenance": result["provenance"],
        })
    return artifacts


def mine_native_sain(
    *,
    topology: str,
    gfsm_json: dict[str, Any],
    fsm_documents: Sequence[tuple[str, dict[str, Any]]],
    components: Sequence[tuple[str, str]],
    fb_to_col: dict[tuple[str, str], str],
    column_map: dict[str, str],
    training_frames: Sequence[pd.DataFrame],
    invariant_documents: Sequence[
        tuple[str, str, Sequence[dict[str, Any]]]
    ] = (),
    min_observations: int = 1,
    quantminer_dir: Path | None = None,
) -> dict[str, Any]:
    """Mine source-derived, per-global-state SAIN templates.

    Only ``training_frames`` quantify the templates.  No validation threshold
    is used: the paper's SMONITOR denies every invariant violation, so the
    passive detector here flags every quantified violation.
    """
    if not training_frames:
        raise SainNativeError("mine_native_sain needs training frames")
    if min_observations < 1:
        raise ValueError("min_observations must be >= 1")
    if not components:
        raise SainNativeError("no GFSM components supplied")
    for component in components:
        if component not in fb_to_col:
            raise SainNativeError(
                f"no dataset column resolved for component {component!r}"
            )

    train_df = pd.concat(list(training_frames), ignore_index=True)
    state_ids = label_frame(train_df, list(components), fb_to_col)
    specs, warnings = derive_source_templates(
        fsm_documents, components, invariant_documents
    )

    states_json = gfsm_json.get("states") or {}
    if not states_json:
        raise SainNativeError("gfsm json has no states")

    states: dict[str, Any] = {}
    for state in sorted(states_json):
        local = _local_ids(state, components)
        mask = (state_ids == state).to_numpy()
        state_df = train_df.loc[mask]
        observations = int(mask.sum())
        active = [
            spec for spec in specs
            if local.get(f"{spec.plc}.{spec.case_var}") == spec.from_local
        ]
        intra = [
            _quantify_spec(spec, state_df, column_map, state, observations)
            for spec in active if spec.kind == "intra_actuation"
        ]
        inter = [
            _quantify_spec(spec, state_df, column_map, state, observations)
            for spec in active if spec.kind == "inter"
        ]
        if observations < min_observations:
            status = "insufficient_data"
            for template in intra:
                template["status"] = "insufficient_data"
        elif any(template["status"] != "ok" for template in intra):
            status = "unquantified"
        else:
            status = "ok"
        entry: dict[str, Any] = {
            "state": state,
            "local_states": local,
            "observations": observations,
            "status": status,
            "intra_source": (
                "pdg_invariant_metadata" if intra else "no_pdg_invariants"
            ),
            "intra": intra,
            "inter": inter,
        }
        states[state] = entry

    intra_count = sum(len(entry["intra"]) for entry in states.values())
    inter_count = sum(len(entry["inter"]) for entry in states.values())
    official_quantminer = []
    if quantminer_dir is not None:
        official_quantminer = _official_quantminer_artifacts(
            specs, states, train_df, state_ids, column_map, Path(quantminer_dir)
        )

    return {
        "schema": SAIN_NATIVE_SCHEMA,
        "status": SAIN_NATIVE_STATUS,
        "topology": topology,
        "upstream": {
            "paper_url": SAIN_NATIVE_PAPER_URL,
            "code_url": SAIN_NATIVE_UPSTREAM_URL,
            "code_commit": SAIN_NATIVE_UPSTREAM_COMMIT,
        },
        "baseline_role": "partial_adapted_source_derived_sain",
        "not_the_niaarm_local_adaptation": True,
        "fully_faithful_sain": False,
        "native_quantminer_reproduction": False,
        "official_quantminer_engine_used": bool(official_quantminer),
        "official_quantminer_artifacts": official_quantminer,
        "adaptations": list(SAIN_NATIVE_ADAPTATIONS),
        "gaps": list(SAIN_NATIVE_GAPS),
        "operational_mode": {
            "mode": "offline_passive_detection",
            "write_interception_supported": False,
            "write_denial_supported": False,
            "smonitor_runtime_implemented": False,
            "unavailable_runtime_features": [
                "external PLC variable write interception",
                "write denial and operator alert",
                "use-def chain instrumentation of server applications",
            ],
        },
        "state_identification": {
            "method": "gfsm_case_selector_signature",
            "components": _component_records(components, fb_to_col),
            "n_global_states": len(states),
        },
        "quantification": {
            "method": "empirical_state_subtrace_envelope",
            "implementation": "project_adaptation_not_quantminer",
            "quantminer_available": bool(official_quantminer),
            "quantminer_backend": (
                "official_vendored_quantminer_java" if official_quantminer
                else "project_adaptation_not_quantminer"
            ),
            "quantminer_note": (
                "the official SAIN artifact inspected at the pinned commit "
                "does not include QuantMiner; unresolved bounds are filled "
                "from the state sub-trace as an explicit adaptation"
            ),
            "source_literals_preserved": True,
            "unresolved_bound_policy": {
                "< / <=": "observed maximum in the global-state sub-trace",
                "> / >=": "observed minimum in the global-state sub-trace",
                "= / <>": "observed equality set; unresolved <> is rejected",
            },
            "min_observations": int(min_observations),
            "fit_rows": int(len(train_df)),
        },
        "templates": {
            "specs": [spec.to_dict() for spec in specs],
            "warnings": warnings,
            "intra_templates": intra_count,
            "inter_templates": inter_count,
            "intra_source": "pdg_invariant_metadata",
            "inter_source": "fsm_pdg_transition_guards",
        },
        "states": states,
        "all_ok": all(entry["status"] == "ok" for entry in states.values()),
    }


def _atom_eval(
    frame: pd.DataFrame,
    atom: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(holds, undecidable)`` for one quantified comparison."""
    n = len(frame)
    if atom.get("status") != "ok":
        return np.zeros(n, dtype=bool), np.ones(n, dtype=bool)
    column = atom.get("column")
    if column is None or column not in frame.columns:
        return np.zeros(n, dtype=bool), np.ones(n, dtype=bool)
    values = frame[column]
    operator = atom.get("operator")
    bound = atom.get("bound")
    if operator == "in":
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        allowed = np.asarray(bound, dtype=float)
        finite = np.isfinite(numeric)
        return np.isin(numeric, allowed) & finite, ~finite
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(numeric)
    if operator == "<=":
        holds = finite & (numeric <= float(bound))
    elif operator == ">=":
        holds = finite & (numeric >= float(bound))
    elif operator == "==":
        holds = finite & (numeric == float(bound))
    elif operator == "!=":
        holds = finite & (numeric != float(bound))
    elif operator == "<":
        holds = finite & (numeric < float(bound))
    elif operator == ">":
        holds = finite & (numeric > float(bound))
    else:
        return np.zeros(n, dtype=bool), np.ones(n, dtype=bool)
    return holds, ~finite


def _action_eval(
    frame: pd.DataFrame,
    actuation: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(holds, undecidable)`` for the source actuation action."""
    n = len(frame)
    if not actuation or actuation.get("status") != "ok":
        return np.zeros(n, dtype=bool), np.ones(n, dtype=bool)
    column = actuation.get("column")
    if column is None or column not in frame.columns:
        return np.zeros(n, dtype=bool), np.ones(n, dtype=bool)
    value = actuation.get("value")
    if isinstance(value, bool):
        target = 1.0 if value else 0.0
    else:
        target = float(value)
    numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(numeric)
    return finite & (numeric == target), ~finite


def _intra_masks(
    frame: pd.DataFrame,
    templates: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate conditional actuation implications.

    A template is violated only when its antecedent holds and the source
    action does not.  A false antecedent never violates the implication.
    """
    n = len(frame)
    violated = np.zeros(n, dtype=bool)
    undecidable = np.zeros(n, dtype=bool)
    for template in templates:
        antecedent_holds = np.ones(n, dtype=bool)
        antecedent_unknown = np.zeros(n, dtype=bool)
        for atom in template.get("atoms", []):
            holds, unknown = _atom_eval(frame, atom)
            antecedent_holds &= holds
            antecedent_unknown |= unknown
        consequent_holds, consequent_unknown = _action_eval(
            frame, template.get("actuation")
        )
        violated |= (
            antecedent_holds & ~consequent_holds & ~consequent_unknown
        )
        undecidable |= antecedent_unknown | (
            antecedent_holds & consequent_unknown
        )
    return violated, undecidable


class SainNativeDetector:
    """Passive detector for the partial source-derived SAIN model."""

    name = "sain_native"

    def __init__(self, model: dict[str, Any]):
        if model.get("schema") != SAIN_NATIVE_SCHEMA:
            raise SainNativeError(
                f"unexpected native SAIN schema {model.get('schema')!r}"
            )
        self._model = model
        self._components = [
            (entry["plc"], entry["case_var"])
            for entry in model["state_identification"]["components"]
        ]
        self._fb_to_col = {
            (entry["plc"], entry["case_var"]): entry["column"]
            for entry in model["state_identification"]["components"]
        }
        self._states = model.get("states") or {}

    @classmethod
    def from_artifact(cls, path: Path | str) -> "SainNativeDetector":
        return cls(json.loads(Path(path).read_text()))

    @property
    def model(self) -> dict[str, Any]:
        return self._model

    def fit(self, calibration_frames: list[pd.DataFrame]) -> "SainNativeDetector":
        return self

    def predict(self, frame: pd.DataFrame) -> StepFlags:
        states = label_frame(frame, self._components, self._fb_to_col)
        n = len(frame)
        flags = np.zeros(n, dtype=int)
        scores = np.zeros(n, dtype=float)

        for state_id in sorted(set(states.tolist())):
            index = np.flatnonzero(states.to_numpy() == state_id)
            entry = self._states.get(state_id)
            if entry is None or entry.get("status") != "ok":
                flags[index] = 1
                scores[index] += 1.0
                continue
            sub = frame.iloc[index]
            violated, undecidable = _intra_masks(sub, entry.get("intra", []))
            if violated.any():
                hit = index[violated]
                flags[hit] = 1
                scores[hit] += 1.0
            if undecidable.any():
                hit = index[undecidable & ~violated]
                flags[hit] = 1
                scores[hit] += 1.0

        state_values = states.tolist()
        for i in range(1, n):
            previous = state_values[i - 1]
            current = state_values[i]
            if previous == current:
                continue
            if previous not in self._states or current not in self._states:
                continue
            entry = self._states.get(previous)
            if entry is None or entry.get("status") != "ok":
                flags[i] = 1
                scores[i] += 1.0
                continue
            if not self._transition_allowed(
                entry, previous, current, frame.iloc[i - 1]
            ):
                flags[i] = 1
                scores[i] += 1.0

        return StepFlags(flags=flags, scores=scores)

    def _transition_allowed(
        self,
        entry: dict[str, Any],
        previous_state: str,
        current_state: str,
        previous_row: pd.Series,
    ) -> bool:
        previous_local = entry.get("local_states") or {}
        current_local = _local_ids(current_state, self._components)
        changed = [
            key for key, value in previous_local.items()
            if current_local.get(key) != value
        ]
        inter = entry.get("inter", [])
        for key in changed:
            plc, case_var = key.split(".", 1)
            target = current_local[key]
            candidates = [
                template for template in inter
                if template.get("plc") == plc
                and template.get("case_var") == case_var
                and template.get("to_local") == target
            ]
            if not candidates:
                return False
            if not any(self._template_holds(previous_row, template)
                       for template in candidates):
                return False
        return True

    def _template_holds(
        self,
        row: pd.Series,
        template: dict[str, Any],
    ) -> bool:
        frame = row.to_frame().T.reset_index(drop=True)
        for atom in template.get("atoms", []):
            holds, unknown = _atom_eval(frame, atom)
            if not bool(holds.all()) or bool(unknown.any()):
                return False
        return True


def load_fsm_documents(
    gfsm_dir: Path | str,
    topology: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Load the per-PLC source FSM tables named by the GFSM manifest."""
    gfsm_dir = Path(gfsm_dir)
    manifest_path = gfsm_dir / f"{topology}_gfsm_manifest.json"
    if not manifest_path.exists():
        raise SainNativeError(f"gfsm manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    documents: list[tuple[str, dict[str, Any]]] = []
    for plc in manifest.get("plcs", []):
        artifacts = plc.get("artifacts") or {}
        fsm_json = artifacts.get("fsm_json")
        if not fsm_json:
            continue
        path = gfsm_dir / fsm_json
        if not path.exists():
            raise SainNativeError(f"fsm json not found: {path}")
        documents.append((str(plc["name"]), json.loads(path.read_text())))
    if not documents:
        raise SainNativeError(
            f"no per-PLC fsm json artifacts listed in {manifest_path}"
        )
    return documents


def _case_vars_by_plc(
    fsm_documents: Sequence[tuple[str, dict[str, Any]]],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for plc, document in fsm_documents:
        case_vars = out.setdefault(plc, [])
        for fb in document.get("function_blocks", []):
            case_var = fb.get("case_variable") or fb.get("name")
            if case_var and case_var not in case_vars:
                case_vars.append(case_var)
    return out


def load_invariant_documents(
    analysis_dir: Path | str,
    topology: str,
    fsm_documents: Sequence[tuple[str, dict[str, Any]]],
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """Load per-PLC PDG invariant metadata from the analysis artifacts.

    Returns ``(plc, case_var, invariants)`` triples. Multi-CASE analysis
    artifacts tag each invariant with its selector. Older untagged artifacts
    are accepted only for single-selector PLCs.
    """
    analysis_dir = Path(analysis_dir)
    manifest_path = analysis_dir / f"{topology}_analysis_manifest.json"
    if not manifest_path.exists():
        raise SainNativeError(f"analysis manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    case_vars = _case_vars_by_plc(fsm_documents)
    documents: list[tuple[str, str, list[dict[str, Any]]]] = []
    for plc in manifest.get("plcs", []):
        name = str(plc["name"])
        artifacts = plc.get("artifacts") or {}
        inv_json = artifacts.get("invariants_json")
        if not inv_json:
            continue
        path = analysis_dir / inv_json
        if not path.exists():
            raise SainNativeError(f"invariants json not found: {path}")
        invariants = json.loads(path.read_text())
        if not isinstance(invariants, list):
            raise SainNativeError(
                f"invariants json {path} must contain a list"
            )
        selectors = case_vars.get(name, [])
        tagged = bool(invariants) and all(
            invariant.get("case_var") for invariant in invariants
        )
        if invariants and len(selectors) != 1 and not tagged:
            raise SainNativeError(
                f"{path} has {len(invariants)} invariants but PLC {name!r} "
                f"has {len(selectors)} CASE selectors; the PDG invariant JSON "
                "does not identify which selector each invariant belongs to"
            )
        for case_var in selectors:
            selected = (
                [inv for inv in invariants if inv["case_var"] == case_var]
                if tagged else invariants
            )
            if selected:
                documents.append((name, case_var, selected))
    return documents


def run_native_sain(
    *,
    topology: str,
    data_root: Path | str = Path("data"),
    gfsm_dir: Path | str | None = None,
    analysis_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    seed: int = 42,
    fraction: float = 0.8,
    min_observations: int = 1,
) -> dict[str, Any]:
    """End-to-end partial native SAIN entry point for later experiments.

    Quantification uses the training side of the standard calibration split;
    evaluation scenarios are untouched.  Nothing is written unless ``out_dir``
    is supplied.
    """
    from dataio.loader import load_topology
    from dataio.splits import split_dataset
    from invariants.state_label import resolve_fb_to_col_from_paths
    from stl.metrics import detection_metrics

    data_root = Path(data_root)
    gfsm_dir = Path(gfsm_dir) if gfsm_dir is not None else (
        data_root / "generated" / topology / "gfsm"
    )
    analysis_dir = Path(analysis_dir) if analysis_dir is not None else (
        gfsm_dir.parent / "analysis"
    )
    dataset = load_topology(topology, data_root=data_root)
    split = split_dataset(dataset, seed=seed, fraction=fraction)
    fb_to_col, components, gfsm_json = resolve_fb_to_col_from_paths(
        gfsm_dir, topology, data_root
    )
    fsm_documents = load_fsm_documents(gfsm_dir, topology)
    invariant_documents = load_invariant_documents(
        analysis_dir, topology, fsm_documents
    )
    column_map = dict(dataset.schema.column_map)
    model = mine_native_sain(
        topology=topology,
        gfsm_json=gfsm_json,
        fsm_documents=fsm_documents,
        components=components,
        fb_to_col=fb_to_col,
        column_map=column_map,
        training_frames=split.train_frames,
        invariant_documents=invariant_documents,
        min_observations=min_observations,
    )
    detector = SainNativeDetector(model)
    scenarios: list[dict[str, Any]] = []
    all_y: list[np.ndarray] = []
    all_flags: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for scenario in dataset.eval_scenarios:
        predicted = detector.predict(scenario.frame)
        y = np.asarray(scenario.labels, dtype=int)
        all_y.append(y)
        all_flags.append(predicted.flags)
        all_scores.append(predicted.scores)
        scenarios.append({
            "name": scenario.name,
            "n": int(len(y)),
            "attack_rows": int(y.sum()),
            "flags": int(predicted.flags.sum()),
            "metrics": detection_metrics(y, predicted.flags, predicted.scores),
        })
    result = {
        "schema": "baselines/sain_native_eval/v1",
        "topology": topology,
        "model_schema": SAIN_NATIVE_SCHEMA,
        "model_status": SAIN_NATIVE_STATUS,
        "split": split.manifest,
        "scenarios": scenarios,
        "summary": detection_metrics(
            np.concatenate(all_y),
            np.concatenate(all_flags),
            np.concatenate(all_scores),
        ) if all_y else {},
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "native_sain_model.json").write_text(
            json.dumps(model, indent=2, sort_keys=True, default=str) + "\n"
        )
        (out / "native_sain_eval.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
        )
    result["model"] = model
    return result
