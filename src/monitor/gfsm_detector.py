"""GFSM δ-conformance anomaly detector.

Loads the persisted GFSM JSON (built by `gfsm-build`) and flags rows whose
composite-state encoding is not a known state, or whose (prev -> curr)
transition is not in G's transition relation. Stutter (state -> same
state) is always permitted. Build-once / reuse-many: `fit` deserializes the
JSON; mining/extraction never happen at predict time.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from invariants.model import InvariantsError
from invariants.state_label import label_frame, load_gfsm_components
from gfsm.signatures import Condition, parse_transition_condition, tokenize

from .model import MonitorError
from .protocol import StepFlags


class GfsmAnomalyDetector:
    name = "gfsm"

    def __init__(
        self,
        *,
        gfsm_dir: Path,
        topology: str,
        fb_to_col: dict[tuple[str, str], str],
    ):
        self._gfsm_dir = Path(gfsm_dir)
        self._topology = topology
        self._fb_to_col = fb_to_col
        self._components: list[tuple[str, str]] = []
        self._known_states: set[str] = set()
        self._allowed: set[tuple[str, str]] = set()
        self._guards: dict[tuple[str, str], list[list[Condition]]] = {}

    def fit(self, calibration_frames: list[pd.DataFrame]) -> "GfsmAnomalyDetector":
        path = self._gfsm_dir / f"{self._topology}.gfsm.json"
        if not path.exists():
            raise MonitorError(f"gfsm json not found: {path}")
        gfsm = json.loads(path.read_text())
        self._components = load_gfsm_components(gfsm)
        self._known_states = set((gfsm.get("states") or {}).keys())
        self._allowed = set()
        self._guards = {}
        for t in gfsm.get("transitions") or []:
            frm = t.get("from_state", t.get("from"))
            to = t.get("to_state", t.get("to"))
            if frm is None or to is None:
                raise MonitorError(
                    f"gfsm transition missing from/to keys: {t!r}")
            self._allowed.add((frm, to))
            condition = t["guard"] if "guard" in t else t.get("condition", "")
            if not isinstance(condition, str):
                raise MonitorError(f"malformed GFSM guard: {condition!r}")
            if condition.strip().lower() in {"true", "no check", ""}:
                self._guards.setdefault((frm, to), []).append([])
                continue
            try:
                tokens = tokenize(condition)
                atom_re = re.compile(
                    r"[A-Za-z_][\w.]*\s*(?:<>|<=|>=|=|<|>)\s*"
                    r"(?:TRUE|FALSE|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
                    re.I,
                )
                if not tokens or any(
                    kind == "COND" and not atom_re.fullmatch(payload or "")
                    for kind, payload in tokens
                ):
                    raise ValueError("unparsed guard atom")
                dnf = parse_transition_condition(condition)
                if not dnf or sum(kind == "COND" for kind, _ in tokens) != sum(
                    len(conjunction) for conjunction in dnf
                ):
                    raise ValueError("unparsed guard text")
            except Exception as exc:
                raise MonitorError(f"malformed GFSM guard {condition!r}: {exc}") from exc
            self._guards.setdefault((frm, to), []).extend(dnf or [[]])
        return self

    def predict(self, frame: pd.DataFrame) -> StepFlags:
        try:
            labels = label_frame(frame, self._components, self._fb_to_col)
        except InvariantsError as exc:
            raise MonitorError(str(exc)) from exc
        flags = np.zeros(len(frame), dtype=int)
        prev: str | None = None
        for i, s in enumerate(labels):
            unknown = s not in self._known_states
            if i == 0:
                flags[i] = int(unknown)
            elif unknown:
                flags[i] = 1
            elif s != prev:
                pair = (prev, s)
                if pair not in self._allowed or not any(
                    self._matches(conj, frame, i) for conj in self._guards[pair]
                ):
                    flags[i] = 1
            prev = s
        return StepFlags(flags=flags, scores=flags.astype(float))

    def _matches(self, conjunction: list[Condition], frame: pd.DataFrame, row: int) -> bool:
        for cond in conjunction:
            col = self._resolve_sensor(cond.variable, frame)
            try:
                actual = frame.iloc[row][col]
                expected = cond.value.strip()
                bool_literal = expected.lower() in {"true", "false"}
                if bool_literal:
                    wanted = expected.lower() == "true"
                    if isinstance(actual, (bool, np.bool_)):
                        a = bool(actual)
                    else:
                        a = float(actual)
                        if not math.isfinite(a) or a not in (0.0, 1.0):
                            raise ValueError
                        a = bool(a)
                    b = wanted
                else:
                    if isinstance(actual, (bool, np.bool_)):
                        actual = int(actual)
                    a, b = float(actual), float(expected)
                    if not math.isfinite(a) or not math.isfinite(b):
                        raise ValueError
            except (TypeError, ValueError, KeyError, IndexError) as exc:
                raise MonitorError(f"missing or nonfinite input for guard sensor {cond.variable!r}") from exc
            funcs = {"=": lambda: a == b, "<>": lambda: a != b, "<": lambda: a < b,
                     "<=": lambda: a <= b, ">": lambda: a > b, ">=": lambda: a >= b}
            if cond.operator not in funcs:
                raise MonitorError(f"unsupported guard operator {cond.operator!r}")
            if not funcs[cond.operator]():
                return False
        return True

    def _resolve_sensor(self, variable: str, frame: pd.DataFrame) -> str:
        aliases: dict[str, set[str]] = {}
        for col in map(str, frame.columns):
            for alias in (col, col.lower(), re.sub(r"^S_", "", col, flags=re.I)):
                aliases.setdefault(alias.lower(), set()).add(col)
        for (_fb, var), col in self._fb_to_col.items():
            match = re.fullmatch(r"([TJ])(.+)", var, re.I)
            canonical = (("l_t" if match.group(1) == "T" else "p_j") + match.group(2)
                         if match else None)
            for alias in (var, f"S_{var}", canonical):
                if alias:
                    aliases.setdefault(alias.lower(), set()).add(col)
        candidates = aliases.get(variable.lower(), set())
        match = re.fullmatch(r"([TJ])(.+)", variable, re.I)
        if match:
            canonical = ("l_t" if match.group(1).upper() == "T" else "p_j") + match.group(2)
            if canonical in frame.columns:
                candidates.add(canonical)
        if len(candidates) != 1:
            reason = "missing" if not candidates else "ambiguous"
            raise MonitorError(f"{reason} guard sensor {variable!r}: {sorted(candidates)}")
        col = next(iter(candidates))
        if col not in frame.columns:
            raise MonitorError(f"missing guard sensor column {col!r}")
        return col
