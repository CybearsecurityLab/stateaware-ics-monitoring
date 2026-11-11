"""Timing-correct GFSM transitions and per-actuator action obligations."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gfsm.signatures import parse_transition_condition

from .gfsm_detector import GfsmAnomalyDetector
from .model import MonitorError
from .protocol import StepFlags


@dataclass
class ObligationPrediction:
    flags: np.ndarray
    scores: np.ndarray
    reasons: list[dict[str, Any]] = field(default_factory=list)


class GfsmObligationDetector(GfsmAnomalyDetector):
    """Check previous-input transitions and, optionally, must-act obligations.

    Observation ``i-1`` supplies controller inputs and the preceding local
    state. Observation ``i`` supplies the resulting actuator feedback. The
    first observation has no command history and produces no new obligation.
    """

    name = "gfsm_obligations"

    def __init__(self, *, check_obligations: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._check_obligations = check_obligations
        self._local_blocks: dict[tuple[str, str], dict[str, Any]] = {}

    def fit(self, calibration_frames: list[pd.DataFrame]) -> "GfsmObligationDetector":
        super().fit(calibration_frames)
        if not self._check_obligations:
            return self
        manifest_path = self._gfsm_dir / f"{self._topology}_gfsm_manifest.json"
        if not manifest_path.is_file():
            raise MonitorError(f"gfsm manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        for plc in manifest.get("plcs", []):
            name = str(plc["name"])
            relative = (plc.get("artifacts") or {}).get("fsm_json")
            if not relative:
                continue
            path = self._gfsm_dir / relative
            if not path.is_file():
                raise MonitorError(f"local fsm json not found: {path}")
            document = json.loads(path.read_text())
            for block in document.get("function_blocks", []):
                case_var = block.get("case_variable") or block.get("name")
                component = (name, str(case_var))
                if component in self._components:
                    self._local_blocks[component] = block
        missing = [component for component in self._components if component not in self._local_blocks]
        if missing:
            raise MonitorError(f"missing local FSM blocks for obligations: {missing}")
        return self

    @staticmethod
    def _local_state(label: str, component: tuple[str, str], components: list[tuple[str, str]]) -> str:
        parts = label.split("|")
        if len(parts) != len(components):
            raise MonitorError(f"malformed composite state {label!r}")
        index = components.index(component)
        left, state_id = parts[index].rsplit(":", 1)
        expected = f"{component[0]}.{component[1]}"
        if left != expected:
            raise MonitorError(f"state segment {parts[index]!r} does not match {expected}")
        return state_id

    def _transition_flags(self, labels: pd.Series, frame: pd.DataFrame) -> np.ndarray:
        flags = np.zeros(len(frame), dtype=int)
        previous: str | None = None
        for i, state in enumerate(labels):
            unknown = state not in self._known_states
            if i == 0:
                flags[i] = int(unknown)
            elif unknown:
                flags[i] = 1
            elif state != previous:
                pair = (previous, state)
                if pair not in self._allowed or not any(
                    self._matches(conjunction, frame, i - 1)
                    for conjunction in self._guards[pair]
                ):
                    flags[i] = 1
            previous = state
        return flags

    def _obligation_reasons(
        self, labels: pd.Series, frame: pd.DataFrame
    ) -> list[dict[str, Any]]:
        reasons: list[dict[str, Any]] = []
        for i in range(1, len(frame)):
            for component in self._components:
                source = self._local_state(labels.iloc[i - 1], component, self._components)
                actual = self._local_state(labels.iloc[i], component, self._components)
                block = self._local_blocks[component]
                enabled = []
                for transition in block.get("transitions", []):
                    if str(transition["from_state"]) != source:
                        continue
                    condition = str(transition.get("condition") or "No Check")
                    if condition.strip().lower() in {"true", "no check", ""}:
                        enabled.append((transition, condition))
                        continue
                    dnf = parse_transition_condition(condition)
                    if any(self._matches(conjunction, frame, i - 1) for conjunction in dnf):
                        enabled.append((transition, condition))
                if not enabled:
                    continue
                targets = {str(transition["to_state"]) for transition, _ in enabled}
                base = {
                    "type": "action_realization",
                    "plc": component[0],
                    "actuator": component[1].removesuffix("_State"),
                    "case_var": component[1],
                    "source_state": source,
                    "actual_state": actual,
                    "input_row": i - 1,
                    "observation_row": i,
                }
                if len(targets) != 1:
                    reasons.append({
                        **base,
                        "expected_states": sorted(targets),
                        "guards": [condition for _, condition in enabled],
                        "reason": "conflicting source obligations",
                    })
                    continue
                expected = next(iter(targets))
                if actual != expected:
                    reasons.append({
                        **base,
                        "expected_state": expected,
                        "guards": [condition for _, condition in enabled],
                        "reason": "enabled action was not realized in the next observation",
                    })
        return reasons

    def predict_detailed(self, frame: pd.DataFrame) -> ObligationPrediction:
        labels = self._labels(frame)
        flags = self._transition_flags(labels, frame)
        reasons = self._obligation_reasons(labels, frame) if self._check_obligations else []
        for reason in reasons:
            flags[int(reason["observation_row"])] = 1
        return ObligationPrediction(
            flags=flags,
            scores=flags.astype(float),
            reasons=reasons,
        )

    def predict(self, frame: pd.DataFrame) -> StepFlags:
        result = self.predict_detailed(frame)
        return StepFlags(flags=result.flags, scores=result.scores)

    def _labels(self, frame: pd.DataFrame) -> pd.Series:
        from invariants.state_label import label_frame
        from invariants.model import InvariantsError

        try:
            return label_frame(frame, self._components, self._fb_to_col)
        except InvariantsError as exc:
            raise MonitorError(str(exc)) from exc
