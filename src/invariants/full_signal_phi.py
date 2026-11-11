"""Global NiaARM invariant detector for full-signal evaluation frames."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from invariants.mine import MinerConfig, mine_state
from invariants.rule_eval import frame_violation_counts
from invariants.state_label import label_frame
from invariants.threshold import select_violation_threshold
from monitor.protocol import StepFlags

EXCLUDED_COLUMNS = frozenset({
    "time_seconds", "timestamp", "attack_active", "attack_class", "label",
    "physical_impact", "delivery_or_command_modified", "observable_attack",
})
EXCLUDED_PREFIXES = ("leak_", "demand_", "emitter_")


def observable_feature_columns(
    frame: pd.DataFrame, state_columns: Sequence[str]
) -> list[str]:
    """Return numeric observed features, excluding states, labels and truth."""
    excluded = set(state_columns)
    return [
        column for column in frame.columns
        if column not in EXCLUDED_COLUMNS
        and column not in excluded
        and not column.startswith(EXCLUDED_PREFIXES)
        and pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
    ]


class FullSignalPhiDetector:
    """Mine and evaluate global composite-state NiaARM invariants."""

    def __init__(self, model: dict[str, Any]):
        self.model = model
        self.components = [tuple(item) for item in model["components"]]
        self.fb_to_col = {
            (plc, case_var): column
            for plc, case_var, column in model["component_columns"]
        }
        self.threshold = max(1, int(model["violation_threshold"]))

    @classmethod
    def fit(
        cls,
        *,
        topology: str,
        training_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        components: Sequence[tuple[str, str]],
        fb_to_col: dict[tuple[str, str], str],
        min_observations: int = 50,
        max_evals: int = 5000,
        seed: int = 42,
        fp_budget: float = 0.01,
    ) -> "FullSignalPhiDetector":
        if training_frame.empty or validation_frame.empty:
            raise ValueError("Phi training and validation frames must be nonempty")
        state_columns = [fb_to_col[component] for component in components]
        feature_cols = observable_feature_columns(training_frame, state_columns)
        if not feature_cols:
            raise ValueError("no observable numeric features available for Phi")
        train_labels = label_frame(training_frame, list(components), fb_to_col)
        states: dict[str, dict[str, Any]] = {}
        all_ok = True
        config = MinerConfig(max_evals=max_evals, seed=seed)
        for state_id in sorted(set(train_labels)):
            state_frame = training_frame.loc[train_labels == state_id]
            observations = len(state_frame)
            if observations < min_observations:
                states[state_id] = {
                    "observations": observations,
                    "status": "insufficient_data",
                    "rules": [],
                }
                continue
            try:
                rules = mine_state(state_frame, feature_cols, config)
            except Exception as exc:  # noqa: BLE001 - mining records failures
                all_ok = False
                states[state_id] = {
                    "observations": observations,
                    "status": "mining_error",
                    "errors": [repr(exc)],
                    "rules": [],
                }
            else:
                states[state_id] = {
                    "observations": observations,
                    "status": "ok",
                    "rules": [rule.to_dict() for rule in rules],
                }

        validation_labels = label_frame(validation_frame, list(components), fb_to_col)
        counts = np.zeros(len(validation_frame), dtype=int)
        for state_id, entry in states.items():
            selected = (validation_labels == state_id).to_numpy()
            rules = entry.get("rules") or []
            if selected.any() and rules:
                counts[selected] = frame_violation_counts(
                    validation_frame.loc[selected], rules
                )
        absent = ~np.isin(validation_labels.to_numpy(), list(states))
        threshold, note = select_violation_threshold(counts, absent, fp_budget)
        model = {
            "schema": "full-signal-phi/v1",
            "topology": topology,
            "granularity": "global_composite_gfsm_state",
            "components": [list(component) for component in components],
            "component_columns": [
                [plc, case_var, fb_to_col[(plc, case_var)]]
                for plc, case_var in components
            ],
            "feature_cols": feature_cols,
            "observable_policy": {
                "excluded_columns": sorted(EXCLUDED_COLUMNS),
                "excluded_prefixes": list(EXCLUDED_PREFIXES),
                "state_columns": state_columns,
            },
            "calibration": {
                "mode": "chronological_train_validation",
                "fit_rows": len(training_frame),
                "threshold_rows": len(validation_frame),
                "threshold_source": "validation",
            },
            "niaarm": {
                "algorithm": "ParticleSwarmAlgorithm",
                "max_evals": int(max_evals),
                "seed": int(seed),
                "min_observations": int(min_observations),
                "confidence_min": 0.7,
                "support_min": 0.1,
            },
            "violation_fp_budget": float(fp_budget),
            "violation_threshold": int(threshold),
            "violation_threshold_note": note,
            "states": states,
            "all_ok": all_ok,
        }
        return cls(model)

    def predict(self, frame: pd.DataFrame) -> StepFlags:
        labels = label_frame(frame, self.components, self.fb_to_col)
        flags = np.zeros(len(frame), dtype=int)
        states = self.model["states"]
        for state_id in sorted(set(labels)):
            selected = np.flatnonzero((labels == state_id).to_numpy())
            entry = states.get(state_id)
            if entry is None:
                flags[selected] = 1
                continue
            rules = entry.get("rules") or []
            if rules:
                counts = frame_violation_counts(frame.iloc[selected], rules)
                flags[selected[counts >= self.threshold]] = 1
        return StepFlags(flags=flags, scores=flags.astype(float))
