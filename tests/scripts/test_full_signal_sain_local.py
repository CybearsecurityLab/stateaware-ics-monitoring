"""Focused tests for the full-signal local SAIN runner."""

from __future__ import annotations

import pandas as pd

from baselines.sain import mine_local_invariants
from scripts.run_full_signal_sain_local import (
    mining_feature_columns,
    pooled_detection_metrics,
)
from stl.metrics import detection_metrics


def test_mining_features_exclude_component_states_and_simulator_truth():
    frame = pd.DataFrame({
        "time_seconds": [0.0, 300.0],
        "T41": [4.0, 4.5],
        "s_P78": [0, 1],
        "demand_J1": [0.1, 0.2],
        "leak_T41": [0.0, 1.0],
        "attack_active": [0, 1],
        "physical_impact": [0.0, 2.0],
        "event_name": ["normal", "attack"],
    })

    assert mining_feature_columns(frame, ["s_P78"]) == ["T41"]


def test_local_state_granularity_mines_components_independently(monkeypatch):
    calls = []

    def fake_mine(slice_df, feature_cols, cfg):
        calls.append((slice_df.copy(), list(feature_cols)))
        return []

    monkeypatch.setattr("baselines.sain.mine_state", fake_mine)
    frame = pd.concat([
        pd.DataFrame({"p1": [0, 0], "p2": [0, 0], "tank": [1.0, 1.1]}),
        pd.DataFrame({"p1": [0, 0], "p2": [1, 1], "tank": [2.0, 2.1]}),
        pd.DataFrame({"p1": [1, 1], "p2": [0, 0], "tank": [3.0, 3.1]}),
    ], ignore_index=True)
    features = mining_feature_columns(frame, ["p1", "p2"])

    model = mine_local_invariants(
        topology="synthetic",
        components=[("PLC1", "A"), ("PLC1", "B")],
        fb_to_col={("PLC1", "A"): "p1", ("PLC1", "B"): "p2"},
        training_frames=[frame],
        validation_frames=[frame],
        feature_cols=features,
        min_observations=1,
        max_evals=5,
        seed=42,
        fp_budget=0.01,
    )

    assert model["granularity"] == "local_per_case_state"
    assert model["feature_cols"] == ["tank"]
    assert set(model["components"]["PLC1.A"]["local_states"]) == {"0", "1"}
    assert set(model["components"]["PLC1.B"]["local_states"]) == {"0", "1"}
    assert len(calls) == 4
    assert all(feature_cols == ["tank"] for _, feature_cols in calls)
    assert all(list(slice_df.columns) == ["p1", "p2", "tank"] for slice_df, _ in calls)


def test_pooled_metrics_recompute_from_renderer_columns(tmp_path):
    predictions = pd.DataFrame({
        "scenario": ["a", "a", "b", "b"],
        "y_true": [0, 0, 1, 1],
        "y_pred": [0, 1, 0, 1],
        "score": [0.1, 0.8, 0.2, 0.9],
    })
    path = tmp_path / "predictions.csv"
    predictions.to_csv(path, index=False)
    written = pd.read_csv(path)

    assert pooled_detection_metrics(written) == detection_metrics(
        written["y_true"], written["y_pred"], written["score"],
    )
