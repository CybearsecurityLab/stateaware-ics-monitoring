import numpy as np
import pandas as pd

from invariants.full_signal_phi import (
    FullSignalPhiDetector,
    observable_feature_columns,
)


COMPONENTS = [("PLC1", "P_State")]
FB_TO_COL = {("PLC1", "P_State"): "s_P"}


def frame(values, states=None):
    n = len(values)
    return pd.DataFrame({
        "time_seconds": np.arange(n) * 300,
        "x": values,
        "s_P": np.zeros(n, dtype=int) if states is None else states,
        "attack_active": np.zeros(n, dtype=int),
        "leak_truth": np.zeros(n),
        "demand_truth": np.zeros(n),
        "emitter_truth": np.zeros(n),
    })


def test_observable_features_exclude_states_labels_time_and_truth():
    assert observable_feature_columns(frame([1, 2]), ["s_P"]) == ["x"]


def test_global_phi_fit_and_prediction_use_validation_threshold(monkeypatch):
    class Rule:
        def to_dict(self):
            return {"antecedent": [], "consequent": [{"col": "x", "op": ">=", "val": 2}]}

    monkeypatch.setattr(
        "invariants.full_signal_phi.mine_state", lambda *args, **kwargs: [Rule()]
    )
    detector = FullSignalPhiDetector.fit(
        topology="test",
        training_frame=frame([1, 2, 3, 4]),
        validation_frame=frame([2, 3, 4, 5]),
        components=COMPONENTS,
        fb_to_col=FB_TO_COL,
        min_observations=1,
    )
    assert detector.model["granularity"] == "global_composite_gfsm_state"
    assert detector.model["calibration"] == {
        "mode": "chronological_train_validation",
        "fit_rows": 4,
        "threshold_rows": 4,
        "threshold_source": "validation",
    }
    result = detector.predict(frame([1, 2, 3], states=[0, 0, 1]))
    assert result.flags.tolist() == [1, 0, 1]
    assert result.scores.tolist() == [1.0, 0.0, 1.0]
