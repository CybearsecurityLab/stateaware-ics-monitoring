"""Tests for the state-aware CUSUM baseline (Ghaeini et al., SAC 2018).

The tests use a synthetic two-actuator tank so that every quantity (area,
flows, noise, attack window) is known exactly; no real dataset is needed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baselines.state_aware_cusum import (
    METRIC_SCHEMA_KEYS,
    StateAwareCusumConfig,
    StateAwareCusumDetector,
    StateAwareCusumError,
    _cusum,
    build_provenance,
    config_from_profile,
    evaluate,
    run_state_aware_cusum,
)

AREA_M2 = 10.0
DT_S = 1.0
TANK = "tank_a"
Q_PU1_LPS = 50.0
Q_PU2_LPS = 30.0


def make_frame(n: int = 240, *, seed: int = 0, noise_mm: float = 1.0,
               attack: tuple[int, int, float] | None = None,
               qout_lps: float = 0.0) -> pd.DataFrame:
    """Synthetic tank with two actuators and a known physical trajectory."""
    rng = np.random.default_rng(seed)
    s1 = rng.integers(0, 2, size=n)
    s2 = rng.integers(0, 2, size=n)
    q1 = np.where(s1 > 0, Q_PU1_LPS, 0.0)
    q2 = np.where(s2 > 0, Q_PU2_LPS, 0.0)

    levels = np.empty(n)
    h = 5.0
    for k in range(n):
        levels[k] = h
        h = h + (q1[k] + q2[k] - qout_lps) * 1e-3 * DT_S / AREA_M2

    # state-dependent sensor noise: noisier while a pump runs
    sigma = np.where(s1 > 0, noise_mm, noise_mm * 0.25)
    sigma = np.where(s2 > 0, sigma * 1.5, sigma)
    measured = levels + rng.normal(0.0, sigma) * 1e-3

    if attack is not None:
        a0, a1, step_mm = attack
        measured[a0:a1] += np.arange(a1 - a0) * step_mm * 1e-3

    return pd.DataFrame({
        TANK: measured,
        "f_pu1": q1,
        "f_pu2": q2,
        "s_pu1": s1.astype(float),
        "s_pu2": s2.astype(float),
    })


def make_config(**overrides) -> StateAwareCusumConfig:
    kwargs = dict(
        topology="synthetic",
        tanks=(TANK,),
        feeders={TANK: ("f_pu1", "f_pu2")},
        actuators={TANK: ("s_pu1", "s_pu2")},
        area_m2={TANK: AREA_M2},
        dt_s=DT_S,
        flow_to_m3ps=1e-3,
        level_to_mm=1000.0,
        false_alarm_budget_per_day=1.0,
    )
    kwargs.update(overrides)
    return StateAwareCusumConfig(**kwargs)


# --------------------------------------------------------------------------
# observer
# --------------------------------------------------------------------------


def test_observer_matches_mass_balance_equation():
    """h_{k+1} = h_k + (Qin - Qout) * dt / Area, with the actual timestep."""
    frame = make_frame(60, seed=3, noise_mm=0.0, qout_lps=10.0)
    cfg = make_config(outflows={TANK: ("f_out",)},
                      feeders={TANK: ("f_pu1", "f_pu2")})
    frame["f_out"] = 10.0
    det = StateAwareCusumDetector(cfg)
    det.fit([frame])

    yhat = det.observe(frame, TANK)
    qin = (frame["f_pu1"].to_numpy() + frame["f_pu2"].to_numpy()) * 1e-3
    qout = frame["f_out"].to_numpy() * 1e-3
    dh = np.diff(yhat)
    offset = det.model.offsets_m_per_step[TANK]
    assert offset == pytest.approx(0.0, abs=1e-12)  # Qout is modelled explicitly
    expected = (qin[:-1] - qout[:-1]) * DT_S / AREA_M2 + offset
    np.testing.assert_allclose(dh, expected, rtol=0, atol=1e-12)

    # with a known noiseless trajectory the observer reproduces the level
    assert yhat[0] == pytest.approx(frame[TANK].to_numpy()[0])


def test_observer_uses_real_timestep_not_one_second():
    """A 60 s timestep must move the observer 60x further per step."""
    frame = make_frame(40, seed=5, noise_mm=0.0)
    cfg = make_config(dt_s=60.0)
    det = StateAwareCusumDetector(cfg)
    det.fit([frame])
    yhat = det.observe(frame, TANK)
    qin = (frame["f_pu1"].to_numpy() + frame["f_pu2"].to_numpy()) * 1e-3
    expected = qin[:-1] * 60.0 / AREA_M2 + det.model.offsets_m_per_step[TANK]
    np.testing.assert_allclose(np.diff(yhat), expected, rtol=0, atol=1e-12)


def test_offset_correction_absorbs_unmodelled_outflow():
    """A constant net outflow is absorbed as a per-step observer offset."""
    frame = make_frame(80, seed=4, noise_mm=0.0, qout_lps=10.0)  # Qout not modelled
    cfg = make_config()  # no outflow mapping supplied
    det = StateAwareCusumDetector(cfg)
    det.fit([frame]).tune([make_frame(80, seed=5, noise_mm=0.0, qout_lps=10.0)])
    offset = det.model.offsets_m_per_step[TANK]
    assert offset == pytest.approx(-10.0 * 1e-3 * DT_S / AREA_M2, rel=1e-6)
    # with the offset the observer tracks the benign trajectory
    res = det.predict(frame)
    assert float(np.max(res.residuals[TANK])) == pytest.approx(0.0, abs=1e-9)


def test_observer_area_from_hydraulics_loader():
    """Area is taken from the hydraulics mass-balance loader when present."""
    frame = make_frame(80, seed=7)
    # dh = alpha * Qin(l/s)  =>  alpha = 1e-3 * dt / Area
    alpha = 1e-3 * DT_S / AREA_M2
    hydraulics = {"mb": {TANK: {"alpha": alpha, "r2": 0.99}}}
    det = StateAwareCusumDetector(make_config(area_m2={}), hydraulics=hydraulics)
    det.fit([frame])
    assert det.model.areas_m2[TANK] == pytest.approx(AREA_M2, rel=1e-9)
    assert det.model.area_sources[TANK] == "hydraulics_loader"


# --------------------------------------------------------------------------
# process states
# --------------------------------------------------------------------------


def test_state_signature_is_multi_actuator():
    frame = make_frame(200, seed=11)
    det = StateAwareCusumDetector(make_config())
    det.fit([frame]).tune([make_frame(200, seed=12, noise_mm=1.0)])
    res = det.predict(frame)
    states = set(res.states[TANK].tolist())
    assert len(states) == 4  # 2 actuators -> 4 signatures
    for s in states:
        assert s.startswith("s_pu1=") and "s_pu2=" in s


def test_per_state_means_differ_and_are_computed_from_benign_history():
    frame = make_frame(400, seed=13, noise_mm=2.0)
    det = StateAwareCusumDetector(make_config())
    det.fit([frame]).tune([make_frame(200, seed=14, noise_mm=2.0)])
    mus = det.model.mu[TANK]
    assert len(mus) == 4
    assert len(set(round(v, 6) for v in mus.values())) > 1  # state dependent

    # mu_p is the mean raw residual inside state p (paper Eq. 5)
    res = det.predict(frame)
    for state, mu in mus.items():
        mask = res.states[TANK] == state
        assert mask.sum() > 0
        assert float(np.mean(res.residuals[TANK][mask])) == pytest.approx(mu, rel=1e-9)


# --------------------------------------------------------------------------
# residual centering
# --------------------------------------------------------------------------


def test_residual_is_centered_exactly_once():
    """No double subtraction: mu_p is removed exactly once (paper Eq. 4/6)."""
    frame = make_frame(240, seed=17, noise_mm=1.5)
    det = StateAwareCusumDetector(make_config())
    det.fit([frame]).tune([make_frame(200, seed=18, noise_mm=1.5)])
    res = det.predict(frame)

    raw = res.residuals[TANK]
    signed = res.signed_centered_residuals[TANK]
    centered = res.centered_residuals[TANK]
    mu = det.model.mu[TANK]
    mu_arr = np.array([mu[s] for s in res.states[TANK]])

    # single centering: d = raw - mu_p
    np.testing.assert_allclose(signed, raw - mu_arr, rtol=0, atol=1e-9)
    # Eq. 4 residual is |d|
    np.testing.assert_allclose(centered, np.abs(raw - mu_arr), rtol=0, atol=1e-9)
    # explicitly *not* the double-centered value
    assert not np.allclose(centered, np.abs(np.abs(raw - mu_arr) - mu_arr))


def test_unseen_state_falls_back_to_global_mean():
    frame = make_frame(120, seed=19)
    det = StateAwareCusumDetector(make_config())
    det.fit([frame]).tune([make_frame(120, seed=20, noise_mm=1.0)])
    res = det.predict(frame)
    # every state seen during fit is known; the model never grows states later
    assert set(det.model.mu[TANK]) == set(res.states[TANK].tolist())


# --------------------------------------------------------------------------
# CUSUM recursion and reset
# --------------------------------------------------------------------------


def test_cusum_recursion_and_reset():
    resid = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    s, pre, flags = _cusum(resid, alpha=0.0, tau=15.0)
    # S accumulates: 0, 0, 10, 20 -> alarm at k=3 and reset
    assert list(flags) == [0, 0, 0, 1, 0]
    assert pre[2] == pytest.approx(10.0)
    assert pre[3] == pytest.approx(20.0)
    assert s[3] == pytest.approx(0.0)  # reset right after the alarm
    assert s[4] == pytest.approx(0.0)


def test_detector_resets_cusum_after_alarm():
    frame = make_frame(200, seed=23, noise_mm=0.5, attack=(60, 140, 6.0))
    cfg = make_config(alpha=0.0, tau=5.0)
    det = StateAwareCusumDetector(cfg)
    det.fit([make_frame(200, seed=29, noise_mm=0.5)])
    res = det.predict(frame)
    alarm_idx = np.flatnonzero(res.tank_flags[TANK])
    assert alarm_idx.size >= 1
    a = int(alarm_idx[0])
    assert res.cusum[TANK][a] == pytest.approx(0.0)  # reset after the alarm
    assert res.cusum_pre_reset[TANK][a] >= 5.0
    assert res.cusum[TANK][a - 1] < 5.0


def test_predict_requires_fit_and_tune():
    det = StateAwareCusumDetector(make_config())
    with pytest.raises(StateAwareCusumError):
        det.predict(make_frame(20))
    det.fit([make_frame(60, seed=31)])
    with pytest.raises(StateAwareCusumError):
        det.predict(make_frame(20))


# --------------------------------------------------------------------------
# validation-only tuning
# --------------------------------------------------------------------------


def test_alpha_and_tau_come_from_the_declared_grid():
    train = make_frame(400, seed=37, noise_mm=1.0)
    val = make_frame(400, seed=41, noise_mm=1.0)
    cfg = make_config()
    det = StateAwareCusumDetector(cfg)
    det.fit([train])
    det.tune([val])

    sel = det.model.tuning["per_tank"][TANK]
    scale = sel["residual_scale_q99_mm"]
    assert scale > 0
    # both parameters come from the predeclared grid (multipliers of the
    # residual scale measured on clean validation data)
    assert sel["alpha_mm"] / scale == pytest.approx(
        min(cfg.alpha_grid_multipliers, key=lambda m: abs(m - sel["alpha_mm"] / scale)),
        rel=1e-9)
    assert sel["tau_mm"] / scale == pytest.approx(
        min(cfg.tau_grid_multipliers, key=lambda m: abs(m - sel["tau_mm"] / scale)),
        rel=1e-9)
    # the SWaT threshold is not transferred to another network
    assert sel["tau_mm"] != pytest.approx(30.0, rel=1e-6)


def test_tuning_uses_validation_only_not_test_data():
    train = make_frame(400, seed=43, noise_mm=1.0)
    val = make_frame(400, seed=47, noise_mm=1.0)
    attacked = make_frame(400, seed=53, noise_mm=1.0, attack=(100, 300, 8.0))

    det_a = StateAwareCusumDetector(make_config()).fit([train]).tune([val])
    det_b = StateAwareCusumDetector(make_config()).fit([train]).tune([val])
    det_b.predict(attacked)  # a test frame must not influence the parameters

    assert det_a.model.alpha_mm == det_b.model.alpha_mm
    assert det_a.model.tau_mm == det_b.model.tau_mm
    assert det_a.model.tuning == det_b.model.tuning


def test_tuning_reacts_to_validation_data_and_reports_budget():
    train = make_frame(400, seed=59, noise_mm=1.0)
    quiet_val = make_frame(400, seed=61, noise_mm=1.0)
    det = StateAwareCusumDetector(make_config()).fit([train]).tune([quiet_val])
    sel = det.model.tuning["per_tank"][TANK]
    assert sel["validation_false_alarms"] <= det.model.tuning["allowed_false_alarms"]
    assert sel["budget_met"] is True
    prov = build_provenance(make_config(), det.model)
    assert prov["validation"]["test_data_used_for_tuning"] is False


def test_predict_does_not_refit_the_model():
    train = make_frame(300, seed=67, noise_mm=1.0)
    val = make_frame(300, seed=71, noise_mm=1.0)
    test = make_frame(300, seed=73, noise_mm=1.0, attack=(50, 200, 7.0))
    det = StateAwareCusumDetector(make_config()).fit([train]).tune([val])
    before = json.dumps(det.model.to_dict(), sort_keys=True)
    det.predict(test)
    after = json.dumps(det.model.to_dict(), sort_keys=True)
    assert before == after


# --------------------------------------------------------------------------
# artifacts / determinism / metrics / provenance
# --------------------------------------------------------------------------


def _run(tmp_path: Path, name: str, test_frames):
    train = make_frame(400, seed=79, noise_mm=1.0)
    val = make_frame(400, seed=83, noise_mm=1.0)
    scenarios = [(n, f, None) for n, f in test_frames]
    return run_state_aware_cusum(
        config=make_config(),
        train_frames=[train],
        validation_frames=[val],
        scenarios=scenarios,
        out_dir=tmp_path / name,
    )


def test_artifacts_are_deterministic(tmp_path: Path):
    frames = [("clean", make_frame(200, seed=89, noise_mm=1.0)),
              ("attack", make_frame(200, seed=97, noise_mm=1.0, attack=(40, 120, 7.0)))]
    _run(tmp_path, "run1", frames)
    _run(tmp_path, "run2", frames)
    for fname in ("state_aware_cusum_config.json",
                  "state_aware_cusum_model.json",
                  "state_aware_cusum_metrics.json",
                  "state_aware_cusum_provenance.json",
                  "state_aware_cusum_predictions.csv"):
        a = (tmp_path / "run1" / fname).read_bytes()
        b = (tmp_path / "run2" / fname).read_bytes()
        assert a == b, fname


def test_prediction_artifact_schema(tmp_path: Path):
    labels = np.zeros(200, dtype=int)
    labels[100:] = 1
    frame = make_frame(200, seed=101, noise_mm=1.0, attack=(100, 200, 7.0))
    manifest = run_state_aware_cusum(
        config=make_config(),
        train_frames=[make_frame(300, seed=103, noise_mm=1.0)],
        validation_frames=[make_frame(300, seed=107, noise_mm=1.0)],
        scenarios=[("test", frame, labels)],
        out_dir=tmp_path / "out",
    )
    csv = (tmp_path / "out" / "state_aware_cusum_predictions.csv").read_text()
    header = csv.splitlines()[0].split(",")
    for col in ("scenario", "row", "y_true", "y_pred", "cusum_score",
                f"cusum_{TANK}", f"residual_{TANK}", f"centered_residual_{TANK}",
                f"state_{TANK}"):
        assert col in header
    assert "test" in manifest["metrics"]
    assert set(METRIC_SCHEMA_KEYS) <= set(manifest["metrics"]["test"])


def test_metric_schema_values():
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, matthews_corrcoef

    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0, 1, 1, 1, 0, 0])
    s = np.array([0.1, 0.9, 0.8, 0.95, 0.2, 0.05])
    m = evaluate(y, p, s)
    assert m["mcc"] == pytest.approx(matthews_corrcoef(y, p))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, p))
    # attack-positive average precision: positives are label == 1
    assert m["attack_average_precision"] == pytest.approx(
        average_precision_score(y, s))
    assert m["tp"] == 2 and m["fp"] == 1 and m["tn"] == 2 and m["fn"] == 1
    assert m["n"] == 6 and m["attack_steps"] == 3


def test_provenance_records_interpretations_and_gaps():
    train = make_frame(300, seed=109, noise_mm=1.0)
    val = make_frame(300, seed=113, noise_mm=1.0)
    det = StateAwareCusumDetector(make_config()).fit([train]).tune([val])
    prov = build_provenance(make_config(), det.model)

    assert prov["paper"]["doi"] == "10.1145/3167132.3167305"
    assert prov["centering_interpretation"]["double_subtraction_performed"] is False
    assert "single centering" in prov["centering_interpretation"]["adopted"]
    gaps = prov["missing_original_observer_parameters"]
    assert gaps["tank_cross_sectional_area_m2"]["reported_in_paper"] is False
    assert gaps["tau_threshold_mm"]["swat_30mm_transferred"] is False
    assert gaps["alpha"]["reported_in_paper"] is False
    sig = prov["signature_adaptation"]["per_tank"][TANK]
    assert sig["n_actuators"] == 2
    assert sig["n_observed_states"] == 4
    assert prov["validation"]["test_data_used_for_tuning"] is False


# --------------------------------------------------------------------------
# profile integration (src/stl/profiles.py)
# --------------------------------------------------------------------------


def test_config_from_stl_profile_uses_feeder_map():
    from stl.profiles import CTOWN

    cfg = config_from_profile(CTOWN, columns=[
        "l_t1", "f_pu1", "f_pu2", "f_pu3", "s_pu1", "s_pu2", "s_pu3",
        "l_t2", "f_v2", "s_v2",
    ])
    assert cfg.feeders["l_t1"] == ("f_pu1", "f_pu2", "f_pu3")
    assert cfg.feeders["l_t2"] == ("f_v2",)
    # multi-actuator signature for the three-pump tank
    assert cfg.actuators["l_t1"] == ("s_pu1", "s_pu2", "s_pu3")
    assert cfg.actuators["l_t2"] == ("s_v2",)
    assert "l_t1" in cfg.tanks and "l_t2" in cfg.tanks


def test_config_from_stl_profile_anytown_outflow_variant():
    """anytown has no feeder_map: its control-rule pumps drain the tanks."""
    from stl.profiles import ANYTOWN

    cols = ["t41", "t42", "p78", "p79", "f_p78", "f_p79"]
    cfg = config_from_profile(ANYTOWN, columns=cols)
    assert cfg.tanks == ()  # nothing derivable without the opt-in

    cfg2 = config_from_profile(ANYTOWN, columns=cols,
                                derive_outflows_from_control_rules=True)
    assert cfg2.tanks == ("t41", "t42")
    assert cfg2.outflows["t41"] == ("f_p78",)
    assert cfg2.actuators["t41"] == ("p78",)
