"""Focused tests for the shared full-signal adapter and CUSUM mapping."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dataio.full_signal import (
    ATTACK_CLASS_TAXONOMY,
    FullSignalError,
    build_cusum_config,
    discover_full_signal_runs,
    load_full_signal_run,
    split_nominal_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ANYTOWN_INP = REPO_ROOT / "scripts/out/logic_attacks/anytown/topology.inp"
ANYTOWN_MANIFEST = REPO_ROOT / "scripts/out/logic_attacks/anytown/nominal/manifest.json"
LTOWN_INP = REPO_ROOT / "scripts/out/full_signal_pdd/ltown/topology.inp"
LTOWN_MANIFEST = REPO_ROOT / "scripts/out/full_signal_pdd/ltown/nominal_two_day/manifest.json"


def _write_bundle(path: Path, *, times: list[float], label_rows: list[dict],
                  events: list[dict] | None = None,
                  with_observed: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if with_observed:
        pd.DataFrame({
            "time_seconds": times,
            "T1": np.linspace(3.0, 3.1, len(times)),
            "f_p1": np.linspace(0.01, 0.02, len(times)),
            "s_p1": np.ones(len(times)),
        }).to_csv(path / "observed.csv", index=False)
    pd.DataFrame(label_rows).to_csv(path / "labels.csv", index=False)
    (path / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "run_id": "synthetic-run",
        "split": "calibration" if path.name == "nominal" else "evaluation",
        "inp_sha256": "0" * 64,
        "controls": [{"id": "c1", "link": "PUMP_1", "sensor": "T1",
                      "relation": "lt", "threshold": 2.4, "value": 1}],
    }) + "\n")
    (path / "events.json").write_text(json.dumps(events or []) + "\n")


def _labels(times: list[float], active: list[bool]) -> list[dict]:
    return [
        {"time_seconds": t, "attack_active": a,
         "delivery_or_command_modified": a, "physical_impact": False}
        for t, a in zip(times, active)
    ]


def test_nominal_split_is_disjoint_chronological_and_hashed():
    n = 100
    times = [float(t) for t in range(n)]
    frame = pd.DataFrame({"time_seconds": times, "T1": np.zeros(n)})
    y = np.zeros(n, dtype=int)
    classes = tuple("normal" for _ in range(n))

    split = split_nominal_frame(frame, y, classes)

    assert (split.train.rows, split.validation.rows, split.test.rows) == (64, 16, 20)
    train_t = set(split.train.frame["time_seconds"])
    val_t = set(split.validation.frame["time_seconds"])
    test_t = set(split.test.frame["time_seconds"])
    assert not (train_t & val_t or train_t & test_t or val_t & test_t)
    assert max(train_t) < min(val_t) < max(val_t) < min(test_t)
    assert set().union(train_t, val_t, test_t) == set(times)
    assert len({split.train.digest, split.validation.digest, split.test.digest}) == 3
    assert split.manifest["boundaries"] == {
        "train_stop_index": 64, "validation_stop_index": 80,
    }
    assert split.test.attack_class == classes[80:]
    assert split.manifest["attack_scenarios_in_split"] is False


@pytest.mark.skipif(not ANYTOWN_INP.is_file() or not ANYTOWN_MANIFEST.is_file(),
                    reason="Anytown resolved artifacts unavailable")
def test_anytown_signed_feeders_use_incident_links_and_manifest_statuses():
    manifest = json.loads(ANYTOWN_MANIFEST.read_text())
    frame = pd.DataFrame({
        "time_seconds": [0.0, 300.0, 600.0],
        "T41": [3.0, 3.1, 3.2],
        "T42": [3.0, 3.1, 3.2],
        "f_PIPE142": [-0.10, -0.20, 0.30],
        "f_PIPE143": [-0.40, 0.50, -0.60],
        "s_P78": [1.0, 0.0, 1.0],
        "s_P79": [0.0, 1.0, 0.0],
    })
    build = build_cusum_config("anytown", frame, manifest, ANYTOWN_INP, dt_s=300.0)
    prepared = build.prepare(frame)

    assert build.config.tanks == ("T41", "T42")
    by_link = {(spec.tank, spec.link): spec for spec in build.feeders}
    assert set(by_link) == {("T41", "PIPE142"), ("T42", "PIPE143")}
    assert build.config.feeders["T41"] == (by_link[("T41", "PIPE142")].signed_column,)
    assert build.config.feeders["T42"] == (by_link[("T42", "PIPE143")].signed_column,)
    assert by_link[("T41", "PIPE142")].sign == 1
    assert by_link[("T42", "PIPE143")].sign == 1
    np.testing.assert_allclose(
        prepared[by_link[("T41", "PIPE142")].signed_column],
        frame["f_PIPE142"].to_numpy(),
    )
    assert build.config.actuators == {"T41": ("s_P78",), "T42": ("s_P79",)}
    assert build.config.area_m2["T41"] == pytest.approx(
        math.pi * 9.9532 ** 2 / 4.0
    )
    assert build.config.flow_to_m3ps == 1.0
    assert build.config.dt_s == 300.0
    assert build.config.dt_source == "fixed"
    assert build.config.offset_correction is False


@pytest.mark.skipif(not LTOWN_INP.is_file() or not LTOWN_MANIFEST.is_file(),
                    reason="L-Town resolved artifacts unavailable")
def test_ltown_t1_feeder_sign_follows_inp_orientation():
    manifest = json.loads(LTOWN_MANIFEST.read_text())
    frame = pd.DataFrame({
        "time_seconds": [0.0, 300.0, 600.0],
        "T1": [3.50, 3.51, 3.52],
        "f_p239": [0.003, 0.004, 0.005],
        "f_PUMP_1": [0.012, 0.013, 0.014],
        "s_PUMP_1": [1.0, 1.0, 1.0],
    })
    build = build_cusum_config("ltown", frame, manifest, LTOWN_INP, dt_s=300.0)
    prepared = build.prepare(frame)

    assert build.config.tanks == ("T1",)
    specs = {spec.link: spec for spec in build.feeders}
    assert set(specs) == {"p239", "PUMP_1"}
    assert specs["p239"].sign == -1
    assert specs["PUMP_1"].sign == 1
    np.testing.assert_allclose(
        prepared[specs["p239"].signed_column], -frame["f_p239"].to_numpy()
    )
    np.testing.assert_allclose(
        prepared[specs["PUMP_1"].signed_column], frame["f_PUMP_1"].to_numpy()
    )
    total = prepared[list(build.config.feeders["T1"])].sum(axis=1).to_numpy()
    np.testing.assert_allclose(total, -frame["f_p239"].to_numpy()
                               + frame["f_PUMP_1"].to_numpy())
    assert build.config.actuators == {"T1": ("s_PUMP_1",)}
    assert build.config.area_m2["T1"] == pytest.approx(
        math.pi * 15.4251 ** 2 / 4.0
    )


def test_non_cylindrical_tank_fails_explicitly(tmp_path):
    inp = tmp_path / "topology.inp"
    inp.write_text(
        "[TANKS]\n"
        " T1 10 3 0 6 12 0 CURVE1\n"
        "[PIPES]\n"
        " p1 T1 n1 10 100 100 0 Open\n"
        "[OPTIONS]\n"
        " UNITS CMH\n"
    )
    frame = pd.DataFrame({
        "time_seconds": [0.0, 300.0], "T1": [3.0, 3.1],
        "f_p1": [0.01, 0.01], "s_PUMP_1": [1.0, 1.0],
    })
    manifest = {"controls": [{"link": "PUMP_1", "sensor": "T1"}]}
    with pytest.raises(FullSignalError, match="not a cylinder"):
        build_cusum_config("ltown", frame, manifest, inp, dt_s=300.0)


def test_legacy_events_derive_classes_without_inventing_leak_labels(tmp_path):
    times = [0.0, 300.0, 600.0, 900.0, 1200.0]
    events = [
        {"kind": "sensor", "mode": "offset", "start": 300, "end": 1500,
         "target": "T1", "value": 0.75},
        {"kind": "actuator", "start": 300, "end": 900,
         "target": "PUMP_1", "value": "OPEN"},
        {"kind": "sensor", "mode": "freeze", "start": 600, "end": 1200,
         "target": "T1"},
    ]
    run_dir = tmp_path / "legacy_run"
    _write_bundle(run_dir, times=times,
                  label_rows=_labels(times, [False, True, True, True, False]),
                  events=events)

    run = load_full_signal_run(run_dir, "ltown")
    assert list(run.attack_class) == [
        "normal", "coordinated", "coordinated", "coordinated", "sensor_offset",
    ]
    assert "leak" not in run.attack_class
    assert all(cls in ATTACK_CLASS_TAXONOMY for cls in run.attack_class)
    assert list(run.labels["attack_class"]) == list(run.attack_class)
    assert list(run.y_true) == [0, 1, 1, 1, 0]


def test_incomplete_and_staging_runs_are_skipped_with_reasons(tmp_path):
    logic_root = tmp_path / "logic_attacks"
    anytown = logic_root / "anytown"
    anytown.mkdir(parents=True)
    times = [0.0, 300.0, 600.0, 900.0]
    _write_bundle(anytown / "nominal", times=times,
                  label_rows=_labels(times, [False] * len(times)))
    _write_bundle(
        anytown / "good_attack", times=times,
        label_rows=_labels(times, [False, True, True, False]),
        events=[{"kind": "sensor", "mode": "offset", "start": 300, "end": 900,
                 "target": "T1", "value": 0.5}],
    )
    (anytown / ".staging_attack").mkdir()
    missing = anytown / "missing_observed"
    _write_bundle(missing, times=times,
                  label_rows=_labels(times, [False] * len(times)),
                  with_observed=False)

    discovery = discover_full_signal_runs(
        "anytown", logic_root=logic_root,
        pdd_root=tmp_path / "pdd", include_full_signal_pdd=False,
    )

    assert discovery.nominal.name == "nominal"
    assert [run.name for run in discovery.attacks] == ["good_attack"]
    reasons = {item.path.name: item.reason for item in discovery.skipped}
    assert "hidden/staging" in reasons[".staging_attack"]
    assert "missing observed.csv" in reasons["missing_observed"]
