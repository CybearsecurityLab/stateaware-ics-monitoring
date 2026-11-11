"""Focused tests for reproducible supervised attack-episode planning."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_supervised_episodes.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_supervised_episodes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


episodes = load_script()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_fixture(tmp_path, *, topology="anytown", demand_model="DDA"):
    step = 300
    scans = 30
    duration = (scans - 1) * step
    times = list(range(0, duration + 1, step))
    t1 = [6, 9, 9, 4, 9, 9, 4, 9, 9, 4, 9, 9, 4, 9, 9,
          4, 9, 9, 4, 9, 9, 4, 9, 9, 4, 9, 9, 4, 9, 9]
    t2 = [6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4,
          6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4, 6, 4, 4]
    frame = pd.DataFrame({
        "time_seconds": times,
        "T1": t1,
        "T2": t2,
        "T3": [1.0] * scans,
        "speed_PU1": [1.0] * scans,
        "speed_PU2": [1.0] * scans,
    })
    nominal = tmp_path / "nominal"
    nominal.mkdir()
    frame.to_csv(nominal / "plant_truth.csv", index=False,
                 float_format="%.17g", lineterminator="\n")
    controls = [
        {"id": "control 1", "link": "PU1", "sensor": "T1", "relation": "gt",
         "threshold": 8.0, "value": 0, "owner": "PLC"},
        {"id": "control 2", "link": "PU1", "sensor": "T1", "relation": "lt",
         "threshold": 5.0, "value": 1, "owner": "PLC"},
        {"id": "control 3", "link": "PU2", "sensor": "T2", "relation": "lt",
         "threshold": 5.0, "value": 1, "owner": "PLC"},
        {"id": "control 4", "link": "V2", "sensor": "T3", "relation": "lt",
         "threshold": 0.5, "value": 1, "owner": "PLC"},
    ]
    manifest = {
        "schema_version": 1,
        "controls": controls,
        "expected_columns": [
            "time_seconds", "T1", "T2", "T3", "speed_PU1", "speed_PU2"],
        "timesteps": {"duration": duration},
        "effective_timing": {"scan_timestep": step},
        "hydraulic_options": {"demand_model": demand_model},
    }
    (nominal / "manifest.json").write_text(json.dumps(manifest))
    source = tmp_path / "source.inp"
    source.write_text("[JUNCTIONS]\n")
    (nominal / "nominal.json").write_text(json.dumps({"source_inp": str(source)}))
    inp = tmp_path / "topology.inp"
    inp.write_text("[JUNCTIONS]\n")
    logic_root = tmp_path / "logic"
    held_dir = logic_root / "anytown"
    held_dir.mkdir(parents=True)
    held = {
        "guard_windows": [
            {"control_id": "control 1", "link": "PU1", "sensor": "T1",
             "time_seconds": 300},
            {"control_id": "control 2", "link": "PU1", "sensor": "T1",
             "time_seconds": 900},
            {"control_id": "control 3", "link": "PU2", "sensor": "T2",
             "time_seconds": 300},
        ],
        "scenarios": [
            {"events": [{"kind": "actuator", "target": "PU1", "start": 300}]},
        ],
    }
    (held_dir / "plan.json").write_text(json.dumps(held))
    spec = episodes.TopologySpec(
        topology, duration, step, nominal, inp,
        "PDA" if topology == "ltown" else None)
    return SimpleNamespace(
        spec=spec, frame=frame, manifest=manifest, nominal=nominal,
        inp=inp, source=source, logic_root=logic_root, held=held,
    )


def build(fixture, **kwargs):
    options = {"logic_root": fixture.logic_root, "replay_period_scans": 2}
    options.update(kwargs)
    return episodes.build_plan(
        fixture.spec.name, fixture.frame, fixture.manifest, fixture.spec, **options)


def held_pairs(fixture):
    pairs = set()
    for window in fixture.held["guard_windows"]:
        for key in ("control_id", "link", "sensor"):
            pairs.add((window[key], window["time_seconds"]))
    for scenario in fixture.held["scenarios"]:
        for event in scenario["events"]:
            pairs.add((event["target"], event["start"]))
    return pairs


def test_plan_has_disjoint_splits_and_full_class_coverage(tmp_path):
    fixture = make_fixture(tmp_path)
    plan = build(fixture)
    scenarios = plan["scenarios"]
    assert [item["attack_class"] for item in scenarios] == list(episodes.ALL_CLASSES)

    training = [item for item in scenarios if item["split"] == "training"]
    validation = [item for item in scenarios if item["split"] == "validation"]
    assert {item["attack_class"] for item in training} == set(episodes.TRAIN_CLASSES)
    assert {item["attack_class"] for item in validation} == set(episodes.VALIDATION_CLASSES)
    assert {item["seed"] for item in training} == {episodes.TRAIN_SEED}
    assert {item["seed"] for item in validation} == {episodes.VALIDATION_SEED}

    train_keys = {(item["selection"]["window"]["control_id"],
                   item["selection"]["window"]["time_seconds"]) for item in training}
    validation_keys = {(item["selection"]["window"]["control_id"],
                        item["selection"]["window"]["time_seconds"]) for item in validation}
    assert len(train_keys) == len(training)
    assert len(validation_keys) == len(validation)
    assert train_keys.isdisjoint(validation_keys)

    excluded = held_pairs(fixture)
    held_times = {window["time_seconds"] for window in fixture.held["guard_windows"]}
    for item in scenarios:
        window = item["selection"]["window"]
        assert (window["control_id"], window["time_seconds"]) not in excluded
        assert window["time_seconds"] not in held_times
        assert item["events"] and all("attack_class" in event for event in item["events"])
        assert item["selection"]["detector_results_consulted"] is False
        assert item["labels"]["inherited_leak_labels"] is False

    by_class = {item["attack_class"]: item["selection"]["window"] for item in scenarios}
    for attack_class in episodes.REQUIRE_GUARD_TRUE:
        assert by_class[attack_class]["guard_state"] is True
    assert by_class["actuator_dos"]["forced_status"] == "CLOSED"
    assert by_class["sensor_replay"]["time_seconds"] >= 2 * fixture.spec.scan_step

    assert plan["provenance"]["fitting_performed"] is False
    assert plan["provenance"]["evaluation_outputs_consulted"] is False
    assert plan["provenance"]["detector_scores_consulted"] is False
    assert plan["provenance"]["new_nominal_simulation"] is False
    assert plan["held_out"]["available"] is True
    assert plan["held_out"]["excluded_time_count"] >= len(held_times)
    assert plan["nominal"]["resolved_inp_sha256"] == sha256(fixture.inp)
    assert plan["nominal"]["source_sha256"] == sha256(fixture.source)


def test_plan_and_manifests_are_deterministic(tmp_path):
    fixture = make_fixture(tmp_path)
    first = build(fixture)
    second = build(fixture)
    assert first == second

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    episodes.write_plan(out_a, "anytown", first)
    episodes.write_plan(out_b, "anytown", first)
    for name in ("plan.json", "manifest.json"):
        assert (out_a / "anytown" / name).read_bytes() == (
            out_b / "anytown" / name).read_bytes()
    episodes.write_plan(out_a, "anytown", first)


def test_execution_uses_split_seed_and_keeps_inputs_immutable(tmp_path):
    fixture = make_fixture(tmp_path)
    plan = build(fixture)
    out = tmp_path / "out"
    episodes.write_plan(out, "anytown", plan)
    watched = [
        fixture.nominal / "plant_truth.csv",
        fixture.nominal / "manifest.json",
        fixture.nominal / "nominal.json",
        fixture.inp,
        fixture.source,
        fixture.logic_root / "anytown" / "plan.json",
    ]
    before = {path: sha256(path) for path in watched}
    calls = []

    def runner(run_spec, events):
        assert (out / "anytown" / "plan.json").is_file()
        assert (out / "anytown" / "manifest.json").is_file()
        calls.append((run_spec, events))
        return object()

    def writer(bundle, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "observed.csv").write_text("time_seconds\n0\n")

    def factory(inp, run_id, seed, split):
        return SimpleNamespace(inp_path=inp, run_id=run_id, seed=seed, split=split)

    outputs = episodes.execute_scenarios(
        "anytown", fixture.spec, plan, out_root=out, runner=runner,
        writer=writer, run_spec_factory=factory)
    assert len(outputs) == len(plan["scenarios"]) == len(calls)

    for (run_spec, events), scenario in zip(calls, plan["scenarios"]):
        assert run_spec.inp_path == fixture.spec.inp_path
        assert run_spec.seed == scenario["seed"]
        assert run_spec.split == scenario["split"]
        assert events == scenario["events"]
        payload = json.loads(
            (out / "anytown" / scenario["dir"] / "scenario.json").read_text())
        assert payload["split"] == scenario["split"]
        assert payload["seed"] == scenario["seed"]
        assert payload["attack_class"] == scenario["attack_class"]
        assert payload["events"] == scenario["events"]
        assert payload["provenance"]["plan_written_before_execution"] is True
        assert payload["provenance"]["evaluation_outputs_consulted"] is False

    after = {path: sha256(path) for path in watched}
    assert before == after
    assert not (fixture.nominal / "training").exists()
    assert not (fixture.logic_root / "anytown" / "training").exists()


def test_no_fitting_or_evaluation_metric_reads(tmp_path):
    text = SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        "logic_attack_predictions", "sklearn", "torch", "joblib", "pickle",
        ".fit(",
    ):
        assert forbidden not in text

    fixture = make_fixture(tmp_path)
    plan = build(fixture)
    assert plan["provenance"]["fitting_performed"] is False
    assert plan["provenance"]["evaluation_outputs_consulted"] is False
    assert plan["provenance"]["detector_scores_consulted"] is False
    assert all(item["selection"]["detector_results_consulted"] is False
               for item in plan["scenarios"])


def test_missing_nominal_and_held_out_plan_fail_explicitly(tmp_path):
    with pytest.raises(episodes.SupervisedEpisodeError, match="incomplete"):
        episodes.load_nominal(tmp_path / "missing")

    fixture = make_fixture(tmp_path)
    with pytest.raises(episodes.SupervisedEpisodeError, match="held-out plan is required"):
        episodes.build_plan(
            "anytown", fixture.frame, fixture.manifest, fixture.spec,
            logic_root=tmp_path / "absent", replay_period_scans=2)


def test_ltown_requires_real_pdd_nominal(tmp_path):
    fixture = make_fixture(tmp_path, topology="ltown", demand_model="DDA")
    with pytest.raises(episodes.SupervisedEpisodeError, match="Real/PDD"):
        episodes.validate_nominal(
            "ltown", fixture.frame, fixture.manifest, fixture.spec)
