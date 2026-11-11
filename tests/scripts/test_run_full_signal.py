import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_full_signal.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_full_signal", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_full_signal = load_script()


def write_topology(path: Path, duration: str = "00:16:40") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[TITLE]\nminimal\n\n[TIMES]\n"
        f" DURATION             {duration}\n"
        " HYDRAULIC TIMESTEP   0:00:01\n"
        " QUALITY TIMESTEP     0:05:00\n"
        " PATTERN TIMESTEP     1:00:00\n"
        " PATTERN START        0:00\n"
         " REPORT TIMESTEP      0:00:01\n"
         " REPORT START         0:00\n\n"
         "[OPTIONS]\n"
         " DEMAND MODEL         DDA\n"
         " QUALITY              NONE mg/L\n\n"
         "[CONTROLS]\n LINK P CLOSED IF NODE T ABOVE 3.9\n",
        encoding="utf-8",
    )
    return path


def test_ltown_existing_scenarios_match_artifact_semantics():
    nominal = run_full_signal.get_scenario("ltown", "nominal_two_day")
    feedback = run_full_signal.get_scenario("ltown", "feedback_two_day")

    assert nominal.run_id == "ltown-nominal-two-day"
    assert nominal.split == "validation"
    assert nominal.duration == 172_800
    assert nominal.scan_step == 300
    assert nominal.events == ()

    assert feedback.run_id == "ltown-feedback-two-day"
    assert feedback.split == "evaluation"
    assert feedback.duration == 172_800
    assert feedback.scan_step == 300
    assert run_full_signal.build_events(feedback) == [
        {"kind": "sensor", "target": "T1", "mode": "constant", "value": 0.0,
         "scope": "both", "start": 86_400, "end": 172_800}
    ]


def test_new_ltown_scenarios_are_distinct_and_start_on_second_day():
    names = ["t1_offset_48h", "t1_replay_48h", "pump1_dos_48h"]
    scenarios = [run_full_signal.get_scenario("ltown", name) for name in names]
    assert len({s.name for s in scenarios}) == len(names)
    assert len({s.run_id for s in scenarios}) == len(names)
    for scenario in scenarios:
        assert scenario.duration == 172_800
        assert scenario.scan_step == 300
        events = run_full_signal.build_events(scenario)
        assert events
        assert all(event["start"] == 86_400 for event in events)
        assert all(event["end"] == 172_800 for event in events)

    offset = run_full_signal.build_events(scenarios[0])
    assert offset == [{"kind": "sensor", "target": "T1", "mode": "offset",
                       "value": 1.0, "scope": "both", "start": 86_400,
                       "end": 172_800}]

    replay = run_full_signal.build_events(scenarios[1])
    assert replay[0]["kind"] == "sensor"
    assert replay[0]["target"] == "T1"
    assert replay[0]["mode"] == "replay"
    assert replay[0]["scope"] == "both"
    period = replay[0]["period"]
    step = scenarios[1].scan_step
    for t in range(86_400, 172_800, step):
        assert t - period * step < 86_400

    dos = run_full_signal.build_events(scenarios[2])
    assert dos == [{"kind": "actuator", "target": "PUMP_1",
                    "value": "CLOSED", "start": 86_400, "end": 172_800}]


def test_existing_anytown_and_ctown_configurations():
    anytown = run_full_signal.scenarios_for("anytown")
    assert set(anytown) == {
        "nominal", "dos", "mitm", "adaptive_bounded_offset",
        "adaptive_coordinated_freeze",
    }
    assert anytown["nominal"].duration == 3_000
    assert anytown["nominal"].split == "calibration"
    assert run_full_signal.build_events(anytown["dos"]) == [
        {"kind": "sensor", "target": "T41", "mode": "freeze",
         "scope": "both", "start": 649, "end": 794}
    ]
    assert run_full_signal.build_events(anytown["mitm"])[0]["value"] == 7.0
    coordinated = run_full_signal.build_events(anytown["adaptive_coordinated_freeze"])
    assert [event["kind"] for event in coordinated] == ["actuator", "sensor"]

    ctown = run_full_signal.scenarios_for("ctown")
    assert set(ctown) == {"nominal_short", "pump_activation", "replay_pump"}
    assert ctown["nominal_short"].split == "calibration"
    replay = run_full_signal.build_events(ctown["replay_pump"])
    assert [event.get("target") for event in replay] == ["PU10", "PU11", "T7"]
    assert replay[-1]["period"] == 10


def test_build_events_returns_independent_copies():
    scenario = run_full_signal.get_scenario("ltown", "t1_offset_48h")
    first = run_full_signal.build_events(scenario)
    first[0]["value"] = 99.0
    assert run_full_signal.build_events(scenario)[0]["value"] == 1.0
    assert "event_id" not in first[0]
    assert "applied" not in first[0]


def test_labels_come_only_from_events_and_physical_impact():
    policy = run_full_signal.LABEL_POLICY
    assert policy["battledim_leak_labels_used"] is False
    assert "physical_impact" in policy
    for scenarios in run_full_signal.SCENARIOS.values():
        for scenario in scenarios.values():
            for event in run_full_signal.build_events(scenario):
                assert "leak" not in json.dumps(event).lower()


def test_resolve_input_prefers_topology_artifact_and_reports_fallback(tmp_path):
    input_root = tmp_path / "in"
    out_root = tmp_path / "out"
    preferred = write_topology(input_root / "ltown" / "topology.inp")
    path, fallback = run_full_signal.resolve_input(
        "ltown", input_root=input_root, out_root=out_root
    )
    assert path == preferred and fallback is False

    empty_input = tmp_path / "empty-in"
    fallback_path, used_fallback = run_full_signal.resolve_input(
        "ltown", input_root=empty_input, out_root=out_root
    )
    assert fallback_path == run_full_signal.LTOWN_FALLBACK_INP
    assert used_fallback is True

    with pytest.raises(run_full_signal.ScenarioError, match="explicit input"):
        run_full_signal.resolve_input(
            "ltown", input_root=input_root, out_root=out_root,
            explicit=tmp_path / "missing.inp",
        )


def test_patch_timing_preserves_other_sections_and_sets_window(tmp_path):
    scenario = run_full_signal.get_scenario("ltown", "t1_replay_48h")
    patched = run_full_signal.patch_timing_text(
        write_topology(tmp_path / "topology.inp").read_text(encoding="utf-8"),
        scenario,
    )
    assert re.search(r"^DURATION\s+48:00:00$", patched, re.MULTILINE)
    assert re.search(r"^HYDRAULIC TIMESTEP\s+00:05:00$", patched, re.MULTILINE)
    assert re.search(r"^REPORT TIMESTEP\s+00:05:00$", patched, re.MULTILINE)
    assert re.search(r"^REPORT START\s+00:00:00$", patched, re.MULTILINE)
    assert re.search(r"^\s*QUALITY TIMESTEP\s+0:05:00$", patched, re.MULTILINE)
    assert re.search(r"^\s*PATTERN TIMESTEP\s+1:00:00$", patched, re.MULTILINE)
    assert "[CONTROLS]\n LINK P CLOSED IF NODE T ABOVE 3.9" in patched


def test_ltown_fallback_is_the_real_benchmark_input():
    fallback = run_full_signal.LTOWN_FALLBACK_INP
    assert fallback == REPO_ROOT / "data" / "ltown" / "raw" / "L-TOWN_Real.inp"
    assert fallback.name == "L-TOWN_Real.inp"
    assert fallback.is_file()


def test_ltown_scenarios_share_pdd_and_other_topologies_are_unchanged():
    ltown = run_full_signal.scenarios_for("ltown")
    assert set(ltown) == {
        "nominal_two_day", "feedback_two_day", "t1_offset_48h",
        "t1_replay_48h", "pump1_dos_48h",
    }
    for scenario in ltown.values():
        demand = scenario.demand_model
        assert demand is not None
        assert demand.model == "PDA"
        assert demand.minimum_pressure == 0.0
        assert demand.required_pressure == 0.07
        assert demand.pressure_exponent == 0.5

    for topology in ("anytown", "ctown"):
        assert all(
            scenario.demand_model is None
            for scenario in run_full_signal.scenarios_for(topology).values()
        )


def test_patch_demand_text_sets_pdd_and_preserves_other_options(tmp_path):
    raw = write_topology(tmp_path / "topology.inp").read_text(encoding="utf-8")
    demand = run_full_signal.get_scenario(
        "ltown", "nominal_two_day"
    ).demand_model
    patched = run_full_signal.patch_demand_text(raw, demand)

    assert re.search(r"^DEMAND MODEL\s+PDA$", patched, re.MULTILINE)
    assert re.search(r"^MINIMUM PRESSURE\s+0\.0$", patched, re.MULTILINE)
    assert re.search(r"^REQUIRED PRESSURE\s+0\.07$", patched, re.MULTILINE)
    assert re.search(r"^PRESSURE EXPONENT\s+0\.5$", patched, re.MULTILINE)
    assert re.search(r"^\s*QUALITY\s+NONE mg/L$", patched, re.MULTILINE)
    assert "[CONTROLS]\n LINK P CLOSED IF NODE T ABOVE 3.9" in patched
    assert run_full_signal.patch_demand_text(patched, demand) == patched
    assert run_full_signal.patch_demand_text(raw, None) == raw


def test_actual_ltown_resolved_configuration_loads_with_wntr(tmp_path):
    wntr = pytest.importorskip("wntr")
    raw_path = run_full_signal.LTOWN_FALLBACK_INP
    if not raw_path.is_file():
        pytest.skip(f"missing raw L-Town input: {raw_path}")
    scenario = run_full_signal.get_scenario("ltown", "nominal_two_day")
    raw_text = raw_path.read_text(encoding="utf-8", errors="strict")
    resolved = run_full_signal.patch_demand_text(
        raw_text, scenario.demand_model
    )
    resolved_path = tmp_path / "ltown" / "topology.inp"
    resolved_path.parent.mkdir(parents=True)
    resolved_path.write_text(resolved, encoding="utf-8")

    network = wntr.network.WaterNetworkModel(str(resolved_path))
    hydraulic = network.options.hydraulic
    assert hydraulic.demand_model in ("PDA", "PDD")
    assert hydraulic.minimum_pressure == pytest.approx(0.0)
    assert hydraulic.required_pressure == pytest.approx(0.07)
    assert hydraulic.pressure_exponent == pytest.approx(0.5)
    assert network.num_junctions == 782


def test_publish_resolved_topology_reuses_identical_and_refuses_different(tmp_path):
    path = tmp_path / "ltown" / "topology.inp"
    path.parent.mkdir(parents=True)
    path.write_text("same\n", encoding="utf-8")

    assert run_full_signal.publish_resolved_topology(
        path, "same\n", overwrite=False
    ) == path
    with pytest.raises(run_full_signal.ScenarioExistsError, match="--overwrite"):
        run_full_signal.publish_resolved_topology(
            path, "different\n", overwrite=False
        )
    assert path.read_text(encoding="utf-8") == "same\n"

    run_full_signal.publish_resolved_topology(path, "different\n", overwrite=True)
    assert path.read_text(encoding="utf-8") == "different\n"

    directory = tmp_path / "not-a-file" / "topology.inp"
    directory.mkdir(parents=True)
    with pytest.raises(run_full_signal.ScenarioError, match="not a file"):
        run_full_signal.publish_resolved_topology(
            directory, "x\n", overwrite=False
        )


def test_run_scenarios_publishes_pdd_topology_and_provenance(tmp_path):
    input_root = tmp_path / "in"
    input_path = write_topology(input_root / "ltown" / "topology.inp")
    out_root = tmp_path / "out"
    record = {}

    outputs = run_full_signal.run_scenarios(
        "ltown",
        ["nominal_two_day", "t1_offset_48h"],
        out_root=out_root,
        input_root=input_root,
        runner=_fake_runner(record),
        writer=_fake_writer(record),
    )

    assert outputs == [
        out_root / "ltown" / "nominal_two_day",
        out_root / "ltown" / "t1_offset_48h",
    ]
    topology = out_root / "ltown" / "topology.inp"
    topology_text = topology.read_text(encoding="utf-8")
    assert re.search(r"^DEMAND MODEL\s+PDA$", topology_text, re.MULTILINE)
    assert re.search(r"^MINIMUM PRESSURE\s+0\.0$", topology_text, re.MULTILINE)
    assert re.search(r"^REQUIRED PRESSURE\s+0\.07$", topology_text, re.MULTILINE)
    assert re.search(r"^PRESSURE EXPONENT\s+0\.5$", topology_text, re.MULTILINE)

    metadata = json.loads(
        (out_root / "ltown" / "nominal_two_day" / "scenario.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["demand_model"] == {
        "model": "PDA",
        "minimum_pressure": 0.0,
        "required_pressure": 0.07,
        "pressure_exponent": 0.5,
    }
    assert metadata["input"]["source_sha256"] == hashlib.sha256(
        input_path.read_bytes()
    ).hexdigest()
    assert metadata["resolved_topology"]["path"] == str(topology)
    assert metadata["resolved_topology"]["sha256"] == hashlib.sha256(
        topology.read_bytes()
    ).hexdigest()
    assert metadata["reproduction"]["legacy_dda_byte_identical"] is False


def test_run_scenarios_refuses_different_resolved_topology(tmp_path):
    input_root = tmp_path / "in"
    write_topology(input_root / "ltown" / "topology.inp")
    out_root = tmp_path / "out"
    topology = out_root / "ltown" / "topology.inp"
    topology.parent.mkdir(parents=True)
    topology.write_text("stale DDA topology\n", encoding="utf-8")
    record = {}

    with pytest.raises(run_full_signal.ScenarioExistsError, match="topology"):
        run_full_signal.run_scenarios(
            "ltown", ["nominal_two_day"],
            out_root=out_root, input_root=input_root,
            runner=_fake_runner(record), writer=_fake_writer(record),
        )
    assert record == {}
    assert topology.read_text(encoding="utf-8") == "stale DDA topology\n"
    assert not (out_root / "ltown" / "nominal_two_day").exists()

    run_full_signal.run_scenarios(
        "ltown", ["nominal_two_day"],
        out_root=out_root, input_root=input_root, overwrite=True,
        runner=_fake_runner(record), writer=_fake_writer(record),
    )
    assert re.search(
        r"^DEMAND MODEL\s+PDA$",
        topology.read_text(encoding="utf-8"), re.MULTILINE,
    )


def test_output_directory_refuses_to_overwrite_without_flag(tmp_path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(run_full_signal.ScenarioExistsError, match="--overwrite"):
        run_full_signal.check_output_dir(out_dir, overwrite=False)
    run_full_signal.check_output_dir(out_dir, overwrite=True)

    empty = tmp_path / "empty"
    empty.mkdir()
    run_full_signal.check_output_dir(empty, overwrite=False)

    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(run_full_signal.ScenarioError, match="not a directory"):
        run_full_signal.check_output_dir(file_path, overwrite=False)


def _fake_runner(record):
    def runner(spec, events, *, plcs_path=None):
        record["spec"] = spec
        record["events"] = events
        record["plcs_path"] = plcs_path
        record["patched"] = Path(spec.inp_path).read_text(encoding="utf-8")
        return object()

    return runner


def _fake_writer(record):
    def writer(bundle, path):
        record["bundle"] = bundle
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "manifest.json").write_text('{"schema_version": 1}\n', encoding="utf-8")

    return writer


def test_execute_scenario_uses_config_and_writes_metadata_atomically(tmp_path):
    input_path = write_topology(tmp_path / "in" / "ltown" / "topology.inp")
    out_dir = tmp_path / "out" / "ltown" / "t1_replay_48h"
    record = {}
    scenario = run_full_signal.get_scenario("ltown", "t1_replay_48h")

    result = run_full_signal.execute_scenario(
        scenario,
        topology="ltown",
        input_path=input_path,
        out_dir=out_dir,
        overwrite=False,
        used_fallback=False,
        runner=_fake_runner(record),
        writer=_fake_writer(record),
    )

    assert result == out_dir
    assert record["spec"].run_id == "ltown-t1-replay-48h"
    assert record["spec"].split == "evaluation"
    assert record["spec"].seed == 42
    assert record["events"][0]["mode"] == "replay"
    assert record["events"][0]["start"] == 86_400
    assert "48:00:00" in record["patched"]
    assert "00:05:00" in record["patched"]
    metadata = json.loads((out_dir / "scenario.json").read_text(encoding="utf-8"))
    assert metadata["scenario"] == "t1_replay_48h"
    assert metadata["labels"]["battledim_leak_labels_used"] is False
    assert metadata["events"][0]["period"] == 288
    assert metadata["input"]["used_fallback"] is False
    assert list(out_dir.iterdir())


def test_execute_scenario_does_not_run_when_output_exists(tmp_path):
    input_path = write_topology(tmp_path / "in" / "ltown" / "topology.inp")
    out_dir = tmp_path / "out" / "ltown" / "pump1_dos_48h"
    out_dir.mkdir(parents=True)
    (out_dir / "old.csv").write_text("old\n", encoding="utf-8")
    record = {}
    scenario = run_full_signal.get_scenario("ltown", "pump1_dos_48h")

    with pytest.raises(run_full_signal.ScenarioExistsError):
        run_full_signal.execute_scenario(
            scenario,
            topology="ltown",
            input_path=input_path,
            out_dir=out_dir,
            overwrite=False,
            runner=_fake_runner(record),
            writer=_fake_writer(record),
        )
    assert record == {}
    assert (out_dir / "old.csv").exists()


def test_execute_scenario_overwrite_replaces_stale_files(tmp_path):
    input_path = write_topology(tmp_path / "in" / "ltown" / "topology.inp")
    out_dir = tmp_path / "out" / "ltown" / "t1_offset_48h"
    out_dir.mkdir(parents=True)
    (out_dir / "stale.txt").write_text("stale\n", encoding="utf-8")
    record = {}
    scenario = run_full_signal.get_scenario("ltown", "t1_offset_48h")

    run_full_signal.execute_scenario(
        scenario,
        topology="ltown",
        input_path=input_path,
        out_dir=out_dir,
        overwrite=True,
        runner=_fake_runner(record),
        writer=_fake_writer(record),
    )
    assert not (out_dir / "stale.txt").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "scenario.json").exists()


def test_run_scenarios_preflights_all_outputs_before_simulating(tmp_path):
    input_path = write_topology(tmp_path / "in" / "ltown" / "topology.inp")
    out_root = tmp_path / "out"
    existing = out_root / "ltown" / "pump1_dos_48h"
    existing.mkdir(parents=True)
    (existing / "manifest.json").write_text("{}", encoding="utf-8")
    record = {}

    with pytest.raises(run_full_signal.ScenarioExistsError):
        run_full_signal.run_scenarios(
            "ltown",
            ["t1_offset_48h", "pump1_dos_48h"],
            out_root=out_root,
            input_root=tmp_path / "in",
            runner=_fake_runner(record),
            writer=_fake_writer(record),
        )
    assert record == {}
    assert not (out_root / "ltown" / "t1_offset_48h").exists()


def test_cli_dry_run_and_safety_without_simulation(tmp_path, capsys):
    input_root = tmp_path / "in"
    write_topology(input_root / "ltown" / "topology.inp")
    out_root = tmp_path / "out"

    assert run_full_signal.main([
        "--topology", "ltown", "--scenario", "t1_offset_48h",
        "--input-root", str(input_root), "--out-root", str(out_root),
        "--dry-run",
    ]) == 0
    captured = capsys.readouterr().out
    assert "t1_offset_48h" in captured
    assert "events=1" in captured
    assert "demand=PDD(PDA min=0 req=0.07 exp=0.5)" in captured
    assert "resolved topology:" in captured
    assert "topology.inp" in captured
    assert not out_root.exists()

    assert run_full_signal.main(["--list-scenarios"]) == 0
    listing = capsys.readouterr().out
    assert "t1_replay_48h" in listing
    assert "demand=PDD(PDA min=0 req=0.07 exp=0.5)" in listing

    assert run_full_signal.main(["--scenario", "nominal_two_day"]) == 2
    assert "topology" in capsys.readouterr().err

    assert run_full_signal.main([
        "--topology", "ltown", "--scenario", "not_a_scenario", "--dry-run",
        "--input-root", str(input_root), "--out-root", str(out_root),
    ]) == 2
    assert "unknown scenario" in capsys.readouterr().err
