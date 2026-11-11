"""Runtime benchmark for the topology-driven STL detector.

The benchmark compares the optimised detector (``stl.topology_detector``)
against the slow reference implementation
(``tests/stl/reference_topology_detector.py``) on the network artifacts of
Anytown, C-Town and L-Town.

For every topology two sample sources are used:

``simulated``
    A physically consistent sample stream produced by EPANET (through WNTR)
    from the artifact INP file.  It carries every channel the compiled
    templates need, so all five template families are really evaluated.

``artifact``
    The dataset files shipped with the artifacts (``.csv``).  Their column
    names do not follow the detector's channel convention, so every template
    is reported as *missing*; the case is reported to show the cost of the
    missing-data path on real files.

Usage::

    python scripts/bench_topology_detector.py --data-root <path> --out runtime.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
import tracemalloc
import warnings

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from hydraulics.loader import load_network
from stl.topology_detector import TopologyStlDetector
from tests.stl.reference_topology_detector import ReferenceTopologyStlDetector

TOPOLOGIES = ("anytown", "ctown", "ltown")

INP_FILES = {
    "anytown": ("anytown", "anytown.inp"),
    "ctown": ("ctown", "ctown.inp"),
    "ltown": ("ltown", os.path.join("raw", "L-TOWN.inp")),
}

#: Status codes reported by EPANET for a link.
STATUS = {0: "CLOSED", 1: "OPEN", 2: "ACTIVE"}


def simulate_frame(inp_path: Path, network, duration_h: float = 48.0, step_h: float = 1.0):
    """Return a frame of observed channels simulated from *inp_path*."""
    import wntr  # noqa: PLC0415 - optional dependency, imported lazily

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wn = wntr.network.WaterNetworkModel(str(inp_path))
        wn.options.time.duration = duration_h
        wn.options.time.report_timestep = step_h
        wn.options.time.hydraulic_timestep = step_h
        wn.options.time.pattern_timestep = step_h
        simulator = wntr.sim.EpanetSimulator(wn)
        with tempfile.TemporaryDirectory() as work:
            results = simulator.run_sim(file_prefix=os.path.join(work, "run"))

    times = [float(stamp) * 3600.0 for stamp in results.link["flowrate"].index]
    data: dict[str, list[float]] = {}
    for link in network.links():
        values = results.link["flowrate"][link.id].to_numpy(dtype=float)
        data[f"f_{link.id}"] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        status = results.link["status"][link.id].to_numpy()
        data[f"s_{link.id}"] = [STATUS.get(int(code), "OPEN") for code in status]
    for junction in network.junctions:
        values = results.node["pressure"][junction.id].to_numpy(dtype=float)
        data[junction.id] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    for tank in network.tanks:
        values = results.node["head"][tank.id].to_numpy(dtype=float)
        data[tank.id] = np.nan_to_num(values - tank.elevation, nan=0.0, posinf=0.0, neginf=0.0)
    for pump in network.pumps:
        data[f"speed_{pump.id}"] = [1.0] * len(times)
    for valve in network.valves:
        setting = results.link["setting"][valve.id].to_numpy(dtype=float)
        data[f"setting_{valve.id}"] = np.nan_to_num(setting, nan=0.0, posinf=0.0, neginf=0.0)
    frame = pd.DataFrame(data, index=pd.Index(times, dtype=float))
    frame.index.name = "time_seconds"
    return frame


def artifact_frames(data_root: Path, topology: str):
    """Return the dataset frames shipped with the artifact (if any)."""
    dataset = data_root / topology / "dataset"
    frames = []
    if topology == "ctown":
        for name in ("calibration/calibration.csv", "evaluation/test.csv"):
            path = dataset / name
            if path.is_file():
                frames.append((f"artifact:{name}", pd.read_csv(path)))
    elif topology == "anytown":
        path = dataset / "calibration" / "cal_0000.csv"
        if path.is_file():
            frames.append((f"artifact:{path.name}", pd.read_csv(path)))
    else:
        path = dataset / "calibration.csv"
        if path.is_file():
            frames.append((f"artifact:{path.name}", pd.read_csv(path)))
    return frames


def _time_predict(detector, frame: pd.DataFrame, events: int, repeats: int):
    """Return (median ms per event, samples) of ``predict_details``."""
    subset = frame.iloc[:events]
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        detector.predict_details(subset)
        durations.append((time.perf_counter() - start) * 1000.0 / max(len(subset), 1))
    return statistics.median(durations), durations


def _peak_memory(detector, frame: pd.DataFrame, events: int):
    """Return the peak traced memory (MiB) of one ``predict_details`` call."""
    subset = frame.iloc[:events]
    tracemalloc.start()
    try:
        detector.predict_details(subset)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / (1024.0 * 1024.0)


def _parity(fast, slow, frame: pd.DataFrame, events: int):
    """Return True when both detectors agree on every evaluated sample."""
    subset = frame.iloc[:events]
    fast_rows = fast.predict_details(subset)
    slow_rows = slow.predict_details(subset)
    if len(fast_rows) != len(slow_rows):
        return False
    for fast_row, slow_row in zip(fast_rows, slow_rows):
        if list(fast_row) != list(slow_row):
            return False
        for key, slow_value in slow_row.items():
            fast_value = fast_row[key]
            if fast_value != slow_value:
                return False
            residual = fast_value.get("residual")
            if isinstance(residual, float) and repr(residual) != repr(slow_value["residual"]):
                return False
    return True


def benchmark_topology(data_root: Path, topology: str, events: int, repeats: int):
    """Benchmark one topology and return its report entry."""
    folder, name = INP_FILES[topology]
    inp_path = data_root / folder / name
    network = load_network(inp_path)
    report: dict = {
        "inp": str(inp_path),
        "network": {
            "junctions": len(network.junctions),
            "pipes": len(network.pipes),
            "tanks": len(network.tanks),
            "pumps": len(network.pumps),
            "valves": len(network.valves),
            "reservoirs": len(network.reservoirs),
        },
        "cases": [],
    }

    frame = simulate_frame(inp_path, network)
    report["events_available"] = int(len(frame))

    fast = TopologyStlDetector(network).fit([frame])
    slow = ReferenceTopologyStlDetector(network).fit([frame])
    report["tolerances"] = len(fast.tolerances)

    cases = [("simulated", frame)] + artifact_frames(data_root, topology)
    for label, case_frame in cases:
        available = min(events, len(case_frame))
        fast_ms, fast_runs = _time_predict(fast, case_frame, available, repeats)
        slow_ms, slow_runs = _time_predict(slow, case_frame, available, repeats)
        fast_peak = _peak_memory(fast, case_frame, available)
        slow_peak = _peak_memory(slow, case_frame, available)
        report["cases"].append({
            "source": label,
            "events": int(available),
            "reference_ms_per_event": round(slow_ms, 4),
            "optimised_ms_per_event": round(fast_ms, 4),
            "speedup": round(slow_ms / fast_ms, 2) if fast_ms else None,
            "reference_peak_mib": round(slow_peak, 2),
            "optimised_peak_mib": round(fast_peak, 2),
            "parity_identical": _parity(fast, slow, case_frame, available),
            "reference_runs_ms": [round(value, 3) for value in slow_runs],
            "optimised_runs_ms": [round(value, 3) for value in fast_runs],
        })
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--topologies", nargs="*", default=list(TOPOLOGIES))
    parser.add_argument("--events", type=int, default=48, help="samples per timing run")
    parser.add_argument("--repeats", type=int, default=3, help="timing repetitions")
    parser.add_argument("--out", type=Path, default=REPO / "runtime.json")
    args = parser.parse_args(argv)

    data_root = args.data_root.expanduser().resolve()
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "data_root": str(data_root),
        "topologies": {},
    }
    for topology in args.topologies:
        print(f"[bench] {topology} ...", flush=True)
        report["topologies"][topology] = benchmark_topology(
            data_root, topology, args.events, args.repeats
        )
        for case in report["topologies"][topology]["cases"]:
            print(
                f"  {case['source']}: reference {case['reference_ms_per_event']:.3f} ms/event, "
                f"optimised {case['optimised_ms_per_event']:.3f} ms/event, "
                f"speedup {case['speedup']}x, parity {case['parity_identical']}",
                flush=True,
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[bench] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
