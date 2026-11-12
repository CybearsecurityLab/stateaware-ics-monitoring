#!/usr/bin/env python3
"""Run C-Town QuantMiner with resumable logs and honest progress estimates."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = Path("/home/resbears/stateaware-quantminer-bitset")
OUT = REPO / "scripts/out/full_signal_sain_local_quantminer_bitset/ctown"
RUNNER_LOG = REPO / "scripts/out/run_logs/sain_local_quantminer_bitset_ctown.log"
STATUS_PATH = Path("/home/resbears/quantminer_ctown_status.json")
SUPERVISOR_PID = Path("/home/resbears/quantminer_ctown_supervisor.pid")
INTERVAL = 30
TOTAL_STATES = 13


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "estimating"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s"


def state_counts() -> tuple[int, int]:
    return (
        len(list(OUT.glob("quantminer/*/state_*/rules.csv"))),
        TOTAL_STATES,
    )


def active_states() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", "QuantMinerCli"], capture_output=True, text=True,
    ).stdout
    prefix = "full_signal_sain_local_quantminer_bitset/ctown/quantminer"
    states = set()
    for line in result.splitlines():
        if prefix not in line or "training.csv" not in line:
            continue
        for token in line.split():
            if token.endswith("/training.csv"):
                states.add(Path(token).parent.name)
    return sorted(states)


def available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    return -1.0


def write_status(started: float, runner: subprocess.Popen, final: dict | None = None) -> None:
    completed, total = state_counts()
    elapsed = time.monotonic() - started
    eta = elapsed / completed * (total - completed) if completed else None
    active = active_states()
    payload = {
        "schema": "quantminer-supervisor/v1",
        "topology": "ctown",
        "supervisor_pid": os.getpid(),
        "runner_pid": runner.pid,
        "updated_utc": utc_now(),
        "elapsed_seconds": int(elapsed),
        "total_states": total,
        "completed_states": completed,
        "percent_complete": round(100 * completed / total, 2) if total else 0.0,
        "active_states": active,
        "eta_seconds": int(eta) if eta is not None else None,
        "eta_policy": "mean elapsed time per completed state; unavailable before first completion",
        "memory_available_gib": round(available_gib(), 2),
        "runner_log": str(RUNNER_LOG),
        "supervisor_log": "/home/resbears/quantminer_ctown_supervisor.log",
    }
    if final:
        payload.update(final)
    STATUS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    bar = "#" * int(20 * completed / total) if total else ""
    bar = f"{bar:20s}"
    eta_text = duration(eta)
    print(
        f"PROGRESS [{bar}] {completed}/{total} states ({100 * completed / total:.1f}%) | "
        f"elapsed {duration(elapsed)} | active {','.join(active) or 'starting'} | "
        f"ETA {eta_text} | MemAvailable {available_gib():.1f} GiB",
        flush=True,
    )


class ExistingRunner:
    def __init__(self, pid: int):
        self.pid = pid
        self.returncode = 0

    def poll(self):
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return self.returncode
        return None


def main() -> int:
    started = time.monotonic()
    SUPERVISOR_PID.write_text(f"{os.getpid()}\n")
    monitor_pid = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--runner-pid" else None
    env = dict(os.environ)
    env.update({
        "STATEAWARE_REPO_ROOT": str(REPO),
        "PYTHONPATH": f"{REPO}:{REPO / 'src'}",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "QUANTMINER_BITSET": "1",
        "QUANTMINER_BITSET_PROFILE": "1",
        "JAVA_TOOL_OPTIONS": "-Xmx3g -Xss512k -XX:MaxDirectMemorySize=512m --enable-native-access=ALL-UNNAMED",
    })
    command = [
        "/home/resbears/stateaware-ics-monitoring/.venv/bin/python",
        "scripts/run_full_signal_sain_local.py",
        "--topology", "ctown", "--engine", "quantminer", "--jobs", "2",
        "--out-root", "scripts/out/full_signal_sain_local_quantminer_bitset",
    ]
    if monitor_pid is not None:
        runner = ExistingRunner(monitor_pid)
    else:
        with RUNNER_LOG.open("a") as log:
            log.write(f"\n=== supervisor restart {utc_now()} ===\n")
            log.flush()
            runner = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)

    def stop(*_):
        try:
            os.killpg(runner.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        write_status(started, runner, {"status": "stopped"})
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"supervisor_pid={os.getpid()} runner_pid={runner.pid}", flush=True)
    print(f"runner_log={RUNNER_LOG}", flush=True)
    print(f"status_json={STATUS_PATH}", flush=True)
    while runner.poll() is None:
        write_status(started, runner)
        time.sleep(INTERVAL)
    code = runner.returncode
    metrics_path = OUT / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    status = metrics.get("status", "failed" if code else "missing_metrics")
    write_status(started, runner, {
        "status": status,
        "runner_exit_code": code,
        "metrics_path": str(metrics_path),
        "finished_utc": utc_now(),
    })
    print(f"FINAL status={status} exit_code={code} metrics={metrics_path}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
