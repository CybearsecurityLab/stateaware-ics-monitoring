#!/usr/bin/env python3
"""Run the state-aware CUSUM baseline over the shared full-signal adapter.

This runner is intentionally thin: :mod:`dataio.full_signal` discovers the
complete bundles, splits the nominal chronologically and builds the per-tank
CUSUM configuration from the resolved INP.  The detector itself is the
existing :class:`baselines.state_aware_cusum.StateAwareCusumDetector` with
unchanged numerical semantics (``fit`` -> ``tune`` -> ``predict``).

Artifacts are written only under ``scripts/out/full_signal_cusum/<topology>``
when this script is executed.  It never writes into ``scripts/out/full_signal``,
``scripts/out/full_signal_pdd``, ``scripts/out/logic_attacks`` or any baseline
directory.

Usage
-----
    python scripts/run_full_signal_cusum.py --topology anytown ctown
    python scripts/run_full_signal_cusum.py --topology ltown --include-full-signal-pdd
    python scripts/run_full_signal_cusum.py --topology all --include-full-signal-pdd
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baselines.state_aware_cusum import (  # noqa: E402
    StateAwareCusumDetector,
    build_provenance,
    evaluate,
)
from dataio.full_signal import (  # noqa: E402
    DEFAULT_LOGIC_ROOT,
    DEFAULT_PDD_ROOT,
    CusumBuild,
    FullSignalError,
    FullSignalRun,
    NominalSplit,
    build_cusum_config,
    discover_full_signal_runs,
    split_nominal_run,
)

DEFAULT_OUT_ROOT = REPO_ROOT / "scripts" / "out" / "full_signal_cusum"
TOPOLOGIES = ("anytown", "ctown", "ltown")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


def _prediction_rows(
    *,
    scenario: str,
    split_name: str,
    run: FullSignalRun | None,
    frame: pd.DataFrame,
    y_true: Sequence[int],
    attack_class: Sequence[str],
    flags: np.ndarray,
    scores: np.ndarray,
    result: Any,
    tanks: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    times = pd.to_numeric(frame["time_seconds"], errors="coerce").to_numpy(dtype=float)
    for i in range(len(frame)):
        row: dict[str, Any] = {
            "scenario": scenario,
            "split": split_name,
            "row": i,
            "time_seconds": float(times[i]),
            "attack_class": str(attack_class[i]),
            "y_true": int(y_true[i]),
            "y_pred": int(flags[i]),
            "cusum_score": float(scores[i]),
        }
        if run is not None:
            row["run_id"] = run.run_id
            row["run_path"] = str(run.path)
        for tank in tanks:
            row[f"cusum_{tank}"] = float(result.cusum[tank][i])
            row[f"cusum_pre_reset_{tank}"] = float(result.cusum_pre_reset[tank][i])
            row[f"residual_{tank}"] = float(result.residuals[tank][i])
            row[f"centered_residual_{tank}"] = float(result.centered_residuals[tank][i])
            row[f"state_{tank}"] = str(result.states[tank][i])
            row[f"flag_{tank}"] = int(result.tank_flags[tank][i])
        rows.append(row)
    return rows


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")


def _scenario_inputs(run: FullSignalRun) -> dict[str, Any]:
    return {
        "name": run.name,
        "path": str(run.path),
        "run_id": run.run_id,
        "manifest_split": run.split,
        "rows": run.rows,
        "input_hashes": dict(sorted(run.input_hashes.items())),
    }


def run_topology(
    topology: str,
    *,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    logic_root: str | Path = DEFAULT_LOGIC_ROOT,
    pdd_root: str | Path = DEFAULT_PDD_ROOT,
    include_full_signal_pdd: bool = False,
    overwrite: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    """Discover, split, fit/tune/predict and export one topology."""
    out_dir = Path(out_root) / topology
    if write and out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FullSignalError(
            f"output directory already populated: {out_dir} (pass --overwrite)"
        )

    discovery = discover_full_signal_runs(
        topology,
        logic_root=logic_root,
        pdd_root=pdd_root,
        include_full_signal_pdd=include_full_signal_pdd,
    )
    nominal = discovery.nominal
    if nominal.inp_path is None:
        raise FullSignalError(f"{topology}: nominal run has no resolved INP")
    split: NominalSplit = split_nominal_run(nominal)
    build: CusumBuild = build_cusum_config(
        topology, split.train.frame, nominal.manifest, nominal.inp_path,
    )

    train = build.prepare(split.train.frame)
    validation = build.prepare(split.validation.frame)
    detector = StateAwareCusumDetector(build.config)
    detector.fit([train])
    detector.tune([validation])

    scenarios: list[tuple[str, str, FullSignalRun | None, pd.DataFrame,
                         np.ndarray, tuple[str, ...]]] = [
        ("nominal_test", "nominal_test", None, split.test.frame,
         split.test.y_true, split.test.attack_class),
    ]
    for run in discovery.attacks:
        scenarios.append((
            run.name, "evaluation", run, run.frame, run.y_true, run.attack_class,
        ))

    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    scenario_manifest: list[dict[str, Any]] = []
    for name, split_name, run, frame, y_true, attack_class in scenarios:
        prepared = build.prepare(frame)
        result = detector.predict(prepared)
        rows.extend(_prediction_rows(
            scenario=name,
            split_name=split_name,
            run=run,
            frame=prepared,
            y_true=y_true,
            attack_class=attack_class,
            flags=result.flags,
            scores=result.scores,
            result=result,
            tanks=build.config.tanks,
        ))
        metrics[name] = evaluate(y_true, result.flags, result.scores)
        scenario_manifest.append({
            "name": name,
            "split": split_name,
            "rows": int(len(prepared)),
            "attack_steps": int(np.asarray(y_true, dtype=int).sum()),
            "attack_classes": sorted(set(attack_class)),
            "run": None if run is None else _scenario_inputs(run),
        })

    data_sources = {
        "topology": topology,
        "nominal": _scenario_inputs(nominal),
        "attack_scenarios": [_scenario_inputs(run) for run in discovery.attacks],
        "scenarios": scenario_manifest,
        "skipped_runs": [item.to_dict() for item in discovery.skipped],
        "input_hashes": {
            "nominal": dict(sorted(nominal.input_hashes.items())),
            "attacks": {
                run.name: dict(sorted(run.input_hashes.items()))
                for run in discovery.attacks
            },
        },
        "split_manifest": dict(split.manifest),
        "split_hashes": {
            part.name: part.digest
            for part in (split.train, split.validation, split.test)
        },
        "cusum_mapping": build.to_dict(),
        "dt": {
            "seconds": float(build.dt_seconds),
            "source": "median of nominal time_seconds deltas",
            "config_dt_source": build.config.dt_source,
        },
        "include_full_signal_pdd": bool(include_full_signal_pdd),
    }
    provenance = build_provenance(
        build.config, detector.model, data_sources=data_sources,
    )

    summary = {
        "topology": topology,
        "out_dir": str(out_dir),
        "rows": {
            item["name"]: int(item["rows"]) for item in scenario_manifest
        },
        "metrics": metrics,
        "scenarios": [item["name"] for item in scenario_manifest],
        "skipped_runs": [item.to_dict() for item in discovery.skipped],
        "inp_path": str(build.inp.path),
        "inp_sha256": build.inp.sha256,
        "dt_seconds": float(build.dt_seconds),
    }

    if write:
        _write_json(out_dir / "config.json", build.config.to_dict())
        _write_json(out_dir / "provenance.json", provenance)
        _write_json(out_dir / "mapping.json", build.to_dict())
        _write_json(out_dir / "metrics.json", metrics)
        _write_json(out_dir / "splits.json", split.manifest)
        _write_json(out_dir / "input_hashes.json", data_sources["input_hashes"])
        _write_json(out_dir / "skipped_runs.json", data_sources["skipped_runs"])
        _write_json(out_dir / "summary.json", summary)
        _write_predictions(out_dir / "predictions.csv", rows)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--topology", nargs="+", default=["all"],
        choices=[*TOPOLOGIES, "all"],
        help="topology name(s) or 'all' (default: all)",
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--logic-root", type=Path, default=DEFAULT_LOGIC_ROOT)
    parser.add_argument("--pdd-root", type=Path, default=DEFAULT_PDD_ROOT)
    parser.add_argument(
        "--include-full-signal-pdd", action="store_true",
        help="also discover attack bundles under scripts/out/full_signal_pdd",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace an already populated topology output directory",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    topologies = list(TOPOLOGIES) if "all" in args.topology else list(dict.fromkeys(args.topology))
    for topology in topologies:
        summary = run_topology(
            topology,
            out_root=args.out_root,
            logic_root=args.logic_root,
            pdd_root=args.pdd_root,
            include_full_signal_pdd=args.include_full_signal_pdd,
            overwrite=args.overwrite,
        )
        print(
            f"{topology}: {len(summary['scenarios'])} scenarios, "
            f"{sum(summary['rows'].values())} rows -> {summary['out_dir']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
