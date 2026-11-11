"""CLI entry point for ics-monitor (hybrid STL + GFSM + Φ invariants)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stl.profiles import PROFILES

from .driver import run_topology
from .model import MonitorError


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ics-monitor",
        description="Hybrid ICS monitor: STL + GFSM + Φ invariants (alarm-union fusion).")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--topology", choices=sorted(PROFILES))
    grp.add_argument("--all", action="store_true")
    p.add_argument("--out", type=Path, default=None,
                   help="output dir (default: <data-root>/generated/<topo>/monitor)")
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--fusion", choices=["or", "intersection", "conjunction"],
                   default="or",
                   help="hybrid fusion rule (default: or [alarm union]; "
                        "intersection/conjunction retains conjunction ablation)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    topologies = sorted(PROFILES) if args.all else [args.topology]
    had_error = False
    overall_ok = True
    for topology in topologies:
        try:
            manifest = run_topology(
                topology=topology, data_root=args.data_root,
                out_dir=args.out if not args.all else None,
                fusion=args.fusion, jobs=args.jobs)
        except MonitorError as exc:
            print(f"error: {exc}", file=sys.stderr)
            if not args.all:
                return 2
            had_error = True
            continue
        if not manifest.get("all_ok", True):
            overall_ok = False
        if args.verbose:
            print(f"{topology}: detectors={manifest.get('detectors')}",
                  file=sys.stderr)
    if had_error:
        return 2
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
