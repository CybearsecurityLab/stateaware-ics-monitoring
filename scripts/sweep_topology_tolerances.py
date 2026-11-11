#!/usr/bin/env python3
"""Sweep topology-STL tolerances over stored full-signal observations."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from invariants.full_signal_phi import FullSignalPhiDetector

try:
    from scripts import export_topology_predictions as export
except ModuleNotFoundError:
    import export_topology_predictions as export

PERCENTILES = (95.0, 99.0, 99.9)
SCALES = (0.0, 1.0, 10.0)
CLOSED_FLOW_EPSILON = 1e-6
OUT_ROOT = export.REPO_ROOT / "scripts" / "out" / "tolerance_sweep"
PHI_TRAIN_FRACTION = 0.64


def grid(percentiles=PERCENTILES, scales=SCALES):
    return list(itertools.product(percentiles, scales))


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_complete_scenario(directory):
    """Return True for a written (non-staging) scenario directory."""
    if directory.name.startswith("."):
        return False
    observed, labels = directory / "observed.csv", directory / "labels.csv"
    return (
        observed.is_file() and labels.is_file()
        and observed.stat().st_size > 0 and labels.stat().st_size > 0
    )


def _load_scenario(directory):
    observed_path, labels_path = directory / "observed.csv", directory / "labels.csv"
    observed = export._read_observed(observed_path)
    labels = pd.read_csv(labels_path)
    if "attack_active" not in labels.columns:
        raise export.ExportError(f"{labels_path} has no 'attack_active' column")
    if len(observed) != len(labels):
        raise export.ExportError(f"{directory}: label and observation lengths differ")
    order = observed["time_seconds"].sort_values(kind="stable").index
    frame = observed.loc[order].reset_index(drop=True)
    if "time_seconds" in labels.columns:
        if (observed["time_seconds"].duplicated().any()
                or labels["time_seconds"].duplicated().any()):
            raise export.ExportError(f"{directory}: duplicate observation or label times")
        try:
            labels = labels.set_index("time_seconds").loc[frame["time_seconds"]]
        except KeyError as exc:
            raise export.ExportError(f"{directory}: observation and label times differ") from exc
        labels = labels.reset_index()
    else:
        labels = labels.loc[order].reset_index(drop=True)
    truth = labels["attack_active"].astype(int).to_numpy()
    return frame, truth, (observed_path, labels_path), labels


def prepare(topology, full_root, data_root, out_root, *, nominal_dir=None,
            nominal_name=None, inp_path=None, skip_incomplete=False):
    """Load independent scenarios and fit the GFSM once per topology.

    The optional ``nominal_dir`` / ``nominal_name`` let the calibration
    nominal live outside ``full_root/topology`` (for example the read-only
    L-Town PDD nominal) and ``inp_path`` overrides the resolved network.
    ``skip_incomplete`` drops hidden/staging or partially written scenario
    directories instead of failing.  Every parameter is optional, so existing
    callers are unchanged.
    """
    config_nominal = export.TOPOLOGY_CONFIG[topology]["nominal_scenario"]
    inp = Path(inp_path) if inp_path is not None else export._resolve_inp(
        topology, full_root, data_root)
    root = Path(full_root) / topology
    if not root.is_dir():
        raise export.ExportError(f"{topology}: no scenario directory at {root}")
    try:
        dirs = export._scenario_dirs(full_root, topology)
    except export.ExportError:
        if not skip_incomplete:
            raise
        dirs = []
    if skip_incomplete:
        dirs = [directory for directory in dirs if _is_complete_scenario(directory)]
    scenarios = {}
    label_frames = {}
    sources = {}
    inputs = {inp}
    for directory in dirs:
        try:
            frame, truth, files, aligned_labels = _load_scenario(directory)
        except (OSError, ValueError, KeyError, export.ExportError):
            if not skip_incomplete:
                raise
            continue
        scenarios[directory.name] = (frame, truth)
        label_frames[directory.name] = aligned_labels
        sources[directory.name] = directory
        inputs.update(files)
    if nominal_dir is not None:
        nominal_path = Path(nominal_dir)
        nominal = nominal_name or nominal_path.name
        known = sources.get(nominal)
        if known is not None and known.resolve() != nominal_path.resolve():
            raise export.ExportError(
                f"{topology}: nominal name {nominal!r} already refers to {known}")
        if nominal not in scenarios:
            frame, truth, files, aligned_labels = _load_scenario(nominal_path)
            scenarios[nominal] = (frame, truth)
            label_frames[nominal] = aligned_labels
            sources[nominal] = nominal_path
            inputs.update(files)
    else:
        nominal = config_nominal
        if nominal not in scenarios:
            raise export.ExportError(f"{topology}: nominal scenario missing")
    nominal_frame = scenarios[nominal][0]
    n_cal = int(len(nominal_frame) * export.DEFAULT_CALIBRATION_FRACTION)
    if n_cal < 2 or n_cal == len(nominal_frame):
        raise export.ExportError(f"{topology}: invalid nominal split {n_cal}")
    if topology in export.EXPLICIT_COLUMN_MAPPING:
        gfsm_dir, _ = export._build_ltown_surrogate(out_root=out_root, inp_path=inp)
    else:
        gfsm_dir = data_root / "generated" / topology / "gfsm"
        manifest_dir = data_root / topology / "dataset"
        if manifest_dir.is_dir():
            inputs.update(p for p in manifest_dir.rglob("*") if p.is_file() and p.suffix in (".yaml", ".yml", ".json"))
    inputs.update(p for p in gfsm_dir.rglob("*") if p.is_file())
    gfsm_json = gfsm_dir / f"{topology}.gfsm.json"
    components = export._gfsm_components(json.loads(gfsm_json.read_text()))
    mapping, mapping_report = export._resolve_gfsm_columns(
        topology=topology, gfsm_dir=gfsm_dir, data_root=data_root,
        components=components, frame_columns=list(nominal_frame.columns),
    )
    gfsm = export.GfsmAnomalyDetector(gfsm_dir=gfsm_dir, topology=topology, fb_to_col=mapping).fit([])
    phi_calibration = nominal_frame.iloc[:n_cal].copy()
    phi_train_end = int(len(nominal_frame) * PHI_TRAIN_FRACTION)
    phi = FullSignalPhiDetector.fit(
        topology=topology,
        training_frame=phi_calibration.iloc[:phi_train_end].copy(),
        validation_frame=phi_calibration.iloc[phi_train_end:].copy(),
        components=components,
        fb_to_col=mapping,
    )
    evaluated = {}
    for name, (frame, truth) in scenarios.items():
        if name == nominal:
            frame = frame.iloc[n_cal:].reset_index(drop=True)
            truth = truth[n_cal:]
        flags = np.asarray(gfsm.predict(frame).flags, dtype=int)
        evaluated[name] = (frame, truth, flags)
    return {
        "network": export.load_network(inp), "calibration": nominal_frame.iloc[:n_cal].copy(),
        "scenarios": evaluated, "label_frames": label_frames,
        "nominal": nominal, "mapping": mapping_report,
        "components": components, "fb_to_col": mapping, "phi": phi,
        "inputs": {str(p): _hash(p) for p in sorted(inputs)},
    }


def evaluate(prepared, percentile, scale):
    additive = {unit: margin * scale for unit, margin in export.DEFAULT_ADDITIVE.items()}
    detector = export.TopologyStlDetector(prepared["network"], {
        "percentile": percentile, "additive": additive,
        "closed_flow_epsilon": CLOSED_FLOW_EPSILON,
        "unknown_policy": export.DEFAULT_UNKNOWN_POLICY,
    }).fit([prepared["calibration"].copy()])
    frames = []
    for name, (frame, truth, gfsm) in prepared["scenarios"].items():
        stl, stl_score, unknown = export._stl_predict(detector, frame)
        phi_result = prepared["phi"].predict(frame)
        phi = np.asarray(phi_result.flags, dtype=int)
        phi_score = np.asarray(phi_result.scores, dtype=float)
        combined = np.logical_or.reduce((stl, gfsm, phi)).astype(int)
        frames.append(pd.DataFrame({
            "scenario": name, "y_true": truth, "gfsm": gfsm, "stl": stl,
            "phi": phi, "phi_score": phi_score,
            "stl_score": stl_score,
            "combined": combined,
            "combined_score": np.maximum.reduce((stl_score, gfsm, phi_score)),
            "unknown": unknown,
        }))
    artifact = detector.to_artifact()
    artifact["phi"] = prepared["phi"].model
    return pd.concat(frames, ignore_index=True), artifact


def metrics(frame, nominal_mask):
    result = {"rows": len(frame), "attack_rows": int(frame.y_true.sum()),
              "unknown_rows": int((frame.unknown > 0).sum()),
              "unknown_residuals": int(frame.unknown.sum())}
    for arm in ("gfsm", "phi", "stl", "combined"):
        scores = frame[arm] if arm == "gfsm" else frame[f"{arm}_score"]
        values = export.detection_metrics(frame.y_true, frame[arm], scores)
        result[arm] = {"mcc": values["mcc"], "balanced_accuracy": values["balanced_accuracy"],
                       "auprc": values["aucpr"],
                       "nominal_false_alarms": int(((frame[arm] == 1) & (frame.y_true == 0) & nominal_mask).sum())}
    return result


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies", nargs="+", choices=list(export.TOPOLOGY_CONFIG), default=list(export.TOPOLOGY_CONFIG))
    parser.add_argument("--full-signal-root", type=Path, default=export.REPO_ROOT / "scripts/out/full_signal")
    parser.add_argument("--data-root", type=Path, default=export.REPO_ROOT / "data")
    args = parser.parse_args(argv)
    if len(set(args.topologies)) != len(args.topologies):
        parser.error("duplicate topologies")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    prepared = {name: prepare(name, args.full_signal_root, args.data_root, OUT_ROOT) for name in args.topologies}
    summary = {"schema": "topology-tolerance-sweep/v1", "configuration": {
        "percentiles": PERCENTILES, "additive_scales": SCALES,
        "base_additive": export.DEFAULT_ADDITIVE, "calibration_fraction": export.DEFAULT_CALIBRATION_FRACTION,
        "unknown_policy": export.DEFAULT_UNKNOWN_POLICY, "closed_flow_epsilon_m3_per_s": CLOSED_FLOW_EPSILON,
        "closed_flow_rule": "calibrated closed tolerance is max(percentile + scaled additive, fixed floor); unseen closed regime uses fixed floor",
        "aggregation": "pooled rows; independent scenario predictions", "scores": "STL violated residual count; combined max(STL score, GFSM flag, Phi flag); Phi is global NiaARM"},
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "code_sha256": {str(p): _hash(p) for p in (
            Path(__file__), Path(export.__file__), export.REPO_ROOT / "src/stl/topology_detector.py",
            export.REPO_ROOT / "src/stl/metrics.py", export.REPO_ROOT / "src/monitor/gfsm_detector.py", export.REPO_ROOT / "src/invariants/full_signal_phi.py")},
        "inputs": {name: data["inputs"] for name, data in prepared.items()},
        "gfsm_mapping": {name: data["mapping"] for name, data in prepared.items()}, "configurations": []}
    csv_rows = []
    for percentile, scale in grid():
        config = {"percentile": percentile, "additive_scale": scale, "topologies": {}}
        pooled = []
        for name, data in prepared.items():
            frame, artifact = evaluate(data, percentile, scale)
            nominal_mask = frame.scenario == data["nominal"]
            topology_result = {"calibration_rows": len(data["calibration"]), "nominal_scenario": data["nominal"],
                               "artifact": artifact, "aggregate": metrics(frame, nominal_mask), "scenarios": {}}
            for scenario, group in frame.groupby("scenario", sort=True):
                topology_result["scenarios"][scenario] = metrics(group, group.scenario == data["nominal"])
            config["topologies"][name] = topology_result
            frame = frame.copy()
            frame["is_nominal"] = nominal_mask
            pooled.append(frame)
        all_rows = pd.concat(pooled, ignore_index=True)
        config["aggregate"] = metrics(all_rows, all_rows.is_nominal)
        summary["configurations"].append(config)
        for topology, item in [("ALL", {"aggregate": config["aggregate"], "scenarios": {}}), *config["topologies"].items()]:
            for scenario, result in [("ALL", item["aggregate"]), *item["scenarios"].items()]:
                for arm in ("gfsm", "phi", "stl", "combined"):
                    csv_rows.append({"topology": topology, "scenario": scenario, "percentile": percentile,
                                     "additive_scale": scale, "arm": arm,
                                     **{k: result[k] for k in ("rows", "attack_rows", "unknown_rows", "unknown_residuals")}, **result[arm]})
    (OUT_ROOT / "summary.json").write_text(json.dumps(_clean(summary), indent=2, sort_keys=True, allow_nan=False) + "\n")
    pd.DataFrame(csv_rows).to_csv(OUT_ROOT / "metrics.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
