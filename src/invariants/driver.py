"""Stage-invariants driver: orchestrate Φ mining for one topology."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

import numpy as np

from .manifest import build_manifest
from .mine import MinerConfig, mine_state
from .model import InvariantsError, MinedRule
from .rule_eval import frame_violation_counts
from .state_label import label_frame, load_gfsm_components
from .threshold import select_violation_threshold


def _load_gfsm_json(gfsm_dir: Path, topology: str) -> dict[str, Any]:
    p = gfsm_dir / f"{topology}.gfsm.json"
    if not p.exists():
        raise InvariantsError(f"gfsm json not found: {p}")
    return json.loads(p.read_text())


def _load_gfsm_manifest(gfsm_dir: Path, topology: str) -> tuple[str, str]:
    p = gfsm_dir / f"{topology}_gfsm_manifest.json"
    if not p.exists():
        raise InvariantsError(f"gfsm manifest not found: {p}")
    return p.read_text(), p.name


def _load_calibration(
    data_root: Path, topology: str
) -> tuple[pd.DataFrame, str, str]:
    ds = data_root / topology / "dataset"
    man = ds / "dataset_manifest.yaml"
    if not man.exists():
        raise InvariantsError(f"dataset manifest not found: {man}")
    man_text = man.read_text()
    cfg = yaml.safe_load(man_text) or {}
    cal_files = (cfg.get("files") or {}).get("calibration") or []
    if not cal_files:
        raise InvariantsError(f"{topology}: no calibration files listed")
    frames = [pd.read_csv(ds / rel) for rel in cal_files]
    df = pd.concat(frames, ignore_index=True)
    return df, man_text, man.name


def mine_topology(
    *,
    topology: str,
    data_root: Path,
    gfsm_dir: Path,
    out_dir: Path,
    fb_to_col: dict[tuple[str, str], str],
    min_observations: int = 50,
    max_evals: int = 5000,
    seed: int = 42,
    feature_cols: list[str] | None = None,
    fp_budget: float = 0.01,
    keep_going: bool = True,
    training_frames: list[pd.DataFrame] | None = None,
    validation_frames: list[pd.DataFrame] | None = None,
    split_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mine the state-aware invariant mapping Φ for one topology.

    Leak-free calibration
    ---------------------
    By default the rules are mined on the whole clean calibration set and the
    violation threshold K is calibrated on those same rows — the threshold
    therefore sees the very data the rules were fitted on, which makes the
    reported clean false-positive rate of K optimistic.

    Passing ``training_frames`` and ``validation_frames`` splits the
    calibration set explicitly:

    * rules are mined on ``training_frames`` only — no validation row ever
      reaches the miner, so no validation information can leak into the
      rules themselves;
    * the violation threshold K is selected on ``validation_frames`` only.

    Both arguments must be given together (or neither): fitting on the full
    calibration set while selecting K on a validation subset would leak the
    validation rows into the rules.

    The exact split (scenario identities, row counts, digests) is recorded in
    the manifest under ``calibration``; pass the manifest produced by
    :func:`dataio.splits.split_calibration_frames` as ``split_manifest`` to
    embed it verbatim.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (training_frames is None) != (validation_frames is None):
        raise InvariantsError(
            "training_frames and validation_frames must be provided "
            "together (or neither): fitting on the full calibration set "
            "while selecting K on a validation subset would leak validation "
            "rows into the mined rules"
        )

    gfsm = _load_gfsm_json(Path(gfsm_dir), topology)
    gfsm_man_text, gfsm_man_name = _load_gfsm_manifest(
        Path(gfsm_dir), topology)
    cal_df, ds_man_text, ds_man_name = _load_calibration(
        Path(data_root), topology)

    components = load_gfsm_components(gfsm)

    if training_frames is None:
        # Backward-compatible mode: mine and calibrate on the whole
        # calibration set (threshold-on-training; kept for reproducibility
        # of already-published artifacts).
        fit_df = cal_df
        thr_df = cal_df
        threshold_source = "calibration"
        calibration_mode = "legacy_full_calibration"
    else:
        fit_df = pd.concat(list(training_frames), ignore_index=True)
        thr_df = pd.concat(list(validation_frames), ignore_index=True)
        threshold_source = "validation"
        calibration_mode = "train_validation_split"

    fit_labels = label_frame(fit_df, components, fb_to_col)

    if feature_cols is None:
        # Feature selection looks at the FITTING rows only (never the
        # validation rows), so the column set cannot leak validation data.
        skip = set(fb_to_col.values())
        numeric = fit_df.select_dtypes(include="number").columns
        feature_cols = [c for c in numeric if c not in skip]
    feature_cols = list(feature_cols)

    cfg = MinerConfig(max_evals=max_evals, seed=seed)
    states: dict[str, dict[str, Any]] = {}
    states_summary: dict[str, dict[str, Any]] = {}
    all_ok = True

    for state_id in sorted(set(fit_labels)):
        slice_df = fit_df.loc[fit_labels == state_id]
        nobs = int(len(slice_df))
        if nobs < min_observations:
            states[state_id] = {"observations": nobs,
                                "status": "insufficient_data", "rules": []}
            states_summary[state_id] = {"observations": nobs, "rules": 0,
                                        "status": "insufficient_data"}
            continue
        try:
            rules: list[MinedRule] = mine_state(slice_df, feature_cols, cfg)
            states[state_id] = {"observations": nobs, "status": "ok",
                                "rules": [r.to_dict() for r in rules]}
            states_summary[state_id] = {"observations": nobs,
                                        "rules": len(rules), "status": "ok"}
        except Exception as exc:  # noqa: BLE001 - mining is best-effort
            all_ok = False
            states[state_id] = {"observations": nobs,
                                "status": "mining_error",
                                "errors": [repr(exc)], "rules": []}
            states_summary[state_id] = {"observations": nobs, "rules": 0,
                                        "status": "mining_error"}
            if not keep_going:
                break

    # --- violation-threshold calibration ---
    # The threshold is selected on the THRESHOLD rows only: the validation
    # split when one was supplied, otherwise the whole calibration set
    # (legacy mode).  Rules are never re-fitted here.
    rules_by_state = {
        s: (e.get("rules") or []) for s, e in states.items()
    }
    thr_labels = label_frame(thr_df, components, fb_to_col)
    labels_arr = (thr_labels.to_numpy()
                  if hasattr(thr_labels, "to_numpy")
                  else np.asarray(list(thr_labels)))
    cal_counts = np.zeros(len(thr_df), dtype=int)
    # A validation row whose composite state was never mined (the state did
    # not occur in the fitting rows) counts as state-absent — exactly the
    # runtime signal the monitor raises for an unknown state.
    cal_absent = ~np.isin(labels_arr, list(rules_by_state.keys()))
    for s, rs in rules_by_state.items():
        if not rs:
            continue
        sel = labels_arr == s
        if sel.any():
            cal_counts[sel] = frame_violation_counts(thr_df[sel], rs)
    k, k_note = select_violation_threshold(
        cal_counts, cal_absent, fp_budget)

    manifest = build_manifest(
        topology=topology,
        gfsm_manifest_name=gfsm_man_name, gfsm_manifest_text=gfsm_man_text,
        dataset_manifest_name=ds_man_name, dataset_manifest_text=ds_man_text,
        niaarm_cfg={"max_evals": max_evals, "confidence_min": 0.7,
                    "support_min": 0.1, "min_observations": min_observations,
                    "seed": seed, "algorithm": "ParticleSwarmAlgorithm"},
        states_summary=states_summary,
        all_ok=all_ok,
    )
    # Explicit calibration provenance: which rows the rules were fitted on,
    # which rows selected the threshold, and the exact feature columns used
    # (so a local/global comparison can be run on identical columns).
    calibration_provenance: dict[str, Any] = {
        "mode": calibration_mode,
        "threshold_source": threshold_source,
        "fit_rows": int(len(fit_df)),
        "threshold_rows": int(len(thr_df)),
        "feature_cols": list(feature_cols),
    }
    if split_manifest is not None:
        calibration_provenance["split"] = split_manifest
    manifest["calibration"] = calibration_provenance
    phi = {
        "schema": "invariants/v1",
        "topology": topology,
        "gfsm_manifest": gfsm_man_name,
        "gfsm_manifest_sha256": manifest["gfsm_manifest_sha256"],
        "dataset_manifest": ds_man_name,
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "niaarm": manifest["niaarm"],
        "generated_at": manifest["generated_at"],
        "violation_threshold": int(k),
        "violation_fp_budget": float(fp_budget),
        "calibration": calibration_provenance,
        "states": states,
    }
    if k_note is not None:
        phi["violation_threshold_note"] = k_note
    (out_dir / f"{topology}_phi.json").write_text(
        json.dumps(phi, indent=2, sort_keys=True, default=str) + "\n"
    )
    (out_dir / f"{topology}_invariants_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    return manifest
