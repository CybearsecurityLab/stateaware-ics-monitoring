#!/usr/bin/env python3
"""Build the DIMVA-revision fact pack from the committed experimental artifacts.

Every number in the output is derived from a file that exists on disk:

* metrics are **recomputed** with :func:`stl.metrics.detection_metrics`
  whenever the artifact stores per-row predictions *and* the ground truth is
  reachable (the normalised evaluation CSVs under ``data/<topology>/dataset``);
* otherwise the artifact-reported metrics are carried over verbatim and are
  flagged with ``metric_source="artifact_reported"`` (never silently
  substituted with a recomputed value);
* a missing artifact yields ``status="unavailable"`` with an explicit
  ``reason`` -- never a number.

The builder reads only; it never re-runs an experiment.

Usage::

    python scripts/build_fact_pack.py [--out scripts/out/fact_pack]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stl.metrics import detection_metrics  # noqa: E402

SCHEMA = "fact_pack/v1"
BUILDER_VERSION = "1.0.0"

#: Metric keys every normalised metric block carries (when defined).
METRIC_KEYS = (
    "precision", "recall", "f1", "mcc", "balanced_accuracy",
    "unadjusted_balanced_accuracy", "aucpr", "aucroc",
    "tp", "fp", "tn", "fn",
)

#: Artefact-independent description of how every metric is produced.
METRIC_DEFINITIONS = {
    "backend": "stl.metrics.detection_metrics",
    "positive_class": "attack (label 1)",
    "aucpr": "sklearn.metrics.average_precision_score(y_true, scores) on the "
             "attack-positive class",
    "aucroc": "sklearn.metrics.roc_auc_score(y_true, scores)",
    "balanced_accuracy": "sklearn.metrics.balanced_accuracy_score -- reported "
                         "unadjusted (no re-balancing of the score threshold)",
    "unadjusted_balanced_accuracy": "alias of balanced_accuracy; the value is "
                                    "never re-tuned or re-balanced",
    "mcc": "sklearn.metrics.matthews_corrcoef",
    "coarse_aucpr": "when a detector exposes no continuous score its binary "
                    "0/1 vector is passed as the score input, so the reported "
                    "average precision is a coarse rule-match AP, not a "
                    "ranking-based AP",
}

TOPOLOGIES = ("anytown", "ctown")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars/NaN/Inf into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if obj is None or isinstance(obj, (str, int)):
        return obj
    return str(obj)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Registry:
    """Records the digest of every artifact the builder consumed."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[str, dict[str, Any]] = {}

    def rel(self, path: Path) -> str:
        path = Path(path)
        try:
            return str(path.resolve().relative_to(self.root))
        except ValueError:
            return str(path)

    def ref(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        if not path.is_file():
            return {"path": self.rel(path), "status": "missing"}
        entry = {
            "path": self.rel(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        self.files[entry["path"]] = entry
        return entry


def unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    """Uniform 'no number here' record."""
    out: dict[str, Any] = {"status": "unavailable", "reason": reason}
    out.update(extra)
    return out


def compute_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    scores: Iterable[float] | None = None,
    *,
    score_kind: str = "average_precision_score",
) -> dict[str, Any]:
    """Normalised metric block produced by the repo's metric backend."""
    yt = np.asarray(list(y_true), dtype=int)
    yp = np.asarray(list(y_pred), dtype=int)
    sc = np.asarray(list(scores), dtype=float) if scores is not None \
        else yp.astype(float)
    raw = detection_metrics(yt, yp, sc)
    out: dict[str, Any] = {}
    for key in METRIC_KEYS:
        out[key] = raw.get(key)
    out["unadjusted_balanced_accuracy"] = raw.get("balanced_accuracy")
    out["aucpr_kind"] = score_kind
    out["n"] = int(len(yt))
    out["n_positive"] = int(yt.sum())
    out["n_predicted_positive"] = int(yp.sum())
    return out


def normalise_artifact_metrics(
    raw: dict[str, Any], *, n: int | None = None,
    n_positive: int | None = None,
) -> dict[str, Any]:
    """Map an artifact-reported metric block onto the canonical key set."""
    out: dict[str, Any] = {}
    for key in ("precision", "recall", "f1", "mcc", "balanced_accuracy",
                "tp", "fp", "tn", "fn"):
        out[key] = raw.get(key)
    out["unadjusted_balanced_accuracy"] = raw.get(
        "unadjusted_balanced_accuracy", raw.get("balanced_accuracy"))
    out["aucpr"] = raw.get("aucpr", raw.get("attack_average_precision",
                                            raw.get("average_precision")))
    out["aucroc"] = raw.get("aucroc", raw.get("roc_auc"))
    out["aucpr_kind"] = raw.get("average_precision_kind", "artifact_reported")
    out["n"] = raw.get("n", n)
    out["n_positive"] = raw.get("n_positive", n_positive)
    out["n_predicted_positive"] = raw.get("predicted_steps")
    return out


STATUS_MARKERS = ("preliminary", "smoke", "pilot", "tentative", "exploratory")
STATUS_KEYS = ("status", "stage", "label", "labels", "maturity", "run_status",
               "note", "notes", "qualification")


def collect_status_labels(
    roots: Iterable[Path], registry: Registry, *, max_bytes: int = 2 << 20,
    exclude_dirs: Iterable[Path] = (),
) -> dict[str, Any]:
    """Find explicit preliminary/smoke style labels in the artifacts.

    The labels are copied verbatim from the artifacts; the builder never adds
    a maturity label of its own. Directories in ``exclude_dirs`` (typically
    the output directory of a previous run) are skipped so the output stays
    deterministic across repeated runs.
    """
    entries: list[dict[str, Any]] = []
    excluded = [Path(d).resolve() for d in exclude_dirs]

    def _walk(node: Any, key_path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                _walk(value, f"{key_path}.{key}" if key_path else str(key))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                _walk(value, f"{key_path}[{i}]")
        elif isinstance(node, str):
            leaf = key_path.split(".")[-1].split("[")[0]
            lowered = node.lower()
            if leaf in STATUS_KEYS and any(m in lowered for m in STATUS_MARKERS):
                entries.append({"key": key_path, "value": node})

    for root_dir in roots:
        root_dir = Path(root_dir)
        if not root_dir.is_dir():
            continue
        for path in sorted(root_dir.rglob("*.json")):
            resolved = path.resolve()
            if any(excluded_path in resolved.parents
                   or resolved == excluded_path
                   for excluded_path in excluded):
                continue
            try:
                if path.stat().st_size > max_bytes:
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            before = len(entries)
            _walk(data, "")
            for entry in entries[before:]:
                entry["path"] = registry.rel(path)
    return {
        "definition": "explicit maturity labels found in the artifacts "
                       "(preliminary / smoke / pilot / tentative / "
                       "exploratory); the builder does not assign them",
        "entries": entries,
    }


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # pragma: no cover - git may be unavailable
        pass
    return None


def _git_dirty(root: Path) -> bool | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except Exception:  # pragma: no cover
        pass
    return None


def _package_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


# --------------------------------------------------------------------------
# ground truth (normalised evaluation frames)
# --------------------------------------------------------------------------
def load_eval_scenarios(
    data_root: Path, topology: str, registry: Registry,
) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Return ``{scenario_name: {"labels": ndarray, "path": str}}``.

    The scenario order is the order of ``files.evaluation`` in the dataset
    manifest, i.e. exactly the order the repo's own dataio loader uses.
    """
    import yaml

    ds_dir = data_root / topology / "dataset"
    manifest_path = ds_dir / "dataset_manifest.yaml"
    if not manifest_path.is_file():
        return {}, f"no dataset manifest at {manifest_path}"
    registry.ref(manifest_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    files = (manifest or {}).get("files", {})
    eval_files = files.get("evaluation") or []
    if not eval_files:
        return {}, f"dataset manifest lists no evaluation files for {topology}"
    out: dict[str, dict[str, Any]] = {}
    for rel in eval_files:
        path = ds_dir / rel
        if not path.is_file():
            return {}, f"evaluation file missing: {path}"
        registry.ref(path)
        df = pd.read_csv(path)
        if "label" not in df.columns:
            return {}, f"evaluation file has no 'label' column: {path}"
        name = Path(rel).stem
        out[name] = {
            "labels": df["label"].astype(int).to_numpy(),
            "path": registry.rel(path),
        }
    return out, None


def _concat_labels(scenarios: dict[str, dict[str, Any]],
                   names: Iterable[str]) -> np.ndarray:
    parts = []
    for name in names:
        parts.append(np.asarray(scenarios[name]["labels"], dtype=int))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=int)


# --------------------------------------------------------------------------
# section: proposed monitor (combined + ablation arms)
# --------------------------------------------------------------------------
def dataset_block(
    data_root: Path, topology: str, registry: Registry,
) -> dict[str, Any]:
    """Split / dataset provenance for one topology."""
    import yaml

    ds_dir = data_root / topology / "dataset"
    manifest_path = ds_dir / "dataset_manifest.yaml"
    block: dict[str, Any] = {"topology": topology}
    if not manifest_path.is_file():
        block["status"] = "unavailable"
        block["reason"] = f"no dataset manifest at {registry.rel(manifest_path)}"
        return block
    registry.ref(manifest_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    files = manifest.get("files", {}) or {}
    cal_files = files.get("calibration") or []
    cal_rows = 0
    for rel in cal_files:
        path = ds_dir / rel
        if not path.is_file():
            continue
        registry.ref(path)
        cal_rows += int(len(pd.read_csv(path)))
    scenarios, err = load_eval_scenarios(data_root, topology, registry)
    if err:
        block["status"] = "unavailable"
        block["reason"] = err
        return block
    block.update({
        "status": "ok",
        "dataset_manifest": registry.rel(manifest_path),
        "dataset_manifest_sha256": manifest.get("sha256") and
        (manifest.get("sha256") or {}).get("dataset_manifest.yaml"),
        "source": manifest.get("source"),
        "calibration_frames": len(cal_files),
        "calibration_rows": cal_rows,
        "evaluation_scenarios": {
            name: {
                "path": info["path"],
                "rows": int(len(info["labels"])),
                "attack_rows": int(np.sum(info["labels"])),
            } for name, info in scenarios.items()
        },
        "attack_windows": manifest.get("attack_windows"),
        "split_policy": "the calibration frames are split into train and "
                        "validation by the detectors; the evaluation "
                        "scenarios are never used for fitting or threshold "
                        "selection (see src/dataio/splits.py)",
    })
    return block


def detector_provenance(
    root: Path, topology: str, registry: Registry,
) -> dict[str, Any]:
    """Paths and split/seed provenance of the detector artifacts."""
    block: dict[str, Any] = {}
    inv_path = (root / "data" / "generated" / topology / "invariants" /
                f"{topology}_invariants_manifest.json")
    if inv_path.is_file():
        registry.ref(inv_path)
        man = json.loads(inv_path.read_text(encoding="utf-8"))
        block["invariants_manifest"] = {
            "path": registry.rel(inv_path),
            "all_ok": man.get("all_ok"),
            "calibration": man.get("calibration"),
            "niaarm": man.get("niaarm"),
            "dataset_manifest_sha256": man.get("dataset_manifest_sha256"),
            "gfsm_manifest_sha256": man.get("gfsm_manifest_sha256"),
            "n_states": len(man.get("states", {}) or {}),
        }
    else:
        block["invariants_manifest"] = {
            "status": "unavailable",
            "reason": f"manifest not found: {registry.rel(inv_path)}"}
    gfsm_path = (root / "data" / "generated" / topology / "gfsm" /
                 f"{topology}_gfsm_manifest.json")
    if gfsm_path.is_file():
        registry.ref(gfsm_path)
        man = json.loads(gfsm_path.read_text(encoding="utf-8"))
        block["gfsm_manifest"] = {
            "path": registry.rel(gfsm_path),
            "all_ok": man.get("all_ok"),
            "gfsm": man.get("gfsm"),
            "source_manifest_sha256": man.get("source_manifest_sha256"),
        }
    else:
        block["gfsm_manifest"] = {
            "status": "unavailable",
            "reason": f"manifest not found: {registry.rel(gfsm_path)}"}
    return block


def build_monitor_section(
    data_root: Path, root: Path, registry: Registry,
) -> dict[str, Any]:
    section: dict[str, Any] = {}
    for topology in TOPOLOGIES:
        pred_path = (root / "data" / "generated" / topology / "monitor" /
                     "predictions.csv")
        man_path = (root / "data" / "generated" / topology / "monitor" /
                    f"{topology}_monitor_manifest.json")
        entry: dict[str, Any] = {
            "topology": topology,
            "method": "state-aware ICS monitor (STL physics + GFSM guard + "
                      "Phi invariants, OR fusion)",
        }
        if not pred_path.is_file():
            entry.update(unavailable(
                f"monitor predictions not found: {registry.rel(pred_path)}"))
            section[topology] = entry
            continue
        registry.ref(pred_path)
        df = pd.read_csv(pred_path)
        required = {"scenario", "row", "y_true", "y_pred_stl", "y_pred_gfsm",
                    "y_pred_invariants", "y_pred"}
        missing = sorted(required - set(df.columns))
        if missing:
            entry.update(unavailable(
                f"monitor predictions lack columns {missing}"))
            section[topology] = entry
            continue

        y_true = df["y_true"].astype(int).to_numpy()
        gfsm = df["y_pred_gfsm"].astype(int).to_numpy()
        inv = df["y_pred_invariants"].astype(int).to_numpy()
        stl = df["y_pred_stl"].astype(int).to_numpy()
        fused = df["y_pred"].astype(int).to_numpy()
        score = (df["anomaly_score"].astype(float).to_numpy()
                 if "anomaly_score" in df.columns else fused.astype(float))

        arms = {
            "monitor_fused_combined": (fused, score,
                                       "average_precision_score"),
            "gfsm_only": (gfsm, gfsm.astype(float), "coarse_binary_rule_match"),
            "invariants_only": (inv, inv.astype(float),
                                "coarse_binary_rule_match"),
            "stl_only": (stl, stl.astype(float), "coarse_binary_rule_match"),
            "gfsm_or_invariants": (gfsm | inv, (gfsm | inv).astype(float),
                                   "coarse_binary_rule_match"),
            "or_combined": (gfsm | inv | stl, (gfsm | inv | stl).astype(float),
                            "coarse_binary_rule_match"),
        }
        arm_block: dict[str, Any] = {}
        for name, (yp, sc, kind) in arms.items():
            per_scenario: dict[str, Any] = {}
            for scen in sorted(set(df["scenario"].tolist())):
                mask = (df["scenario"] == scen).to_numpy()
                per_scenario[scen] = {"metrics": compute_metrics(
                    y_true[mask], yp[mask], sc[mask], score_kind=kind)}
            arm_block[name] = {
                "status": "ok",
                "metric_source": "recomputed_from_predictions",
                "metrics": compute_metrics(y_true, yp, sc, score_kind=kind),
                "per_scenario": per_scenario,
            }

        entry.update({
            "status": "ok",
            "metric_source": "recomputed_from_predictions",
            "prediction_path": registry.rel(pred_path),
            "n_rows": int(len(df)),
            "attack_rows": int(y_true.sum()),
            "arms": arm_block,
            "dataset": dataset_block(data_root, topology, registry),
            "detector_provenance": detector_provenance(root, topology, registry),
            "seed": None,
            "seed_note": "the monitor manifest does not record a seed; the "
                         "per-detector manifests below record their own",
        })

        # Cross-check the fused arm against the recorded monitor manifest.
        if man_path.is_file():
            registry.ref(man_path)
            man = json.loads(man_path.read_text(encoding="utf-8"))
            recorded = {}
            for sc in man.get("scenarios", []):
                recorded[sc.get("name")] = sc.get("metrics", {})
            checks = {}
            for scen, rec in recorded.items():
                got = arm_block["monitor_fused_combined"]["per_scenario"].get(
                    scen, {}).get("metrics", {})
                checks[scen] = {
                    "recomputed_matches_recorded": all(
                        got.get(k) is not None and rec.get(k) is not None
                        and abs(float(got[k]) - float(rec[k])) < 1e-9
                        for k in ("mcc", "balanced_accuracy", "aucpr")),
                }
            entry["manifest_cross_check"] = {
                "manifest_path": registry.rel(man_path),
                "detectors": man.get("detectors"),
                "fusion": man.get("fusion"),
                "per_scenario": checks,
            }
        section[topology] = entry
    return section


# --------------------------------------------------------------------------
# section: baselines
# --------------------------------------------------------------------------
def build_sain(root: Path, registry: Registry) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for topology in TOPOLOGIES:
        path = (root / "scripts" / "out" / "sain" / topology /
                "sain_local_vs_global.json")
        if not path.is_file():
            out[topology] = unavailable(
                f"SAIN comparison artifact not found: {registry.rel(path)}")
            continue
        registry.ref(path)
        art = json.loads(path.read_text(encoding="utf-8"))
        summary = art.get("summary", {})
        block: dict[str, Any] = {
            "status": "ok",
            "method": "SAIN local vs global (published baseline)",
            "metric_source": "artifact_reported",
            "metric_source_reason": "the SAIN artifact stores summary metrics "
                                    "only; no per-row predictions are kept, "
                                    "so the metrics cannot be recomputed",
            "prediction_path": registry.rel(path),
            "feature_cols": art.get("feature_cols"),
            "scenarios": [],
            "local": {
                "status": "ok",
                "metric_source": "artifact_reported",
                "metrics": normalise_artifact_metrics(summary.get("local", {})),
            },
            "global": {
                "status": "ok",
                "metric_source": "artifact_reported",
                "metrics": normalise_artifact_metrics(summary.get("global", {})),
            },
        }
        for sc in art.get("scenarios", []):
            block["scenarios"].append({
                "name": sc.get("name"),
                "n": sc.get("n"),
                "attack_rows": sc.get("attack_rows"),
                "local": {
                    "flags": (sc.get("local") or {}).get("flags"),
                    "metrics": normalise_artifact_metrics(
                        (sc.get("local") or {}).get("metrics", {}),
                        n=sc.get("n")),
                },
                "global": {
                    "flags": (sc.get("global") or {}).get("flags"),
                    "metrics": normalise_artifact_metrics(
                        (sc.get("global") or {}).get("metrics", {}),
                        n=sc.get("n")),
                },
            })
        block["notes"] = [
            "SAIN local/global summary metrics are aggregated over the "
            "evaluation scenarios of the topology.",
        ]
        out[topology] = block
    out["ltown"] = unavailable(
        "no SAIN artifact exists for L-Town: the repository ships no "
        "L-Town normalised dataset (only the raw L-Town EPANET input), so no "
        "SAIN comparison could be produced")
    return out


def build_imdisco(root: Path, registry: Registry) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for topology in TOPOLOGIES:
        base = root / "scripts" / "out" / "imdisco" / topology
        metrics_path = base / "metrics.json"
        model_path = base / "model.json"
        if not metrics_path.is_file():
            out[topology] = unavailable(
                f"IM-DISCO metrics artifact not found: "
                f"{registry.rel(metrics_path)}")
            continue
        registry.ref(metrics_path)
        art = json.loads(metrics_path.read_text(encoding="utf-8"))
        block: dict[str, Any] = {
            "status": "ok",
            "method": "IM-DISCO (association-rule anomaly detection)",
            "metric_source": "artifact_reported",
            "metric_source_reason": "the IM-DISCO artifact stores metrics and "
                                    "the mined rule model only; no per-row "
                                    "predictions are kept, so the metrics "
                                    "cannot be recomputed",
            "prediction_path": registry.rel(metrics_path),
            "metrics": normalise_artifact_metrics(
                art, n=art.get("n"), n_positive=art.get("n", 0) -
                art.get("fn", 0)),
        }
        if model_path.is_file():
            registry.ref(model_path)
            model = json.loads(model_path.read_text(encoding="utf-8"))
            prov = model.get("provenance", {})
            block["model_artifact"] = registry.rel(model_path)
            block["training"] = model.get("training")
            block["software"] = {"mlxtend": prov.get("mlxtend_version")}
            block["reproduction_gaps"] = prov.get("reproduction_gaps", [])
            block["notes"] = [
                "IM-DISCO flagged no evaluation row "
                f"(tp={art.get('tp')}, fp={art.get('fp')}), so its average "
                "precision equals the attack-positive rate "
                f"({art.get('average_precision')}).",
            ]
        out[topology] = block
    out["ltown"] = unavailable(
        "no IM-DISCO artifact exists for L-Town: the repository ships no "
        "L-Town normalised dataset (only the raw L-Town EPANET input), so "
        "the baseline could not be evaluated on L-Town")
    return out


def build_state_aware_cusum(
    data_root: Path, root: Path, registry: Registry,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for topology in TOPOLOGIES:
        base = (root / "scripts" / "out" / "state_aware_cusum" / topology)
        pred_path = base / "state_aware_cusum_predictions.csv"
        metrics_path = base / "state_aware_cusum_metrics.json"
        cfg_path = base / "state_aware_cusum_config.json"
        prov_path = base / "state_aware_cusum_provenance.json"
        if not pred_path.is_file():
            out[topology] = unavailable(
                f"state-aware CUSUM predictions not found: "
                f"{registry.rel(pred_path)}")
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) \
            if cfg_path.is_file() else {}
        registry.ref(pred_path)
        for extra_path in (metrics_path, cfg_path, prov_path):
            if extra_path.is_file():
                registry.ref(extra_path)

        tanks = cfg.get("tanks") or []
        if not tanks:
            out[topology] = unavailable(
                "not applicable for this topology: the state-aware CUSUM "
                "observer has no per-tank state model here "
                f"(config tanks={tanks}); the paper's observer needs tank "
                "level columns plus the actuator status that feeds them, and "
                "the missing tank flows make the Eq. 8 observer and the "
                "per-state residual undefined, so no detection number can be "
                "reported for this topology",
                metric_source="artifact_reported",
                prediction_path=registry.rel(pred_path),
                artifact_reported_metrics=(
                    json.loads(metrics_path.read_text(encoding="utf-8"))
                    if metrics_path.is_file() else None),
                config=cfg)
            continue

        df = pd.read_csv(pred_path)
        y_true = df["y_true"].astype(int).to_numpy()
        y_pred = df["y_pred"].astype(int).to_numpy()
        scores = (df["cusum_score"].astype(float).to_numpy()
                  if "cusum_score" in df.columns else y_pred.astype(float))
        per_scenario: dict[str, Any] = {}
        for scen in sorted(set(df["scenario"].tolist())):
            mask = (df["scenario"] == scen).to_numpy()
            per_scenario[scen] = {"metrics": compute_metrics(
                y_true[mask], y_pred[mask], scores[mask])}
        block: dict[str, Any] = {
            "status": "ok",
            "method": "state-aware CUSUM (SAC 2018 baseline)",
            "metric_source": "recomputed_from_predictions",
            "prediction_path": registry.rel(pred_path),
            "metrics": compute_metrics(y_true, y_pred, scores),
            "per_scenario": per_scenario,
            "config": cfg,
        }
        if metrics_path.is_file():
            art = json.loads(metrics_path.read_text(encoding="utf-8"))
            block["artifact_reported_metrics"] = art
            checks = {}
            for scen, rec in art.items():
                got = per_scenario.get(scen, {}).get("metrics", {})
                checks[scen] = {
                    "recomputed_matches_recorded": all(
                        got.get(k) is not None and rec.get(k) is not None
                        and abs(float(got[k]) - float(rec[k])) < 1e-9
                        for k in ("mcc", "balanced_accuracy", "precision")),
                    "aucpr_matches_recorded": (
                        got.get("aucpr") is not None
                        and rec.get("attack_average_precision") is not None
                        and abs(float(got["aucpr"]) -
                                float(rec["attack_average_precision"])) < 1e-9),
                }
            block["artifact_cross_check"] = checks
        if prov_path.is_file():
            prov = json.loads(prov_path.read_text(encoding="utf-8"))
            block["validation"] = prov.get("validation")
            block["missing_original_observer_parameters"] = prov.get(
                "missing_original_observer_parameters")
        out[topology] = block
    out["ltown"] = unavailable(
        "no state-aware CUSUM artifact exists for L-Town: the repository ships no "
        "L-Town normalised dataset (only the raw L-Town EPANET input), so "
        "the baseline could not be evaluated on L-Town")
    return out


def build_conv_bilstm_ae(
    data_root: Path, root: Path, registry: Registry,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for topology in TOPOLOGIES:
        base = root / "scripts" / "out" / "conv_bilstm_ae" / topology
        npz_path = base / "predictions.npz"
        meta_path = base / "metadata.json"
        if not npz_path.is_file():
            out[topology] = unavailable(
                f"ConvBiLSTM-AE predictions not found: "
                f"{registry.rel(npz_path)}")
            continue
        registry.ref(npz_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8")) \
            if meta_path.is_file() else {}
        if meta_path.is_file():
            registry.ref(meta_path)

        with np.load(npz_path) as z:
            keys = set(z.files)
            pred_key = "row_predictions" if "row_predictions" in keys \
                else ("predictions" if "predictions" in keys else None)
            err_key = "row_errors" if "row_errors" in keys else (
                "errors" if "errors" in keys else None)
            if pred_key is None:
                out[topology] = unavailable(
                    f"ConvBiLSTM-AE artifact has no prediction array "
                    f"(keys={sorted(keys)})")
                continue
            y_pred = np.asarray(z[pred_key], dtype=int)
            errors = np.asarray(z[err_key], dtype=float) if err_key else None

        scenarios, err = load_eval_scenarios(data_root, topology, registry)
        if err:
            out[topology] = unavailable(err)
            continue
        names = list(scenarios)
        total = sum(len(scenarios[n]["labels"]) for n in names)
        if total != len(y_pred):
            out[topology] = unavailable(
                f"ConvBiLSTM-AE stores {len(y_pred)} scored rows but the "
                f"normalised evaluation frames hold {total} rows "
                f"({names}); the alignment cannot be established")
            continue
        y_true = _concat_labels(scenarios, names)
        metrics = compute_metrics(
            y_true, y_pred, errors if errors is not None else y_pred)
        per_scenario: dict[str, Any] = {}
        offset = 0
        for name in names:
            n = len(scenarios[name]["labels"])
            sl = slice(offset, offset + n)
            offset += n
            per_scenario[name] = {"metrics": compute_metrics(
                y_true[sl], y_pred[sl],
                errors[sl] if errors is not None else y_pred[sl])}
        block: dict[str, Any] = {
            "status": "ok",
            "method": "ConvBiLSTM-AE (reconstruction-error baseline)",
            "metric_source": "recomputed_from_predictions",
            "prediction_path": registry.rel(npz_path),
            "ground_truth": {
                "source": "data/<topology>/dataset/evaluation/*.csv (label "
                          "column), concatenated in dataset-manifest order",
                "scenarios": names,
            },
            "metrics": metrics,
            "per_scenario": per_scenario,
            "dataset": meta.get("dataset"),
            "seed": meta.get("seed"),
            "threshold": meta.get("threshold"),
            "threshold_rule": meta.get("threshold_rule"),
            "software": meta.get("software"),
            "hardware": meta.get("hardware"),
            "implementation_notes": meta.get("implementation_notes"),
            "notes": [
                "The artifact stores reconstruction errors and binary "
                "predictions but no labels; the ground truth is taken from "
                "the normalised evaluation frames.",
            ],
        }
        if len(names) > 1:
            label_vectors = [np.asarray(scenarios[n]["labels"], dtype=int)
                             for n in names]
            identical = all((label_vectors[0] == lv).all()
                            for lv in label_vectors[1:])
            block["notes"].append(
                "The evaluation scenarios of this topology share an identical "
                "label vector, so the block order of the scenarios does not "
                "change any metric."
                if identical else
                "The evaluation scenarios have different label vectors; the "
                "scenario block order is the dataset-manifest order.")
        out[topology] = block
    out["ltown"] = unavailable(
        "no ConvBiLSTM-AE artifact exists for L-Town: the repository ships "
        "no L-Town normalised dataset (only the raw L-Town EPANET input), "
        "so the baseline could not be evaluated on L-Town")
    return out


def build_hybrid_ensemble(
    data_root: Path, root: Path, registry: Registry,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    base = root / "scripts" / "out" / "hybrid_ensemble" / "ctown"
    run_path = base / "run_result.json"
    npz_path = base / "test_predictions.npz"
    if not run_path.is_file() or not npz_path.is_file():
        out["ctown"] = unavailable(
            f"hybrid-ensemble artifact not found: "
            f"{registry.rel(run_path if not run_path.is_file() else npz_path)}")
        out["anytown"] = unavailable(
            "the hybrid ensemble was only reproduced on the native BATADAL "
            "C-Town dataset; no Anytown run exists "
            "(the baseline is defined over BATADAL only)")
        out["ltown"] = unavailable(
            "the hybrid ensemble was only reproduced on the native BATADAL "
            "C-Town dataset; no L-Town run exists")
        return out
    registry.ref(run_path)
    registry.ref(npz_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))

    scenarios, err = load_eval_scenarios(data_root, "ctown", registry)
    if err:
        out["ctown"] = unavailable(err)
        out["anytown"] = unavailable(
            "the hybrid ensemble was only reproduced on the native BATADAL "
            "C-Town dataset; no Anytown run exists")
        out["ltown"] = unavailable(
            "the hybrid ensemble was only reproduced on the native BATADAL "
            "C-Town dataset; no L-Town run exists")
        return out
    name = list(scenarios)[0]
    labels = np.asarray(scenarios[name]["labels"], dtype=int)

    with np.load(npz_path) as z:
        probs = np.asarray(z["probabilities"], dtype=float)
        stored_labels = (np.asarray(z["labels"], dtype=int)
                         if "labels" in z.files else None)

    reported = run.get("metrics_test", {})
    dropped = int(reported.get("dropped_rows", 0))
    threshold = float(reported.get("threshold", 0.5))
    if len(labels) - dropped != len(probs):
        out["ctown"] = unavailable(
            f"the scored test rows ({len(probs)}) do not match the official "
            f"test file ({len(labels)} rows) minus the {dropped} dropped "
            "head rows recorded in the run result")
        out["anytown"] = unavailable(
            "the hybrid ensemble was only reproduced on the native BATADAL "
            "C-Town dataset; no Anytown run exists")
        out["ltown"] = unavailable(
            "the hybrid ensemble was only reproduced on the native BATADAL "
            "C-Town dataset; no L-Town run exists")
        return out
    y_true = labels[dropped:]
    y_pred = (probs >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, probs)

    notes: list[str] = []
    if stored_labels is not None and (stored_labels == y_true).all():
        notes.append(
            "test_predictions.npz stores the ground-truth labels in its "
            "'labels' array; the detector output is derived from the "
            "'probabilities' array thresholded at the recorded threshold "
            f"({threshold}). The recomputed confusion counts match the run "
            "result.")
    block = {
        "status": "ok",
        "method": "hybrid ensemble (RF/XGB + LSTM stack, native BATADAL)",
        "metric_source": "recomputed_from_predictions",
        "prediction_path": registry.rel(npz_path),
        "ground_truth": {
            "source": "data/ctown/dataset/evaluation/test.csv (label column)",
            "rows_used": int(len(y_true)),
            "rows_dropped": dropped,
        },
        "metrics": metrics,
        "artifact_reported_metrics": reported,
        "dataset": run.get("artifacts"),
        "seed": run.get("provenance", {}).get("seed"),
        "software": run.get("provenance", {}).get("library_versions"),
        "adaptations": run.get("provenance", {}).get("adaptations"),
        "notes": notes,
    }
    out["ctown"] = block
    out["anytown"] = unavailable(
        "the hybrid ensemble was only reproduced on the native BATADAL "
        "C-Town dataset; no Anytown run exists (the baseline is defined "
        "over BATADAL only)")
    out["ltown"] = unavailable(
        "the hybrid ensemble was only reproduced on the native BATADAL "
        "C-Town dataset; no L-Town run exists")
    return out


def build_sicams(
    data_root: Path, root: Path, registry: Registry,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for topology in TOPOLOGIES:
        path = (root / "scripts" / "out" / "sicams" / topology /
                "evaluation.json")
        if not path.is_file():
            out[topology] = unavailable(
                f"SICAMS artifact not found: {registry.rel(path)}")
            continue
        registry.ref(path)
        art = json.loads(path.read_text(encoding="utf-8"))
        scenarios, err = load_eval_scenarios(data_root, topology, registry)
        if err:
            out[topology] = unavailable(err)
            continue
        preds = art.get("predictions", {})
        flags = np.asarray(preds.get("flags", []), dtype=int)
        scores = np.asarray(preds.get("moving_average", []), dtype=float)
        if flags.size == 0:
            out[topology] = unavailable(
                "the SICAMS artifact stores no per-row flags")
            continue
        # SICAMS evaluates one evaluation scenario; candidates are the
        # scenarios with the same row count. The scenario is identified by
        # checking which candidate reproduces the artifact-reported metrics.
        candidates = [n for n, s in scenarios.items()
                      if len(s["labels"]) == flags.size]
        if not candidates:
            out[topology] = unavailable(
                f"cannot map the {flags.size} SICAMS rows onto an evaluation "
                f"scenario (available: {sorted(scenarios)})")
            continue
        reported = art.get("metrics", {}) or {}
        matching: list[str] = []
        for cand in candidates:
            cand_metrics = compute_metrics(
                np.asarray(scenarios[cand]["labels"], dtype=int), flags, scores)
            same = all(
                cand_metrics.get(k) is not None
                and reported.get(src) is not None
                and abs(float(cand_metrics[k]) - float(reported[src])) < 1e-9
                for k, src in (("tp", "tp"), ("fp", "fp"), ("fn", "fn"),
                               ("tn", "tn"), ("mcc", "mcc"),
                               ("aucpr", "attack_average_precision")))
            if same:
                matching.append(cand)
        if not matching:
            out[topology] = unavailable(
                f"none of the candidate evaluation scenarios {sorted(candidates)} "
                "reproduces the SICAMS artifact-reported metrics, so the "
                "artifact rows cannot be aligned with the ground truth")
            continue
        scen_name = sorted(matching)[0]
        y_true = np.asarray(scenarios[scen_name]["labels"], dtype=int)
        metrics = compute_metrics(y_true, flags, scores)
        prov = art.get("provenance", {})
        clusters = (prov.get("temporal_clusters", {}) or {}).get("clusters", [])
        degenerate = [c for c in clusters
                      if c.get("covariance_rank") is not None
                      and c.get("s") is not None
                      and c["covariance_rank"] < c["s"]]
        cluster_ids = np.asarray(preds.get("cluster_ids", []), dtype=int)
        rows_degenerate = int(np.isin(
            cluster_ids, [c["cluster_id"] for c in degenerate]).sum()) \
            if degenerate and cluster_ids.size else 0
        block = {
            "status": "ok",
            "method": "SICAMS (change-point / whitening baseline)",
            "metric_source": "recomputed_from_predictions",
            "prediction_path": registry.rel(path),
            "ground_truth": {
                "source": "data/<topology>/dataset/evaluation/*.csv (label "
                          "column)",
                "scenario": scen_name,
                "scenario_alignment": {
                    "candidates_by_row_count": sorted(candidates),
                    "matching_candidates": sorted(matching),
                    "method": "the candidate scenario is the one that "
                              "reproduces the artifact-reported metrics; "
                              "several candidates match when their label "
                              "vectors are identical, in which case the "
                              "metrics are identical too",
                },
            },
            "metrics": metrics,
            "artifact_reported_metrics": art.get("metrics"),
            "config": art.get("config"),
            "temporal_clusters": {
                "n_clusters": (prov.get("temporal_clusters", {}) or {}).get(
                    "n_clusters"),
                "assignment": (prov.get("temporal_clusters", {}) or {}).get(
                    "assignment"),
                "degenerate_clusters": [c["cluster_id"] for c in degenerate],
                "evaluation_rows_in_degenerate_clusters": rows_degenerate,
                "evaluation_rows_total": int(cluster_ids.size),
            },
            "missing_original_settings": prov.get("missing_original_settings"),
            "omitted_native_components": prov.get("omitted_native_components"),
            "native_vs_adaptation": prov.get("native_vs_adaptation"),
        }
        out[topology] = block
    out["ltown"] = unavailable(
        "no SICAMS artifact exists for L-Town: the repository ships no "
        "L-Town normalised dataset (only the raw L-Town EPANET input file), "
        "so the native L-Town SICAMS configuration could not be evaluated")
    return out


# --------------------------------------------------------------------------
# section: full-signal simulation runs
# --------------------------------------------------------------------------
def build_full_signal(root: Path, registry: Registry) -> dict[str, Any]:
    base = root / "scripts" / "out" / "full_signal"
    out: dict[str, Any] = {}
    if not base.is_dir():
        return {"status": "unavailable",
                "reason": f"no full-signal run directory at {registry.rel(base)}",
                "runs": {}}
    for topology_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        topology = topology_dir.name
        for run in sorted(p for p in topology_dir.iterdir() if p.is_dir()):
            name = f"{topology}/{run.name}"
            labels_path = run / "labels.csv"
            events_path = run / "events.json"
            man_path = run / "manifest.json"
            if not labels_path.is_file():
                out[name] = unavailable(
                    f"labels.csv not found in {registry.rel(run)}")
                continue
            registry.ref(labels_path)
            labels = pd.read_csv(labels_path)

            def _count(col: str) -> int:
                if col not in labels.columns:
                    return 0
                return int(labels[col].astype(bool).sum())

            attack_rows = _count("attack_active")
            impact_rows = _count("physical_impact")
            modified_rows = _count("delivery_or_command_modified")
            times = labels["time_seconds"].astype(float) if \
                "time_seconds" in labels.columns else None
            attack_window: dict[str, Any] = {"n_steps": int(len(labels))}
            if attack_rows and times is not None:
                mask = labels["attack_active"].astype(bool).to_numpy()
                attack_window.update({
                    "start_s": float(times.to_numpy()[mask].min()),
                    "end_s": float(times.to_numpy()[mask].max()),
                })
            else:
                attack_window.update({"start_s": None, "end_s": None})

            events_summary: dict[str, Any] = {"n_events": 0}
            if events_path.is_file():
                registry.ref(events_path)
                events = json.loads(events_path.read_text(encoding="utf-8"))
                kinds: dict[str, int] = {}
                modes: dict[str, int] = {}
                targets: list[str] = []
                starts: list[float] = []
                ends: list[float] = []
                for ev in events:
                    kinds[ev.get("kind", "?")] = kinds.get(ev.get("kind", "?"), 0) + 1
                    mode = ev.get("mode") or "n/a"
                    modes[mode] = modes.get(mode, 0) + 1
                    tgt = ev.get("target")
                    if tgt and tgt not in targets:
                        targets.append(tgt)
                    if ev.get("start") is not None:
                        starts.append(float(ev["start"]))
                    if ev.get("end") is not None:
                        ends.append(float(ev["end"]))
                events_summary = {
                    "n_events": len(events),
                    "kinds": kinds,
                    "modes": modes,
                    "targets": sorted(targets),
                    "start_s": min(starts) if starts else None,
                    "end_s": max(ends) if ends else None,
                }

            manifest_block: dict[str, Any] = {}
            if man_path.is_file():
                registry.ref(man_path)
                man = json.loads(man_path.read_text(encoding="utf-8"))
                manifest_block = {
                    "run_id": man.get("run_id"),
                    "seed": man.get("seed"),
                    "split": man.get("split"),
                    "timesteps": man.get("timesteps"),
                    "engine": man.get("engine"),
                    "wntr_version": man.get("wntr_version"),
                    "inp_sha256": man.get("inp_sha256"),
                    "plc_config_sha256": man.get("plc_config_sha256"),
                    "sensor_modes": man.get("sensor_modes"),
                    "artifact_path": registry.rel(man_path),
                }

            detector_path = (root / "scripts" / "out" / "topology_predictions" /
                             topology / "detector_predictions.csv")
            detection: dict[str, Any]
            if detector_path.is_file():
                registry.ref(detector_path)
                predictions = pd.read_csv(detector_path)
                predictions = predictions[predictions["scenario"] == run.name]
                nominal_holdout = (
                    "nominal" in run.name
                    and attack_rows == 0
                    and 0 < len(predictions) < len(labels)
                    and not predictions["y_true"].astype(bool).any()
                )
                if len(predictions) != len(labels) and not nominal_holdout:
                    detection = unavailable(
                        f"detector predictions for {run.name} have {len(predictions)} "
                        f"rows but labels.csv has {len(labels)}")
                else:
                    detection = {
                        "source": registry.rel(detector_path),
                        "gfsm": compute_metrics(
                            predictions["y_true"], predictions["y_pred_gfsm"],
                            predictions["y_pred_gfsm"].astype(float),
                            score_kind="coarse_binary_guard"),
                        "stl": compute_metrics(
                            predictions["y_true"], predictions["y_pred_stl"],
                            predictions["anomaly_score"],
                            score_kind="fused_anomaly_score"),
                        "combined": compute_metrics(
                            predictions["y_true"], predictions["y_pred"],
                            predictions["anomaly_score"],
                            score_kind="fused_anomaly_score"),
                        "unknown_rows": int((predictions["uncalibrated_count"] > 0).sum()),
                        "unknown_residuals": int(predictions["uncalibrated_count"].sum()),
                        "metric_source": "recomputed",
                    }
                    if nominal_holdout:
                        detection["evaluation_scope"] = "held_out_nominal_rows"
                        detection["source_trace_rows"] = int(len(labels))
            else:
                detection = unavailable(
                    f"no detector predictions are stored for {run.name}; "
                    f"expected {registry.rel(detector_path)}")

            out[name] = {
                "status": "ok",
                "topology": topology,
                "run": run.name,
                "artifact_path": registry.rel(run),
                "labels_path": registry.rel(labels_path),
                "n_steps": int(len(labels)),
                "attack_window": attack_window,
                "attack_steps": attack_rows,
                "delivery_or_command_modified_steps": modified_rows,
                "physical_impact_steps": impact_rows,
                "events": events_summary,
                "manifest": manifest_block,
                "detection_metrics": detection,
            }
    return {"status": "ok", "runs": out}


# --------------------------------------------------------------------------
# section: regime counts (nominal false alarms, uncalibrated regimes)
# --------------------------------------------------------------------------
def build_regime_counts(root: Path, registry: Registry) -> dict[str, Any]:
    section: dict[str, Any] = {
        "nominal_false_alarms": {
            "definition": "number of alarms raised on attack-free "
                          "(nominal / clean-validation) data",
            "entries": [],
        },
        "uncalibrated_regime": {
            "definition": "number of evaluation rows for which no calibrated "
                          "model was available (no per-tank state model, or a "
                          "degenerate covariance cluster)",
            "entries": [],
        },
    }

    # -- state-aware CUSUM: clean-validation false alarms + per-tank model
    for topology in TOPOLOGIES:
        base = root / "scripts" / "out" / "state_aware_cusum" / topology
        prov_path = base / "state_aware_cusum_provenance.json"
        cfg_path = base / "state_aware_cusum_config.json"
        if not prov_path.is_file():
            section["nominal_false_alarms"]["entries"].append({
                "method": "state-aware CUSUM", "topology": topology,
                **unavailable(f"provenance artifact not found: "
                              f"{registry.rel(prov_path)}")})
            continue
        registry.ref(prov_path)
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) \
            if cfg_path.is_file() else {}
        if cfg_path.is_file():
            registry.ref(cfg_path)
        validation = prov.get("validation", {}) or {}
        selection = validation.get("selection", {}) or {}
        per_tank = selection.get("per_tank", {}) or {}
        total = sum(int(v.get("validation_false_alarms", 0))
                    for v in per_tank.values())
        tanks = cfg.get("tanks") or []
        if not per_tank:
            pred_rows = None
            pred_path = base / "state_aware_cusum_predictions.csv"
            if pred_path.is_file():
                registry.ref(pred_path)
                pred_rows = int(len(pd.read_csv(pred_path)))
            section["nominal_false_alarms"]["entries"].append({
                "method": "state-aware CUSUM", "topology": topology,
                **unavailable(
                    "no per-tank calibration was possible for this topology "
                    f"(config tanks={tanks}); the observer therefore has no "
                    "calibrated regime and no nominal false-alarm count can "
                    "be reported",
                    clean_validation_false_alarms_total=None,
                    clean_validation_steps=selection.get("validation_steps"),
                    false_alarm_budget_per_day=validation.get(
                        "false_alarm_budget_per_day"))})
            section["uncalibrated_regime"]["entries"].append({
                "method": "state-aware CUSUM", "topology": topology,
                "status": "ok",
                "calibrated_tanks": 0,
                "tanks": tanks,
                "uncalibrated_rows": pred_rows,
                "evaluation_rows_total": pred_rows,
                "note": "no per-tank observer model exists for this topology, "
                        "so every evaluated row is outside a calibrated "
                        "regime",
                "source": registry.rel(cfg_path)})
            continue
        section["nominal_false_alarms"]["entries"].append({
            "method": "state-aware CUSUM", "topology": topology,
            "status": "ok",
            "clean_validation_false_alarms_total": total,
            "clean_validation_steps": selection.get("validation_steps"),
            "clean_validation_days": selection.get("validation_days"),
            "allowed_false_alarms": selection.get("allowed_false_alarms"),
            "false_alarm_budget_per_day": validation.get(
                "false_alarm_budget_per_day"),
            "per_tank": per_tank,
            "source": registry.rel(prov_path)})
        section["uncalibrated_regime"]["entries"].append({
            "method": "state-aware CUSUM", "topology": topology,
            "status": "ok",
            "calibrated_tanks": len(per_tank),
            "tanks": sorted(per_tank),
            "uncalibrated_rows": 0,
            "note": "every tank state observed in the benign history was "
                    "also observed during calibration, so no evaluation row "
                    "falls outside a calibrated regime",
            "source": registry.rel(prov_path)})

    # -- SICAMS: degenerate (uncalibrated) covariance clusters
    for topology in TOPOLOGIES:
        path = root / "scripts" / "out" / "sicams" / topology / "evaluation.json"
        if not path.is_file():
            section["uncalibrated_regime"]["entries"].append({
                "method": "SICAMS", "topology": topology,
                **unavailable(f"artifact not found: {registry.rel(path)}")})
            continue
        registry.ref(path)
        art = json.loads(path.read_text(encoding="utf-8"))
        clusters = ((art.get("provenance", {}) or {})
                    .get("temporal_clusters", {}) or {}).get("clusters", [])
        degenerate = [c["cluster_id"] for c in clusters
                      if c.get("covariance_rank") is not None
                      and c.get("s") is not None
                      and c["covariance_rank"] < c["s"]]
        cluster_ids = np.asarray(
            (art.get("predictions", {}) or {}).get("cluster_ids", []), dtype=int)
        rows = int(np.isin(cluster_ids, degenerate).sum()) \
            if degenerate and cluster_ids.size else 0
        section["uncalibrated_regime"]["entries"].append({
            "method": "SICAMS", "topology": topology, "status": "ok",
            "uncalibrated_rows": rows,
            "evaluation_rows_total": int(cluster_ids.size),
            "degenerate_clusters": degenerate,
            "source": registry.rel(path)})

    # -- full-signal nominal runs: attack-free runs carry no detector output
    fs_base = root / "scripts" / "out" / "full_signal"
    if fs_base.is_dir():
        for topology_dir in sorted(p for p in fs_base.iterdir() if p.is_dir()):
            topology = topology_dir.name
            for run in sorted(p for p in topology_dir.iterdir()
                              if p.is_dir()):
                if "nominal" not in run.name:
                    continue
                labels_path = run / "labels.csv"
                events_path = run / "events.json"
                n_steps = 0
                if labels_path.is_file():
                    registry.ref(labels_path)
                    n_steps = int(len(pd.read_csv(labels_path)))
                n_events = 0
                if events_path.is_file():
                    registry.ref(events_path)
                    n_events = len(json.loads(
                        events_path.read_text(encoding="utf-8")))
                detector_path = (root / "scripts" / "out" / "topology_predictions" /
                                 topology / "detector_predictions.csv")
                if detector_path.is_file():
                    registry.ref(detector_path)
                    predictions = pd.read_csv(detector_path)
                    predictions = predictions[predictions["scenario"] == run.name]
                    section["nominal_false_alarms"]["entries"].append({
                        "method": "full-signal simulation", "topology": topology,
                        "run": run.name, "status": "ok",
                        "clean_validation_false_alarms_total": int(predictions["y_pred"].sum()),
                        "unknown_residuals": int(predictions["uncalibrated_count"].sum()),
                        "attack_free_steps": int(len(predictions)),
                        "calibration_rows_excluded": max(n_steps - len(predictions), 0),
                        "injected_events": n_events,
                        "source": registry.rel(detector_path)})
                else:
                    section["nominal_false_alarms"]["entries"].append({
                        "method": "full-signal simulation", "topology": topology,
                        "run": run.name, "status": "unavailable",
                        "reason": f"no nominal detector predictions at {registry.rel(detector_path)}",
                        "attack_free_steps": n_steps,
                        "injected_events": n_events,
                        "source": registry.rel(run)})
    return section


# --------------------------------------------------------------------------
# top-level build
# --------------------------------------------------------------------------
def build_fact_pack(
    root: Path, data_root: Path, artifacts_root: Path,
    exclude_dirs: Iterable[Path] = (),
) -> dict[str, Any]:
    root = Path(root).resolve()
    data_root = Path(data_root).resolve()
    artifacts_root = Path(artifacts_root).resolve()
    registry = Registry(root)

    monitor = build_monitor_section(data_root, root, registry)
    baselines = {
        "sain_local_vs_global": build_sain(root, registry),
        "imdisco": build_imdisco(root, registry),
        "state_aware_cusum": build_state_aware_cusum(
            data_root, root, registry),
        "conv_bilstm_ae": build_conv_bilstm_ae(data_root, root, registry),
        "hybrid_ensemble": build_hybrid_ensemble(data_root, root, registry),
        "sicams": build_sicams(data_root, root, registry),
    }
    full_signal = build_full_signal(root, registry)
    regime_counts = build_regime_counts(root, registry)

    unavailable_items: list[dict[str, Any]] = []

    def _collect(section: Any, prefix: str) -> None:
        if isinstance(section, dict):
            if section.get("status") == "unavailable":
                unavailable_items.append({
                    "item": prefix,
                    "reason": section.get("reason"),
                })
                return
            for key, value in section.items():
                if isinstance(value, dict):
                    _collect(value, f"{prefix}.{key}")
        elif isinstance(section, list):
            for i, value in enumerate(section):
                _collect(value, f"{prefix}[{i}]")

    for topology, entry in monitor.items():
        _collect(entry, f"proposed_monitor.{topology}")
    for name, entry in baselines.items():
        _collect(entry, f"baselines.{name}")
    _collect(full_signal, "full_signal")
    _collect(regime_counts, "regime_counts")

    pack: dict[str, Any] = {
        "schema": SCHEMA,
        "builder": {
            "script": "scripts/build_fact_pack.py",
            "version": BUILDER_VERSION,
            "metric_backend": "stl.metrics.detection_metrics",
            "recomputes": "every metric for which per-row predictions and the "
                          "ground truth are both available",
        },
        "environment": {
            "repo_root": str(root),
            "data_root": str(data_root),
            "artifacts_root": str(artifacts_root),
            "git_commit": _git_commit(root),
            "git_dirty": _git_dirty(root),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": _package_version("numpy"),
            "pandas_version": _package_version("pandas"),
            "scikit_learn_version": _package_version("sklearn"),
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "proposed_monitor": monitor,
        "baselines": baselines,
        "full_signal": full_signal,
        "regime_counts": regime_counts,
        "status_labels": collect_status_labels(
            [artifacts_root, root / "data" / "generated"], registry,
            exclude_dirs=exclude_dirs),
        "unavailable": unavailable_items,
        "caveats": build_caveats(root, registry),
    }
    pack["source_files"] = registry.files
    return _json_safe(pack)


def build_caveats(root: Path, registry: Registry) -> list[dict[str, Any]]:
    """Reproduction caveats carried over from the artifacts themselves."""
    caveats: list[dict[str, Any]] = []

    def _add(scope: str, text: str, source: str | None = None) -> None:
        caveats.append({"scope": scope, "caveat": text, "source": source})

    ablation_path = root / "scripts" / "run_ablation.py"
    caveat_text = None
    if ablation_path.is_file():
        registry.ref(ablation_path)
        source = ablation_path.read_text(encoding="utf-8")
        match = re.search(r"CAVEAT\s*=\s*\((.*?)\)\s*\n", source, re.S)
        if match:
            caveat_text = re.sub(
                r"\s+", " ",
                " ".join(part.strip().strip('"')
                         for part in match.group(1).split("\n")
                         if part.strip())).strip()
    _add("proposed_monitor",
         caveat_text or
         "The GFSM composition state depends on the Stage-4 composition fix; "
         "the ablation arms computed here are built from the per-detector "
         "columns of the recorded monitor predictions.",
         "scripts/run_ablation.py")

    for topology in TOPOLOGIES:
        path = (root / "scripts" / "out" / "sicams" / topology /
                "evaluation.json")
        if not path.is_file():
            continue
        registry.ref(path)
        art = json.loads(path.read_text(encoding="utf-8"))
        prov = art.get("provenance", {}) or {}
        for item in prov.get("missing_original_settings", []) or []:
            _add(f"baselines.sicams.{topology}", item, registry.rel(path))
        for item in prov.get("omitted_native_components", []) or []:
            _add(f"baselines.sicams.{topology}", item, registry.rel(path))
        adapt = (prov.get("native_vs_adaptation", {}) or {}).get(
            "adaptations", {}) or {}
        for key, value in adapt.items():
            _add(f"baselines.sicams.{topology}", f"{key}: {value}",
                 registry.rel(path))

    for topology in TOPOLOGIES:
        path = (root / "scripts" / "out" / "imdisco" / topology / "model.json")
        if not path.is_file():
            continue
        registry.ref(path)
        model = json.loads(path.read_text(encoding="utf-8"))
        for item in (model.get("provenance", {}) or {}).get(
                "reproduction_gaps", []) or []:
            _add(f"baselines.imdisco.{topology}", item, registry.rel(path))

    for topology in TOPOLOGIES:
        path = (root / "scripts" / "out" / "state_aware_cusum" / topology /
                "state_aware_cusum_provenance.json")
        if not path.is_file():
            continue
        registry.ref(path)
        prov = json.loads(path.read_text(encoding="utf-8"))
        note = (prov.get("missing_original_observer_parameters", {}) or {}).get(
            "note")
        if note:
            _add(f"baselines.state_aware_cusum.{topology}", note,
                 registry.rel(path))
        centering = prov.get("centering_interpretation", {}) or {}
        if centering.get("paper_ambiguity"):
            _add(f"baselines.state_aware_cusum.{topology}",
                 f"centering ambiguity: {centering['paper_ambiguity']} "
                 f"(adopted: {centering.get('adopted')})", registry.rel(path))

    hybrid_path = root / "scripts" / "out" / "hybrid_ensemble" / "ctown" / \
        "run_result.json"
    if hybrid_path.is_file():
        registry.ref(hybrid_path)
        run = json.loads(hybrid_path.read_text(encoding="utf-8"))
        for adapt in (run.get("provenance", {}) or {}).get(
                "adaptations", []) or []:
            _add("baselines.hybrid_ensemble.ctown",
                 f"{adapt.get('id')}: ours={adapt.get('ours')}; "
                 f"paper={adapt.get('paper')}; reason={adapt.get('reason')}",
                 registry.rel(hybrid_path))

    for topology in TOPOLOGIES:
        path = (root / "scripts" / "out" / "conv_bilstm_ae" / topology /
                "metadata.json")
        if not path.is_file():
            continue
        registry.ref(path)
        meta = json.loads(path.read_text(encoding="utf-8"))
        for key, value in (meta.get("implementation_notes", {}) or {}).items():
            _add(f"baselines.conv_bilstm_ae.{topology}", f"{key}: {value}",
                 registry.rel(path))
    return caveats


# --------------------------------------------------------------------------
# markdown rendering
# --------------------------------------------------------------------------
def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


METRIC_COLUMNS = ("mcc", "balanced_accuracy", "aucpr", "precision", "recall",
                  "f1", "tp", "fp", "fn", "tn")


def _fmt_time(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.1f}"


def _metric_row(label: str, metrics: dict[str, Any]) -> str:
    cells = [_fmt(metrics.get(col)) for col in METRIC_COLUMNS]
    return "| " + " | ".join([label] + cells) + " |"


def render_markdown(pack: dict[str, Any]) -> str:
    env = pack.get("environment", {})
    lines: list[str] = []
    lines.append("# DIMVA revision fact pack")
    lines.append("")
    lines.append(
        "Every number below is derived from the artifacts listed in "
        "`fact_pack.json` (`source_files` carries the SHA-256 of each "
        "consumed file). Metrics are recomputed with "
        "`stl.metrics.detection_metrics` whenever per-row predictions and the "
        "ground truth are available; otherwise the artifact-reported value "
        "is carried over and flagged `metric_source=\"artifact_reported\"`. "
        "Missing artifacts are reported as `unavailable` with a reason.")
    lines.append("")
    lines.append(f"- builder: `{pack.get('builder', {}).get('script')}` "
                 f"v{pack.get('builder', {}).get('version')}")
    lines.append(f"- repo root: `{env.get('repo_root')}`")
    lines.append(f"- data root: `{env.get('data_root')}`")
    lines.append(f"- artifacts root: `{env.get('artifacts_root')}`")
    lines.append(f"- git commit: `{env.get('git_commit')}` "
                 f"(dirty: {_fmt(env.get('git_dirty'))})")
    lines.append(f"- python {env.get('python_version')}, numpy "
                 f"{env.get('numpy_version')}, pandas "
                 f"{env.get('pandas_version')}, scikit-learn "
                 f"{env.get('scikit_learn_version')}")
    lines.append("")

    # --- proposed monitor
    lines.append("## 1. Proposed monitor (combined + ablation arms)")
    lines.append("")
    lines.append("Metric columns: " + ", ".join(METRIC_COLUMNS) + ".")
    lines.append("")
    lines.append("Arms without a continuous score (`gfsm_only`, "
                 "`invariants_only`, `stl_only`, `gfsm_or_invariants`, "
                 "`or_combined`) report a coarse average precision: their "
                 "binary 0/1 vector is used as the score input. Only "
                 "`monitor_fused_combined` has a real score "
                 "(`anomaly_score`), so its AP is ranking-based.")
    lines.append("")
    for topology, entry in sorted(
            pack.get("proposed_monitor", {}).items()):
        lines.append(f"### {topology}")
        lines.append("")
        if entry.get("status") != "ok":
            lines.append(f"- **unavailable** — {entry.get('reason')}")
            lines.append("")
            continue
        lines.append(f"- predictions: `{entry.get('prediction_path')}` "
                     f"({entry.get('n_rows')} rows, "
                     f"{entry.get('attack_rows')} attack rows)")
        lines.append("")
        lines.append("| arm | " + " | ".join(METRIC_COLUMNS) + " |")
        lines.append("|" + "---|" * (len(METRIC_COLUMNS) + 1))
        for arm, block in entry.get("arms", {}).items():
            lines.append(_metric_row(arm, block.get("metrics", {})))
        lines.append("")
        for scen, block in entry.get("arms", {}).get(
                "monitor_fused_combined", {}).get(
                    "per_scenario", {}).items():
            lines.append(f"- scenario `{scen}` (combined monitor): " +
                         ", ".join(f"{col}={_fmt(block['metrics'].get(col))}"
                                   for col in METRIC_COLUMNS))
        dataset = entry.get("dataset", {})
        if dataset.get("status") == "ok":
            lines.append(f"- dataset: {dataset.get('calibration_frames')} "
                         f"calibration frames ({dataset.get('calibration_rows')} "
                         "rows), evaluation scenarios: " +
                         ", ".join(
                             f"{name} ({info.get('rows')} rows, "
                             f"{info.get('attack_rows')} attack)"
                             for name, info in (dataset.get(
                                 "evaluation_scenarios") or {}).items()))
        prov = entry.get("detector_provenance", {}) or {}
        inv = prov.get("invariants_manifest", {}) or {}
        if inv.get("path"):
            cal = inv.get("calibration") or {}
            niaarm = inv.get("niaarm") or {}
            split = cal.get("split") or {}
            split_summary = {
                k: (v if isinstance(v, (int, float, str, bool))
                    else f"<{type(v).__name__}>")
                for k, v in split.items()
            } if isinstance(split, dict) else split
            lines.append(
                f"- invariants detector: `{inv.get('path')}` "
                f"(calibration mode `{cal.get('mode')}`, "
                f"fit rows {cal.get('fit_rows')}, "
                f"threshold rows {cal.get('threshold_rows')} from "
                f"`{cal.get('threshold_source')}`, "
                f"split {split_summary}, "
                f"seed {niaarm.get('seed')}, "
                f"{inv.get('n_states')} mined states)")
        gfsm = prov.get("gfsm_manifest", {}) or {}
        if gfsm.get("path"):
            gfsm_state = gfsm.get("gfsm") or {}
            lines.append(f"- GFSM detector: `{gfsm.get('path')}` "
                         f"(states {gfsm_state.get('states')}, "
                         f"transitions {gfsm_state.get('transitions')})")
        lines.append("")

    # --- baselines
    lines.append("## 2. Published baselines")
    lines.append("")
    for name, entry in sorted(pack.get("baselines", {}).items()):
        lines.append(f"### {name}")
        lines.append("")
        for topology, block in sorted(entry.items()):
            if block.get("status") != "ok":
                lines.append(f"- **{topology}**: unavailable — "
                             f"{block.get('reason')}")
                continue
            lines.append(f"- **{topology}** ({block.get('method')}), "
                         f"metric source: {block.get('metric_source')}")
            lines.append("")
            lines.append("| setting | " + " | ".join(METRIC_COLUMNS) + " |")
            lines.append("|" + "---|" * (len(METRIC_COLUMNS) + 1))
            if name == "sain_local_vs_global":
                lines.append(_metric_row("local (SAIN)",
                                         block.get("local", {}).get(
                                             "metrics", {})))
                lines.append(_metric_row("global (SAIN)",
                                         block.get("global", {}).get(
                                             "metrics", {})))
                for sc in block.get("scenarios", []):
                    lines.append(_metric_row(
                        f"local / {sc.get('name')}",
                        sc.get("local", {}).get("metrics", {})))
                    lines.append(_metric_row(
                        f"global / {sc.get('name')}",
                        sc.get("global", {}).get("metrics", {})))
            else:
                lines.append(_metric_row("overall",
                                         block.get("metrics", {})))
                for scen, sc_block in block.get("per_scenario", {}).items():
                    lines.append(_metric_row(
                        f"{scen}", sc_block.get("metrics", {})))
            lines.append("")
            extra = []
            if block.get("prediction_path"):
                extra.append(f"artifact: `{block['prediction_path']}`")
            if block.get("dataset"):
                extra.append(f"dataset/split: "
                             f"{json.dumps(block['dataset'])[:200]}")
            if block.get("seed") is not None:
                extra.append(f"seed: {block['seed']}")
            if block.get("software"):
                extra.append(f"software: "
                             f"{json.dumps(block['software'])[:200]}")
            for item in extra:
                lines.append(f"- {item}")
            for note in block.get("notes", []) or []:
                lines.append(f"- note: {note}")
            lines.append("")
        lines.append("")

    # --- full signal
    lines.append("## 3. Full-signal simulation runs")
    lines.append("")
    lines.append("| run | steps | attack steps | physical-impact steps | "
                 "events | attack window (s) | detection metrics |")
    lines.append("|---|---|---|---|---|---|---|")
    full_signal = pack.get("full_signal", {})
    if full_signal.get("status") != "ok":
        lines.append(f"- unavailable: {full_signal.get('reason')}")
        lines.append("")
    for name, block in sorted((full_signal.get("runs") or {}).items()):
        if block.get("status") != "ok":
            lines.append(f"| {name} | n/a | n/a | n/a | n/a | n/a | "
                         f"unavailable: {block.get('reason')} |")
            continue
        win = block.get("attack_window", {})
        detection = block.get("detection_metrics", {})
        if "combined" in detection:
            metrics = detection["combined"]
            summary = (f"mcc={_fmt(metrics.get('mcc'))}, "
                       f"BA={_fmt(metrics.get('balanced_accuracy'))}, "
                       f"AUPRC={_fmt(metrics.get('aucpr'))}")
            unknown = detection.get("unknown_residuals")
            if unknown:
                summary += f", unknown={unknown}"
        else:
            summary = f"unavailable: {detection.get('reason')}"
        lines.append(
            f"| {name} | {block.get('n_steps')} | "
            f"{block.get('attack_steps')} | "
            f"{block.get('physical_impact_steps')} | "
            f"{block.get('events', {}).get('n_events')} | "
            f"{_fmt_time(win.get('start_s'))} -> "
            f"{_fmt_time(win.get('end_s'))} | {summary} |")
    lines.append("")

    # --- regime counts
    lines.append("## 4. Regime counts")
    lines.append("")
    for key, block in pack.get("regime_counts", {}).items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append(f"{block.get('definition', '')}")
        lines.append("")
        for entry in block.get("entries", []):
            label = f"{entry.get('method')} / {entry.get('topology', '')}"
            if entry.get("run"):
                label += f" / {entry['run']}"
            if entry.get("status") != "ok":
                lines.append(f"- **{label}**: unavailable — "
                             f"{entry.get('reason')}")
                continue
            bits = []
            for field in ("clean_validation_false_alarms_total",
                          "clean_validation_steps",
                          "clean_validation_days",
                          "allowed_false_alarms",
                          "false_alarm_budget_per_day",
                          "uncalibrated_rows", "evaluation_rows_total",
                          "calibrated_tanks", "tanks",
                          "degenerate_clusters", "attack_free_steps",
                          "injected_events"):
                if entry.get(field) is not None:
                    bits.append(f"{field}={_fmt(entry[field])}")
            lines.append(f"- **{label}**: " + ", ".join(bits))
            if entry.get("note"):
                lines.append(f"  - note: {entry['note']}")
        lines.append("")

    # --- unavailable
    lines.append("## 5. Unavailable / missing cases")
    lines.append("")
    if pack.get("unavailable"):
        for item in pack["unavailable"]:
            lines.append(f"- `{item.get('item')}` — {item.get('reason')}")
    else:
        lines.append("- (none)")
    lines.append("")

    # --- status labels
    lines.append("## 6. Status labels (preliminary / smoke)")
    lines.append("")
    status = pack.get("status_labels", {})
    lines.append(status.get("definition", ""))
    lines.append("")
    if status.get("entries"):
        for entry in status["entries"]:
            lines.append(f"- `{entry.get('path')}` `{entry.get('key')}` = "
                         f"{entry.get('value')}")
    else:
        lines.append("- (no artifact carries an explicit preliminary/smoke "
                     "label)")
    lines.append("")

    # --- caveats
    lines.append("## 7. Reproduction caveats")
    lines.append("")
    for item in pack.get("caveats", []):
        src = f" (source: `{item['source']}`)" if item.get("source") else ""
        lines.append(f"- **{item.get('scope')}** — {item.get('caveat')}{src}")
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_fact_pack",
        description="Build the artifact-backed DIMVA revision fact pack.")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT,
                        help="repository root (default: the checkout that "
                             "contains this script)")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="normalised dataset root (default: "
                             "<repo-root>/data)")
    parser.add_argument("--artifacts-root", type=Path, default=None,
                        help="baseline artifact root (default: "
                             "<repo-root>/scripts/out)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: "
                             "<repo-root>/scripts/out/fact_pack)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root \
        else root / "data"
    artifacts_root = Path(args.artifacts_root).resolve() \
        if args.artifacts_root else root / "scripts" / "out"
    out_dir = Path(args.out).resolve() if args.out \
        else artifacts_root / "fact_pack"

    pack = build_fact_pack(root, data_root, artifacts_root,
                           exclude_dirs=[out_dir])
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fact_pack.json"
    md_path = out_dir / "fact_pack.md"
    json_path.write_text(
        json.dumps(pack, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    md_path.write_text(render_markdown(pack), encoding="utf-8")
    n_unavailable = len(pack.get("unavailable", []))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"unavailable items: {n_unavailable}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
