"""Tests for scripts/build_fact_pack.py.

The tests build tiny temporary artifacts (never the real experiments) and
check the fact pack's schema, the numerical correctness of the recomputed
metrics, the handling of missing artifacts, and the determinism of the
output.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
)


def _load_builder():
    path = _REPO_ROOT / "scripts" / "build_fact_pack.py"
    spec = importlib.util.spec_from_file_location("build_fact_pack", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


# --------------------------------------------------------------------------
# fixture: a miniature repository with hand-checkable artifacts
# --------------------------------------------------------------------------
MITM_LABELS = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
DOS_LABELS = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
CTOWN_LABELS = [0, 0, 1, 1, 0, 0, 0, 0]

MITM_STL = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
MITM_GFSM = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
MITM_INV = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
MITM_FUSED = [0, 0, 1, 0, 1, 1, 0, 0, 0, 0]
MITM_SCORE = [0.0, 0.0, 0.9, 0.0, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0]

DOS_STL = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
DOS_GFSM = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
DOS_INV = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
DOS_FUSED = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
DOS_SCORE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0]

CTOWN_STL = [0, 0, 1, 1, 0, 0, 0, 0]
CTOWN_GFSM = [0, 0, 0, 0, 0, 0, 0, 0]
CTOWN_INV = [0, 0, 0, 0, 0, 0, 0, 0]
CTOWN_FUSED = [0, 0, 1, 1, 0, 0, 0, 0]
CTOWN_SCORE = [0.0, 0.0, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8")


def _write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _confusion(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(
        list(y_true), list(y_pred), labels=[0, 1]).ravel()
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def _artifact_metrics(y_true, y_pred, scores) -> dict:
    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
    counts = _confusion(y_true, y_pred)
    return {
        **counts,
        "mcc": float(matthews_corrcoef(list(y_true), list(y_pred))),
        "balanced_accuracy": float(
            balanced_accuracy_score(list(y_true), list(y_pred))),
        "precision": float(precision_score(list(y_true), list(y_pred),
                                           zero_division=0)),
        "recall": float(recall_score(list(y_true), list(y_pred),
                                     zero_division=0)),
        "f1": float(f1_score(list(y_true), list(y_pred), zero_division=0)),
        "attack_average_precision": float(
            average_precision_score(list(y_true), list(scores))),
    }


def _make_dataset(root: Path, topology: str, scenarios: dict) -> None:
    ds = root / "data" / topology / "dataset"
    (ds / "calibration").mkdir(parents=True, exist_ok=True)
    (ds / "evaluation").mkdir(parents=True, exist_ok=True)
    cal = pd.DataFrame({"x": [1.0, 2.0, 3.0], "label": [0, 0, 0]})
    cal.to_csv(ds / "calibration" / "cal_0000.csv", index=False)
    eval_files = []
    for name, labels in scenarios.items():
        rel = f"evaluation/{name}.csv"
        pd.DataFrame({"x": [1.0] * len(labels), "label": labels}).to_csv(
            ds / rel, index=False)
        eval_files.append(rel)
    _write_json(ds / "dataset_manifest.yaml", {
        "schema": "dataio/v1",
        "files": {"calibration": ["calibration/cal_0000.csv"],
                  "evaluation": eval_files},
    })


def _make_monitor(root: Path, topology: str, rows, scenarios) -> None:
    out = root / "data" / "generated" / topology / "monitor"
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "predictions.csv", rows)
    manifest = {"schema": "monitor/v1", "topology": topology,
                "detectors": ["stl", "gfsm", "invariants"], "fusion": "or",
                "all_ok": True, "scenarios": []}
    for name, (y_true, y_pred, score) in scenarios.items():
        manifest["scenarios"].append({
            "name": name,
            "n": len(y_true),
            "attack_rows": int(sum(y_true)),
            "metrics": {
                **_artifact_metrics(y_true, y_pred, score),
                "aucpr": _artifact_metrics(
                    y_true, y_pred, score)["attack_average_precision"],
                "aucroc": 0.5,
            },
        })
    _write_json(out / f"{topology}_monitor_manifest.json", manifest)


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()

    _make_dataset(root, "anytown", {"mitm": MITM_LABELS, "dos": DOS_LABELS})
    _make_dataset(root, "ctown", {"test": CTOWN_LABELS})

    def monitor_rows(scenarios):
        rows = []
        for name, (y_true, stl, gfsm, inv, fused, score) in scenarios:
            for i, (t, s, g, v, f, sc) in enumerate(
                    zip(y_true, stl, gfsm, inv, fused, score)):
                rows.append({
                    "scenario": name, "row": i, "y_true": t,
                    "y_pred_stl": s, "y_pred_gfsm": g, "y_pred_invariants": v,
                    "y_pred": f, "anomaly_score": sc})
        return rows

    _make_monitor(root, "anytown", monitor_rows([
        ("mitm", (MITM_LABELS, MITM_STL, MITM_GFSM, MITM_INV, MITM_FUSED,
                   MITM_SCORE)),
        ("dos", (DOS_LABELS, DOS_STL, DOS_GFSM, DOS_INV, DOS_FUSED,
                 DOS_SCORE)),
    ]), {
        "mitm": (MITM_LABELS, MITM_FUSED, MITM_SCORE),
        "dos": (DOS_LABELS, DOS_FUSED, DOS_SCORE),
    })
    _make_monitor(root, "ctown", monitor_rows([
        ("test", (CTOWN_LABELS, CTOWN_STL, CTOWN_GFSM, CTOWN_INV,
                  CTOWN_FUSED, CTOWN_SCORE)),
    ]), {
        "test": (CTOWN_LABELS, CTOWN_FUSED, CTOWN_SCORE),
    })

    out = root / "scripts" / "out"

    # -- SAIN (artifact-reported metrics only, no per-row predictions)
    for topology, scen in (("anytown", "mitm"), ("ctown", "test")):
        labels = MITM_LABELS if topology == "anytown" else CTOWN_LABELS
        local_pred = [1 if i % 3 == 0 else 0 for i in range(len(labels))]
        global_pred = [1 if i % 5 == 0 else 0 for i in range(len(labels))]
        _write_json(out / "sain" / topology / "sain_local_vs_global.json", {
            "schema": "baselines/compare/v1",
            "feature_cols": ["x"],
            "scenarios": [{
                "name": scen, "n": len(labels),
                "attack_rows": int(sum(labels)),
                "local": {"flags": int(sum(local_pred)),
                          "metrics": _artifact_metrics(
                              labels, local_pred, local_pred)},
                "global": {"flags": int(sum(global_pred)),
                           "metrics": _artifact_metrics(
                               labels, global_pred, global_pred)},
            }],
            "summary": {
                "local": _artifact_metrics(labels, local_pred, local_pred),
                "global": _artifact_metrics(labels, global_pred, global_pred),
            },
        })

    # -- IM-DISCO (artifact-reported metrics only)
    for topology, labels in (("anytown", MITM_LABELS),
                             ("ctown", CTOWN_LABELS)):
        counts = _confusion(labels, [0] * len(labels))
        _write_json(out / "imdisco" / topology / "metrics.json", {
            "n": len(labels), **counts,
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "mcc": 0.0,
            "balanced_accuracy": 0.5, "unadjusted_balanced_accuracy": 0.5,
            "average_precision": float(np.mean(labels)),
            "average_precision_kind": "coarse_binary_rule_match",
        })
        _write_json(out / "imdisco" / topology / "model.json", {
            "schema_version": "imdisco-baseline-v1",
            "training": {"n_rows": 3},
            "provenance": {"mlxtend_version": "0.25.0",
                           "reproduction_gaps": ["rule order unknown"]},
        })

    # -- state-aware CUSUM: C-Town has tanks, Anytown has none (N/A)
    cusum_ctown = root / "scripts" / "out" / "state_aware_cusum" / "ctown"
    cusum_pred = [0, 1, 0, 0, 1, 0, 0, 0]
    _write_csv(cusum_ctown / "state_aware_cusum_predictions.csv", [
        {"scenario": "test", "row": i, "y_pred": p, "cusum_score": 0.5 * p,
         "y_true": t}
        for i, (p, t) in enumerate(zip(cusum_pred, CTOWN_LABELS))])
    _write_json(cusum_ctown / "state_aware_cusum_metrics.json", {
        "test": {
            **_artifact_metrics(CTOWN_LABELS, cusum_pred, cusum_pred),
            "false_alarms": sum(
                1 for p, t in zip(cusum_pred, CTOWN_LABELS) if p == 1 and t == 0),
            "predicted_steps": sum(cusum_pred),
            "attack_steps": int(sum(CTOWN_LABELS)), "n": len(CTOWN_LABELS),
        }})
    _write_json(cusum_ctown / "state_aware_cusum_config.json", {
        "schema": "baselines/state_aware_cusum/v1",
        "tanks": ["l_t1"], "alpha": 1.0, "tau": 2.0,
    })
    _write_json(cusum_ctown / "state_aware_cusum_provenance.json", {
        "validation": {"false_alarm_budget_per_day": 1.0,
                       "selection": {"validation_steps": 8,
                                     "allowed_false_alarms": 1,
                                     "per_tank": {
                                         "l_t1": {"validation_false_alarms": 1,
                                                  "alpha_mm": 1.0,
                                                  "tau_mm": 2.0}}}},
        "missing_original_observer_parameters": {
            "note": "the SAC 2018 paper does not report the tank area"},
    })
    cusum_anytown = root / "scripts" / "out" / "state_aware_cusum" / "anytown"
    _write_csv(cusum_anytown / "state_aware_cusum_predictions.csv", [
        {"scenario": "mitm", "row": i, "y_pred": 0, "cusum_score": 0.0,
         "y_true": t} for i, t in enumerate(MITM_LABELS)])
    _write_json(cusum_anytown / "state_aware_cusum_metrics.json", {
        "mitm": {**_confusion(MITM_LABELS, [0] * 10),
                 "attack_average_precision": float(np.mean(MITM_LABELS)),
                 "mcc": 0.0, "balanced_accuracy": 0.5, "precision": 0.0,
                 "recall": 0.0, "f1": 0.0, "n": 10,
                 "attack_steps": 2, "predicted_steps": 0,
                 "false_alarms": 0, "false_alarm_rate": 0.0}})
    _write_json(cusum_anytown / "state_aware_cusum_config.json", {
        "schema": "baselines/state_aware_cusum/v1", "tanks": [],
        "area_m2": {}, "flows": {}})
    _write_json(cusum_anytown / "state_aware_cusum_provenance.json", {
        "validation": {"selection": {"per_tank": {}}}})

    # -- ConvBiLSTM-AE: predictions + errors, no labels
    conv = root / "scripts" / "out" / "conv_bilstm_ae"
    conv_pred = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    conv_err = [0.1, 0.9, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1]
    for topology, labels in (("anytown", MITM_LABELS),
                             ("ctown", CTOWN_LABELS)):
        base = conv / topology
        base.mkdir(parents=True, exist_ok=True)
        np.savez(base / "predictions.npz",
                 predictions=np.asarray(conv_pred[:len(labels)], dtype=np.int64),
                 errors=np.asarray(conv_err[:len(labels)], dtype=float),
                 row_predictions=np.asarray(conv_pred[:len(labels)],
                                           dtype=np.int64),
                 row_errors=np.asarray(conv_err[:len(labels)], dtype=float))
        _write_json(base / "metadata.json", {
            "seed": 42, "threshold": 0.5,
            "threshold_rule": "quantile(0.995) of validation errors",
            "dataset": {"n_train_rows": 3, "n_validation_rows": 2},
            "software": {"numpy_version": "2.4.5"},
        })

    # -- hybrid ensemble: C-Town only, probabilities + stored labels
    hybrid = root / "scripts" / "out" / "hybrid_ensemble" / "ctown"
    hybrid.mkdir(parents=True, exist_ok=True)
    hybrid_true = CTOWN_LABELS[1:]           # one dropped head row
    hybrid_prob = [0.1, 0.9, 0.2, 0.1, 0.1, 0.0, 0.0]
    np.savez(hybrid / "test_predictions.npz",
             probabilities=np.asarray(hybrid_prob, dtype=float),
             labels=np.asarray(hybrid_true, dtype=np.int64))
    _write_json(hybrid / "run_result.json", {
        "schema": "baselines/hybrid_ensemble/run/v1",
        "artifacts": {"train_rows": 3},
        "metrics_test": {
            "schema": "baselines/hybrid_ensemble/metrics/v1",
            "threshold": 0.5, "threshold_tuned": False,
            "dropped_rows": 1, "n_samples": len(hybrid_true),
            "n_positive": int(sum(hybrid_true)),
            **_artifact_metrics(hybrid_true,
                                [1 if p >= 0.5 else 0 for p in hybrid_prob],
                                hybrid_prob),
            "average_precision": float(average_precision_score(
                hybrid_true, hybrid_prob)),
            "accuracy": 0.5,
            "pr_auc": 0.0, "roc_auc": 0.5,
            "precision_attack": 0.0, "recall_attack": 0.0,
            "f1_attack": 0.0, "partition": "official_test:test",
        },
        "provenance": {"seed": 42,
                       "library_versions": {"numpy": "2.4.5"},
                       "adaptations": [{"id": "fixed_threshold",
                                        "ours": "0.5", "paper": "0.5",
                                        "reason": "untuned"}]},
    })

    # -- SICAMS: per-row flags + moving average
    sicams = root / "scripts" / "out" / "sicams"
    sicams_flags = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    sicams_score = [0.9, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1]
    for topology, labels in (("anytown", MITM_LABELS),
                             ("ctown", CTOWN_LABELS)):
        base = sicams / topology
        base.mkdir(parents=True, exist_ok=True)
        flags = sicams_flags[:len(labels)]
        score = sicams_score[:len(labels)]
        _write_json(base / "evaluation.json", {
            "schema": "baselines/sicams/v1",
            "config": {"significance": 0.05, "hysteresis": True},
            "metrics": {
                **_artifact_metrics(labels, flags, score),
                "n_positive": int(sum(labels)),
                "n_samples": len(labels), "positive_class": "attack",
                "scope": "pointwise_binary_detection_only",
                "score_for_ap": "causal_moving_average_tf2",
                "balanced_accuracy_adjusted": False,
            },
            "predictions": {
                "flags": flags, "moving_average": score,
                "cluster_ids": [0] * len(labels),
                "lower_threshold": [1.0] * len(labels),
                "upper_threshold": [2.0] * len(labels),
                "t2": [1.0] * len(labels), "tf2": [1.0] * len(labels),
            },
            "provenance": {
                "method": "SICAMS",
                "temporal_clusters": {"n_clusters": 1,
                                      "assignment": "explicit_cluster_ids",
                                      "clusters": [
                                          {"cluster_id": 0, "n_k": 10, "s": 1,
                                           "covariance_rank": 1,
                                           "lower_threshold": 1.0,
                                           "upper_threshold": 2.0}]},
                "missing_original_settings": ["alpha not reported"],
                "omitted_native_components": ["PELT change-point detection"],
            },
        })

    # -- full-signal runs
    fs = root / "scripts" / "out" / "full_signal" / "anytown" / "mitm"
    fs.mkdir(parents=True, exist_ok=True)
    _write_csv(fs / "labels.csv", [
        {"time_seconds": t, "attack_active": bool(t >= 3),
         "delivery_or_command_modified": bool(t >= 3),
         "physical_impact": bool(t >= 4)} for t in range(6)])
    _write_json(fs / "events.json", [
        {"event_id": 0, "kind": "sensor", "mode": "offset", "target": "T41",
         "start": 3, "end": 5, "time_seconds": 3, "value": 7.0}])
    _write_json(fs / "manifest.json", {
        "schema_version": 1, "run_id": "anytown-mitm-full", "seed": 42,
        "split": "evaluation", "engine": "WNTRSimulator", "wntr_version": "1.5.0",
        "timesteps": {"duration": 5.0},
    })
    nominal = root / "scripts" / "out" / "full_signal" / "anytown" / "nominal"
    nominal.mkdir(parents=True, exist_ok=True)
    _write_csv(nominal / "labels.csv", [
        {"time_seconds": t, "attack_active": False,
         "delivery_or_command_modified": False, "physical_impact": False}
        for t in range(4)])
    _write_json(nominal / "events.json", [])
    return root


def _build(root: Path, out_dir: Path) -> dict:
    rc = builder.main([
        "--repo-root", str(root),
        "--data-root", str(root / "data"),
        "--artifacts-root", str(root / "scripts" / "out"),
        "--out", str(out_dir),
    ])
    assert rc == 0
    return json.loads((out_dir / "fact_pack.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------
def test_pack_schema(mini_repo: Path, tmp_path: Path) -> None:
    pack = _build(mini_repo, tmp_path / "out")
    assert pack["schema"] == "fact_pack/v1"
    for key in ("builder", "environment", "metric_definitions",
                "proposed_monitor", "baselines", "full_signal",
                "regime_counts", "unavailable", "caveats", "source_files"):
        assert key in pack, key
    assert pack["builder"]["metric_backend"] == "stl.metrics.detection_metrics"
    # every metric block carries the canonical keys
    arm = pack["proposed_monitor"]["anytown"]["arms"]["stl_only"]["metrics"]
    for key in ("precision", "recall", "f1", "mcc", "balanced_accuracy",
                "unadjusted_balanced_accuracy", "aucpr", "aucroc",
                "tp", "fp", "tn", "fn"):
        assert key in arm, key
    # digests are recorded for the consumed artifacts
    assert pack["source_files"]
    for entry in pack["source_files"].values():
        assert len(entry["sha256"]) == 64


def test_markdown_is_written(mini_repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _build(mini_repo, out)
    md = (out / "fact_pack.md").read_text(encoding="utf-8")
    assert "# DIMVA revision fact pack" in md
    assert "## 1. Proposed monitor" in md
    assert "stl_only" in md
    assert "unavailable" in md


# --------------------------------------------------------------------------
# numerical correctness
# --------------------------------------------------------------------------
def test_monitor_arms_are_recomputed(mini_repo: Path, tmp_path: Path) -> None:
    pack = _build(mini_repo, tmp_path / "out")
    arms = pack["proposed_monitor"]["anytown"]["arms"]
    assert arms["stl_only"]["metric_source"] == "recomputed_from_predictions"

    mitm = arms["stl_only"]["per_scenario"]["mitm"]["metrics"]
    assert (mitm["tp"], mitm["fp"], mitm["fn"], mitm["tn"]) == (2, 0, 0, 8)
    assert mitm["mcc"] == pytest.approx(
        float(matthews_corrcoef(MITM_LABELS, MITM_STL)))

    gfsm = arms["gfsm_only"]["per_scenario"]["mitm"]["metrics"]
    assert (gfsm["tp"], gfsm["fp"], gfsm["fn"], gfsm["tn"]) == (0, 1, 2, 7)

    inv = arms["invariants_only"]["per_scenario"]["mitm"]["metrics"]
    assert (inv["tp"], inv["fp"], inv["fn"], inv["tn"]) == (1, 0, 1, 8)

    fused = arms["monitor_fused_combined"]["per_scenario"]["mitm"]["metrics"]
    assert (fused["tp"], fused["fp"], fused["fn"], fused["tn"]) == (2, 1, 0, 7)
    assert fused["aucpr"] == pytest.approx(
        float(average_precision_score(MITM_LABELS, MITM_SCORE)))

    combined = arms["or_combined"]["per_scenario"]["mitm"]["metrics"]
    assert (combined["tp"], combined["fp"], combined["fn"], combined["tn"]) \
        == (2, 1, 0, 7)

    # the recomputed fused arm must agree with the recorded monitor manifest
    entry = pack["proposed_monitor"]["anytown"]
    assert entry["manifest_cross_check"]["per_scenario"]["mitm"][
        "recomputed_matches_recorded"] is True
    assert entry["dataset"]["status"] == "ok"
    assert entry["dataset"]["evaluation_scenarios"]["mitm"]["rows"] == 10


def test_cusum_metrics_recomputed_and_cross_checked(
        mini_repo: Path, tmp_path: Path) -> None:
    pack = _build(mini_repo, tmp_path / "out")
    entry = pack["baselines"]["state_aware_cusum"]["ctown"]
    assert entry["status"] == "ok"
    assert entry["metric_source"] == "recomputed_from_predictions"
    metrics = entry["metrics"]
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) \
        == (0, 2, 2, 4)
    check = entry["artifact_cross_check"]["test"]
    assert check["recomputed_matches_recorded"] is True
    assert check["aucpr_matches_recorded"] is True


def test_sicams_metrics_recomputed(mini_repo: Path, tmp_path: Path) -> None:
    pack = _build(mini_repo, tmp_path / "out")
    entry = pack["baselines"]["sicams"]["anytown"]
    assert entry["status"] == "ok"
    metrics = entry["metrics"]
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) \
        == (1, 1, 1, 7)
    assert entry["ground_truth"]["scenario"] == "mitm"
    assert entry["ground_truth"]["scenario_alignment"]["matching_candidates"] \
        == ["mitm"]


def test_hybrid_ensemble_recomputed(mini_repo: Path, tmp_path: Path) -> None:
    pack = _build(mini_repo, tmp_path / "out")
    entry = pack["baselines"]["hybrid_ensemble"]["ctown"]
    assert entry["status"] == "ok"
    assert entry["metric_source"] == "recomputed_from_predictions"
    reported = entry["artifact_reported_metrics"]
    metrics = entry["metrics"]
    assert metrics["tp"] == reported["tp"]
    assert metrics["fp"] == reported["fp"]
    assert metrics["aucpr"] == pytest.approx(reported["average_precision"])
    assert entry["ground_truth"]["rows_dropped"] == 1


def test_conv_bilstm_uses_dataset_labels(mini_repo: Path,
                                         tmp_path: Path) -> None:
    pack = _build(mini_repo, tmp_path / "out")
    entry = pack["baselines"]["conv_bilstm_ae"]["ctown"]
    assert entry["status"] == "ok"
    metrics = entry["metrics"]
    assert metrics["n"] == len(CTOWN_LABELS)
    assert (metrics["tp"], metrics["fp"]) == (0, 2)


def test_artifact_reported_metrics_are_labelled(
        mini_repo: Path, tmp_path: Path) -> None:
    pack = _build(mini_repo, tmp_path / "out")
    imdisco = pack["baselines"]["imdisco"]["ctown"]
    assert imdisco["metric_source"] == "artifact_reported"
    assert imdisco["metrics"]["mcc"] == 0.0
    assert imdisco["metrics"]["aucpr"] == pytest.approx(
        float(np.mean(CTOWN_LABELS)))
    sain = pack["baselines"]["sain_local_vs_global"]["ctown"]
    assert sain["metric_source"] == "artifact_reported"
    assert sain["local"]["metrics"]["tp"] == 1  # local flags every 3rd row
    assert sain["local"]["metrics"]["fp"] == 2


# --------------------------------------------------------------------------
# N/A handling
# --------------------------------------------------------------------------
def test_cusum_anytown_is_not_applicable(mini_repo: Path,
                                         tmp_path: Path) -> None:
    pack = _build(mini_repo, tmp_path / "out")
    entry = pack["baselines"]["state_aware_cusum"]["anytown"]
    assert entry["status"] == "unavailable"
    assert "reason" in entry and entry["reason"]
    assert "metrics" not in entry
    items = {i["item"] for i in pack["unavailable"]}
    assert "baselines.state_aware_cusum.anytown" in items


def test_missing_artifact_is_unavailable(mini_repo: Path,
                                         tmp_path: Path) -> None:
    sicams = mini_repo / "scripts" / "out" / "sicams" / "ctown"
    (sicams / "evaluation.json").unlink()
    pack = _build(mini_repo, tmp_path / "out")
    entry = pack["baselines"]["sicams"]["ctown"]
    assert entry["status"] == "unavailable"
    assert entry["reason"]
    assert "metrics" not in entry


def test_hybrid_anytown_is_unavailable(mini_repo: Path,
                                       tmp_path: Path) -> None:
    pack = _build(mini_repo, tmp_path / "out")
    entry = pack["baselines"]["hybrid_ensemble"]["anytown"]
    assert entry["status"] == "unavailable"
    assert "BATADAL" in entry["reason"]


def test_full_signal_detection_is_unavailable(mini_repo: Path,
                                              tmp_path: Path) -> None:
    pack = _build(mini_repo, tmp_path / "out")
    runs = pack["full_signal"]["runs"]
    run = runs["anytown/mitm"]
    assert run["status"] == "ok"
    assert run["attack_steps"] == 3
    assert run["physical_impact_steps"] == 2
    assert run["events"]["n_events"] == 1
    assert run["detection_metrics"]["status"] == "unavailable"
    nominal = runs["anytown/nominal"]
    assert nominal["attack_steps"] == 0
    assert nominal["events"]["n_events"] == 0


def test_nominal_holdout_predictions_use_held_out_denominator(
        mini_repo: Path, tmp_path: Path) -> None:
    pred_dir = (mini_repo / "scripts" / "out" / "topology_predictions" /
                "anytown")
    pred_dir.mkdir(parents=True)
    pd.DataFrame({
        "scenario": ["nominal", "nominal"],
        "row": [0, 1],
        "y_true": [0, 0],
        "y_pred_stl": [0, 1],
        "y_pred_gfsm": [0, 0],
        "y_pred": [0, 1],
        "anomaly_score": [0.0, 1.0],
        "uncalibrated_count": [0, 0],
    }).to_csv(pred_dir / "detector_predictions.csv", index=False)

    pack = _build(mini_repo, tmp_path / "out")
    detection = pack["full_signal"]["runs"]["anytown/nominal"]["detection_metrics"]
    assert detection["evaluation_scope"] == "held_out_nominal_rows"
    assert detection["source_trace_rows"] == 4
    assert detection["combined"]["fp"] == 1
    entries = pack["regime_counts"]["nominal_false_alarms"]["entries"]
    nominal = next(e for e in entries if e.get("method") == "full-signal simulation")
    assert nominal["attack_free_steps"] == 2
    assert nominal["calibration_rows_excluded"] == 2
    assert nominal["clean_validation_false_alarms_total"] == 1


def test_regime_counts(mini_repo: Path, tmp_path: Path) -> None:
    pack = _build(mini_repo, tmp_path / "out")
    nominal = pack["regime_counts"]["nominal_false_alarms"]["entries"]
    ctown = [e for e in nominal
             if e.get("method") == "state-aware CUSUM"
             and e.get("topology") == "ctown"]
    assert ctown and ctown[0]["status"] == "ok"
    assert ctown[0]["clean_validation_false_alarms_total"] == 1
    anytown = [e for e in nominal
               if e.get("method") == "state-aware CUSUM"
               and e.get("topology") == "anytown"]
    assert anytown and anytown[0]["status"] == "unavailable"
    uncal = pack["regime_counts"]["uncalibrated_regime"]["entries"]
    sicams = [e for e in uncal if e.get("method") == "SICAMS"]
    assert sicams and all(e["uncalibrated_rows"] == 0 for e in sicams)


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------
def test_output_is_deterministic(mini_repo: Path, tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    _build(mini_repo, out_a)
    _build(mini_repo, out_b)
    assert (out_a / "fact_pack.json").read_text(encoding="utf-8") == \
        (out_b / "fact_pack.json").read_text(encoding="utf-8")
    assert (out_a / "fact_pack.md").read_text(encoding="utf-8") == \
        (out_b / "fact_pack.md").read_text(encoding="utf-8")
