"""SAIN-style *local* per-CASE-state invariant baseline.

This module is the separately named **local-vs-global adaptation**
(:class:`SainLocalDetector`), not the paper-faithful native SAIN path.  The
source-derived, per-global-state implementation is in
:mod:`baselines.sain_native`; NiaARM is used only here and must never be
described as native SAIN.

Upstream reference (official, cited here so the baseline is traceable)
---------------------------------------------------------------------
* Paper: "SAIN: Improving ICS Attack Detection Sensitivity via State-Aware
  Invariants", USENIX Security 2024 (Abbas et al.).
  https://www.usenix.org/conference/usenixsecurity24/presentation/abbas
* Official code/artifacts: https://github.com/purseclab/SAIN
  pinned commit ``0a55724c7a5a352635d235d5f1a2e741045c4bed``
  (2024-07-18, "Add files via upload").

What this module is — and is not
--------------------------------
SAIN derives *state-aware* invariants: the plant traces are partitioned into
state-specific sub-traces and the invariant bounds are quantified per state,
so the bounds are tighter than state-agnostic ones.  The official
implementation does that for the *global* ICS state of its two testbeds and
enforces the invariants inside a PLC runtime monitor.

This module implements a **SAIN-style local** variant of that idea on top of
the repository's existing machinery:

* the partition is **local per CASE state**: each GFSM component (one PLC
  CASE selector) gets its own state partition and its own rule set, mined
  separately, instead of the cross-product composite state used by the
  ``invariants`` stage;
* the rules are mined with the repository's existing NiaARM miner
  (:func:`invariants.mine.mine_state`) and the same quality filter
  (confidence >= 0.7, support >= 0.1);
* the violation threshold is selected per component on a **validation**
  split, so no threshold is calibrated on the rows its rules were fitted on.

This is **not** a byte-exact reproduction of the official SAIN artifact and
must not be described as one.  The adaptations are:

1. **Passive per-row monitor.**  SAIN's SMONITOR is an active PLC-runtime
   agent that intercepts external variable-write requests and *denies* them
   when an invariant would be violated.  Here the monitor is passive: it
   scores recorded trace rows after the fact.  Nothing is blocked, and no
   request-level causality is modelled.
2. **NiaARM instead of Quantminer.**  The official implementation quantifies
   invariant templates with Quantminer.  Quantminer's configuration is not
   reproducible here, so the association-rule miner already vendored in this
   repository (NiaARM + Niapy PSO, with the same confidence/support filter)
   is used instead.  The mined rules are therefore association rules, not
   the paper's single-variable/multi-variable bound templates.
3. **Surrogate PLC ST.**  The official artifact ships the real Structured
   Text of its Fischertechnik and Tennessee Eastman testbeds.  Here the PLC
   programs are the surrogate/generated ST of this repository, so the state
   partition comes from the GFSM CASE selectors rather than from SAIN's own
   PDG/FSM extraction of a real program.
4. **Local granularity.**  SAIN quantifies invariants per global ICS state.
   This baseline quantifies them per local CASE state of each component, so
   that the local and the global partition can be compared on identical data.

Runtime semantics
-----------------
Per component the detector keeps

* the rules mined for each *local* state of that component, and
* a per-component violation threshold selected on the validation split.

At prediction time a row is flagged when **any** component flags it (the
OR of the local detectors).  A row whose local state was never mined for a
component is flagged as an *unknown local state*.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd

from invariants.mine import MinerConfig, mine_state
from invariants.model import InvariantsError, MinedRule
from invariants.rule_eval import frame_violation_counts
from invariants.threshold import select_violation_threshold

from monitor.protocol import StepFlags

__all__ = [
    "SAIN_CODE_COMMIT",
    "SAIN_CODE_URL",
    "SAIN_PAPER_URL",
    "SAIN_ADAPTATIONS",
    "SAIN_SCHEMA",
    "ComparisonScenario",
    "SainLocalDetector",
    "compare_local_vs_global",
    "local_state_ids",
    "mine_local_invariants",
    "run_sain_local_comparison",
]

#: Official upstream code/artifacts for the SAIN paper.
SAIN_CODE_URL = "https://github.com/purseclab/SAIN"
SAIN_CODE_COMMIT = "0a55724c7a5a352635d235d5f1a2e741045c4bed"
SAIN_PAPER_URL = (
    "https://www.usenix.org/conference/usenixsecurity24/presentation/abbas"
)

#: The adaptations listed in the module docstring, kept machine-readable so
#: the artifact and the docs cannot drift apart.
SAIN_ADAPTATIONS: tuple[str, ...] = (
    "this class is the local per-CASE-state adaptation, not native SAIN; "
    "the source-derived per-global-state path is baselines.sain_native",
    "passive per-row monitor (upstream SMONITOR is an active PLC-runtime "
    "write-request interceptor; here violations are only scored on recorded "
    "rows, nothing is denied)",
    "NiaARM association rules replace the upstream Quantminer invariant-"
    "template quantification (Quantminer configuration not reproducible "
    "here); confidence>=0.7 / support>=0.1 filters are kept",
    "surrogate/generated PLC Structured Text replaces the upstream testbed "
    "programs; the state partition comes from the GFSM CASE selectors, not "
    "from SAIN's PDG-based FSM extraction",
    "local per-CASE-state granularity: each component is mined and "
    "calibrated separately instead of per global ICS state",
)

SAIN_SCHEMA = "baselines/sain_local/v1"


def local_state_ids(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Per-row local state id of one component, as strings.

    The id is the integer value of the component's CASE-selector column,
    exactly the id the GFSM composite label uses for that component
    (``<plc>.<case_var>:<sid>``), so local and global state ids are
    directly comparable.
    """
    if column not in frame.columns:
        raise InvariantsError(
            f"local state column {column!r} missing from frame "
            f"(columns={list(frame.columns)[:8]})"
        )
    values = frame[column].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise InvariantsError(
            f"non-finite value in local state column {column!r}; cannot "
            "derive the local state"
        )
    return values.astype(int).astype(str)


@dataclass
class _ComponentModel:
    """Per-component local invariant model."""

    plc: str
    case_var: str
    column: str
    states: dict[str, dict[str, Any]] = field(default_factory=dict)
    threshold: int = 1
    threshold_note: str | None = None

    @property
    def key(self) -> str:
        return f"{self.plc}.{self.case_var}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "plc": self.plc,
            "case_var": self.case_var,
            "column": self.column,
            "violation_threshold": int(self.threshold),
            "violation_threshold_note": self.threshold_note,
            "local_states": self.states,
        }


def _mine_component(
    *,
    plc: str,
    case_var: str,
    column: str,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: MinerConfig,
    min_observations: int,
    keep_going: bool,
) -> tuple[_ComponentModel, bool]:
    """Mine the rules of one component, per local state."""
    model = _ComponentModel(plc=plc, case_var=case_var, column=column)
    ids = local_state_ids(train_df, column)
    all_ok = True
    for state_id in sorted(set(ids.tolist())):
        slice_df = train_df.loc[ids == state_id]
        nobs = int(len(slice_df))
        if nobs < min_observations:
            model.states[state_id] = {
                "observations": nobs,
                "status": "insufficient_data",
                "rules": [],
            }
            continue
        try:
            rules: list[MinedRule] = mine_state(slice_df, feature_cols, cfg)
            model.states[state_id] = {
                "observations": nobs,
                "status": "ok",
                "rules": [r.to_dict() for r in rules],
            }
        except Exception as exc:  # noqa: BLE001 - mining is best-effort
            all_ok = False
            model.states[state_id] = {
                "observations": nobs,
                "status": "mining_error",
                "errors": [repr(exc)],
                "rules": [],
            }
            if not keep_going:
                break
    return model, all_ok


def _calibrate_component(
    model: _ComponentModel,
    val_df: pd.DataFrame,
    fp_budget: float,
) -> None:
    """Select the component's violation threshold on the validation rows.

    The threshold is chosen on validation rows only — never on the rows the
    rules were fitted on.  A validation row whose local state was never
    mined counts as *state-absent*, mirroring the runtime signal the monitor
    raises for an unknown state.
    """
    ids = local_state_ids(val_df, model.column)
    counts = np.zeros(len(val_df), dtype=int)
    known = np.zeros(len(val_df), dtype=bool)
    for state_id, entry in model.states.items():
        sel = ids == state_id
        if not sel.any():
            continue
        known |= sel
        rules = entry.get("rules") or []
        if not rules:
            continue
        counts[sel] = frame_violation_counts(val_df.loc[sel], rules)
    absent = ~known
    threshold, note = select_violation_threshold(counts, absent, fp_budget)
    model.threshold = int(threshold)
    model.threshold_note = note


def mine_local_invariants(
    *,
    topology: str,
    components: Sequence[tuple[str, str]],
    fb_to_col: dict[tuple[str, str], str],
    training_frames: Sequence[pd.DataFrame],
    validation_frames: Sequence[pd.DataFrame],
    feature_cols: Sequence[str],
    min_observations: int = 50,
    max_evals: int = 5000,
    seed: int = 42,
    fp_budget: float = 0.01,
    keep_going: bool = True,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Mine the local (per-CASE-state) SAIN-style invariant model.

    Rules are mined on ``training_frames`` only; the per-component violation
    thresholds are selected on ``validation_frames`` only.  The two frame
    lists must be disjoint in practice — nothing here can enforce that, but
    the caller is expected to pass the output of
    :func:`dataio.splits.split_calibration_frames`.

    ``feature_cols`` is used verbatim for every component, so the local
    model and the global ``invariants`` stage can be compared on identical
    feature columns.
    """
    if not training_frames:
        raise InvariantsError("mine_local_invariants needs training frames")
    if not validation_frames:
        raise InvariantsError(
            "mine_local_invariants needs validation frames: the violation "
            "threshold must be selected on rows the rules were not fitted on"
        )
    train_df = pd.concat(list(training_frames), ignore_index=True)
    val_df = pd.concat(list(validation_frames), ignore_index=True)
    feature_cols = list(feature_cols)

    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1")
    cfg = MinerConfig(max_evals=max_evals, seed=seed)
    models: dict[str, _ComponentModel] = {}
    all_ok = True

    jobs = []
    for plc, case_var in components:
        column = fb_to_col.get((plc, case_var))
        if column is None:
            raise InvariantsError(
                f"no dataset column resolved for component "
                f"(plc={plc}, case_var={case_var!r})"
            )
        jobs.append(dict(
            plc=plc, case_var=case_var, column=column,
            train_df=train_df, feature_cols=feature_cols, cfg=cfg,
            min_observations=min_observations, keep_going=keep_going,
        ))
    if n_jobs == 1:
        results = [_mine_component(**job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(_mine_component, **job) for job in jobs]
            results = [future.result() for future in futures]
    for model, ok in results:
        models[model.key] = model
        all_ok = all_ok and ok

    for model in models.values():
        _calibrate_component(model, val_df, fp_budget)

    return {
        "schema": SAIN_SCHEMA,
        "topology": topology,
        "upstream": {
            "paper_url": SAIN_PAPER_URL,
            "code_url": SAIN_CODE_URL,
            "code_commit": SAIN_CODE_COMMIT,
            "reproduction": "not byte-exact; see adaptations",
        },
        "adaptations": list(SAIN_ADAPTATIONS),
        "granularity": "local_per_case_state",
        "niaarm": {
            "max_evals": int(max_evals),
            "confidence_min": 0.7,
            "support_min": 0.1,
            "min_observations": int(min_observations),
            "seed": int(seed),
            "algorithm": "ParticleSwarmAlgorithm",
            "n_jobs": int(n_jobs),
        },
        "violation_fp_budget": float(fp_budget),
        "feature_cols": feature_cols,
        "calibration": {
            "mode": "train_validation_split",
            "threshold_source": "validation",
            "fit_rows": int(len(train_df)),
            "threshold_rows": int(len(val_df)),
        },
        "components": {k: m.to_dict() for k, m in models.items()},
        "all_ok": all_ok,
    }


class SainLocalDetector:
    """Runtime detector for the local SAIN-style model.

    Flags a row when *any* component flags it (OR of the local detectors).
    A component flags a row when the row's local state for that component
    was never mined (*unknown local state*), or when the number of violated
    rules of that local state reaches the component's validation-selected
    threshold.  ``scores`` is the number of flagging components, aligned
    row-by-row with ``flags`` (``flags > 0`` iff ``scores > 0``).
    """

    name = "sain_local"

    def __init__(self, model: dict[str, Any]):
        self._model = model
        self._components = model.get("components") or {}

    @classmethod
    def from_artifact(cls, path: Path | str) -> "SainLocalDetector":
        return cls(json.loads(Path(path).read_text()))

    @property
    def model(self) -> dict[str, Any]:
        return self._model

    @property
    def feature_cols(self) -> list[str]:
        return list(self._model.get("feature_cols") or [])

    def fit(self, calibration_frames: list[pd.DataFrame]
            ) -> "SainLocalDetector":
        """No-op fit: the model is already mined. Present for the monitor
        detector protocol."""
        return self

    def predict(self, frame: pd.DataFrame) -> StepFlags:
        n = len(frame)
        flags = np.zeros(n, dtype=int)
        scores = np.zeros(n, dtype=float)
        for entry in self._components.values():
            column = entry["column"]
            ids = local_state_ids(frame, column)
            states = entry.get("local_states") or {}
            threshold = max(1, int(entry.get("violation_threshold", 1)))
            for state_id in sorted(set(ids.tolist())):
                sel = np.where(ids == state_id)[0]
                mined = states.get(state_id)
                if mined is None:
                    # Unknown local state for this component.
                    flags[sel] = 1
                    scores[sel] += 1.0
                    continue
                rules = mined.get("rules") or []
                if not rules:
                    continue
                sub = frame.iloc[sel]
                counts = frame_violation_counts(sub, rules)
                fired = counts >= threshold
                if fired.any():
                    flags[sel[fired]] = 1
                    scores[sel[fired]] += 1.0
        return StepFlags(flags=flags, scores=scores)


@dataclass
class ComparisonScenario:
    """One evaluation scenario: a frame plus its ground-truth labels."""

    name: str
    frame: pd.DataFrame
    labels: np.ndarray


class _DetectorLike(Protocol):
    name: str

    def predict(self, frame: pd.DataFrame) -> StepFlags: ...


def compare_local_vs_global(
    *,
    scenarios: Sequence[ComparisonScenario],
    local_detector: _DetectorLike,
    global_detector: _DetectorLike,
    feature_cols: Sequence[str] | None = None,
    out_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Reusable comparison entry point: local SAIN vs. global invariants.

    Both detectors are evaluated on *identical* frames and, when
    ``feature_cols`` is given, the comparison records the feature columns
    both models were built on so a mismatch is visible in the artifact.

    The metric schema is the one used by the hybrid monitor
    (:func:`stl.metrics.detection_metrics`), so the comparison output can be
    consumed by the same tooling as the monitor manifest.
    """
    from stl.metrics import detection_metrics

    rows: list[dict[str, Any]] = []
    ys: list[np.ndarray] = []
    local_p: list[np.ndarray] = []
    local_s: list[np.ndarray] = []
    global_p: list[np.ndarray] = []
    global_s: list[np.ndarray] = []
    for sc in scenarios:
        y = np.asarray(sc.labels).astype(int)
        local = local_detector.predict(sc.frame)
        glob = global_detector.predict(sc.frame)
        local_scores = (local.scores if local.scores is not None
                        else local.flags.astype(float))
        glob_scores = (glob.scores if glob.scores is not None
                       else glob.flags.astype(float))
        ys.append(y)
        local_p.append(local.flags)
        local_s.append(local_scores)
        global_p.append(glob.flags)
        global_s.append(glob_scores)
        rows.append({
            "name": sc.name,
            "n": int(len(y)),
            "attack_rows": int(y.sum()),
            "local": {
                "flags": int(local.flags.sum()),
                "metrics": detection_metrics(y, local.flags, local_scores),
            },
            "global": {
                "flags": int(glob.flags.sum()),
                "metrics": detection_metrics(y, glob.flags, glob_scores),
            },
        })

    def _agg(ps: list[np.ndarray], ss: list[np.ndarray]) -> dict[str, Any]:
        if not ys:
            return {}
        return detection_metrics(
            np.concatenate(ys), np.concatenate(ps), np.concatenate(ss))

    result: dict[str, Any] = {
        "schema": "baselines/compare/v1",
        "scenarios": rows,
        "summary": {
            "local": _agg(local_p, local_s),
            "global": _agg(global_p, global_s),
        },
    }
    if feature_cols is not None:
        result["feature_cols"] = list(feature_cols)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "sain_local_vs_global.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
        )
    return result


def run_sain_local_comparison(
    *,
    topology: str,
    data_root: Path | str = Path("data"),
    gfsm_dir: Path | str | None = None,
    invariants_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    seed: int = 42,
    fraction: float = 0.8,
    min_observations: int = 50,
    max_evals: int = 5000,
    fp_budget: float = 0.01,
) -> dict[str, Any]:
    """End-to-end local-vs-global comparison for one topology.

    This is the reusable entry point: it loads the dataset, splits the
    calibration frames with :func:`dataio.splits.split_calibration_frames`,
    mines the global invariants and the local SAIN model on the *same*
    train/validation split with the *same* feature columns, and evaluates
    both on the untouched evaluation scenarios.

    Nothing here rewrites the driver wiring — the global side goes through
    :func:`invariants.driver.mine_topology` and the existing runtime
    detector, the local side through this module.
    """
    from dataio.loader import load_topology
    from dataio.model import DataIoError
    from dataio.splits import split_dataset
    from invariants.driver import mine_topology
    from invariants.state_label import resolve_fb_to_col_from_paths
    from monitor.invariants_detector import InvariantsAnomalyDetector

    data_root = Path(data_root)
    gfsm_dir = Path(gfsm_dir) if gfsm_dir is not None else (
        data_root / "generated" / topology / "gfsm")
    invariants_dir = Path(invariants_dir) if invariants_dir is not None else (
        data_root / "generated" / topology / "invariants")

    try:
        dataset = load_topology(topology, data_root=data_root)
    except DataIoError as exc:
        raise InvariantsError(str(exc)) from exc

    split = split_dataset(dataset, seed=seed, fraction=fraction)

    try:
        fb_to_col, components, _gfsm = resolve_fb_to_col_from_paths(
            gfsm_dir, topology, data_root)
    except InvariantsError as exc:
        raise InvariantsError(str(exc)) from exc

    # Identical feature columns for both sides.  The global driver derives
    # them from the fitting rows when none are given; derive them the same
    # way here and hand the very same list to both miners.
    skip = set(fb_to_col.values())
    fit_df = pd.concat(split.train_frames, ignore_index=True)
    numeric = fit_df.select_dtypes(include="number").columns
    feature_cols = [c for c in numeric if c not in skip]

    mine_topology(
        topology=topology, data_root=data_root, gfsm_dir=gfsm_dir,
        out_dir=invariants_dir, fb_to_col=fb_to_col,
        min_observations=min_observations, max_evals=max_evals, seed=seed,
        feature_cols=feature_cols, fp_budget=fp_budget,
        training_frames=split.train_frames,
        validation_frames=split.validation_frames,
        split_manifest=split.manifest,
    )

    local_model = mine_local_invariants(
        topology=topology, components=components, fb_to_col=fb_to_col,
        training_frames=split.train_frames,
        validation_frames=split.validation_frames,
        feature_cols=feature_cols,
        min_observations=min_observations, max_evals=max_evals, seed=seed,
        fp_budget=fp_budget,
    )

    global_det = InvariantsAnomalyDetector(
        invariants_dir=invariants_dir, gfsm_dir=gfsm_dir,
        data_root=data_root, topology=topology, components=components,
        fb_to_col=fb_to_col,
    ).fit([])
    local_det = SainLocalDetector(local_model)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "sain_local_model.json").write_text(
            json.dumps(local_model, indent=2, sort_keys=True, default=str) + "\n"
        )
        (out / "split_manifest.json").write_text(
            json.dumps(split.manifest, indent=2, sort_keys=True, default=str) + "\n"
        )

    scenarios = [
        ComparisonScenario(name=sc.name, frame=sc.frame,
                           labels=np.asarray(sc.labels))
        for sc in dataset.eval_scenarios
    ]
    return compare_local_vs_global(
        scenarios=scenarios,
        local_detector=local_det,
        global_detector=global_det,
        feature_cols=feature_cols,
        out_dir=out_dir,
    )
