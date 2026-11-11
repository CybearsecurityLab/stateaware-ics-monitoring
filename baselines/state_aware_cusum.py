"""State-Aware CUSUM baseline (Ghaeini et al., SAC 2018).

Faithful re-implementation of

    H. R. Ghaeini, D. Antonioli, F. Brasser, A.-R. Sadeghi, N. O. Tippenhauer,
    "State-Aware Anomaly Detection for Industrial Control Systems",
    SAC 2018 (Pau, France). doi:10.1145/3167132.3167305

Method summary (equations numbered as in the paper):

* Observer (Eq. 8, discretised with the *actual* timestep ``dt``)::

      h_{k+1} = h_k + (Qin_k - Qout_k) * dt / Area          [SI units]

* Residual (Eq. 1): ``x_k = |y_k - yhat_k|`` (absolute deviation between the
  reported sensor level and the observer estimate).
* Process-state dependent residual (Eq. 4/5): every tank is monitored in a
  *process state* ``p`` defined by the status of the actuators that feed that
  tank.  ``mu_p`` is the mean residual observed in state ``p`` during benign
  history.
* CUSUM (Eq. 6): ``S_k = max(0, S_{k-1} + r[t,p]_k - alpha)``; an alarm is
  raised when ``S_k >= tau`` and the statistic is reset (``S_{k-1} = 0``).

Paper ambiguity (documented, resolved by a single-centering interpretation)
--------------------------------------------------------------------------
Eq. 4 already centers the residual with ``mu_p`` (``r[t,p] = ||y - yhat| -
mu_p|``), while Eq. 6 writes ``|r_k - mu_p|``, which reads as a *second*
subtraction if ``r_k`` is taken to be Eq. 4's residual.  We therefore
subtract ``mu_p`` exactly once:

    x_k   = |y_k - yhat_k|                (raw deviation, Eq. 1)
    d_k   = x_k - mu_p                    (single centering)
    r[t,p] = |d_k|                       (Eq. 4, default increment)
    increment_k = r[t,p]_k - alpha        (default: "centered_abs")
    increment_k = d_k - alpha            (optional: "centered_signed")

No double subtraction is performed anywhere in this module; the single
centering happens in exactly one place (``_center_residual``).

Units
-----
Everything is computed in SI internally (levels in m, flows in m^3/s, area in
m^2) and reported in millimetres (the paper's reporting unit for SWaT).
``dt`` is taken from the data when the frame carries a time column, otherwise
from the configuration; it is never silently assumed to be 1 s.

Integration
-----------
``config_from_profile`` turns a ``stl.profiles.TopologyProfile`` into a
configuration: per-tank feeder mappings come from ``profile.feeder_map``,
the per-tank actuator signature from the pump/valve status columns of the
same profile, and the tank geometry (cross-sectional area) from the
hydraulics mass-balance loader (``params["mb"][tank]["alpha"]``) when it is
available, otherwise from a regression on benign training data.  Networks
whose profile has no feeder mapping (e.g. anytown, whose pumps *drain* the
tanks) need an explicit feeder/outflow mapping from the caller; see
``derive_outflows_from_control_rules`` for an opt-in derivation.

Not transferred from SWaT
-------------------------
The paper's 30 mm threshold is SWaT-specific and its ``alpha`` value is not
reported.  Neither is transferred here: ``alpha`` and ``tau`` are selected on
*clean validation data only*, from a grid that is declared in the
configuration, subject to a declared false-alarm budget.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

PAPER: dict[str, Any] = {
    "title": "State-Aware Anomaly Detection for Industrial Control Systems",
    "authors": [
        "Hamid Reza Ghaeini",
        "Daniele Antonioli",
        "Ferdinand Brasser",
        "Ahmad-Reza Sadeghi",
        "Nils Ole Tippenhauer",
    ],
    "venue": "SAC 2018 - Symposium on Applied Computing, Pau, France",
    "year": 2018,
    "doi": "10.1145/3167132.3167305",
}

SCHEMA = "baselines/state_aware_cusum/v1"

_TIME_COLUMN_CANDIDATES = ("datetime", "DATETIME", "timestamp", "time", "DateTime")


class StateAwareCusumError(Exception):
    """Unrecoverable state-aware CUSUM error."""


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StateAwareCusumConfig:
    """Declarative, serialisable configuration (fully deterministic).

    ``feeders``/``outflows``/``actuators`` map a tank id to the columns used
    for that tank:

    * ``feeders``  -> flow columns that *fill* the tank (Qin),
    * ``outflows`` -> flow columns that *drain* the tank (Qout),
    * ``actuators``-> status columns defining the tank's process state.
    """

    topology: str = "custom"
    tanks: tuple[str, ...] = ()
    feeders: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    outflows: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    actuators: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    area_m2: Mapping[str, float] = field(default_factory=dict)

    # physical / unit conventions
    dt_s: float = 1.0
    dt_source: str = "auto"  # "auto" -> infer from a time column, else dt_s
    flow_to_m3ps: float = 1e-3  # flow units -> m^3/s (L/s -> m^3/s)
    level_to_mm: float = 1000.0  # level units -> mm (m -> mm)
    initial_level: str = "frame_first"  # or "train_median"
    initial_level_samples: int = 200
    offset_correction: bool = True  # absorb the constant net-outflow bias

    # detector
    increment_mode: str = "centered_abs"  # or "centered_signed"
    alpha: float | None = None  # fixed slack (mm); None -> tune on validation
    tau: float | None = None  # fixed threshold (mm); None -> tune on validation

    # validation grid (declared before any data is seen)
    residual_quantile: float = 0.99
    alpha_grid_multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5)
    tau_grid_multipliers: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)
    false_alarm_budget_per_day: float = 1.0
    min_validation_steps: int = 30

    # state handling
    unseen_state_policy: str = "global"  # "global" | "skip"
    status_on_threshold: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "topology": self.topology,
            "tanks": list(self.tanks),
            "feeders": {k: list(v) for k, v in sorted(self.feeders.items())},
            "outflows": {k: list(v) for k, v in sorted(self.outflows.items())},
            "actuators": {k: list(v) for k, v in sorted(self.actuators.items())},
            "area_m2": {k: float(v) for k, v in sorted(self.area_m2.items())},
            "dt_s": float(self.dt_s),
            "dt_source": self.dt_source,
            "flow_to_m3ps": float(self.flow_to_m3ps),
            "level_to_mm": float(self.level_to_mm),
            "initial_level": self.initial_level,
            "initial_level_samples": int(self.initial_level_samples),
            "offset_correction": bool(self.offset_correction),
            "increment_mode": self.increment_mode,
            "alpha": None if self.alpha is None else float(self.alpha),
            "tau": None if self.tau is None else float(self.tau),
            "residual_quantile": float(self.residual_quantile),
            "alpha_grid_multipliers": [float(x) for x in self.alpha_grid_multipliers],
            "tau_grid_multipliers": [float(x) for x in self.tau_grid_multipliers],
            "false_alarm_budget_per_day": float(self.false_alarm_budget_per_day),
            "min_validation_steps": int(self.min_validation_steps),
            "unseen_state_policy": self.unseen_state_policy,
            "status_on_threshold": float(self.status_on_threshold),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StateAwareCusumConfig":
        d = dict(d)
        d.pop("schema", None)
        kwargs: dict[str, Any] = {
            "topology": d.get("topology", "custom"),
            "tanks": tuple(d.get("tanks", ())),
            "feeders": {k: tuple(v) for k, v in d.get("feeders", {}).items()},
            "outflows": {k: tuple(v) for k, v in d.get("outflows", {}).items()},
            "actuators": {k: tuple(v) for k, v in d.get("actuators", {}).items()},
            "area_m2": {k: float(v) for k, v in d.get("area_m2", {}).items()},
            "dt_s": float(d.get("dt_s", 1.0)),
            "dt_source": d.get("dt_source", "auto"),
            "flow_to_m3ps": float(d.get("flow_to_m3ps", 1e-3)),
            "level_to_mm": float(d.get("level_to_mm", 1000.0)),
            "initial_level": d.get("initial_level", "frame_first"),
            "initial_level_samples": int(d.get("initial_level_samples", 200)),
            "offset_correction": bool(d.get("offset_correction", True)),
            "increment_mode": d.get("increment_mode", "centered_abs"),
            "alpha": d.get("alpha"),
            "tau": d.get("tau"),
            "residual_quantile": float(d.get("residual_quantile", 0.99)),
            "alpha_grid_multipliers": tuple(
                float(x) for x in d.get("alpha_grid_multipliers", (0.25, 0.5, 1.0, 1.5))
            ),
            "tau_grid_multipliers": tuple(
                float(x) for x in d.get("tau_grid_multipliers", (1.0, 2.0, 4.0, 8.0, 16.0))
            ),
            "false_alarm_budget_per_day": float(d.get("false_alarm_budget_per_day", 1.0)),
            "min_validation_steps": int(d.get("min_validation_steps", 30)),
            "unseen_state_policy": d.get("unseen_state_policy", "global"),
            "status_on_threshold": float(d.get("status_on_threshold", 0.5)),
        }
        return cls(**kwargs)


def config_from_profile(
    profile: Any,
    *,
    columns: Sequence[str] | None = None,
    dt_s: float = 1.0,
    flow_to_m3ps: float = 1e-3,
    level_to_mm: float = 1000.0,
    derive_outflows_from_control_rules: bool = False,
    **overrides: Any,
) -> StateAwareCusumConfig:
    """Build a config from a ``stl.profiles.TopologyProfile``.

    Per-tank feeder mappings come from ``profile.feeder_map``; the actuator
    (status) columns that define each tank's process state are derived from
    the same profile (pump status format, valve status map) and, when the
    profile has no feeder for a tank, from ``profile.control_rules``.
    """
    cols = set(columns) if columns is not None else None

    def _keep(name: str) -> bool:
        return cols is None or name in cols

    feeders: dict[str, tuple[str, ...]] = {}
    for tank, flows in profile.feeder_map.items():
        kept = tuple(f for f in flows if _keep(f))
        if kept:
            feeders[tank] = kept

    actuators: dict[str, list[str]] = {}

    def _add(tank: str, status: str | None) -> None:
        if not tank or not status or not _keep(status):
            return
        bucket = actuators.setdefault(tank, [])
        if status not in bucket:
            bucket.append(status)

    # actuators implied by the feeder columns (pump/valve flow -> status)
    for tank, flows in profile.feeder_map.items():
        for f in flows:
            _add(tank, _status_column_for(profile, f))

    # actuators implied by the control rules (actuator -> tank)
    for rule in profile.control_rules:
        actuator, tank = rule[0], rule[1]
        _add(tank, _status_column_for(profile, actuator))

    outflows: dict[str, tuple[str, ...]] = {}
    if derive_outflows_from_control_rules:
        # For topologies whose profile has no feeder_map (e.g. anytown, where the
        # pumps drain the tanks) the control-rule actuators are the only known
        # flows; they are offered as *outflows*.  The sign convention is an
        # assumption of the caller, hence it is opt-in.
        for rule in profile.control_rules:
            actuator, tank = rule[0], rule[1]
            if tank in feeders:
                continue
            flow = _flow_column_for(profile, actuator)
            if flow and _keep(flow):
                outflows.setdefault(tank, ())
                if flow not in outflows[tank]:
                    outflows[tank] = outflows[tank] + (flow,)

    # A tank can only be monitored when at least one of its flows is known:
    # the observer needs Qin and/or Qout.  Tanks that only carry a process
    # signature (but no flow) are dropped and reported by the caller.
    tanks = sorted({t for t in profile.tanks if t in feeders or t in outflows})
    actuators = {t: v for t, v in actuators.items() if t in tanks}
    return StateAwareCusumConfig(
        topology=getattr(profile, "name", "custom"),
        tanks=tuple(tanks),
        feeders=feeders,
        outflows={k: tuple(v) for k, v in sorted(outflows.items())},
        actuators={k: tuple(v) for k, v in sorted(actuators.items())},
        dt_s=dt_s,
        flow_to_m3ps=flow_to_m3ps,
        level_to_mm=level_to_mm,
        **overrides,
    )


def _flow_column_for(profile: Any, name: str) -> str | None:
    """Map an actuator name onto its flow column, using the profile."""
    if name in profile.valves:
        return profile.valves[name]["flow"]
    for vid, v in profile.valves.items():
        if name == vid:
            return v["flow"]
    if name.startswith("s_pu") and name[1:].isdigit():
        pid = int(name[1:])
    elif name.startswith("p") and name[1:].isdigit():
        pid = int(name[1:])
    elif name.startswith("f_pu"):
        pid = int(name[4:])
    elif name.startswith("f_p"):
        pid = int(name[3:])
    else:
        return None
    try:
        return profile.pump_flow_fmt.format(pid=pid)
    except (ValueError, KeyError):
        return None


def _status_column_for(profile: Any, name: str) -> str | None:
    """Map a feeder/actuator name onto its status column, using the profile."""
    if name in profile.valves:
        return profile.valves[name]["status"]
    for vid, v in profile.valves.items():
        if name == vid or name == v["flow"]:
            return v["status"]
    if name.startswith("f_pu"):
        pid = name[4:]
    elif name.startswith("f_p"):
        pid = name[3:]
    elif name.startswith("s_pu") and name[1:].isdigit():
        return name
    elif name.startswith("p") and name[1:].isdigit():
        pid = name[1:]
    else:
        return None
    try:
        return profile.pump_status_fmt.format(pid=int(pid))
    except (ValueError, KeyError):
        return None


# --------------------------------------------------------------------------
# fitted model
# --------------------------------------------------------------------------


@dataclass
class StateAwareCusumModel:
    """Everything learned from benign data (train + clean validation only)."""

    areas_m2: dict[str, float] = field(default_factory=dict)
    area_sources: dict[str, str] = field(default_factory=dict)
    offsets_m_per_step: dict[str, float] = field(default_factory=dict)
    area_fit_quality: dict[str, float] = field(default_factory=dict)
    mu: dict[str, dict[str, float]] = field(default_factory=dict)  # tank -> state -> mu (mm)
    global_mu: dict[str, float] = field(default_factory=dict)
    state_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    alpha_mm: dict[str, float] = field(default_factory=dict)
    tau_mm: dict[str, float] = field(default_factory=dict)
    tuning: dict[str, Any] = field(default_factory=dict)
    dt_s: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "areas_m2": {k: float(v) for k, v in sorted(self.areas_m2.items())},
            "area_sources": dict(sorted(self.area_sources.items())),
            "area_fit_quality_r2": {
                k: float(v) for k, v in sorted(self.area_fit_quality.items())
            },
            "offsets_m_per_step": {
                k: float(v) for k, v in sorted(self.offsets_m_per_step.items())
            },
            "mu_mm": {
                t: {s: float(v) for s, v in sorted(states.items())}
                for t, states in sorted(self.mu.items())
            },
            "global_mu_mm": {k: float(v) for k, v in sorted(self.global_mu.items())},
            "state_counts": {
                t: {s: int(c) for s, c in sorted(states.items())}
                for t, states in sorted(self.state_counts.items())
            },
            "alpha_mm": {k: float(v) for k, v in sorted(self.alpha_mm.items())},
            "tau_mm": {k: float(v) for k, v in sorted(self.tau_mm.items())},
            "tuning": self.tuning,
            "dt_s": float(self.dt_s),
        }


@dataclass
class StateAwareCusumResult:
    """Per-step output of the detector."""

    flags: np.ndarray  # 0/1, 1 when any tank's CUSUM crosses tau
    scores: np.ndarray  # native CUSUM score (max over tanks, pre-reset)
    cusum: dict[str, np.ndarray]  # tank -> CUSUM statistic after reset
    cusum_pre_reset: dict[str, np.ndarray]  # tank -> statistic before reset
    residuals: dict[str, np.ndarray]  # tank -> raw |y - yhat| (mm)
    centered_residuals: dict[str, np.ndarray]  # tank -> |x - mu_p| (mm)
    signed_centered_residuals: dict[str, np.ndarray]  # tank -> x - mu_p (mm)
    states: dict[str, np.ndarray]  # tank -> process-state id per step
    tank_flags: dict[str, np.ndarray]  # tank -> 0/1


# --------------------------------------------------------------------------
# detector
# --------------------------------------------------------------------------


class StateAwareCusumDetector:
    """State-aware CUSUM detector.

    Lifecycle: ``fit(benign_train)`` -> ``tune(clean_validation)`` ->
    ``predict(frame)``.  ``predict`` never refits anything.
    """

    def __init__(self, config: StateAwareCusumConfig,
                 hydraulics: Mapping[str, Any] | None = None):
        self.config = config
        self.hydraulics = hydraulics
        self.model = StateAwareCusumModel()
        self._fitted = False
        self._tuned = False
        self._train_initial_levels: dict[str, float] = {}

    # -- helpers ----------------------------------------------------------

    def _tanks(self) -> list[str]:
        return list(self.config.tanks)

    def _columns(self, frame: pd.DataFrame, tank: str) -> list[str]:
        need = [tank] + list(self.config.feeders.get(tank, ())) \
            + list(self.config.outflows.get(tank, ())) \
            + list(self.config.actuators.get(tank, ()))
        missing = [c for c in need if c not in frame.columns]
        if missing:
            raise StateAwareCusumError(
                f"tank {tank!r}: missing columns {missing} in frame")
        return need

    def _dt(self, frame: pd.DataFrame) -> float:
        if self.config.dt_source == "auto":
            inferred = infer_timestep_seconds(frame)
            if inferred is not None and inferred > 0:
                return float(inferred)
        return float(self.config.dt_s)

    def _flows(self, frame: pd.DataFrame, tank: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (Qin, Qout) in m^3/s."""
        scale = self.config.flow_to_m3ps
        f_in = [c for c in self.config.feeders.get(tank, ())]
        f_out = [c for c in self.config.outflows.get(tank, ())]
        qin = (frame[f_in].to_numpy(dtype=float).sum(axis=1) * scale
               if f_in else np.zeros(len(frame)))
        qout = (frame[f_out].to_numpy(dtype=float).sum(axis=1) * scale
                if f_out else np.zeros(len(frame)))
        return qin, qout

    def _states(self, frame: pd.DataFrame, tank: str) -> np.ndarray:
        cols = list(self.config.actuators.get(tank, ()))
        if not cols:
            return np.full(len(frame), "default", dtype=object)
        thr = self.config.status_on_threshold
        bits = [np.where(frame[c].to_numpy(dtype=float) > thr, 1, 0) for c in cols]
        stacked = np.column_stack(bits)
        return np.array(
            [";".join(f"{c}={int(v)}" for c, v in zip(cols, row)) for row in stacked],
            dtype=object,
        )

    def _initial_level(self, frame: pd.DataFrame, tank: str) -> float:
        if self.config.initial_level == "train_median":
            if tank in self._train_initial_levels:
                return self._train_initial_levels[tank]
        return float(frame[tank].to_numpy(dtype=float)[0])

    def observe(self, frame: pd.DataFrame, tank: str) -> np.ndarray:
        """Observer estimate of the tank level (m) for every step.

        Implements Eq. 8 with the actual timestep::

            h_{k+1} = h_k + (Qin_k - Qout_k) * dt / Area
        """
        area = self.model.areas_m2.get(tank)
        if area is None:
            raise StateAwareCusumError(f"no area for tank {tank!r}; call fit() first")
        dt = self._dt(frame)
        qin, qout = self._flows(frame, tank)
        offset = float(self.model.offsets_m_per_step.get(tank, 0.0))
        h = self._initial_level(frame, tank)
        out = np.empty(len(frame), dtype=float)
        for k in range(len(frame)):
            out[k] = h
            h = h + (qin[k] - qout[k]) * dt / area + offset
        return out

    # -- fitting ----------------------------------------------------------

    def fit(self, train_frames: Iterable[pd.DataFrame],
            hydraulics: Mapping[str, Any] | None = None) -> "StateAwareCusumDetector":
        """Learn the observer geometry and the per-state residual means.

        Only benign history may be passed here.  Nothing here looks at
        validation or test data.
        """
        frames = [f for f in train_frames if f is not None and len(f)]
        if not frames:
            raise StateAwareCusumError("fit() needs at least one benign frame")
        if hydraulics is not None:
            self.hydraulics = hydraulics
        cfg = self.config
        self.model = StateAwareCusumModel()
        self.model.dt_s = self._dt(frames[0])

        # 1. tank geometry (cross-sectional area) -----------------------------
        for tank in self._tanks():
            area, source, r2 = self._resolve_area(tank, frames)
            self.model.areas_m2[tank] = area
            self.model.area_sources[tank] = source
            self.model.area_fit_quality[tank] = r2

        # 2. constant net-outflow offset (unmodelled Qout) -------------------
        for tank in self._tanks():
            self.model.offsets_m_per_step[tank] = (
                self._estimate_offset(tank, frames) if cfg.offset_correction else 0.0
            )

        # 3. initial level anchor (train median variant) ----------------------
        if cfg.initial_level == "train_median":
            for tank in self._tanks():
                vals = np.concatenate(
                    [f[tank].to_numpy(dtype=float) for f in frames
                     if tank in f.columns])
                tail = vals[-cfg.initial_level_samples:]
                self._train_initial_levels[tank] = float(np.median(tail)) if len(tail) else 0.0

        # 4. per-state residual means (Eq. 5) ---------------------------------
        sums: dict[str, dict[str, float]] = {t: {} for t in self._tanks()}
        counts: dict[str, dict[str, int]] = {t: {} for t in self._tanks()}
        all_vals: dict[str, list[float]] = {t: [] for t in self._tanks()}
        for frame in frames:
            for tank in self._tanks():
                self._columns(frame, tank)
                y = frame[tank].to_numpy(dtype=float) * cfg.level_to_mm
                yhat = self.observe(frame, tank) * cfg.level_to_mm
                raw = np.abs(y - yhat)
                states = self._states(frame, tank)
                all_vals[tank].extend(raw.tolist())
                for s in np.unique(states):
                    mask = states == s
                    sums[tank][s] = sums[tank].get(s, 0.0) + float(raw[mask].sum())
                    counts[tank][s] = counts[tank].get(s, 0) + int(mask.sum())

        for tank in self._tanks():
            vals = np.asarray(all_vals[tank], dtype=float)
            self.model.global_mu[tank] = float(vals.mean()) if len(vals) else 0.0
            self.model.mu[tank] = {
                s: sums[tank][s] / counts[tank][s]
                for s in sorted(sums[tank])
                if counts[tank][s] > 0
            }
            self.model.state_counts[tank] = {
                s: int(c) for s, c in sorted(counts[tank].items())
            }
        self._fitted = True
        self._tuned = False
        return self

    def _estimate_offset(self, tank: str, frames: list[pd.DataFrame]) -> float:
        """Constant per-step offset (m) of the observed level change.

        The paper measures Qout explicitly; in the evaluated networks the
        outflow of a tank is not observable, so the constant net outflow shows
        up as a constant per-step bias of the measured level change.  It is
        estimated once on benign training data (median, robust to noise).
        """
        area = self.model.areas_m2[tank]
        residuals = []
        for frame in frames:
            if tank not in frame.columns:
                continue
            dt = self._dt(frame)
            h = frame[tank].to_numpy(dtype=float)
            dh = np.diff(h)
            qin, qout = self._flows(frame, tank)
            # the observer update at step k uses the flow of step k, so the
            # measured change dh[k] must be compared with the flow of step k
            modeled = (qin[:-1] - qout[:-1]) * dt / area
            mask = np.isfinite(dh) & np.isfinite(modeled)
            residuals.append(dh[mask] - modeled[mask])
        if not residuals:
            return 0.0
        allr = np.concatenate(residuals)
        if not len(allr):
            return 0.0
        # The mean (not the median) is used on purpose: it telescopes to
        # (h_end - h_start - sum(modeled)) / n, whose sampling error is an
        # order of magnitude smaller than the median's.  A biased offset would
        # accumulate into a linear drift over a long frame.
        return float(np.mean(allr))

    def _resolve_area(self, tank: str, frames: list[pd.DataFrame]) -> tuple[float, str, float]:
        """Cross-sectional area (m^2) and where it came from."""
        cfg = self.config
        if tank in cfg.area_m2 and cfg.area_m2[tank]:
            return float(cfg.area_m2[tank]), "config", float("nan")

        # (a) hydraulics loader (mass-balance calibration) --------------------
        mb = (self.hydraulics or {}).get("mb") or {}
        entry = mb.get(tank)
        if entry and entry.get("alpha"):
            alpha = float(entry["alpha"])
            dt = self._dt(frames[0])
            area = cfg.flow_to_m3ps * dt / alpha
            if area > 0:
                return area, "hydraulics_loader", float(entry.get("r2", float("nan")))

        # (b) regress the observed level change on the inflow -----------------
        dh_all, q_all = [], []
        for frame in frames:
            if tank not in frame.columns:
                continue
            h = frame[tank].to_numpy(dtype=float)
            dh = np.diff(h)
            qin, _ = self._flows(frame, tank)
            qv = qin[1:] * self._dt(frame)
            mask = np.isfinite(dh) & np.isfinite(qv)
            dh_all.append(dh[mask])
            q_all.append(qv[mask])
        if dh_all:
            dh = np.concatenate(dh_all)
            qv = np.concatenate(q_all)
            if len(dh) >= 2 and np.ptp(qv) > 0:
                x = qv.reshape(-1, 1)
                a, b = np.polyfit(x, dh, 1)
                if a > 0:
                    area = 1.0 / a  # dh = Qin*dt/Area
                    ss_res = float(np.sum((dh - (a * qv + b)) ** 2))
                    ss_tot = float(np.sum((dh - dh.mean()) ** 2))
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                    return float(area), "regressed_on_benign_train", r2
        raise StateAwareCusumError(
            f"cannot determine the cross-sectional area of tank {tank!r}: "
            "supply config.area_m2 or a hydraulics mb entry")

    # -- tuning ------------------------------------------------------------

    def tune(self, validation_frames: Iterable[pd.DataFrame]) -> "StateAwareCusumDetector":
        """Select ``alpha`` and ``tau`` on clean validation data only.

        The grid is declared in the configuration; the selection rule is
        fixed and documented in ``build_provenance``.  No test data is used.
        """
        if not self._fitted:
            raise StateAwareCusumError("tune() called before fit()")
        frames = [f for f in validation_frames if f is not None and len(f)]
        if not frames:
            raise StateAwareCusumError("tune() needs at least one clean validation frame")
        cfg = self.config
        n_steps = int(sum(len(f) for f in frames))
        if n_steps < cfg.min_validation_steps:
            raise StateAwareCusumError(
                f"validation too short: {n_steps} steps < {cfg.min_validation_steps}")
        dt = self._dt(frames[0])
        days = max(n_steps * dt / 86400.0, 1.0 / 24.0)
        allowed_fa = int(np.floor(cfg.false_alarm_budget_per_day * days))

        per_tank: dict[str, Any] = {}
        for tank in self._tanks():
            raw_all, state_all = [], []
            for frame in frames:
                self._columns(frame, tank)
                y = frame[tank].to_numpy(dtype=float) * cfg.level_to_mm
                yhat = self.observe(frame, tank) * cfg.level_to_mm
                raw_all.append(np.abs(y - yhat))
                state_all.append(self._states(frame, tank))
            raw = np.concatenate(raw_all)
            states = np.concatenate(state_all)
            mu = self._mu_for(tank, states)
            d = raw - mu  # single centering (exactly one mu_p subtraction)
            centered = np.abs(d)  # Eq. 4
            increments = d if cfg.increment_mode == "centered_signed" else centered
            q = float(np.quantile(centered, cfg.residual_quantile))
            if not np.isfinite(q) or q <= 0:
                q = float(np.mean(centered)) if len(centered) else 1.0
            if not np.isfinite(q) or q <= 0:
                q = 1.0

            alpha_candidates = (
                [float(cfg.alpha)] if cfg.alpha is not None
                else [m * q for m in cfg.alpha_grid_multipliers]
            )
            tau_candidates = (
                [float(cfg.tau)] if cfg.tau is not None
                else [m * q for m in cfg.tau_grid_multipliers]
            )

            best = None
            feasible = 0
            for alpha in sorted(alpha_candidates):
                for tau in sorted(tau_candidates):
                    fa = self._count_false_alarms(increments, alpha, tau)
                    ok = fa <= allowed_fa
                    feasible += int(ok)
                    key = (0 if ok else 1, tau, alpha, fa)
                    if best is None or key < best[0]:
                        best = (key, alpha, tau, fa, ok)
            assert best is not None
            _, alpha, tau, fa, ok = best
            self.model.alpha_mm[tank] = float(alpha)
            self.model.tau_mm[tank] = float(tau)
            per_tank[tank] = {
                "alpha_mm": float(alpha),
                "tau_mm": float(tau),
                "residual_scale_q99_mm": q,
                "validation_false_alarms": int(fa),
                "feasible_grid_points": int(feasible),
                "budget_met": bool(ok),
                "grid_points_evaluated": int(len(alpha_candidates) * len(tau_candidates)),
            }
        self.model.tuning = {
            "validation_steps": n_steps,
            "validation_days": days,
            "allowed_false_alarms": allowed_fa,
            "false_alarm_budget_per_day": float(cfg.false_alarm_budget_per_day),
            "alpha_grid_multipliers": [float(x) for x in cfg.alpha_grid_multipliers],
            "tau_grid_multipliers": [float(x) for x in cfg.tau_grid_multipliers],
            "residual_quantile": float(cfg.residual_quantile),
            "selection_rule": (
                "among grid points whose clean-validation false-alarm count is "
                "within the declared budget, choose the smallest tau, then the "
                "smallest alpha; if no point is feasible, choose the fewest "
                "false alarms, then the smallest tau, then the smallest alpha"
            ),
            "per_tank": per_tank,
        }
        self._tuned = True
        return self

    # -- prediction --------------------------------------------------------

    def predict(self, frame: pd.DataFrame) -> StateAwareCusumResult:
        if not self._fitted:
            raise StateAwareCusumError("predict() called before fit()")
        if not self._tuned:
            # A fully specified configuration needs no validation-based
            # selection; anything else must be tuned on clean validation data
            # before it is applied (never on test data).
            if self.config.alpha is None or self.config.tau is None:
                raise StateAwareCusumError("predict() called before tune()")
            for tank in self._tanks():
                self.model.alpha_mm[tank] = float(self.config.alpha)
                self.model.tau_mm[tank] = float(self.config.tau)
        cfg = self.config
        n = len(frame)
        flags = np.zeros(n, dtype=int)
        tank_flags: dict[str, np.ndarray] = {}
        cusum: dict[str, np.ndarray] = {}
        pre: dict[str, np.ndarray] = {}
        residuals: dict[str, np.ndarray] = {}
        centered: dict[str, np.ndarray] = {}
        signed: dict[str, np.ndarray] = {}
        states_out: dict[str, np.ndarray] = {}
        score = np.zeros(n, dtype=float)

        for tank in self._tanks():
            self._columns(frame, tank)
            y = frame[tank].to_numpy(dtype=float) * cfg.level_to_mm
            yhat = self.observe(frame, tank) * cfg.level_to_mm
            raw = np.abs(y - yhat)
            states = self._states(frame, tank)
            mu = self._mu_for(tank, states)
            d = raw - mu  # single centering (exactly one mu_p subtraction)
            r = np.abs(d)  # Eq. 4
            alpha = self.model.alpha_mm[tank]
            tau = self.model.tau_mm[tank]
            s, pre_s, tf = _cusum(d if cfg.increment_mode == "centered_signed" else r,
                                  alpha, tau)
            cusum[tank] = s
            pre[tank] = pre_s
            tank_flags[tank] = tf
            residuals[tank] = raw
            centered[tank] = r
            signed[tank] = d
            states_out[tank] = states
            flags = np.maximum(flags, tf)
            score = np.maximum(score, pre_s)

        return StateAwareCusumResult(
            flags=flags,
            scores=score,
            cusum=cusum,
            cusum_pre_reset=pre,
            residuals=residuals,
            centered_residuals=centered,
            signed_centered_residuals=signed,
            states=states_out,
            tank_flags=tank_flags,
        )

    # -- internals ---------------------------------------------------------

    def _mu_for(self, tank: str, states: np.ndarray) -> np.ndarray:
        """Per-step mu_p (mm) for the given process states."""
        table = self.model.mu.get(tank, {})
        glob = self.model.global_mu.get(tank, 0.0)
        out = np.empty(len(states), dtype=float)
        for i, s in enumerate(states):
            if s in table:
                out[i] = table[s]
            elif self.config.unseen_state_policy == "global":
                out[i] = glob
            else:  # "skip": neutral (zero) mean, no state-specific centring
                out[i] = 0.0
        return out

    def _count_false_alarms(self, increments: np.ndarray, alpha: float,
                            tau: float) -> int:
        _, _, tf = _cusum(increments, alpha, tau)
        return int(tf.sum())


def _cusum(residual: np.ndarray, alpha: float, tau: float):
    """Positive-recursive CUSUM with reset after alarm (paper Eq. 6).

    ``S_k = max(0, S_{k-1} + residual_k - alpha)``; alarm when ``S_k >= tau``;
    the statistic is restarted (``S_{k-1} = 0``) right after the alarm.
    """
    n = len(residual)
    s = np.zeros(n, dtype=float)
    pre = np.zeros(n, dtype=float)
    flags = np.zeros(n, dtype=int)
    acc = 0.0
    for k in range(n):
        acc = acc + float(residual[k]) - float(alpha)
        if acc < 0.0:
            acc = 0.0
        if acc >= tau:
            flags[k] = 1
            pre[k] = acc
            acc = 0.0  # reset after alarm
        else:
            pre[k] = acc
        s[k] = acc
    return s, pre, flags


def infer_timestep_seconds(frame: pd.DataFrame) -> float | None:
    """Infer the sampling period (s) from a time column, if present."""
    for col in _TIME_COLUMN_CANDIDATES:
        if col in frame.columns:
            ts = pd.to_datetime(frame[col], errors="coerce")
            if ts.notna().sum() > 2:
                diffs = ts.diff().dt.total_seconds().dropna()
                diffs = diffs[diffs > 0]
                if len(diffs):
                    return float(diffs.median())
    return None


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def evaluate(y_true: Sequence[int], flags: Sequence[int],
             scores: Sequence[float]) -> dict[str, Any]:
    """Metric schema: MCC, unadjusted balanced accuracy, attack-positive AP.

    Positives are *attacks* (``label == 1``); the average precision is
    therefore the attack-positive average precision and the score used for
    ranking is the native CUSUM score.
    """
    from sklearn.metrics import (  # noqa: PLC0415 - keep module import light
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(flags, dtype=int)
    s = np.asarray(scores, dtype=float)
    out: dict[str, Any] = {
        "mcc": float(matthews_corrcoef(y, p)) if len(y) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)) if len(y) else float("nan"),
        "attack_average_precision": (
            float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
        ),
        "precision": float(precision_score(y, p, zero_division=0)) if len(y) else float("nan"),
        "recall": float(recall_score(y, p, zero_division=0)) if len(y) else float("nan"),
        "f1": float(f1_score(y, p, zero_division=0)) if len(y) else float("nan"),
        "n": int(len(y)),
        "attack_steps": int(y.sum()),
        "predicted_steps": int(p.sum()),
        "false_alarms": int(((p == 1) & (y == 0)).sum()),
        "false_alarm_rate": float(((p == 1) & (y == 0)).sum() / len(y)) if len(y) else float("nan"),
        "tp": int(((p == 1) & (y == 1)).sum()),
        "fp": int(((p == 1) & (y == 0)).sum()),
        "tn": int(((p == 0) & (y == 0)).sum()),
        "fn": int(((p == 0) & (y == 1)).sum()),
    }
    return out


METRIC_SCHEMA_KEYS = (
    "mcc", "balanced_accuracy", "attack_average_precision", "precision", "recall",
    "f1", "n", "attack_steps", "predicted_steps", "false_alarms",
    "false_alarm_rate", "tp", "fp", "tn", "fn",
)


# --------------------------------------------------------------------------
# provenance + artifact runner
# --------------------------------------------------------------------------


def build_provenance(
    config: StateAwareCusumConfig,
    model: StateAwareCusumModel,
    *,
    data_sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Provenance record: paper mapping, interpretations, and known gaps."""
    cfg = config
    signature = {}
    for tank in sorted(cfg.tanks):
        acts = list(cfg.actuators.get(tank, ()))
        observed = sorted(model.state_counts.get(tank, {}))
        signature[tank] = {
            "actuator_status_columns": acts,
            "n_actuators": len(acts),
            "n_observed_states": len(observed),
            "observed_states": observed,
            "states_never_seen_in_benign_history": len(observed) == 0,
        }
    return {
        "schema": SCHEMA,
        "paper": dict(PAPER),
        "equations_implemented": {
            "observer_eq8": "h_{k+1} = h_k + (Qin_k - Qout_k) * dt / Area  [SI]",
            "residual_eq1": "x_k = |y_k - yhat_k|",
            "state_centered_residual_eq4": "r[t,p] = ||y - yhat| - mu_p|",
            "state_mean_eq5": "mu_p = E(x) over benign history in state p",
            "cusum_eq6": "S_k = max(0, S_{k-1} + r[t,p]_k - alpha); reset after alarm",
        },
        "centering_interpretation": {
            "paper_ambiguity": (
                "Eq. 4 already centers the residual with mu_p, while Eq. 6 writes "
                "|r_k - mu_p|, which reads as a second subtraction if r_k is taken "
                "to be Eq. 4's residual."
            ),
            "adopted": (
                "single centering: x = |y - yhat|; d = x - mu_p (mu_p subtracted "
                "exactly once); the CUSUM increment is |d| (default, equals Eq. 4's "
                "r[t,p]) or d (centered_signed mode)."
            ),
            "double_subtraction_performed": False,
            "increment_mode": cfg.increment_mode,
        },
        "missing_original_observer_parameters": {
            "note": (
                "The SAC 2018 paper does not report the SWaT tank cross-sectional "
                "area, the LDS observer coefficients, the value of alpha, or the "
                "per-state residual scale; the 30 mm threshold is SWaT-specific."
            ),
            "tank_cross_sectional_area_m2": {
                "reported_in_paper": False,
                "resolution": (
                    "taken from the hydraulics (mass-balance) loader when available, "
                    "otherwise regressed on benign training data"
                ),
                "per_tank": dict(sorted(model.area_sources.items())),
            },
            "alpha": {
                "reported_in_paper": False,
                "resolution": "selected on clean validation from the declared grid",
            },
            "tau_threshold_mm": {
                "reported_in_paper": False,
                "swat_30mm_transferred": False,
                "resolution": "selected on clean validation from the declared grid",
            },
            "dt_s": {
                "reported_in_paper": "one second (SWaT discretisation)",
                "here": model.dt_s,
                "resolution": "actual timestep of the evaluated data",
            },
            "qout": {
                "reported_in_paper": "yes (SWaT PLC1 tank)",
                "here": "not observable in the evaluated networks unless an explicit "
                        "outflow mapping is supplied; the residual constant net-outflow "
                        "bias is absorbed by the estimated observer offset and by the "
                        "per-state mean mu_p",
            },
        },
        "observer_adaptations": {
            "offset_correction": {
                "enabled": bool(cfg.offset_correction),
                "description": (
                    "the constant net-outflow bias of the observed level change "
                    "(the paper measures Qout explicitly; the evaluated networks do "
                    "not expose it) is estimated once as a per-step offset on benign "
                    "training data and added to the Eq. 8 update"
                ),
                "per_tank_offset_m_per_step": {
                    k: float(v) for k, v in sorted(model.offsets_m_per_step.items())
                },
            },
            "initial_level": {
                "mode": cfg.initial_level,
                "description": (
                    "observer anchor; with 'frame_first' the first reported level of "
                    "the evaluated frame is used, with 'train_median' the median of "
                    "the tail of the benign training history"
                ),
            },
        },
        "signature_adaptation": {
            "paper": "SWaT PLC1 tank: one pump and one valve -> 4 states (A-D)",
            "here": (
                "per-tank process state = the status tuple of every actuator that "
                "feeds that tank (multi-actuator signature); tanks without any "
                "actuator fall back to a single 'default' state"
            ),
            "per_tank": signature,
        },
        "units": {
            "level": "m internally, reported in mm",
            "flow": "dataset flow units scaled by flow_to_m3ps into m^3/s",
            "area": "m^2",
            "alpha": "mm",
            "tau": "mm",
            "flow_to_m3ps": cfg.flow_to_m3ps,
            "level_to_mm": cfg.level_to_mm,
        },
        "validation": {
            "grid": {
                "alpha_grid_multipliers": [float(x) for x in cfg.alpha_grid_multipliers],
                "tau_grid_multipliers": [float(x) for x in cfg.tau_grid_multipliers],
                "residual_quantile": cfg.residual_quantile,
                "scaling": "multipliers are applied to the per-tank residual scale "
                           "(the declared quantile of the state-centred residual "
                           "measured on clean validation data)",
            },
            "false_alarm_budget_per_day": cfg.false_alarm_budget_per_day,
            "selection": model.tuning,
            "tuned_on": "clean validation frames only",
            "test_data_used_for_tuning": False,
        },
        "data_sources": dict(data_sources or {}),
        "determinism": {
            "config_serialised": "JSON with sorted keys",
            "predictions_serialised": "CSV with fixed float formatting",
            "no_wall_clock_in_artifacts": True,
        },
    }


def run_state_aware_cusum(
    *,
    config: StateAwareCusumConfig,
    train_frames: Sequence[pd.DataFrame],
    validation_frames: Sequence[pd.DataFrame],
    scenarios: Sequence[tuple[str, pd.DataFrame, Sequence[int] | None]],
    out_dir: Path | None = None,
    hydraulics: Mapping[str, Any] | None = None,
    data_sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit -> tune -> predict, and (optionally) write deterministic artifacts.

    ``scenarios`` is a sequence of ``(name, frame, labels)``; ``labels`` may be
    ``None`` for unlabelled data (metrics are then omitted for that scenario).
    """
    det = StateAwareCusumDetector(config, hydraulics=hydraulics)
    det.fit(train_frames)
    det.tune(validation_frames)

    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    for name, frame, labels in scenarios:
        res = det.predict(frame)
        tanks = list(config.tanks)
        for i in range(len(frame)):
            row: dict[str, Any] = {
                "scenario": name,
                "row": i,
                "y_pred": int(res.flags[i]),
                "cusum_score": float(res.scores[i]),
            }
            for tank in tanks:
                row[f"cusum_{tank}"] = float(res.cusum[tank][i])
                row[f"cusum_pre_reset_{tank}"] = float(res.cusum_pre_reset[tank][i])
                row[f"residual_{tank}"] = float(res.residuals[tank][i])
                row[f"centered_residual_{tank}"] = float(res.centered_residuals[tank][i])
                row[f"state_{tank}"] = str(res.states[tank][i])
                row[f"flag_{tank}"] = int(res.tank_flags[tank][i])
            if labels is not None:
                row["y_true"] = int(labels[i])
            rows.append(row)
        if labels is not None:
            metrics[name] = evaluate(labels, res.flags, res.scores)

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "config": config.to_dict(),
        "model": det.model.to_dict(),
        "metrics": metrics,
        "provenance": build_provenance(config, det.model, data_sources=data_sources),
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "state_aware_cusum_config.json", config.to_dict())
        _write_json(out / "state_aware_cusum_model.json", det.model.to_dict())
        _write_json(out / "state_aware_cusum_metrics.json", metrics)
        _write_json(out / "state_aware_cusum_provenance.json", manifest["provenance"])
        _write_predictions(out / "state_aware_cusum_predictions.csv", rows)
    return manifest


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    cols: list[str] = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    for r in rows:
        for c in cols:
            if c not in r:
                r[c] = ""
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
