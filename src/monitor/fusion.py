"""Combine per-detector step flags and anomaly scores.

Safe behavior is LogicValid AND PhysicsValid, therefore an alarm is any logical
OR physical violation.

fuse_or: default alarm-union fusion across detectors.
fuse_intersection: conjunction mode ablation / backward-compatible option.
compute_fused_score: deterministic fused score with a 1.0 alarm threshold.
"""

from __future__ import annotations

import numpy as np


def fuse_or(flag_arrays: list[np.ndarray]) -> np.ndarray:
    """Alarm-union fusion: safe behavior is LogicValid AND PhysicsValid,
    so an alarm is any logical OR physical violation.
    """
    if not flag_arrays:
        raise ValueError("fuse_or needs at least one flag array")
    out = np.zeros_like(np.asarray(flag_arrays[0]), dtype=int)
    for fa in flag_arrays:
        out = np.logical_or(out, np.asarray(fa).astype(int)).astype(int)
    return out


def fuse_intersection(logical: list[np.ndarray],
                      physical: list[np.ndarray]) -> np.ndarray:
    """Conjunction mode ablation: (any logical detector fires)
    AND (any physical detector fires).

    Returns an all-zeros vector if either side has no detectors (a side
    with no evidence cannot contribute to the AND). Length is inferred
    from whichever side is non-empty; an empty result if both are empty.
    """
    if not logical or not physical:
        ref = logical or physical
        if not ref:
            return np.zeros(0, dtype=int)
        return np.zeros(len(ref[0]), dtype=int)
    lg = np.any(np.column_stack(logical), axis=1)
    ph = np.any(np.column_stack(physical), axis=1)
    return (lg & ph).astype(int)


def compute_fused_score(fused_flags: np.ndarray,
                        stl_scores: np.ndarray | None = None) -> np.ndarray:
    """Deterministic non-probabilistic fused anomaly score.

    Alarms score 1 plus their STL score; non-alarms are capped just below 1.
    """
    flags = np.asarray(fused_flags, dtype=float)
    if flags.ndim != 1 or not np.all((flags == 0) | (flags == 1)):
        raise ValueError("fused_flags must be a one-dimensional binary vector")
    if stl_scores is None:
        return flags
    scores = np.asarray(stl_scores, dtype=float)
    if scores.ndim != 1 or scores.shape != flags.shape:
        raise ValueError("stl_scores must be one-dimensional and match fused_flags")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("stl_scores must be finite and nonnegative")
    return np.where(flags == 1, 1.0 + scores,
                    np.minimum(scores, np.nextafter(1.0, 0.0)))
