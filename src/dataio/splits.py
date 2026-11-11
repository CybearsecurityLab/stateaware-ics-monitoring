"""Leak-free train/validation splits of the clean calibration frames.

Why this exists
---------------
The invariants stage mines rules from clean traces and then calibrates a
violation threshold `K` from those same traces.  Doing both on the same rows
lets the threshold see the very data the rules were fitted on, so the
reported false-positive rate of `K` is optimistic (threshold-on-training
leakage).  This module produces an explicit, auditable split:

* **Anytown** — whole-scenario split.  The Anytown calibration set is a pool
  of independent scenarios (``network_anomalies/<scenario>/scada_values.csv``),
  so a *scenario* is the atomic unit: each scenario lands wholly in the train
  or the validation side.  Scenarios are ordered by their *scenario identity*
  (the scenario name when the caller supplies one, otherwise a stable
  content-derived id), so the split does not depend on the order in which the
  frames happen to be listed.
* **Ctown** — chronological row split.  The C-Town (BATADAL) calibration set
  is a single time series, so the atomic unit is a *row*: the first
  ``fraction`` of the rows (in file order, which is chronological) are used
  for fitting, the remaining tail for validation.

Evaluation scenarios are never touched: this module only ever splits
*calibration* frames, and the returned manifest records the evaluation
scenarios separately so a caller can assert they were left alone.

The manifest is the exact, reproducible description of the split (scenario
identities, row counts, content digests, chronological cut point).  It is
designed to be embedded verbatim in the invariants provenance manifest.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import pandas as pd

from .model import DataIoError

SPLIT_SCHEMA = "dataio/splits/v1"

#: Topologies with a dedicated split strategy.
SUPPORTED_TOPOLOGIES = ("anytown", "ctown")


def _digest(df: pd.DataFrame) -> str:
    """Stable content digest of a frame (column order + values)."""
    payload = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scenario_name(entry: Any, index: int) -> str:
    """Identity of a calibration frame.

    Accepted entry forms:

    * a ``pandas.DataFrame``            -> ``frame_<index>_<digest[:16]>``
    * a ``(name, DataFrame)`` pair      -> the supplied name
    """
    if isinstance(entry, pd.DataFrame):
        return f"frame_{index:04d}_{_digest(entry)[:16]}"
    if (
        isinstance(entry, tuple)
        and len(entry) == 2
        and isinstance(entry[0], str)
        and isinstance(entry[1], pd.DataFrame)
    ):
        return entry[0]
    raise DataIoError(
        "calibration frames must be DataFrames or (name, DataFrame) pairs; "
        f"got {type(entry).__name__} at index {index}"
    )


def _entries(
    frames: Sequence[Any],
) -> list[tuple[str, pd.DataFrame]]:
    out: list[tuple[str, pd.DataFrame]] = []
    seen: set[str] = set()
    for i, entry in enumerate(frames):
        name = _scenario_name(entry, i)
        df = entry if isinstance(entry, pd.DataFrame) else entry[1]
        if name in seen:
            raise DataIoError(
                f"duplicate calibration scenario identity {name!r}; "
                "scenario identities must be unique"
            )
        seen.add(name)
        out.append((name, df))
    return out


def _eval_names(evaluation_scenarios: Iterable[Any] | None) -> list[str]:
    if evaluation_scenarios is None:
        return []
    names: list[str] = []
    for sc in evaluation_scenarios:
        if isinstance(sc, str):
            names.append(sc)
        elif hasattr(sc, "name"):
            names.append(str(sc.name))
        else:
            names.append(str(sc))
    return names


@dataclass(frozen=True)
class CalibrationSplit:
    """Result of :func:`split_calibration_frames`."""

    topology: str
    strategy: str
    train_frames: list[pd.DataFrame]
    validation_frames: list[pd.DataFrame]
    manifest: dict[str, Any]

    @property
    def train_rows(self) -> int:
        return int(self.manifest["train"]["rows"])

    @property
    def validation_rows(self) -> int:
        return int(self.manifest["validation"]["rows"])


def _split_scenarios(
    entries: list[tuple[str, pd.DataFrame]],
    seed: int,
    fraction: float,
) -> tuple[list[tuple[str, pd.DataFrame]], list[tuple[str, pd.DataFrame]]]:
    """Whole-scenario split (Anytown).

    Scenarios are ordered by scenario identity (canonical ordering, so the
    result does not depend on the order the frames were listed in), then a
    seeded permutation of that canonical order decides which scenarios go to
    the validation side.  The permutation is seeded, hence reproducible for
    a given ``seed`` and independent of the input ordering.
    """
    ordered = sorted(entries, key=lambda kv: kv[0])
    n = len(ordered)
    if n == 0:
        return [], []
    n_train = int(round(fraction * n))
    if n >= 2:
        n_train = min(max(n_train, 1), n - 1)
    else:
        n_train = 1
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    train_idx = sorted(idx[:n_train])
    val_idx = sorted(idx[n_train:])
    train = [ordered[i] for i in train_idx]
    val = [ordered[i] for i in val_idx]
    return train, val


def _split_chronological(
    entries: list[tuple[str, pd.DataFrame]],
    fraction: float,
) -> tuple[
    list[tuple[str, pd.DataFrame]],
    list[tuple[str, pd.DataFrame]],
    list[dict[str, Any]],
    int,
    int,
]:
    """Chronological row split (Ctown).

    Frames are assumed to be listed in chronological order (the C-Town
    calibration CSV is a single time series in file order).  The cut point
    is computed over the concatenated row count and applied back to the
    original frame boundaries, so no row is duplicated or dropped.
    """
    total = sum(len(df) for _, df in entries)
    if total == 0:
        raise DataIoError("calibration frames contain no rows")
    cut = int(total * fraction)
    if total >= 2:
        cut = min(max(cut, 1), total - 1)
    else:
        cut = total
    train: list[tuple[str, pd.DataFrame]] = []
    val: list[tuple[str, pd.DataFrame]] = []
    detail: list[dict[str, Any]] = []
    offset = 0
    for name, df in entries:
        n = len(df)
        n_train_rows = max(0, min(offset + n, cut) - offset)
        n_val_rows = n - n_train_rows
        if n_train_rows:
            train.append((name, df.iloc[:n_train_rows]))
        if n_val_rows:
            val.append((name, df.iloc[n_train_rows:]))
        detail.append({
            "scenario": name,
            "rows": int(n),
            "train_rows": int(n_train_rows),
            "validation_rows": int(n_val_rows),
            "digest": _digest(df),
        })
        offset += n
    return train, val, detail, cut, total


def split_calibration_frames(
    topology: str,
    frames: Sequence[Any],
    seed: int = 42,
    fraction: float = 0.8,
    *,
    evaluation_scenarios: Iterable[Any] | None = None,
) -> CalibrationSplit:
    """Split clean calibration frames into a train and a validation side.

    Parameters
    ----------
    topology:
        ``"anytown"`` (whole-scenario split) or ``"ctown"`` (chronological
        row split).  Any other topology is rejected — the split strategy is
        topology-specific and silently guessing one would hide a leak.
    frames:
        The calibration frames.  Each entry is either a ``DataFrame`` or a
        ``(scenario_name, DataFrame)`` pair.  Supplying names makes the split
        manifest readable and pins the scenario ordering.
    seed:
        Seed of the scenario permutation used by the Anytown strategy.
    fraction:
        Fraction of the calibration material used for *fitting*.  The
        remainder is used for *validation* (threshold selection).
    evaluation_scenarios:
        Optional evaluation scenarios, recorded in the manifest and checked
        against the calibration identities.  They are never modified.

    Returns
    -------
    CalibrationSplit
        Carries the train/validation frames and the exact split manifest.
    """
    if topology not in SUPPORTED_TOPOLOGIES:
        raise DataIoError(
            f"unsupported topology for calibration splitting: {topology!r}; "
            f"expected one of {SUPPORTED_TOPOLOGIES}"
        )
    if not (0.0 < fraction < 1.0):
        raise DataIoError(
            f"split fraction must be in (0, 1), got {fraction!r}")
    entries = _entries(frames)
    if not entries:
        raise DataIoError("no calibration frames to split")

    eval_names = _eval_names(evaluation_scenarios)
    cal_names = {name for name, _ in entries}
    leaked = sorted(set(eval_names) & cal_names)
    if leaked:
        raise DataIoError(
            "evaluation scenario(s) also present in the calibration pool: "
            f"{leaked}; refusing to split"
        )

    if topology == "anytown":
        strategy = "scenario_split"
        (train, val) = _split_scenarios(entries, seed, fraction)
        detail = [
            {
                "scenario": name,
                "rows": int(len(df)),
                "digest": _digest(df),
                "partition": (
                    "train" if (name, df) in train else "validation"
                ),
            }
            for name, df in sorted(entries, key=lambda kv: kv[0])
        ]
        chrono: dict[str, Any] | None = None
    else:
        strategy = "chronological_split"
        train, val, detail, cut, total = _split_chronological(entries, fraction)
        chrono = {
            "total_rows": int(total),
            "cut_row": int(cut),
            "order": "calibration file order (chronological)",
            "frames": detail,
        }

    train_names = [n for n, _ in train]
    val_names = [n for n, _ in val]
    both = sorted(set(train_names) & set(val_names))
    if strategy == "scenario_split":
        # A scenario must land wholly on one side; a name on both sides
        # would mean the same scenario was used for fitting and for
        # threshold selection.
        if both:
            raise DataIoError(
                f"split produced overlapping scenarios {both}; refusing")
        shared: list[str] = []
        cut_scenarios: list[str] = []
    else:
        # Chronological split: a scenario may be cut at the row level, but
        # never at the row level *and* with overlapping rows.
        shared = []
        cut_scenarios = both
    train_rows = sum(len(df) for _, df in train)
    val_rows = sum(len(df) for _, df in val)
    if train_rows == 0 or val_rows == 0:
        raise DataIoError(
            "split produced an empty side "
            f"(train_rows={train_rows}, validation_rows={val_rows})")

    manifest: dict[str, Any] = {
        "schema": SPLIT_SCHEMA,
        "topology": topology,
        "strategy": strategy,
        "seed": int(seed),
        "fraction": float(fraction),
        "train": {
            "scenarios": train_names,
            "frames": int(len(train)),
            "rows": int(train_rows),
        },
        "validation": {
            "scenarios": val_names,
            "frames": int(len(val)),
            "rows": int(val_rows),
        },
        "scenarios": detail,
        "isolation": {
            "shared_scenarios": shared,
            "cut_scenarios": cut_scenarios,
            "train_rows": int(train_rows),
            "validation_rows": int(val_rows),
            "disjoint": True,
            "level": "scenario" if strategy == "scenario_split" else "row",
        },
        "evaluation": {
            "scenarios": eval_names,
            "count": int(len(eval_names)),
            "untouched": True,
            "included_in_split": False,
        },
    }
    if chrono is not None:
        manifest["chronological"] = chrono

    return CalibrationSplit(
        topology=topology,
        strategy=strategy,
        train_frames=[df for _, df in train],
        validation_frames=[df for _, df in val],
        manifest=manifest,
    )


def split_dataset(
    dataset: Any,
    seed: int = 42,
    fraction: float = 0.8,
) -> CalibrationSplit:
    """Split a :class:`dataio.model.TopologyDataset`'s calibration frames.

    Evaluation scenarios of the dataset are left untouched; they are only
    *recorded* in the split manifest.
    """
    return split_calibration_frames(
        dataset.topology,
        dataset.calibration_frames,
        seed=seed,
        fraction=fraction,
        evaluation_scenarios=[
            getattr(sc, "name", None) or str(sc)
            for sc in getattr(dataset, "eval_scenarios", [])
        ],
    )
