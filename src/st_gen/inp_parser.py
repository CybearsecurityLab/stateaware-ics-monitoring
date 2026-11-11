"""EPANET INP parser — only the sections st_gen needs."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from .model import (
    Control,
    MalformedInpError,
    Network,
    Pump,
    Tank,
    UnsupportedControlError,
    UnsupportedRulesError,
    Valve,
)


_SECTION_RE = re.compile(r"^\s*\[([A-Za-z_]+)\]\s*$")


def _iter_section_lines(text: str):
    """Yield (section_name, line_number, raw_line, stripped) for each data line.

    Skips blank lines and full-line comments. Trailing `;...` is stripped from
    data lines. `section_name` is uppercased.
    """
    section = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        m = _SECTION_RE.match(raw)
        if m:
            section = m.group(1).upper()
            continue
        if section is None:
            continue
        stripped = raw.strip()
        if not stripped or stripped.startswith(";"):
            continue
        # strip inline ";..." trailing comment
        semi = stripped.find(";")
        data = stripped[:semi].strip() if semi != -1 else stripped
        if not data:
            continue
        yield section, lineno, raw, data


def parse_inp(path: str | Path) -> Network:
    """Parse an EPANET INP file into a Network.

    Consumes [TANKS], [PUMPS], [VALVES], [CONTROLS]; checks [RULES] is empty;
    ignores everything else.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    tanks: list[Tank] = []
    pumps: list[Pump] = []
    valves: list[Valve] = []
    controls: list[Control] = []
    statuses: dict[str, str] = {}

    for section, lineno, _raw, data in _iter_section_lines(text):
        if section == "TANKS":
            tanks.append(_parse_tank(data, lineno))
        elif section == "PUMPS":
            pumps.append(_parse_pump(data, lineno))
        elif section == "VALVES":
            valves.append(_parse_valve(data, lineno))
        elif section == "CONTROLS":
            controls.append(_parse_control(data, lineno))
        elif section == "STATUS":
            link_id, status = _parse_status(data, lineno)
            statuses[link_id.upper()] = status
        elif section == "RULES":
            raise UnsupportedRulesError(
                f"[RULES] section is non-empty at line {lineno}: {data!r}. "
                "st_gen does not support [RULES] yet."
            )

    if statuses:
        pumps = [
            replace(p, initial_status=_lookup_status(statuses, p.id, p.initial_status))
            for p in pumps
        ]
        valves = [
            replace(v, initial_status=_lookup_status(statuses, v.id, v.initial_status))
            for v in valves
        ]

    return Network(tanks=tanks, pumps=pumps, valves=valves, controls=controls)


def _parse_tank(data: str, lineno: int) -> Tank:
    # EPANET [TANKS]: ID Elev InitLevel MinLevel MaxLevel Diameter [MinVol] [VolCurve] [Overflow]
    parts = data.split()
    if len(parts) < 6:
        raise MalformedInpError(
            f"[TANKS] line {lineno} has too few fields: {data!r}"
        )
    return Tank(id=parts[0], min_level=float(parts[3]), max_level=float(parts[4]))


def _parse_pump(data: str, lineno: int) -> Pump:
    parts = data.split()
    if len(parts) < 3:
        raise MalformedInpError(
            f"[PUMPS] line {lineno} has too few fields: {data!r}"
        )
    return Pump(id=parts[0], node1=parts[1], node2=parts[2])


def _parse_valve(data: str, lineno: int) -> Valve:
    # ID Node1 Node2 Diameter Type Setting [MinorLoss]
    parts = data.split()
    if len(parts) < 6:
        raise MalformedInpError(
            f"[VALVES] line {lineno} has too few fields: {data!r}"
        )
    return Valve(
        id=parts[0],
        node1=parts[1],
        node2=parts[2],
        vtype=parts[4].upper(),
        setting=float(parts[5]),
    )


_CONTROL_RE = re.compile(
    r"^\s*LINK\s+(\S+)\s+(OPEN|CLOSED)\s+IF\s+NODE\s+(\S+)\s+(BELOW|ABOVE)\s+(-?(?:\d+\.?\d*|\.\d+))\s*$",
    re.IGNORECASE,
)


def _parse_control(data: str, lineno: int) -> Control:
    m = _CONTROL_RE.match(data)
    if not m:
        raise UnsupportedControlError(
            f"[CONTROLS] line {lineno} is not a simple LINK ... IF NODE ... "
            f"BELOW/ABOVE form: {data!r}"
        )
    link_id, target_state, sensor, comparator, threshold = m.groups()
    return Control(
        link_id=link_id,
        target_state=target_state.upper(),
        sensor_node=sensor,
        comparator=comparator.upper(),
        threshold=float(threshold),
    )


def _parse_status(data: str, lineno: int) -> tuple[str, str]:
    """Parse one ``[STATUS]`` line into ``(link_id, "OPEN"|"CLOSED")``.

    EPANET writes ``<id> <status>`` where the status is ``OPEN``/``CLOSED``
    for pumps and valves, or a numeric setting for some valve types.  A
    numeric ``0`` means closed, anything else means open.
    """
    parts = data.split()
    if len(parts) < 2:
        raise MalformedInpError(
            f"[STATUS] line {lineno} has too few fields: {data!r}"
        )
    link_id, raw_status = parts[0], parts[1]
    value = raw_status.upper()
    if value in ("OPEN", "ACTIVE"):
        return link_id, "OPEN"
    if value == "CLOSED":
        return link_id, "CLOSED"
    try:
        number = float(raw_status)
    except ValueError:
        raise MalformedInpError(
            f"[STATUS] line {lineno} has an invalid status: {raw_status!r}"
        ) from None
    return link_id, "CLOSED" if number == 0.0 else "OPEN"


def _lookup_status(
    statuses: dict[str, str], link_id: str, default: str = "OPEN"
) -> str:
    """Return the initial status of *link_id* (case-insensitive).

    Falls back to *default* (the EPANET default, ``OPEN``) when the link has
    no ``[STATUS]`` entry.
    """
    if link_id in statuses:
        return statuses[link_id]
    upper = link_id.upper()
    for key, value in statuses.items():
        if key.upper() == upper:
            return value
    return default
