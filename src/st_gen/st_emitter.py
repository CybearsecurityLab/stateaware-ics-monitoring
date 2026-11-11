"""Generate IEC 61131-3 Structured Text (CASE-FSM) files per PLC."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .model import (
    Control,
    Manifest,
    MultiStateActuatorError,
    Network,
    Plc,
)
from .provenance import (
    CONTROL_LIMITATIONS,
    GENERATOR_NAME,
    GENERATOR_VERSION,
    SURROGATE_HEADER_LINES,
    SURROGATE_STATEMENT,
    sha256_file,
    upstream_block,
)


def _iec_ident(name: str) -> str:
    """Return a valid IEC 61131-3 identifier for *name*.

    IEC 61131-3 identifiers must start with a letter or underscore.
    If *name* starts with a digit, prefix with ``v_``.
    The logical name (used in the manifest) is unchanged.
    """
    return f"v_{name}" if name and name[0].isdigit() else name


def emit(
    network: Network,
    plcs: list[Plc],
    out_dir: str | Path,
    topology: str,
    write_manifest: bool = True,
    inp_filename: str = "",
    plcs_filename: str = "",
    inp_path: str | Path | None = None,
    plcs_path: str | Path | None = None,
) -> Manifest:
    """Emit one .st file per PLC and (optionally) a manifest.

    ``inp_path`` / ``plcs_path`` are the *full paths* of the source files.  When
    given, the manifest records them together with their SHA256 digests.  When
    omitted (older programmatic callers) the manifest falls back to the bare
    filenames and records a null digest, so existing usage keeps working.

    Returns the Manifest dataclass.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sensor_owner = _build_sensor_owner_index(plcs)
    controls_by_link = _index_controls_by_link(network.controls)
    initial_states = _initial_state_index(network)
    actuator_states = _resolve_actuator_states(plcs, initial_states)

    plc_entries: list[dict] = []
    for plc in plcs:
        file_name = f"{topology}_{plc.name.lower()}.st"
        content, fsms, sensors_referenced = _render_plc(
            plc,
            controls_by_link,
            sensor_owner,
            topology,
            inp_filename,
            plcs_filename,
            actuator_states,
        )
        (out / file_name).write_text(content, encoding="utf-8")
        plc_entries.append(
            {
                "name": plc.name,
                "file": file_name,
                "actuators": list(plc.actuators),
                "sensors_referenced": sensors_referenced,
                "fsms": fsms,
            }
        )

    manifest = Manifest(
        topology=topology,
        inp=inp_filename,
        plcs_yaml=plcs_filename,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        plcs=plc_entries,
        provenance=_build_provenance(
            inp_filename=inp_filename,
            plcs_filename=plcs_filename,
            inp_path=inp_path,
            plcs_path=plcs_path,
            actuator_states=actuator_states,
        ),
    )
    if write_manifest:
        (out / f"{topology}_manifest.json").write_text(
            json.dumps(_manifest_to_dict(manifest), indent=2) + "\n",
            encoding="utf-8",
        )
    return manifest


def _build_provenance(
    *,
    inp_filename: str,
    plcs_filename: str,
    inp_path: str | Path | None,
    plcs_path: str | Path | None,
    actuator_states: dict[str, str],
) -> dict:
    """Build the surrogate-provenance block recorded in the manifest."""
    return {
        "surrogate": True,
        "statement": SURROGATE_STATEMENT,
        "generator": {"name": GENERATOR_NAME, "version": GENERATOR_VERSION},
        "upstream": upstream_block(),
        "inputs": {
            "inp": _input_entry(inp_filename, inp_path),
            "plcs_yaml": _input_entry(plcs_filename, plcs_path),
        },
        "actuator_initial_states": dict(sorted(actuator_states.items())),
        "control_limitations": list(CONTROL_LIMITATIONS),
    }


def _input_entry(filename: str, path: str | Path | None) -> dict:
    """Describe one input file: recorded path + SHA256 digest (if readable)."""
    recorded = str(path) if path is not None else filename
    digest = sha256_file(path) if path is not None else None
    return {"path": recorded, "sha256": digest}


def _initial_state_index(network: Network) -> dict[str, str]:
    """Map link id (upper-cased) -> initial OPEN/CLOSED state."""
    states: dict[str, str] = {}
    for link in list(network.pumps) + list(network.valves):
        states[link.id.upper()] = link.initial_status.upper()
    return states


def _resolve_actuator_states(
    plcs: list[Plc], initial_states: dict[str, str]
) -> dict[str, str]:
    """Per-actuator initial state, keyed by the id used in plcs.yaml.

    Actuators that are not declared in the INP fall back to ``OPEN``, matching
    the EPANET default for links absent from [STATUS].
    """
    resolved: dict[str, str] = {}
    for plc in plcs:
        for actuator in plc.actuators:
            resolved[actuator] = initial_states.get(actuator.upper(), "OPEN")
    return resolved


def _state_initial_value(status: str) -> int:
    """Map an initial status to the ``_State`` integer encoding (1=OPEN, 0=CLOSED)."""
    return 1 if status.upper() == "OPEN" else 0


def _manifest_to_dict(m: Manifest) -> dict:
    return {
        "topology": m.topology,
        "inp": m.inp,
        "plcs_yaml": m.plcs_yaml,
        "generated_at": m.generated_at,
        "plcs": m.plcs,
        "provenance": m.provenance,
    }


def _build_sensor_owner_index(plcs: list[Plc]) -> dict[str, str]:
    """Map sensor_id -> owning PLC name. Last wins if duplicated (yaml shouldn't)."""
    owner: dict[str, str] = {}
    for plc in plcs:
        for s in plc.sensors:
            owner[s] = plc.name
    return owner


def _index_controls_by_link(controls: list[Control]) -> dict[str, list[Control]]:
    by: dict[str, list[Control]] = defaultdict(list)
    for c in controls:
        by[c.link_id].append(c)
    return by


def _render_plc(
    plc: Plc,
    controls_by_link: dict[str, list[Control]],
    sensor_owner: dict[str, str],
    topology: str,
    inp_filename: str = "",
    plcs_filename: str = "",
    actuator_states: dict[str, str] | None = None,
) -> tuple[str, list[dict], list[dict]]:
    """Return (file_text, fsms_meta, sensors_referenced_meta)."""
    case_blocks: list[str] = []
    state_var_decls: list[str] = []
    output_decls: list[str] = []
    sensors_used: list[str] = []
    fsms: list[dict] = []
    states = actuator_states or {}

    for act in plc.actuators:
        ctrls = controls_by_link.get(act, [])
        if not ctrls:
            continue  # empty-shell handled in Task 8
        block, fsm_meta, sensors_in_block = _render_case_block(act, ctrls)
        case_blocks.append(block)
        act_id = _iec_ident(act)
        init_state = _state_initial_value(states.get(act, "OPEN"))
        state_var_decls.append(f"    {act_id}_State : INT := {init_state};")
        output_decls.append(f"    {act_id} : BOOL;")
        sensors_used.extend(sensors_in_block)
        fsms.append(fsm_meta)

    # Unique-preserving order for sensors_used
    seen: set[str] = set()
    sensors_ordered: list[str] = []
    for s in sensors_used:
        if s not in seen:
            seen.add(s)
            sensors_ordered.append(s)

    var_input_lines = [
        _render_var_input_line(s, sensor_owner, plc.name) for s in sensors_ordered
    ]

    sensors_referenced = [
        {"id": s, "owner_plc": sensor_owner.get(s, "")} for s in sensors_ordered
    ]

    header = _render_header(topology, plc.name, inp_filename, plcs_filename)

    if not case_blocks:
        return (
            header
            + f"\nPROGRAM {plc.name}\n"
            f"(* No actuators with [CONTROLS] entries -- sensor-only or "
            f"static-setting PLC. *)\n"
            f"END_PROGRAM\n",
            fsms,
            sensors_referenced,
        )

    body = "\n".join(case_blocks) + "\n"
    footer = "END_PROGRAM\n"
    return header + f"\nPROGRAM {plc.name}\n" + _var_blocks(
        var_input_lines, output_decls, state_var_decls
    ) + "\n" + body + footer, fsms, sensors_referenced


def _render_header(
    topology: str, plc_name: str, inp_filename: str, plcs_filename: str
) -> str:
    """Return the ST header comment, including the surrogate provenance note."""
    note = "\n".join(f"   {line}" for line in SURROGATE_HEADER_LINES)
    return (
        f"(* Auto-generated by {GENERATOR_NAME} {GENERATOR_VERSION} "
        f"from {inp_filename} + {plcs_filename}\n"
        f"   Topology: {topology}   PLC: {plc_name}\n"
        f"{note}\n"
        f"   DO NOT EDIT. *)\n"
    )


def _var_blocks(
    var_input_lines: list[str], output_decls: list[str], state_var_decls: list[str]
) -> str:
    """Return the VAR_INPUT / VAR_OUTPUT / VAR declaration blocks."""
    out = ""
    if var_input_lines:
        out += "VAR_INPUT\n" + "\n".join(var_input_lines) + "\nEND_VAR\n"
    if output_decls:
        out += "VAR_OUTPUT\n" + "\n".join(output_decls) + "\nEND_VAR\n"
    if state_var_decls:
        out += "VAR\n" + "\n".join(state_var_decls) + "\nEND_VAR\n"
    return out


def _render_var_input_line(sensor: str, owner: dict[str, str], this_plc: str) -> str:
    if sensor not in owner:
        annot = "(* unassigned in plcs.yaml *)"
    elif owner[sensor] == this_plc:
        annot = "(* owned by this PLC *)"
    else:
        annot = f"(* sourced from {owner[sensor]} *)"
    sensor_id = _iec_ident(sensor)
    return f"    {sensor_id} : REAL;            {annot}"


def _render_case_block(
    actuator: str, ctrls: list[Control]
) -> tuple[str, dict, list[str]]:
    """Render one CASE block for a 2-state (OPEN/CLOSED) actuator.

    Raises MultiStateActuatorError if the same (comparator, threshold) pair maps
    to both OPEN and CLOSED — that is ambiguous and cannot be represented as a
    deterministic 2-state FSM.  Any number of distinct pairs is fine as long as
    each pair maps to exactly one target state.
    """
    pair_to_states: dict[tuple[str, float], set[str]] = {}
    for c in ctrls:
        key = (c.comparator, c.threshold)
        pair_to_states.setdefault(key, set()).add(c.target_state)
    ambiguous = {k: v for k, v in pair_to_states.items() if len(v) > 1}
    if ambiguous:
        raise MultiStateActuatorError(
            f"actuator {actuator!r} has ambiguous (comparator,threshold) pairs that map "
            f"to both OPEN and CLOSED: {ambiguous}"
        )

    open_trans = [c for c in ctrls if c.target_state == "OPEN"]
    closed_trans = [c for c in ctrls if c.target_state == "CLOSED"]

    act_id = _iec_ident(actuator)
    sensors: list[str] = []
    lines: list[str] = [f"(* ----- FSM for actuator {actuator} ----- *)"]
    lines.append(f"CASE {act_id}_State OF")

    # state 0 (CLOSED): transitions that flip to OPEN
    lines.append("    0:  (* CLOSED *)")
    lines.append(f"        {act_id} := FALSE;")
    for c in open_trans:
        sensors.append(c.sensor_node)
        op = "<" if c.comparator == "BELOW" else ">"
        sensor_id = _iec_ident(c.sensor_node)
        lines.append(f"        IF {sensor_id} {op} {c.threshold:g} THEN")
        lines.append(f"            {act_id}_State := 1;")
        lines.append("        END_IF;")

    # state 1 (OPEN): transitions that flip to CLOSED
    lines.append("    1:  (* OPEN *)")
    lines.append(f"        {act_id} := TRUE;")
    for c in closed_trans:
        sensors.append(c.sensor_node)
        op = "<" if c.comparator == "BELOW" else ">"
        sensor_id = _iec_ident(c.sensor_node)
        lines.append(f"        IF {sensor_id} {op} {c.threshold:g} THEN")
        lines.append(f"            {act_id}_State := 0;")
        lines.append("        END_IF;")

    lines.append("END_CASE;")
    block = "\n".join(lines) + "\n"
    fsm_meta = {
        "actuator": actuator,
        "states": 2,
        "transitions": len(open_trans) + len(closed_trans),
    }
    return block, fsm_meta, sensors
