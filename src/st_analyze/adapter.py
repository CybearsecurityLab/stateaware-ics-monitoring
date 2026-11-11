"""The only module that imports the vendored iec_st_compiler analyzer.

One parse of the ST source yields every Stage-2 artifact. The analyzer
swallows PDG/invariant exceptions and prints DEBUG/tracebacks to stdout
while returning None/partial results, so we capture its output and treat
a missing PDG as a failure signal.
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stderr, redirect_stdout

from iec_st_compiler import core, invariants as inv_mod, pdg as pdg_mod

from .model import AnalyzeResult

# st_gen emits only simple (* ... *) / { ... } comments. This is the
# analyzer's own DEFAULT_COMMENT_PATTERN (defined in its cli.py); we
# replicate it here so the pristine analyzer is never imported via cli.
_COMMENT_PATTERN = re.compile(r"(\(\*.*?\*\))|(\{.*?})", re.S)


_CASE_BLOCK_PATTERN = re.compile(
    r"CASE\s+(\w+_State)\s+OF.*?END_CASE;", flags=re.S
)
_PROGRAM_PATTERN = re.compile(r"\bPROGRAM\s+([A-Za-z_]\w*)")


def analyze_source(source: str) -> AnalyzeResult:
    """Run the vendored analyzer, retaining every actuator CASE separately."""
    case_blocks = list(_CASE_BLOCK_PATTERN.finditer(source))
    result = _analyze_single_source(source)
    if len(case_blocks) <= 1 or (not result.ast_xml and not result.ok):
        return result

    # The vendored analyzer keys graphs only by state ID, so its full-program
    # result contains the final CASE's 0/1/... graphs. Keep those top-level
    # entries for existing readers, but build the complete representation from
    # independent single-CASE analyses rather than recursing through this
    # multi-CASE wrapper.
    components: list[dict] = []
    invariants: list[dict] = []
    errors = list(result.errors)
    for case_index, block in enumerate(case_blocks):
        reduced_source = (
            source[: case_blocks[0].start()]
            + block.group(0)
            + source[case_blocks[-1].end() :]
        )
        block_result = _analyze_single_source(reduced_source)
        case_var = block_result.state_variable or block.group(1)
        components.append({
            "plc": _program_name(source),
            "case_var": case_var,
            "case_index": case_index,
            "states": _state_graphs(block_result.pdg_structured),
        })
        for invariant in block_result.invariants:
            tagged = dict(invariant)
            tagged["case_var"] = case_var
            invariants.append(tagged)
        errors.extend(block_result.errors)

    result.invariants = invariants
    result.pdg_components = components
    result.pdg_structured = {
        **{
            key: graph
            for key, graph in result.pdg_structured.items()
            if key != "components"
        },
        "components": components,
    }
    try:
        result.pdg_dot = _export_components_to_graphviz(components)
    except Exception as exc:  # noqa: BLE001 - keep analyzer API non-throwing
        result.pdg_dot = ""
        errors.append(f"pdg export failed: {exc!r}")
    result.errors = errors
    result.ok = not errors
    return result


def _analyze_single_source(source: str) -> AnalyzeResult:
    """Analyze source without applying the multi-CASE aggregation wrapper."""
    errors: list[str] = []
    captured = io.StringIO()

    with redirect_stdout(captured), redirect_stderr(captured):
        # compile_to_ast and compile_to_xml each parse from scratch (a
        # deliberate, spec-approved double parse: public API only, .st
        # files are small). A parse failure here must not escape.
        try:
            ast = core.compile_to_ast(source, _COMMENT_PATTERN)
            ast_xml = core.compile_to_xml(
                source, _COMMENT_PATTERN, pretty_print=True
            )
        except Exception as exc:  # noqa: BLE001 - analyzer raises broadly
            return AnalyzeResult(
                ast_xml="",
                invariants=[],
                pdg_dot="",
                pdg_structured={},
                programs=[],
                state_variable=None,
                ok=False,
                errors=[f"parse failed: {exc!r}"],
            )

        pdgs = None
        state_variable = None
        try:
            pdgs, state_variable = pdg_mod.build_all_pdgs(ast)
        except Exception as exc:  # noqa: BLE001 - analyzer raises broadly
            errors.append(f"build_all_pdgs failed: {exc!r}")

        invariant_objs: dict = {}
        if pdgs:
            try:
                invariant_objs = inv_mod.extract_invariants_from_all_pdgs(
                    pdgs, state_variable
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"extract_invariants failed: {exc!r}")

        legacy_pdgs: dict = {}
        if pdgs:
            try:
                legacy_pdgs = {
                    name: graph.to_dict() for name, graph in pdgs.items()
                }
            except Exception as exc:  # noqa: BLE001
                errors.append(f"pdg export failed: {exc!r}")

    components: list[dict] = []
    if state_variable is not None:
        components.append({
            "plc": _program_name(source),
            "case_var": state_variable,
            "case_index": 0,
            "states": _state_graphs(legacy_pdgs),
        })

    pdg_dot = ""
    if components:
        try:
            pdg_dot = _export_components_to_graphviz(components)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"pdg export failed: {exc!r}")

    invariants = []
    for inv_list in (invariant_objs or {}).values():
        for inv in inv_list:
            tagged = inv.to_dict()
            if state_variable is not None:
                tagged["case_var"] = state_variable
            invariants.append(tagged)
    analyzer_noise = captured.getvalue()
    if "Traceback (most recent call last)" in analyzer_noise:
        errors.append("analyzer emitted a traceback (see captured output)")

    pdg_structured = dict(legacy_pdgs)
    if components:
        pdg_structured["components"] = components
    return AnalyzeResult(
        ast_xml=ast_xml,
        invariants=invariants,
        pdg_dot=pdg_dot,
        pdg_structured=pdg_structured,
        pdg_components=components,
        programs=sorted((pdgs or {}).keys()),
        state_variable=state_variable,
        ok=not errors,
        errors=errors,
    )


def _program_name(source: str) -> str:
    match = _PROGRAM_PATTERN.search(source)
    return match.group(1) if match else ""


def _state_graphs(pdg_structured: dict) -> list[dict]:
    """Return state graphs in state order, excluding additive metadata."""
    return [
        graph
        for state_id, graph in sorted(pdg_structured.items())
        if state_id != "components"
    ]


def _dot_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(value))


def _dot_string(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace(
        "\n", "\\n"
    )


def _export_components_to_graphviz(components: list[dict]) -> str:
    """Export complete component PDGs with collision-free DOT identifiers."""
    lines = [
        "digraph PDGs {",
        "  rankdir=TB;",
        "  compound=true;",
        "  node [shape=box];",
        "",
    ]
    for component_index, component in enumerate(components):
        plc = component["plc"]
        case_var = component["case_var"]
        namespace = (
            f"c{component_index}_{_dot_name(plc)}_{_dot_name(case_var)}"
        )
        for state_index, graph in enumerate(component["states"]):
            state_id = graph["state_id"]
            state_namespace = f"{namespace}_s{state_index}"
            lines.extend([
                f"  subgraph cluster_{state_namespace} {{",
                f'    label="{_dot_string(plc)} / {_dot_string(case_var)}'
                f' / State {_dot_string(state_id)}";',
                "    style=dashed;",
                "    color=blue;",
                "",
            ])
            node_names = {
                node["id"]: f"{state_namespace}_n{node['id']}"
                for node in graph["nodes"]
            }
            for node in graph["nodes"]:
                color = {
                    "assignment": "lightblue",
                    "condition": "lightyellow",
                    "if_block": "lightgreen",
                }.get(node["type"], "white")
                lines.append(
                    f"    {node_names[node['id']]} "
                    f'[label="{_dot_string(node["statement"])}", '
                    f"fillcolor={color}, style=filled];"
                )
            for edge in graph["edges"]:
                style = "solid" if edge["type"] == "control" else "dashed"
                color = "red" if edge["type"] == "control" else "blue"
                label = edge.get("label") or edge.get("variable") or ""
                lines.append(
                    f"    {node_names[edge['from']]} -> "
                    f"{node_names[edge['to']]} [style={style}, "
                    f'color={color}, label="{_dot_string(label)}"];'
                )
            lines.extend(["  }", ""])
    lines.append("}")
    return "\n".join(lines) + "\n"
