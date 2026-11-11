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

from iec_st_compiler import ast_writer, core, invariants as inv_mod, pdg as pdg_mod

from .model import AnalyzeResult

# st_gen emits only simple (* ... *) / { ... } comments. This is the
# analyzer's own DEFAULT_COMMENT_PATTERN (defined in its cli.py); we
# replicate it here so the pristine analyzer is never imported via cli.
_COMMENT_PATTERN = re.compile(r"(\(\*.*?\*\))|(\{.*?})", re.S)


def analyze_source(source: str) -> AnalyzeResult:
    """Run the vendored analyzer over one .st source string."""
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

        pdg_dot = ""
        pdg_structured: dict = {}
        if pdgs:
            try:
                pdg_dot = ast_writer.export_pdgs_to_graphviz(pdgs)
                pdg_structured = {
                    name: graph.to_dict() for name, graph in pdgs.items()
                }
            except Exception as exc:  # noqa: BLE001
                errors.append(f"pdg export failed: {exc!r}")

    case_blocks = list(
        re.finditer(r"CASE\s+\w+_State\s+OF.*?END_CASE;", source, flags=re.S)
    )
    if len(case_blocks) > 1:
        invariants = []
        for block in case_blocks:
            reduced_source = (
                source[: case_blocks[0].start()]
                + block.group(0)
                + source[case_blocks[-1].end() :]
            )
            block_result = analyze_source(reduced_source)
            for invariant in block_result.invariants:
                tagged = dict(invariant)
                tagged["case_var"] = block_result.state_variable
                invariants.append(tagged)
            errors.extend(block_result.errors)
    else:
        invariants = [
            inv.to_dict()
            for inv_list in (invariant_objs or {}).values()
            for inv in inv_list
        ]

    analyzer_noise = captured.getvalue()
    if "Traceback (most recent call last)" in analyzer_noise:
        errors.append("analyzer emitted a traceback (see captured output)")

    ok = not errors
    return AnalyzeResult(
        ast_xml=ast_xml,
        invariants=invariants,
        pdg_dot=pdg_dot,
        pdg_structured=pdg_structured,
        programs=sorted((pdgs or {}).keys()),
        state_variable=state_variable,
        ok=ok,
        errors=errors,
    )
