"""AST-XML -> raw parsed structures. Faithful port of src/xml_parser.rs.

roxmltree -> lxml.etree mapping:
- Document::parse        -> etree.fromstring(bytes)
- node.descendants()     -> elem.iter()  (self first, then doc-order)
- node.tag_name().name() -> elem.tag     (no namespaces in AST files)
- node.text()            -> elem.text    (Optional[str])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from .model import GfsmError


@dataclass
class Assignment:
    variable: str
    value: str


@dataclass
class IfStatement:
    condition: str
    assignments: list[Assignment] = field(default_factory=list)


@dataclass
class CaseElement:
    state_id: str
    if_statements: list[IfStatement] = field(default_factory=list)


@dataclass
class FunctionBlockData:
    name: str
    case_variable: str
    case_elements: list[CaseElement] = field(default_factory=list)


def _first(root: etree._Element, tag: str) -> etree._Element | None:
    """First descendant (incl. root) whose tag == `tag`, document order.

    Mirrors roxmltree `node.descendants().find(|n| n.tag_name().name()==tag)`.
    """
    for el in root.iter():
        if el.tag == tag:
            return el
    return None


def _text(el: etree._Element | None) -> str | None:
    return None if el is None else el.text


def _preprocess(content: str) -> str:
    """Faithful port of xml_parser.rs:16-18 — opening-tag-only rewrite.

    Rust rewrites ONLY the opening pair `<expression><integer-literal>` /
    `<expression><boolean-literal>` to `<value>...`; the matching
    `</expression>` close tag is deliberately left unchanged (Rust does the
    same). On the real, pretty-printed AST these adjacent substrings never
    occur, so this is a no-op there — byte-for-byte matching Rust. Do NOT
    add a closing-tag rewrite; that would diverge from the Rust port.
    """
    return content.replace(
        "<expression><integer-literal>", "<value><integer-literal>"
    ).replace(
        "<expression><boolean-literal>", "<value><boolean-literal>"
    )


class XmlParser:
    def __init__(self, content: str) -> None:
        # xml_parser.rs:16-18 — raw string replace BEFORE parsing.
        content = _preprocess(content)
        self._content = content
        try:
            self._root = etree.fromstring(content.encode("utf-8"))
        except etree.XMLSyntaxError as exc:
            raise GfsmError(f"XML parsing error: {exc}") from exc

    @classmethod
    def from_path(cls, xml_path: Path) -> "XmlParser":
        return cls(Path(xml_path).read_text())

    def find_function_blocks(self) -> list[str]:
        blocks: list[str] = []
        for node in self._root.iter():
            if node.tag == "function-block-declaration":
                name = _text(_first(node, "derived-function-block-name"))
                if name is not None:
                    blocks.append(name)
            elif node.tag == "program-declaration":
                name = _text(_first(node, "program-type-name"))
                if name is not None:
                    blocks.append(name)
        return blocks

    def _find_block_node(self, name: str) -> etree._Element | None:
        for node in self._root.iter():
            if node.tag == "function-block-declaration":
                cur = _text(_first(node, "derived-function-block-name"))
            elif node.tag == "program-declaration":
                cur = _text(_first(node, "program-type-name"))
            else:
                cur = None
            if cur is not None and cur == name:
                return node
        return None

    def extract_function_blocks(self, name: str) -> list[FunctionBlockData]:
        """One FunctionBlockData per CASE construct in the program (paper
        Def 1: one FSM per CASE). Case-statements are siblings in the
        st_gen-emitted program body; doc-order is stable.
        """
        fb_node = self._find_block_node(name)
        if fb_node is None:
            raise GfsmError(f"Function block '{name}' not found")
        case_stmts = [
            n for n in fb_node.iter() if n.tag == "case-statement"
        ]
        if not case_stmts:
            raise GfsmError(
                f"No case statement found in function block '{name}'"
            )
        out: list[FunctionBlockData] = []
        for case_stmt in case_stmts:
            out.append(FunctionBlockData(
                name=name,
                case_variable=self._extract_case_variable(case_stmt),
                case_elements=self._extract_case_elements(case_stmt),
            ))
        return out

    def extract_function_block(self, name: str) -> FunctionBlockData:
        """First CASE only (backward-compat for single-CASE callers)."""
        return self.extract_function_blocks(name)[0]

    def _extract_case_variable(self, case_stmt: etree._Element) -> str:
        vn = _text(_first(case_stmt, "variable-name"))
        if vn is None:
            raise GfsmError("XML parsing error: Case variable not found")
        return vn

    def _extract_case_elements(
        self, case_stmt: etree._Element
    ) -> list[CaseElement]:
        elements: list[CaseElement] = []
        for node in case_stmt.iter():
            if node.tag == "case-element":
                try:
                    elements.append(self._parse_case_element(node))
                except GfsmError:
                    # Rust: `if let Ok(element) = ...` — skip on error.
                    continue
        return elements

    def _parse_case_element(self, element_node: etree._Element) -> CaseElement:
        state_id = self._extract_state_id(element_node)
        if_statements = self._extract_if_statements(element_node)
        return CaseElement(state_id=state_id, if_statements=if_statements)

    def _extract_state_id(self, element_node: etree._Element) -> str:
        for node in element_node.iter():
            if node.tag == "case-list-element":
                for child in node.iter():
                    if child.tag == "integer-literal" and child.text is not None:
                        return child.text
        raise GfsmError("XML parsing error: State ID not found")

    def _extract_if_statements(
        self, element_node: etree._Element
    ) -> list[IfStatement]:
        statements: list[IfStatement] = []
        for node in element_node.iter():
            if node.tag == "if-statement":
                statements.append(self._parse_if_statement(node))
        return statements

    def _parse_if_statement(self, if_node: etree._Element) -> IfStatement:
        condition = self._extract_expression(if_node)
        assignments = self._extract_assignments(if_node)
        return IfStatement(condition=condition, assignments=assignments)

    def _extract_expression(self, node: etree._Element) -> str:
        expr = _first(node, "expression")
        if expr is None:
            return ""
        return self._parse_expression_node(expr)

    def _parse_expression_node(self, expr_node: etree._Element) -> str:
        parts: list[str] = []
        in_not = False
        for n in expr_node.iter():
            tag = n.tag
            if tag == "logical-not":
                in_not = True
            elif tag == "logical-and":
                parts.append(" AND ")
            elif tag == "logical-or":
                parts.append(" OR ")
            elif tag == "equal":
                parts.append(" = ")
            elif tag == "not-equal":
                parts.append(" <> ")
            elif tag == "less-than":
                parts.append(" < ")
            elif tag == "less-or-equal":
                parts.append(" <= ")
            elif tag == "greater-than":
                parts.append(" > ")
            elif tag == "greater-or-equal":
                parts.append(" >= ")
            elif tag == "adding":
                parts.append(" + ")
            elif tag == "subtracting":
                parts.append(" - ")
            elif tag == "variable-name":
                if n.text is not None:
                    if in_not:
                        parts.append("NOT ")
                        in_not = False
                    parts.append(n.text)
            elif tag in ("integer-literal", "real-literal", "boolean-literal"):
                if n.text is not None:
                    parts.append(n.text)
        return "".join(parts).strip()

    def _extract_assignments(self, if_node: etree._Element) -> list[Assignment]:
        assignments: list[Assignment] = []
        for node in if_node.iter():
            if node.tag == "assignment-statement":
                assignments.append(self._parse_assignment(node))
        return assignments

    def _parse_assignment(self, assign_node: etree._Element) -> Assignment:
        variable = ""
        value = ""
        for n in assign_node.iter():
            if not variable and n.tag == "variable-name" and n.text is not None:
                variable = n.text
                break
        for n in assign_node.iter():
            if n.tag in ("integer-literal", "real-literal", "boolean-literal") and n.text is not None:
                value = n.text
                break
        return Assignment(variable=variable, value=value)
