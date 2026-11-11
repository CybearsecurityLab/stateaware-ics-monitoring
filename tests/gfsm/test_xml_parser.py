import pytest

from gfsm.model import GfsmError
from gfsm.xml_parser import XmlParser

# NOTE: real Stage-2 AST XML is pretty-printed (a newline + indent between
# every open tag and its first child), so `<expression><integer-literal>`
# adjacency never occurs and the faithful opening-only `_preprocess` is a
# no-op on real data (exactly like the Rust `.replace`). These fixtures
# mirror that real formatting so they stay well-formed after preprocessing.
PROGRAM_XML = """<iec-source>
  <program-declaration>
    <program-type-name>PLC1</program-type-name>
    <function-block-body><statement-list>
      <case-statement>
        <expression>
          <variable-name>P78_State</variable-name>
        </expression>
        <case-element>
          <case-list><case-list-element>
            <integer-literal>0</integer-literal>
          </case-list-element></case-list>
          <statement-list>
            <if-statement>
              <expression>
                <variable-name>T41</variable-name>
                <less-than/>
                <integer-literal>5</integer-literal>
              </expression>
              <statement-list>
                <assignment-statement>
                  <variable-name>P78_State</variable-name>
                  <expression>
                    <integer-literal>1</integer-literal>
                  </expression>
                </assignment-statement>
              </statement-list>
            </if-statement>
          </statement-list>
        </case-element>
      </case-statement>
    </statement-list></function-block-body>
  </program-declaration>
</iec-source>"""

FB_XML = """<iec-source>
  <function-block-declaration>
    <derived-function-block-name>FB1</derived-function-block-name>
    <case-statement>
      <expression><variable-name>S</variable-name></expression>
      <case-element>
        <case-list><case-list-element>
          <integer-literal>10</integer-literal>
        </case-list-element></case-list>
      </case-element>
    </case-statement>
  </function-block-declaration>
</iec-source>"""


def test_find_program_declaration():
    p = XmlParser(PROGRAM_XML)
    assert p.find_function_blocks() == ["PLC1"]


def test_find_function_block_declaration():
    p = XmlParser(FB_XML)
    assert p.find_function_blocks() == ["FB1"]


def test_preprocess_is_faithful_opening_only():
    from gfsm.xml_parser import _preprocess

    # Exactly the Rust .replace (xml_parser.rs:16-18): ONLY the opening
    # "<expression><integer-literal>"/"<expression><boolean-literal>" pair
    # is rewritten to "<value>...". The matching close tag stays
    # "</expression>" (Rust does NOT rewrite it). "<expression><variable-name>"
    # is never touched.
    src = (
        "<r>"
        "<expression><integer-literal>1</integer-literal></expression>"
        "<expression><boolean-literal>TRUE</boolean-literal></expression>"
        "<expression><variable-name>S</variable-name></expression>"
        "</r>"
    )
    assert _preprocess(src) == (
        "<r>"
        "<value><integer-literal>1</integer-literal></expression>"
        "<value><boolean-literal>TRUE</boolean-literal></expression>"
        "<expression><variable-name>S</variable-name></expression>"
        "</r>"
    )


def test_preprocess_noop_on_pretty_printed_real_style():
    from gfsm.xml_parser import _preprocess

    # Real AST is pretty-printed: no adjacency -> preprocessing is a no-op,
    # so the parsed content is byte-identical to the input (matches Rust).
    assert _preprocess(PROGRAM_XML) == PROGRAM_XML
    assert XmlParser(PROGRAM_XML)._content == PROGRAM_XML


def test_malformed_xml_raises_gfsmerror():
    with pytest.raises(GfsmError):
        XmlParser("<iec-source><unclosed>")


def test_extract_case_variable_is_selector_not_inner():
    p = XmlParser(PROGRAM_XML)
    fbd = p.extract_function_block("PLC1")
    assert fbd.name == "PLC1"
    assert fbd.case_variable == "P78_State"
    assert [ce.state_id for ce in fbd.case_elements] == ["0"]


def test_block_not_found_raises():
    p = XmlParser(PROGRAM_XML)
    with pytest.raises(GfsmError, match="not found"):
        p.extract_function_block("NOPE")


def test_no_case_statement_raises():
    xml = (
        "<iec-source><program-declaration>"
        "<program-type-name>P</program-type-name>"
        "</program-declaration></iec-source>"
    )
    p = XmlParser(xml)
    with pytest.raises(GfsmError, match="No case statement"):
        p.extract_function_block("P")


def test_if_condition_flattened():
    p = XmlParser(PROGRAM_XML)
    fbd = p.extract_function_block("PLC1")
    ce = fbd.case_elements[0]
    assert len(ce.if_statements) == 1
    assert ce.if_statements[0].condition == "T41 < 5"


def test_real_literal_condition_flattened():
    xml = PROGRAM_XML.replace(
        "<integer-literal>5</integer-literal>",
        "<real-literal>1.5</real-literal>",
    )
    fbd = XmlParser(xml).extract_function_block("PLC1")
    assert fbd.case_elements[0].if_statements[0].condition == "T41 < 1.5"


def test_logical_and_not_flattening():
    xml = """<iec-source><function-block-declaration>
      <derived-function-block-name>FB</derived-function-block-name>
      <case-statement>
        <expression><variable-name>S</variable-name></expression>
        <case-element>
          <case-list><case-list-element>
            <integer-literal>10</integer-literal>
          </case-list-element></case-list>
          <statement-list><if-statement>
            <expression>
              <variable-name>A</variable-name><equal/><integer-literal>1</integer-literal>
              <logical-and/>
              <logical-not/><variable-name>B</variable-name>
            </expression>
            <statement-list></statement-list>
          </if-statement></statement-list>
        </case-element>
      </case-statement>
    </function-block-declaration></iec-source>"""
    fbd = XmlParser(xml).extract_function_block("FB")
    assert fbd.case_elements[0].if_statements[0].condition == (
        "A = 1 AND NOT B"
    )


def test_empty_condition_when_no_expression():
    xml = """<iec-source><function-block-declaration>
      <derived-function-block-name>FB</derived-function-block-name>
      <case-statement>
        <expression><variable-name>S</variable-name></expression>
        <case-element>
          <case-list><case-list-element>
            <integer-literal>10</integer-literal>
          </case-list-element></case-list>
          <statement-list><if-statement>
            <statement-list></statement-list>
          </if-statement></statement-list>
        </case-element>
      </case-statement>
    </function-block-declaration></iec-source>"""
    fbd = XmlParser(xml).extract_function_block("FB")
    assert fbd.case_elements[0].if_statements[0].condition == ""


def test_assignment_extracted():
    p = XmlParser(PROGRAM_XML)
    fbd = p.extract_function_block("PLC1")
    asn = fbd.case_elements[0].if_statements[0].assignments
    assert len(asn) == 1
    assert asn[0].variable == "P78_State"
    assert asn[0].value == "1"


def test_assignment_defaults_empty_strings():
    xml = """<iec-source><function-block-declaration>
      <derived-function-block-name>FB</derived-function-block-name>
      <case-statement>
        <expression><variable-name>S</variable-name></expression>
        <case-element>
          <case-list><case-list-element>
            <integer-literal>10</integer-literal>
          </case-list-element></case-list>
          <statement-list><if-statement>
            <expression><variable-name>X</variable-name></expression>
            <statement-list><assignment-statement>
            </assignment-statement></statement-list>
          </if-statement></statement-list>
        </case-element>
      </case-statement>
    </function-block-declaration></iec-source>"""
    fbd = XmlParser(xml).extract_function_block("FB")
    a = fbd.case_elements[0].if_statements[0].assignments[0]
    assert a.variable == ""
    assert a.value == ""


# ---------------------------------------------------------------------------
# Tests for extract_function_blocks (plural) — one FBD per CASE construct
# ---------------------------------------------------------------------------

MULTI_CASE_XML = """<iec-source>
  <program-declaration>
    <program-type-name>PLC1</program-type-name>
    <function-block-body><statement-list>
      <case-statement>
        <expression>
          <variable-name>P78_State</variable-name>
        </expression>
        <case-element>
          <case-list><case-list-element>
            <integer-literal>0</integer-literal>
          </case-list-element></case-list>
        </case-element>
      </case-statement>
      <case-statement>
        <expression>
          <variable-name>P79_State</variable-name>
        </expression>
        <case-element>
          <case-list><case-list-element>
            <integer-literal>0</integer-literal>
          </case-list-element></case-list>
        </case-element>
      </case-statement>
    </statement-list></function-block-body>
  </program-declaration>
</iec-source>"""


def test_extract_function_blocks_one_per_case():
    p = XmlParser(MULTI_CASE_XML)
    fbds = p.extract_function_blocks("PLC1")
    assert [f.case_variable for f in fbds] == ["P78_State", "P79_State"]
    assert all(f.name == "PLC1" for f in fbds)
    assert [ce.state_id for ce in fbds[0].case_elements] == ["0"]


def test_extract_function_blocks_single_case():
    p = XmlParser(PROGRAM_XML)
    fbds = p.extract_function_blocks("PLC1")
    assert len(fbds) == 1
    assert fbds[0].case_variable == "P78_State"


def test_extract_function_blocks_not_found_raises():
    p = XmlParser(PROGRAM_XML)
    with pytest.raises(GfsmError, match="not found"):
        p.extract_function_blocks("NOPE")


def test_extract_function_blocks_no_case_raises():
    xml = (
        "<iec-source><program-declaration>"
        "<program-type-name>P</program-type-name>"
        "</program-declaration></iec-source>"
    )
    with pytest.raises(GfsmError, match="No case statement"):
        XmlParser(xml).extract_function_blocks("P")


def test_extract_function_block_still_returns_first_case():
    # Backward-compat: existing single-CASE callers/tests unchanged.
    p = XmlParser(MULTI_CASE_XML)
    fbd = p.extract_function_block("PLC1")
    assert fbd.case_variable == "P78_State"
