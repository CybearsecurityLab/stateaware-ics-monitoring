from xml.etree import ElementTree as ET

from st_analyze.adapter import analyze_source

TINY_ST = """\
PROGRAM PLC1
VAR
    P1_State : INT := 0;
END_VAR
CASE P1_State OF
    0:
        IF T1 < 5 THEN
            P1_State := 1;
        END_IF;
END_CASE;
END_PROGRAM
"""

MULTI_ACTUATOR_ST = """\
PROGRAM PLC1
VAR_INPUT
    T1 : REAL;
    T2 : REAL;
END_VAR
VAR_OUTPUT
    P1 : BOOL;
    P2 : BOOL;
END_VAR
VAR
    P1_State : INT := 1;
    P2_State : INT := 1;
END_VAR
CASE P1_State OF
    0:
        P1 := FALSE;
        IF T1 < 5 THEN
            P1_State := 1;
        END_IF;
    1:
        P1 := TRUE;
        IF T1 > 8 THEN
            P1_State := 0;
        END_IF;
END_CASE;
CASE P2_State OF
    0:
        P2 := FALSE;
        IF T2 < 5 THEN
            P2_State := 1;
        END_IF;
    1:
        P2 := TRUE;
        IF T2 > 8 THEN
            P2_State := 0;
        END_IF;
END_CASE;
END_PROGRAM
"""

EMPTY_ST = "PROGRAM PLC2\n(* no actuators *)\nEND_PROGRAM\n"


def test_analyze_source_returns_well_formed_xml():
    r = analyze_source(TINY_ST)
    ET.fromstring(r.ast_xml)  # raises if malformed
    assert r.ast_xml.strip()


def test_analyze_source_produces_pdg_and_invariant_containers():
    r = analyze_source(TINY_ST)
    assert isinstance(r.invariants, list)
    assert isinstance(r.pdg_structured, dict)
    assert "digraph" in r.pdg_dot.lower() or r.pdg_dot == ""
    assert r.ok is True
    assert r.programs  # PDG state keys present (e.g. ["0"])


def test_multi_actuator_analysis_aggregates_every_case_block():
    r = analyze_source(MULTI_ACTUATOR_ST)
    assert r.ok is True, r.errors
    actuation = [inv for inv in r.invariants if inv.get("actuation_var")]
    assert {inv["actuation_var"] for inv in actuation} == {"P1", "P2"}
    assert {inv["case_var"] for inv in actuation} == {"P1_State", "P2_State"}
    assert len(actuation) == 4


def test_analyze_empty_program_is_ok_not_crash():
    r = analyze_source(EMPTY_ST)
    assert r.ok is True
    assert isinstance(r.invariants, list)


def test_invariants_are_json_serializable_dicts():
    import json

    r = analyze_source(TINY_ST)
    json.dumps(r.invariants)  # raises if not serializable
    for inv in r.invariants:
        assert "type" in inv and "id" in inv


def test_malformed_source_sets_not_ok_without_raising():
    # Garbage that the analyzer grammar cannot parse must not escape
    # as an unhandled exception; analyze_source must return ok=False.
    r = analyze_source("@@@ this is definitely not structured text @@@")
    assert r.ok is False
    assert r.errors
    assert r.ast_xml == ""
    assert r.invariants == []
    assert r.pdg_structured == {}
