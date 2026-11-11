"""Focused tests for the partial source-derived native SAIN baseline."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from baselines.sain import SainLocalDetector
from baselines.sain_native import (
    SAIN_NATIVE_GAPS,
    SAIN_NATIVE_SCHEMA,
    SAIN_NATIVE_STATUS,
    SainNativeDetector,
    SainTemplateError,
    derive_source_templates,
    load_invariant_documents,
    mine_native_sain,
    parse_transition_condition,
)

COMPONENTS = [("P1", "S"), ("P2", "T")]
FB_TO_COL = {("P1", "S"): "p1", ("P2", "T"): "p2"}
COLUMN_MAP = {"T41": "t41", "T42": "t42", "P78": "p78", "P79": "p79"}


def _fsm_documents(with_p2_reverse: bool = True):
    p2_transitions = [
        {"id": "0_to_1", "from_state": "0", "to_state": "1",
         "condition": "T42 < 5"},
    ]
    if with_p2_reverse:
        p2_transitions.append(
            {"id": "1_to_0", "from_state": "1", "to_state": "0",
             "condition": "T42 > 8"}
        )
    return [
        ("P1", {"function_blocks": [{
            "name": "S", "case_variable": "S",
            "transitions": [
                {"id": "0_to_1", "from_state": "0", "to_state": "1",
                 "condition": "T41 < 5"},
                {"id": "1_to_0", "from_state": "1", "to_state": "0",
                 "condition": "T41 > 8"},
            ],
        }]}),
        ("P2", {"function_blocks": [{
            "name": "T", "case_variable": "T",
            "transitions": p2_transitions,
        }]}),
    ]


def _invariant_documents():
    """Synthetic PDG invariant metadata (shape of InvariantTemplate.to_dict)."""
    return [
        ("P1", "S", [
            {
                "id": "single_0_T41_P78", "type": "single", "state_id": "0",
                "variables": ["T41", "P78"], "confidence": 1.0,
                "structure": "IF T41 < [#] THEN P78 = TRUE",
                "sensing_var": "T41", "sensing_var_type": "sensing",
                "actuation_var": "P78", "operator": "<",
                "actuation_value": "TRUE",
            },
            {
                "id": "multi_1_T41_P78", "type": "multi", "state_id": "1",
                "variables": ["T41", "P78"], "confidence": 1.0,
                "structure": "IF T41 > 8 THEN P78 := FALSE",
                "sensing_vars": ["T41"], "configuration_vars": [],
                "actuation_var": "P78", "condition": "T41 > 8",
                "action": "P78 := FALSE",
            },
        ]),
    ]


def _gfsm():
    return {
        "initial": "P1.S:0|P2.T:0",
        "states": {
            "P1.S:0|P2.T:0": ["0", "0"],
            "P1.S:1|P2.T:0": ["1", "0"],
            "P1.S:0|P2.T:1": ["0", "1"],
            "P1.S:1|P2.T:1": ["1", "1"],
        },
        "transitions": [],
        "metadata": {"source_file": "synthetic.st", "extraction_date": "",
                     "total_states": 4, "total_transitions": 0},
        "max_states": 100,
    }


def _training_frame() -> pd.DataFrame:
    rows = [
        # state (0, 0): T41 max 4, action P78=TRUE
        (0, 0, 1.0, 1.0, 1.0, 0.0),
        (0, 0, 4.0, 4.0, 1.0, 0.0),
        # state (1, 0): T41 min 9, action P78=FALSE
        (1, 0, 9.0, 4.0, 0.0, 0.0),
        (1, 0, 10.0, 4.5, 0.0, 0.0),
        # state (0, 1)
        (0, 1, 4.0, 9.0, 1.0, 1.0),
        (0, 1, 4.5, 10.0, 1.0, 1.0),
        # state (1, 1)
        (1, 1, 9.0, 9.0, 0.0, 1.0),
        (1, 1, 10.0, 10.0, 0.0, 1.0),
    ]
    return pd.DataFrame(
        rows, columns=["p1", "p2", "t41", "t42", "p78", "p79"]
    )


def _model(**overrides):
    kwargs = dict(
        topology="synthetic",
        gfsm_json=_gfsm(),
        fsm_documents=_fsm_documents(),
        components=COMPONENTS,
        fb_to_col=FB_TO_COL,
        column_map=COLUMN_MAP,
        training_frames=[_training_frame()],
        invariant_documents=_invariant_documents(),
        min_observations=1,
    )
    kwargs.update(overrides)
    return mine_native_sain(**kwargs)


def _frame(rows, columns=("p1", "p2", "t41", "t42", "p78", "p79")):
    return pd.DataFrame(rows, columns=list(columns))


def test_parse_single_and_multi_variable_conditions():
    single = parse_transition_condition("T41 < 5")
    assert len(single) == 1
    assert single[0].variable == "T41"
    assert single[0].operator == "<"
    assert single[0].literal == 5.0

    multi = parse_transition_condition("(T41 < 5) AND (T42 > 8)")
    assert [(c.variable, c.operator, c.literal) for c in multi] == [
        ("T41", "<", 5.0), ("T42", ">", 8.0),
    ]


def test_unsupported_guards_fail_explicitly():
    with pytest.raises(SainTemplateError, match="unsupported OR"):
        parse_transition_condition("T41 < 5 OR T42 < 5")
    with pytest.raises(SainTemplateError, match="right-hand side"):
        parse_transition_condition("T41 < T42")
    with pytest.raises(
        SainTemplateError, match="unsupported token|comparison operator"
    ):
        parse_transition_condition("T41 + 1 < 5")


def test_outgoing_fsm_guards_are_inter_state_not_intra_state():
    specs, _ = derive_source_templates(_fsm_documents(), COMPONENTS)
    assert specs
    assert {spec.kind for spec in specs} == {"inter"}
    assert not any(spec.kind == "intra_actuation" for spec in specs)

    with_pdg, _ = derive_source_templates(
        _fsm_documents(), COMPONENTS, _invariant_documents()
    )
    intra = [spec for spec in with_pdg if spec.kind == "intra_actuation"]
    assert {spec.source_invariant_type for spec in intra} == {"single", "multi"}
    assert all(spec.from_local in {"0", "1"} for spec in intra)


def test_intra_quantification_is_per_state_and_labels_adaptation():
    model = _model()
    state0 = model["states"]["P1.S:0|P2.T:0"]
    state1 = model["states"]["P1.S:1|P2.T:0"]
    single0 = [t for t in state0["intra"]
               if t["source_invariant_type"] == "single"][0]
    multi1 = [t for t in state1["intra"]
              if t["source_invariant_type"] == "multi"][0]

    upper = single0["atoms"][0]
    assert upper["operator"] == "<=" and upper["bound"] == 4.0
    assert upper["bound_origin"] == "empirical_state_subtrace"
    assert single0["actuation"] == {
        "variable": "P78", "value": True, "column": "p78", "status": "ok",
    }

    source_literal = multi1["atoms"][0]
    assert source_literal["operator"] == ">" and source_literal["bound"] == 8.0
    assert source_literal["bound_origin"] == "source_literal"

    assert model["status"] == SAIN_NATIVE_STATUS == "partial_adapted"
    assert model["quantification"]["implementation"] == (
        "project_adaptation_not_quantminer"
    )
    assert model["quantification"]["quantminer_available"] is False
    json.dumps(model)


def test_staying_in_state_with_false_exit_guard_is_not_a_violation():
    detector = SainNativeDetector(_model())
    frame = _frame([
        (0, 0, 7.0, 1.0, 1.0, 0.0),
        (0, 0, 8.0, 1.0, 1.0, 0.0),
        (0, 0, 9.0, 1.0, 1.0, 0.0),
    ])
    result = detector.predict(frame)
    assert result.flags.tolist() == [0, 0, 0]


def test_conditional_actuation_implication_semantics():
    detector = SainNativeDetector(_model())
    # P1 state 0: IF T41 <= 4 (empirical) THEN P78 = TRUE.
    state0 = _frame([
        (0, 0, 4.0, 1.0, 1.0, 0.0),   # antecedent true, action true
        (0, 0, 4.0, 1.0, 0.0, 0.0),   # antecedent true, action false
        (0, 0, 9.0, 1.0, 0.0, 0.0),   # antecedent false, action false
    ])
    assert detector.predict(state0).flags.tolist() == [0, 1, 0]

    # P1 state 1: IF T41 > 8 THEN P78 = FALSE.
    state1 = _frame([
        (1, 0, 9.0, 1.0, 0.0, 0.0),   # antecedent true, action true
        (1, 0, 9.0, 1.0, 1.0, 0.0),   # antecedent true, action false
        (1, 0, 5.0, 1.0, 1.0, 0.0),   # antecedent false, action true
    ])
    assert detector.predict(state1).flags.tolist() == [0, 1, 0]


def test_unconditional_source_action_is_enforced():
    docs = [("P1", "S", [{
        "id": "state_0_P79", "type": "single", "state_id": "0",
        "variables": ["P79"], "confidence": 1.0,
        "structure": "In State 0, P79 = FALSE",
        "sensing_var": "STATE", "sensing_var_type": "internal",
        "actuation_var": "P79", "operator": "=",
        "actuation_value": "FALSE",
    }])]
    detector = SainNativeDetector(_model(invariant_documents=docs))
    frame = _frame([
        (0, 0, 1.0, 1.0, 1.0, 0.0),
        (0, 0, 1.0, 1.0, 1.0, 1.0),
    ])
    assert detector.predict(frame).flags.tolist() == [0, 1]


def test_genuine_illegal_transition_is_detected():
    detector = SainNativeDetector(_model())

    legal = _frame([
        (0, 0, 4.0, 1.0, 1.0, 0.0),
        (1, 0, 9.0, 1.0, 0.0, 0.0),
    ])
    assert detector.predict(legal).flags.tolist() == [0, 0]

    illegal = _frame([
        (0, 0, 7.0, 1.0, 1.0, 0.0),   # exit guard T41 < 5 is false
        (1, 0, 9.0, 1.0, 0.0, 0.0),
    ])
    result = detector.predict(illegal)
    assert result.flags.tolist() == [0, 1]


def test_transition_to_unsourced_target_is_illegal():
    detector = SainNativeDetector(_model(
        fsm_documents=_fsm_documents(with_p2_reverse=False)
    ))
    frame = _frame([
        (0, 1, 4.0, 9.0, 1.0, 1.0),
        (0, 0, 4.0, 4.0, 1.0, 0.0),
    ])
    result = detector.predict(frame)
    assert result.flags.tolist() == [0, 1]


def test_no_check_transition_is_unconditional():
    documents = _fsm_documents()
    documents[1][1]["function_blocks"][0]["transitions"][0]["condition"] = (
        "No Check"
    )
    detector = SainNativeDetector(_model(fsm_documents=documents))
    frame = _frame([
        (0, 0, 4.0, 9.0, 1.0, 0.0),
        (0, 1, 4.0, 9.0, 1.0, 1.0),
    ])
    assert detector.predict(frame).flags.tolist() == [0, 0]


def test_unsupported_source_expressions_fail_at_derivation():
    arithmetic = [("P1", "S", [{
        "id": "bad", "type": "multi", "state_id": "0",
        "variables": ["T41", "P78"], "actuation_var": "P78",
        "condition": "T41 + 1 < 5", "action": "P78 := TRUE",
    }])]
    with pytest.raises(
        SainTemplateError, match="unsupported token|comparison operator"
    ):
        derive_source_templates(_fsm_documents(), COMPONENTS, arithmetic)

    expression_action = [("P1", "S", [{
        "id": "bad_action", "type": "multi", "state_id": "0",
        "variables": ["T41", "P78"], "actuation_var": "P78",
        "condition": "T41 < 5", "action": "P78 := (S = 1)",
    }])]
    with pytest.raises(SainTemplateError, match="unsupported actuation value"):
        derive_source_templates(
            _fsm_documents(), COMPONENTS, expression_action
        )


def test_multi_case_invariants_are_loaded_for_their_own_selector(tmp_path):
    (tmp_path / "test_analysis_manifest.json").write_text(json.dumps({
        "plcs": [{
            "name": "PLC1",
            "artifacts": {"invariants_json": "plc1.invariants.json"},
        }],
    }))
    (tmp_path / "plc1.invariants.json").write_text(json.dumps([
        {"id": "p1", "case_var": "P1_State"},
        {"id": "p2", "case_var": "P2_State"},
    ]))
    fsm_documents = [("PLC1", {"function_blocks": [
        {"case_variable": "P1_State"},
        {"case_variable": "P2_State"},
    ]})]
    documents = load_invariant_documents(tmp_path, "test", fsm_documents)
    assert [(case_var, [inv["id"] for inv in invariants])
            for _, case_var, invariants in documents] == [
        ("P1_State", ["p1"]),
        ("P2_State", ["p2"]),
    ]


def test_missing_pdg_invariants_do_not_flag_intra_rows():
    model = _model(invariant_documents=())
    assert all(
        entry["intra_source"] == "no_pdg_invariants"
        for entry in model["states"].values()
    )
    assert model["templates"]["intra_templates"] == 0
    detector = SainNativeDetector(model)
    frame = _frame([(0, 0, 7.0, 1.0, 0.0, 0.0)])
    assert detector.predict(frame).flags.tolist() == [0]


def test_missing_source_column_is_unquantified_not_silently_dropped():
    model = _model(column_map={"T41": "t41", "T42": "t42", "P79": "p79"})
    state0 = model["states"]["P1.S:0|P2.T:0"]
    assert state0["status"] == "unquantified"
    detector = SainNativeDetector(model)
    frame = _frame([(0, 0, 1.0, 1.0, 1.0, 0.0)])
    assert detector.predict(frame).flags.tolist() == [1]


def test_unknown_state_is_flagged():
    detector = SainNativeDetector(_model())
    frame = _frame([(2, 0, 1.0, 1.0, 1.0, 0.0)])
    assert detector.predict(frame).flags.tolist() == [1]


def test_artifact_records_partial_status_and_exact_gaps():
    model = _model()
    mode = model["operational_mode"]
    assert mode["mode"] == "offline_passive_detection"
    assert mode["write_interception_supported"] is False
    assert mode["write_denial_supported"] is False
    assert mode["smonitor_runtime_implemented"] is False
    assert model["schema"] == SAIN_NATIVE_SCHEMA
    assert model["fully_faithful_sain"] is False
    assert model["native_quantminer_reproduction"] is False
    assert any("QuantMiner" in gap for gap in SAIN_NATIVE_GAPS)
    assert any("SMONITOR" in gap for gap in SAIN_NATIVE_GAPS)
    assert model["gaps"] == list(SAIN_NATIVE_GAPS)
    adaptations = " ".join(model["adaptations"]).lower()
    assert "direct implementation" not in adaptations


def test_partial_path_is_not_the_niaarm_local_adaptation():
    model = _model()
    assert model["baseline_role"] == "partial_adapted_source_derived_sain"
    assert model["not_the_niaarm_local_adaptation"] is True
    assert SainLocalDetector.name == "sain_local"
    assert SainNativeDetector.name == "sain_native"
