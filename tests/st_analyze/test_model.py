import pytest

from st_analyze.model import AnalyzeResult, StAnalyzeError


def test_st_analyze_error_is_exception():
    with pytest.raises(StAnalyzeError, match="boom"):
        raise StAnalyzeError("boom")


def test_analyze_result_defaults():
    r = AnalyzeResult(
        ast_xml="<x/>",
        invariants=[],
        pdg_dot="digraph {}",
        pdg_structured={},
    )
    assert r.ok is True
    assert r.errors == []
    assert r.programs == []
    assert r.pdg_components == []
