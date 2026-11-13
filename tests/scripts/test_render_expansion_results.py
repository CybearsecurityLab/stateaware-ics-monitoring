import json

from scripts import render_expansion_results as render


def test_baseline_bolding_is_per_topology_and_includes_ties(monkeypatch):
    def fake_baseline(method, topology):
        if method == "imdisco":
            return {"status": "unavailable"}
        values = {"mcc": 0.0, "balanced_accuracy": 0.0, "aucpr": 0.0}
        if topology == "anytown" and method in {"cusum", "sicams"}:
            values["mcc"] = 0.9
        if topology == "ctown" and method == "sicams":
            values["balanced_accuracy"] = 0.8
        if topology == "ltown" and method == "conv":
            values["aucpr"] = 0.7
        return {"status": "ok", "metrics": values}

    monkeypatch.setattr(render, "baseline_one", fake_baseline)
    lines = render.baselines().splitlines()
    cusum = next(line for line in lines if line.startswith("CUSUM"))
    sicams = next(line for line in lines if line.startswith("SICAMS"))
    conv = next(line for line in lines if line.startswith("CAE"))
    assert "\\textbf{0.900}" in cusum
    assert "\\textbf{0.900}" in sicams
    assert "\\textbf{0.800}" in sicams
    assert "\\textbf{0.700}" in conv
    assert "\\cite{ghaeini2018state}" in cusum
    assert "\\cite{sicams2025}" in sicams
    assert "\\cite{lee2024advancing}" in conv
    framework = next(line for line in lines if line.startswith("Ours"))
    assert "\\cite" not in framework
    assert "\\shortstack" not in cusum
    assert not any(line.startswith("IM-DISCO") for line in lines)
    assert any(line.startswith("IMD-max") for line in lines)
    assert not any(line.startswith("O+O") for line in lines)
    assert not any(line.startswith("ctown") or line.startswith("ltown") for line in render.tolerance().splitlines())
    assert "Ref &" not in render.ablation()


def test_historical_quantminer_failure_is_pending_for_corrected_run(
    tmp_path, monkeypatch,
):
    output = tmp_path / "scripts/out/full_signal_sain_local_quantminer/ctown"
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(json.dumps({
        "status": "mining_errors",
        "config": {"quantminer": {"max_attributes": 3}},
    }))
    (output / "predictions.csv").write_text("y_true,y_pred,score\n0,0,0\n")
    monkeypatch.setattr(render, "ROOT", tmp_path)

    assert render.baseline_one("sain_local", "ctown") == {"status": "pending"}


def test_current_quantminer_failure_is_not_relabelled(tmp_path, monkeypatch):
    output = tmp_path / "scripts/out/full_signal_sain_local_quantminer_bitset/ctown"
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(json.dumps({
        "status": "mining_errors",
        "config": {"quantminer": {
            "max_attributes": render.SAIN_QUANTMINER_CONFIG.max_attributes,
        }},
    }))
    monkeypatch.setattr(render, "ROOT", tmp_path)

    assert render.baseline_one("sain_local", "ctown") == {"status": "mining_errors"}
