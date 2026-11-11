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
    cusum = next(line for line in lines if line.startswith("State-aware CUSUM"))
    sicams = next(line for line in lines if line.startswith("SICAMS"))
    conv = next(line for line in lines if line.startswith("ConvBiLSTM-AE"))
    assert "MCC \\textbf{0.9000}" in cusum
    assert "MCC \\textbf{0.9000}" in sicams
    assert "BA \\textbf{0.8000}" in sicams
    assert "AUPRC \\textbf{0.7000}" in conv
