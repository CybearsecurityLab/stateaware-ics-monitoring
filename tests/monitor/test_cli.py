from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("sklearn")

from dataio.manifest import build_dataset_manifest
from monitor.cli import _build_parser, main


def test_cli_parser_defaults_to_or():
    parser = _build_parser()
    args = parser.parse_args(["--topology", "anytown"])
    assert args.fusion == "or"


def test_cli_parser_ablation_options():
    parser = _build_parser()
    assert parser.parse_args(["--topology", "anytown", "--fusion", "or"]).fusion == "or"
    assert parser.parse_args(["--topology", "anytown", "--fusion", "intersection"]).fusion == "intersection"
    assert parser.parse_args(["--topology", "anytown", "--fusion", "conjunction"]).fusion == "conjunction"


def test_cli_parser_rejects_invalid_fusion():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--topology", "anytown", "--fusion", "invalid"])


def _seed(data_root: Path):
    ds = data_root / "ctown" / "dataset"
    (ds / "calibration").mkdir(parents=True)
    (ds / "evaluation").mkdir(parents=True)
    rng = np.random.default_rng(0)
    pd.DataFrame({"l_t1": rng.uniform(2.0, 3.0, 1200)}).to_csv(
        ds / "calibration" / "calibration.csv", index=False)
    pd.DataFrame({"l_t1": [2.5] * 6 + [999.0] * 6,
                  "label": [0] * 6 + [1] * 6}).to_csv(
        ds / "evaluation" / "test.csv", index=False)
    m = build_dataset_manifest(
        topology="ctown", source_name="synthetic", source_note="C-Town synthetic",
        fmt="csv", root=ds, calibration_files=["calibration/calibration.csv"],
        evaluation_files=["evaluation/test.csv"], column_map={}, attack_windows=[])
    (ds / "dataset_manifest.yaml").write_text(yaml.safe_dump(m, sort_keys=True))


def test_ics_monitor_runs_with_fusion_flag_if_data_present(tmp_path):
    pytest.importorskip("rtamt")
    repo = Path(__file__).resolve().parents[2]
    gfsm = repo / "data" / "generated" / "anytown" / "gfsm" / "anytown.gfsm.json"
    inv = repo / "data" / "generated" / "anytown" / "invariants" / "anytown_phi.json"
    ds = repo / "data" / "anytown" / "dataset" / "dataset_manifest.yaml"
    if not (gfsm.exists() and inv.exists() and ds.exists()):
        pytest.skip("anytown gfsm/Φ/dataset not populated")
    rc = main(["--topology", "anytown", "--data-root", str(repo / "data"),
               "--out", str(tmp_path / "m"), "--fusion", "intersection"])
    assert rc in (0, 1)
    assert (tmp_path / "m" / "anytown_monitor_manifest.json").exists()
