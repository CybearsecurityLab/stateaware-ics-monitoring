from pathlib import Path

from baselines.sicams_topology import build_sicams_topology_inputs


FIXTURE = Path(__file__).parents[1] / "hydraulics/fixtures/synthetic.inp"


def test_inp_adapter_maps_only_observed_node_channels_and_all_link_types():
    result = build_sicams_topology_inputs(
        FIXTURE,
        ["J1", "J2", "T1", "R1", "f_P1", "s_PU1"],
    )
    assert result.sensor_nodes == {
        "J1": "J1", "J2": "J2", "T1": "T1", "R1": "R1"
    }
    assert set(result.adjacency["R1"]) == {"J1", "J3"}
    assert result.mapped_feature_count == 4
    assert result.unmapped_features == ("f_P1", "s_PU1")
    assert len(result.inp_sha256) == 64


def test_inp_adapter_supports_documented_sparse_sensor_inventory():
    result = build_sicams_topology_inputs(
        FIXTURE,
        ["J1", "J2", "T1"],
        support_nodes={"J1"},
        policy="test_sparse_pressure_inventory",
    )
    assert result.sensor_nodes == {"J1": "J1"}
    assert result.policy == "test_sparse_pressure_inventory"
    assert result.unmapped_features == ("J2", "T1")
