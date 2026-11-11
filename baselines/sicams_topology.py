"""INP-backed graph and sensor mappings for SICAMS prelocalization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Mapping, Sequence

import wntr


@dataclass(frozen=True)
class SicamsTopologyInputs:
    adjacency: Mapping[str, tuple[str, ...]]
    sensor_nodes: Mapping[str, str]
    policy: str
    inp_path: Path
    inp_sha256: str
    node_count: int
    link_count: int
    mapped_feature_count: int
    unmapped_features: tuple[str, ...]

    def to_dict(self, *, include_graph: bool = True) -> dict:
        result = {
            "policy": self.policy,
            "inp_path": str(self.inp_path),
            "inp_sha256": self.inp_sha256,
            "node_count": self.node_count,
            "link_count": self.link_count,
            "mapped_feature_count": self.mapped_feature_count,
            "unmapped_features": list(self.unmapped_features),
            "graph_links": "all EPANET pipes, pumps, and valves; undirected",
            "sensor_nodes": dict(sorted(self.sensor_nodes.items())),
        }
        if include_graph:
            result["adjacency"] = {
                node: list(neighbours)
                for node, neighbours in sorted(self.adjacency.items())
            }
        return result


def build_sicams_topology_inputs(
    inp_path: str | Path,
    feature_names: Sequence[str],
    *,
    support_nodes: Collection[str] | None = None,
    policy: str = "all_observed_node_channels",
) -> SicamsTopologyInputs:
    """Map observed node channels onto an undirected EPANET topology.

    Only exact feature/node identifier matches become fixed Laplacian support
    values. Link flow, actuator status, and other non-node channels remain
    unmapped. ``support_nodes`` optionally restricts those matches to a
    documented sparse sensor inventory.
    """
    path = Path(inp_path).resolve()
    network = wntr.network.WaterNetworkModel(str(path))
    graph = network.to_graph().to_undirected()
    adjacency = {
        str(node): tuple(sorted(str(neighbour) for neighbour in graph.neighbors(node)))
        for node in sorted(graph.nodes, key=str)
    }
    nodes = set(adjacency)
    allowed = nodes if support_nodes is None else nodes & set(support_nodes)
    names = [str(name) for name in feature_names]
    sensor_nodes = {name: name for name in names if name in allowed}
    unmapped = tuple(name for name in names if name not in sensor_nodes)
    return SicamsTopologyInputs(
        adjacency=adjacency,
        sensor_nodes=sensor_nodes,
        policy=policy,
        inp_path=path,
        inp_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        node_count=len(adjacency),
        link_count=int(graph.number_of_edges()),
        mapped_feature_count=len(sensor_nodes),
        unmapped_features=unmapped,
    )
