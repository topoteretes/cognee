"""Scope a graph read to one dataset when all datasets share one graph.

With backend access control on, every dataset has its own graph database, and a
read inside the dataset's database context sees only that dataset. With it off,
every dataset writes into one shared graph and that context is a no-op, so a
dataset's view has to be cut down to the nodes and edges the dataset owns.
"""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select

from cognee.context_global_variables import backend_access_control_enabled
from cognee.infrastructure.databases.graph.graph_db_interface import EdgeData, Node
from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.graph.models import Node as NodeRow

EdgeKey = tuple[str, str, str]


@dataclass(frozen=True)
class DatasetScope:
    """The nodes and edges one dataset owns in a shared graph.

    ``edge_keys`` is None when only node ownership is known (graphs that keep
    ownership in the relational ledger, whose relationship names do not always
    match the graph's); an edge is then kept when both its endpoints are.
    """

    node_ids: frozenset[str]
    edge_keys: frozenset[EdgeKey] | None

    def read_order(self, seeds: list[str], max_nodes: int) -> list[str]:
        """The dataset's own seeds first, then its other nodes, at most ``max_nodes``.

        A bounded read admits its seeds before any neighbour, so with these as
        seeds a dataset of ``max_nodes`` or more fills the whole budget with
        its own nodes.
        """
        own_seeds = [seed for seed in seeds if seed in self.node_ids]
        rest = sorted(self.node_ids.difference(own_seeds))
        return (own_seeds + rest)[:max_nodes]

    def keep(self, nodes: list[Node], edges: list[EdgeData]) -> tuple[list[Node], list[EdgeData]]:
        """Drop the nodes and edges another dataset owns."""
        kept_nodes = [node for node in nodes if str(node[0]) in self.node_ids]
        kept_edges = [
            edge
            for edge in edges
            if str(edge[0]) in self.node_ids
            and str(edge[1]) in self.node_ids
            and (self.edge_keys is None or (str(edge[0]), str(edge[1]), edge[2]) in self.edge_keys)
        ]
        return kept_nodes, kept_edges


async def get_shared_graph_scope(graph_engine, dataset_id: UUID) -> DatasetScope | None:
    """The scope a read of ``dataset_id`` needs, or None when its graph is its own."""
    if backend_access_control_enabled():
        return None
    return await get_dataset_scope(graph_engine, dataset_id)


async def get_dataset_scope(graph_engine, dataset_id: UUID) -> DatasetScope:
    """What ``dataset_id`` owns in the shared graph, from wherever ownership is kept."""
    # Community registration permits duck-typed adapters, which may not know provenance.
    if hasattr(graph_engine, "get_graph_metadata") and await stores_provenance_in_graph(
        graph_engine
    ):
        nodes = await graph_engine.find_node_source_refs_by_dataset(str(dataset_id))
        edges = await graph_engine.find_edge_source_refs_by_dataset(str(dataset_id))
        return DatasetScope(
            node_ids=frozenset(str(node_id) for node_id in nodes),
            edge_keys=frozenset(
                (str(edge.source_id), str(edge.target_id), edge.relationship_name) for edge in edges
            ),
        )

    # Graphs written before provenance moved into the graph keep it in the ledger.
    async with get_relational_engine().get_async_session() as session:
        slugs = await session.scalars(select(NodeRow.slug).where(NodeRow.dataset_id == dataset_id))
        return DatasetScope(node_ids=frozenset(str(slug) for slug in slugs), edge_keys=None)
