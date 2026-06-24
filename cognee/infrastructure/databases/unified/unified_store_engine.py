from __future__ import annotations

from typing import Optional, cast

from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface

from .capabilities import EngineCapability
from .graph_vector_store_interface import GraphVectorStoreInterface
from .provenance_delete_planner import execute_source_ref_removal


class UnifiedStoreEngine(GraphVectorStoreInterface):
    """Facade that wraps graph and vector engines with capability flags.

    For separate backends (e.g. Ladybug + LanceDB), holds two independent engine
    instances.  For hybrid backends (e.g. Neptune Analytics), both properties
    point to the same adapter object.

    The pipeline can check ``has_capability()`` to decide whether to optimise
    writes or searches for hybrid backends.
    """

    def __init__(
        self,
        graph_engine: Optional[GraphDBInterface] = None,
        vector_engine: Optional[VectorDBInterface] = None,
        capabilities: EngineCapability = EngineCapability.NONE,
    ):
        self._graph = graph_engine
        self._vector = vector_engine
        self._capabilities = capabilities

    @property
    def capabilities(self) -> EngineCapability:
        return self._capabilities

    def has_capability(self, cap: EngineCapability) -> bool:
        return bool(self._capabilities & cap)

    @property
    def graph(self) -> GraphDBInterface:
        if not self.has_capability(EngineCapability.GRAPH) or self._graph is None:
            raise RuntimeError(
                "This UnifiedStoreEngine has no GRAPH capability. "
                "Check has_capability(EngineCapability.GRAPH) before accessing .graph"
            )
        return self._graph

    @property
    def vector(self) -> VectorDBInterface:
        if not self.has_capability(EngineCapability.VECTOR) or self._vector is None:
            raise RuntimeError(
                "This UnifiedStoreEngine has no VECTOR capability. "
                "Check has_capability(EngineCapability.VECTOR) before accessing .vector"
            )
        return cast(VectorDBInterface, self._vector)

    @property
    def is_hybrid(self) -> bool:
        return self.has_capability(EngineCapability.HYBRID_WRITE) or self.has_capability(
            EngineCapability.HYBRID_SEARCH
        )

    @property
    def is_same_backend(self) -> bool:
        return self._graph is not None and self._graph is self._vector

    def supports_graph_native_delete(self) -> bool:
        """Return True when this engine can perform graph-native delete/rollback.

        Requires both GRAPH and VECTOR capabilities and present engines. Routing
        additionally checks ``is_graph_native_graph`` on the graph; unsupported
        provenance reads raise ``UnsupportedProvenanceCapability`` from the
        adapter (there is no separate provenance-capability flag).
        """
        return (
            self.has_capability(EngineCapability.GRAPH)
            and self.has_capability(EngineCapability.VECTOR)
            and self._graph is not None
            and self._vector is not None
        )

    async def delete_by_source_ref(self, source_ref_key: str) -> None:
        """Delete artifacts owned only by the given source ref; detach the rest."""
        if not self.supports_graph_native_delete():
            raise UnsupportedProvenanceCapability()
        graph = self.graph
        vector = self.vector

        node_ids = await graph.find_nodes_by_source_ref(source_ref_key)
        edges = await graph.find_edges_by_source_ref(source_ref_key)

        node_data = await graph.get_node_delete_data(node_ids)
        edge_data = await graph.get_edge_delete_data(edges)

        refs_by_node = {node_id: [source_ref_key] for node_id in node_data}
        refs_by_edge = {edge: [source_ref_key] for edge in edge_data}

        await execute_source_ref_removal(
            graph,
            vector,
            node_data=node_data,
            edge_data=edge_data,
            refs_by_node=refs_by_node,
            refs_by_edge=refs_by_edge,
        )

    async def delete_by_dataset_id(self, dataset_id: str) -> None:
        """Remove the dataset's source refs; delete artifacts left unowned."""
        if not self.supports_graph_native_delete():
            raise UnsupportedProvenanceCapability()
        graph = self.graph
        vector = self.vector

        refs_by_node = await graph.find_node_source_refs_by_dataset(dataset_id)
        refs_by_edge = await graph.find_edge_source_refs_by_dataset(dataset_id)

        node_data = await graph.get_node_delete_data(list(refs_by_node.keys()))
        edge_data = await graph.get_edge_delete_data(list(refs_by_edge.keys()))

        await execute_source_ref_removal(
            graph,
            vector,
            node_data=node_data,
            edge_data=edge_data,
            refs_by_node=refs_by_node,
            refs_by_edge=refs_by_edge,
        )

    async def rollback_by_pipeline_run_id(self, pipeline_run_id: str) -> None:
        """Remove the refs a run attached; delete artifacts left unowned."""
        if not self.supports_graph_native_delete():
            raise UnsupportedProvenanceCapability()
        graph = self.graph
        vector = self.vector

        refs_by_node = await graph.find_node_source_refs_by_pipeline_run(pipeline_run_id)
        refs_by_edge = await graph.find_edge_source_refs_by_pipeline_run(pipeline_run_id)

        node_data = await graph.get_node_delete_data(list(refs_by_node.keys()))
        edge_data = await graph.get_edge_delete_data(list(refs_by_edge.keys()))

        await execute_source_ref_removal(
            graph,
            vector,
            node_data=node_data,
            edge_data=edge_data,
            refs_by_node=refs_by_node,
            refs_by_edge=refs_by_edge,
        )
