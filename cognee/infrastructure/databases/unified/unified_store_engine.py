from __future__ import annotations

from typing import Optional, cast

from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface

from .capabilities import EngineCapability
from .graph_vector_store_interface import GraphVectorStoreInterface
from .provenance_delete_planner import SourceRefRemovalResult, execute_source_ref_removal


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

    def supports_graph_provenance_delete(self) -> bool:
        """Return True when this engine can perform graph-provenance delete/rollback.

        Requires both GRAPH and VECTOR capabilities and present engines. Routing
        additionally checks ``stores_provenance_in_graph`` on the graph; unsupported
        provenance reads raise ``UnsupportedProvenanceCapability`` from the
        adapter (there is no separate provenance-capability flag).
        """
        return (
            self.has_capability(EngineCapability.GRAPH)
            and self.has_capability(EngineCapability.VECTOR)
            and self._graph is not None
            and self._vector is not None
        )

    async def delete_by_source_ref(self, source_ref_key: str) -> "SourceRefRemovalResult":
        """Delete artifacts owned only by the given source ref; detach the rest.

        Returns the hard-deleted node/edge identities so callers can invalidate
        derived caches (e.g. session entries that used the deleted elements).
        """
        if not self.supports_graph_provenance_delete():
            raise UnsupportedProvenanceCapability()
        graph = self.graph
        vector = self.vector

        node_ids = await graph.find_nodes_by_source_ref(source_ref_key)
        edges = await graph.find_edges_by_source_ref(source_ref_key)

        node_data = await graph.get_node_delete_data(node_ids)
        edge_data = await graph.get_edge_delete_data(edges)

        refs_by_node = {node_id: [source_ref_key] for node_id in node_data}
        refs_by_edge = {edge: [source_ref_key] for edge in edge_data}

        return await execute_source_ref_removal(
            graph,
            vector,
            node_data=node_data,
            edge_data=edge_data,
            refs_by_node=refs_by_node,
            refs_by_edge=refs_by_edge,
        )

    async def delete_by_source_refs(self, source_ref_keys) -> "SourceRefRemovalResult":
        """Remove MANY source refs in one planner pass (chunk-level updates).

        Reads the dataset-independent ref maps once through the per-ref finders
        and hands the planner every retired key together, so an artifact owned
        by several retired chunks is detached or deleted in a single decision
        and the post-delete cleanup (orphaned EdgeTypes, NodeSet tags) runs
        once per call instead of once per chunk.
        """
        if not self.supports_graph_provenance_delete():
            raise UnsupportedProvenanceCapability()
        graph = self.graph
        vector = self.vector

        keys = list(dict.fromkeys(source_ref_keys))
        refs_by_node: dict = {}
        refs_by_edge: dict = {}
        for key in keys:
            for node_id in await graph.find_nodes_by_source_ref(key):
                refs_by_node.setdefault(node_id, []).append(key)
            for edge in await graph.find_edges_by_source_ref(key):
                refs_by_edge.setdefault(edge, []).append(key)

        node_data = await graph.get_node_delete_data(list(refs_by_node.keys()))
        edge_data = await graph.get_edge_delete_data(list(refs_by_edge.keys()))

        return await execute_source_ref_removal(
            graph,
            vector,
            node_data=node_data,
            edge_data=edge_data,
            refs_by_node=refs_by_node,
            refs_by_edge=refs_by_edge,
        )

    async def delete_by_document(self, dataset_id: str, data_id: str) -> "SourceRefRemovalResult":
        """Remove EVERY ref a document owns — v1 doc-scope AND v2 chunk-scope.

        Chunk-scoped ownership (source_ref:v2) means a document's artifacts
        may carry only their producing chunk's ref; deleting by the v1 key
        alone would strand them. The dataset's ref maps are filtered to refs
        whose data id is this document — any version — and the planner
        deletes what is left unowned, detaching shared output.
        """
        if not self.supports_graph_provenance_delete():
            raise UnsupportedProvenanceCapability()
        from cognee.infrastructure.databases.provenance import parse_source_ref_key

        graph = self.graph
        vector = self.vector

        def _document_refs(refs) -> list:
            selected = []
            for ref in refs:
                try:
                    parsed = parse_source_ref_key(ref)
                except ValueError:
                    continue
                if str(parsed.data_id) == str(data_id):
                    selected.append(ref)
            return selected

        refs_by_node = {
            node_id: document_refs
            for node_id, refs in (await graph.find_node_source_refs_by_dataset(dataset_id)).items()
            if (document_refs := _document_refs(refs))
        }
        refs_by_edge = {
            edge: document_refs
            for edge, refs in (await graph.find_edge_source_refs_by_dataset(dataset_id)).items()
            if (document_refs := _document_refs(refs))
        }

        node_data = await graph.get_node_delete_data(list(refs_by_node.keys()))
        edge_data = await graph.get_edge_delete_data(list(refs_by_edge.keys()))

        return await execute_source_ref_removal(
            graph,
            vector,
            node_data=node_data,
            edge_data=edge_data,
            refs_by_node=refs_by_node,
            refs_by_edge=refs_by_edge,
        )

    async def delete_by_dataset_id(self, dataset_id: str) -> None:
        """Remove the dataset's source refs; delete artifacts left unowned."""
        if not self.supports_graph_provenance_delete():
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
        if not self.supports_graph_provenance_delete():
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
