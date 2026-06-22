from __future__ import annotations

from typing import Optional, cast
from uuid import UUID

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface
from cognee.modules.graph.provenance.constants import (
    DATASET_IDS_KEY,
    SOURCE_REFS_KEY,
    SOURCE_RUN_REFS_KEY,
)
from cognee.modules.graph.provenance.refs import make_source_run_ref
from cognee.modules.graph.provenance.results import ProvenanceDeleteResult

from .capabilities import EngineCapability
from .graph_vector_store_interface import GraphVectorStoreInterface
from .provenance_delete_planner import execute_ref_removal


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

    # ------------------------------------------------------------------
    # Graph-native delete / rollback (GraphVectorStoreInterface, Part 2)
    # ------------------------------------------------------------------

    def supports_graph_native_delete(self) -> bool:
        """True only when this engine can read graph provenance and delete vectors.

        Requires GRAPH + VECTOR capabilities and a graph backend that implements
        the Part 0 provenance read primitives. Backends without those keep
        delete/rollback on the relational-ledger path.
        """
        if not (
            self.has_capability(EngineCapability.GRAPH)
            and self.has_capability(EngineCapability.VECTOR)
        ):
            return False
        if self._graph is None:
            return False
        return self._graph.supports_graph_native_provenance()

    async def delete_by_source_ref(self, source_ref: str) -> ProvenanceDeleteResult:
        nodes = await self.graph.get_nodes_delete_data_by_source_ref(source_ref)
        edges = await self.graph.get_edges_delete_data_by_source_ref(source_ref)

        # A node/edge survives if another source ref still owns it.
        def node_survives(node) -> bool:
            return bool(set(node.source_refs) - {source_ref})

        def edge_survives(edge) -> bool:
            return bool(set(edge.source_refs) - {source_ref})

        return await execute_ref_removal(
            self.graph,
            self.vector,
            nodes=nodes,
            edges=edges,
            property_key=SOURCE_REFS_KEY,
            refs_to_remove=[source_ref],
            node_survives=node_survives,
            edge_survives=edge_survives,
        )

    async def delete_by_dataset_id(self, dataset_id: UUID) -> ProvenanceDeleteResult:
        dataset_key = str(dataset_id)
        nodes = await self.graph.get_nodes_delete_data_by_dataset_id(dataset_id)
        edges = await self.graph.get_edges_delete_data_by_dataset_id(dataset_id)

        # An artifact survives if it still belongs to another dataset.
        def node_survives(node) -> bool:
            return bool(set(node.dataset_ids) - {dataset_key})

        def edge_survives(edge) -> bool:
            return bool(set(edge.dataset_ids) - {dataset_key})

        return await execute_ref_removal(
            self.graph,
            self.vector,
            nodes=nodes,
            edges=edges,
            property_key=DATASET_IDS_KEY,
            refs_to_remove=[dataset_key],
            node_survives=node_survives,
            edge_survives=edge_survives,
        )

    async def rollback_by_pipeline_run_id(
        self, pipeline_run_id: UUID, dataset_id: UUID
    ) -> ProvenanceDeleteResult:
        run_ref = make_source_run_ref(dataset_id, pipeline_run_id)
        nodes = await self.graph.get_nodes_delete_data_by_source_run_ref(run_ref)
        edges = await self.graph.get_edges_delete_data_by_source_run_ref(run_ref)

        # Hard-delete only what the run solely introduced: no surviving source
        # ref and no other source-run ref keeps the artifact alive.
        def node_survives(node) -> bool:
            return bool(node.source_refs) or bool(set(node.source_run_refs) - {run_ref})

        def edge_survives(edge) -> bool:
            return bool(edge.source_refs) or bool(set(edge.source_run_refs) - {run_ref})

        return await execute_ref_removal(
            self.graph,
            self.vector,
            nodes=nodes,
            edges=edges,
            property_key=SOURCE_RUN_REFS_KEY,
            refs_to_remove=[run_ref],
            node_survives=node_survives,
            edge_survives=edge_survives,
        )
