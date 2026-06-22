"""Graph-native delete/rollback orchestration interface (Part 0 contract).

``GraphVectorStoreInterface`` is the high-level surface that ``delete`` and
``rollback`` will call once they are graph-native (Part 2). It spans both the
graph and vector stores — removing a node means deleting it from the graph and
its embeddings from the vector collections — which is why it lives next to
``UnifiedStoreEngine`` rather than on either single-store interface.

Part 0 ships the interface only. Every method has a default body that raises
``UnsupportedProvenanceCapability``; the methods are deliberately NOT
``@abstractmethod`` so the interface can be mixed into existing engines (and
subclassed by test fakes) without forcing an implementation before Part 1/2.
No live delete, rollback, add, search, or retrieval path calls these yet.
"""

from __future__ import annotations

from uuid import UUID

from cognee.modules.graph.provenance.exceptions import UnsupportedProvenanceCapability
from cognee.modules.graph.provenance.results import ProvenanceDeleteResult


class GraphVectorStoreInterface:
    """Provenance-based deletion across the graph and vector stores.

    Implementations read the provenance stamped on graph artifacts (see
    ``cognee.modules.graph.provenance``) to find what to remove, delete it from
    both stores, and strip the targeted ref — hard-deleting an artifact only
    once its last source ref / source-run ref is gone.
    """

    def supports_graph_native_delete(self) -> bool:
        """Whether this store can serve graph-native delete/rollback.

        Defaults to ``False``; real implementations override to ``True``. Lets
        ``delete``/``rollback`` decide between the graph-native path and the
        relational-ledger fallback without catching
        ``UnsupportedProvenanceCapability``.

        Distinct from ``GraphDBInterface.supports_graph_native_provenance``:
        that flag covers only the graph-side read primitives; this one covers
        full graph+vector delete/rollback orchestration and so requires both the
        graph reads and the vector-side deletes to be in place.
        """
        return False

    async def delete_by_source_ref(self, source_ref: str) -> ProvenanceDeleteResult:
        """Delete the artifacts owned by one ingestion source.

        Removes ``source_ref`` from every node/edge carrying it and hard-deletes
        those left with no remaining source ref. Graph-native equivalent of
        deleting a single data item from a dataset.

        Parameters:
        -----------
            - source_ref (str): A source ref (see provenance.make_source_ref).
        """
        raise UnsupportedProvenanceCapability("delete_by_source_ref")

    async def delete_by_dataset_id(self, dataset_id: UUID) -> ProvenanceDeleteResult:
        """Delete the artifacts belonging to one dataset.

        Graph-native equivalent of deleting a whole dataset's graph/vector
        contents.

        Parameters:
        -----------
            - dataset_id (UUID): The dataset to delete.
        """
        raise UnsupportedProvenanceCapability("delete_by_dataset_id")

    async def rollback_by_pipeline_run_id(
        self, pipeline_run_id: UUID, dataset_id: UUID
    ) -> ProvenanceDeleteResult:
        """Roll back the artifacts a single pipeline run introduced.

        Removes the run's source-run ref (derived from ``dataset_id`` and
        ``pipeline_run_id``) from every artifact it touched and hard-deletes
        those it solely created. Graph-native equivalent of
        ``cognify_rollback_handler``.

        Parameters:
        -----------
            - pipeline_run_id (UUID): The run to roll back.
            - dataset_id (UUID): The dataset the run wrote to; combined with the
              run id to form the source-run ref.
        """
        raise UnsupportedProvenanceCapability("rollback_by_pipeline_run_id")


__all__ = ["GraphVectorStoreInterface"]
