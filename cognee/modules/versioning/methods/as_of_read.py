"""Time-travel reads: filter the live store down to the runs completed by T.

**Semantics — forward-faithful filter, not full reconstruction.** An "as of T"
read returns the artifacts whose provenance shows they were attached by a run
completed at or before T *and that still exist in the live store*. Nothing is
copied or mutated. The boundary this implies: a destructive operation executed
*after* T (a forget, a rollback) shadows earlier state — the artifact is gone
from the live store, so an as-of-T read cannot return it. Reversible
operations recover the full view: undo the forget (or the rollback) and the
as-of read is exact again. Reconstructing past state *through* an un-undone
destructive op is exactly the replay cost Approach 1 trades away (see issue
#3650); it is documented and tested, not silently approximated.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID

from cognee.infrastructure.databases.provenance import EdgeIdentity
from cognee.infrastructure.databases.provenance.source_refs import (
    get_dataset_id_from_source_ref_key,
    get_pipeline_run_id_from_source_run_ref,
    get_source_ref_key_from_source_run_ref,
)
from cognee.modules.versioning.methods.timeline import (
    get_allowed_run_ids,
    resolve_as_of_time,
)


def _is_visible(
    source_run_refs: List[str], allowed_run_ids: Set[str], dataset_id: Optional[str]
) -> bool:
    """An artifact existed at T when some run completed by T attached it.

    Model A records a run ref only for the run that *first* attached each key,
    so "any run ref within the allowed set" is exactly "first attached at or
    before T". When ``dataset_id`` is given, only refs owned by that dataset
    count.
    """
    for run_ref in source_run_refs:
        if str(get_pipeline_run_id_from_source_run_ref(run_ref)) not in allowed_run_ids:
            continue
        if dataset_id is not None:
            key = get_source_ref_key_from_source_run_ref(run_ref)
            if str(get_dataset_id_from_source_ref_key(key)) != dataset_id:
                continue
        return True
    return False


async def get_visible_artifacts_as_of(
    graph_engine,
    dataset_id: UUID,
    as_of: Union[str, datetime],
) -> Tuple[Set[str], Set[EdgeIdentity]]:
    """Node ids and edge identities visible at ``as_of`` for the dataset."""
    as_of_time = await resolve_as_of_time(dataset_id, as_of)
    allowed = await get_allowed_run_ids(dataset_id, as_of_time)
    if not allowed:
        return set(), set()

    dataset_id_str = str(dataset_id)

    node_run_refs = await graph_engine.find_all_node_source_run_refs()
    visible_nodes = {
        node_id
        for node_id, run_refs in node_run_refs.items()
        if _is_visible(run_refs, allowed, dataset_id_str)
    }

    edge_run_refs = await graph_engine.find_all_edge_source_run_refs()
    visible_edges = {
        edge
        for edge, run_refs in edge_run_refs.items()
        if _is_visible(run_refs, allowed, dataset_id_str)
        # Guard: an edge is only renderable when both endpoints are visible.
        and edge.source_id in visible_nodes
        and edge.target_id in visible_nodes
    }

    return visible_nodes, visible_edges


async def get_graph_as_of(
    graph_engine,
    dataset_id: UUID,
    as_of: Union[str, datetime],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Materialize the visible subgraph at ``as_of`` as plain node/edge dicts."""
    visible_nodes, visible_edges = await get_visible_artifacts_as_of(
        graph_engine, dataset_id, as_of
    )

    node_data = await graph_engine.get_node_delete_data(sorted(visible_nodes))
    nodes = [
        {
            "id": data.node_id,
            "type": data.node_type,
            "properties": data.node_properties,
        }
        for data in node_data.values()
    ]

    edge_data = await graph_engine.get_edge_delete_data(sorted(visible_edges, key=str))
    edges = [
        {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "relationship_name": edge.relationship_name,
            "properties": data.edge_properties,
        }
        for edge, data in edge_data.items()
    ]

    return nodes, edges


async def search_chunks_as_of(
    graph_engine,
    vector_engine,
    dataset_id: UUID,
    query_text: str,
    as_of: Union[str, datetime],
    top_k: int = 15,
    collection_name: str = "DocumentChunk_text",
) -> List[Any]:
    """Chunk vector search restricted to the artifacts visible at ``as_of``.

    Chunk vectors are keyed by their graph node id (the provenance the delete
    path relies on), so the graph-derived visible set is the exact filter for
    the vector hits: post-filter, then truncate to ``top_k``.
    """
    visible_nodes, _visible_edges = await get_visible_artifacts_as_of(
        graph_engine, dataset_id, as_of
    )
    if not visible_nodes:
        return []

    # Over-fetch (unbounded) then post-filter: hits removed by the visibility
    # filter must not shrink the result below top_k when older ones qualify.
    hits = await vector_engine.search(collection_name, query_text=query_text, limit=None)

    visible_hits = [hit for hit in hits if str(hit.id) in visible_nodes]
    return visible_hits[:top_k]
