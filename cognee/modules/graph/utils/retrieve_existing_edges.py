from collections.abc import Collection
from typing import Any, Optional

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance import (
    EdgeIdentity,
    graph_provenance_write_kwargs,
)


async def find_existing_edge_identities(
    edge_identities: Collection[EdgeIdentity],
    ctx: Optional[Any] = None,
) -> set[EdgeIdentity]:
    """Return the supplied edge identities that already exist in graph storage.

    When a pipeline context is provided, graph-native source and run refs are
    attached to the relationships that already exist.
    """
    if not edge_identities:
        return set()

    graph_engine = await get_graph_engine()
    existing_edges = await graph_engine.has_edges(
        [
            (identity.source_id, identity.target_id, identity.relationship_name)
            for identity in edge_identities
        ]
    )
    existing_edge_identities = {
        EdgeIdentity(
            source_id=str(source_id),
            target_id=str(target_id),
            relationship_name=relationship_name,
        )
        for source_id, target_id, relationship_name in existing_edges
    }

    # attach_new_edges_to_data_points deliberately leaves existing relationships
    # out of the DataPoint graph, which prevents duplicate graph writes and
    # vector re-indexing. On graph-provenance graphs, preserve the later
    # source's ownership by attaching its ref directly to those relationships
    # in one batch. The run mapping makes this early attach rollback-safe if a
    # later pipeline task fails.
    if existing_edge_identities and ctx is not None:
        provenance_kwargs = await graph_provenance_write_kwargs(graph_engine, ctx)
        source_ref_key = provenance_kwargs["source_ref_key"]
        if source_ref_key is not None:
            await graph_engine.attach_edge_source_refs(
                list(existing_edge_identities),
                [source_ref_key],
                provenance_kwargs["pipeline_run_id"],
            )

    return existing_edge_identities
