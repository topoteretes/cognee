from collections.abc import Collection

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance import EdgeIdentity


async def find_existing_edge_identities(
    edge_identities: Collection[EdgeIdentity],
) -> set[EdgeIdentity]:
    """Return the supplied edge identities that already exist in graph storage."""
    if not edge_identities:
        return set()

    graph_engine = await get_graph_engine()
    existing_edges = await graph_engine.has_edges(
        [
            (identity.source_id, identity.target_id, identity.relationship_name)
            for identity in edge_identities
        ]
    )
    return {
        EdgeIdentity(
            source_id=str(source_id),
            target_id=str(target_id),
            relationship_name=relationship_name,
        )
        for source_id, target_id, relationship_name in existing_edges
    }
