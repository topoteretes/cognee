"""Chunk search restricted to a typed graph scope.

A chunk search scoped by ``node_type`` other than ``NodeSet`` has two steps, the
same two ``brute_force_triplet_search`` takes for a typed scope:

1. The graph decides what is in scope. ``get_nodeset_subgraph`` projects the named
   nodes of ``node_type`` and their one-hop neighbours; the ``DocumentChunk`` nodes
   among them are the scope. For a document that is its chunks (``is_part_of``),
   for an entity the chunks that mention it (``contains``).
2. The vector store scores exactly those chunk ids against the query.

The vector store never learns what a document or an entity is: it only ever gets
chunk ids. The ``node_name`` vector filter means node-set membership and is never
applied to a typed scope.
"""

from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.modules.retrieval.utils.node_edge_vector_search import engine_supports_id_scoring

CHUNK_COLLECTION = "DocumentChunk_text"
CHUNK_NODE_TYPE = "DocumentChunk"


async def scoped_chunk_ids(
    graph_engine,
    node_type: type,
    node_name: list[str],
    node_name_filter_operator: str,
) -> list[str]:
    """The ids of the chunk nodes in the projected scope."""
    nodes, _edges = await graph_engine.get_nodeset_subgraph(
        node_type=node_type,
        node_name=node_name,
        node_name_filter_operator=node_name_filter_operator,
    )
    return [str(node_id) for node_id, data in nodes if data.get("type") == CHUNK_NODE_TYPE]


async def search_scoped_chunks(
    vector_engine,
    graph_engine,
    query: str,
    node_type: type,
    node_name: list[str],
    node_name_filter_operator: str,
    limit: int | None,
) -> list[ScoredResult]:
    """The scope's chunks nearest to ``query``, with their payloads, nearest first.

    Raises ``CollectionNotFoundError`` like ``search`` does when there are no chunks.
    """
    chunk_ids = await scoped_chunk_ids(
        graph_engine, node_type, node_name, node_name_filter_operator
    )
    if not chunk_ids:
        return []

    query_vector = (await vector_engine.embedding_engine.embed_text([query]))[0]
    if engine_supports_id_scoring(vector_engine):
        scored = await vector_engine.score_by_ids(CHUNK_COLLECTION, chunk_ids, query_vector)
    else:
        # Same fallback as brute_force_triplet_search: an adapter without
        # ``score_by_ids`` scans the collection unfiltered and the scope is kept
        # here. Correct, only slower; the adapter's fix is to add score_by_ids.
        in_scope = set(chunk_ids)
        scored = [
            result
            for result in await vector_engine.search(
                CHUNK_COLLECTION, query_vector=query_vector, limit=None
            )
            if str(result.id) in in_scope
        ]

    scored.sort(key=lambda result: result.score)
    if limit is not None:
        scored = scored[:limit]
    if not scored:
        return []

    # Scoring carries no payload; the chunk retrievers render one.
    rows = await vector_engine.retrieve(CHUNK_COLLECTION, [str(result.id) for result in scored])
    payloads = {str(row.id): row.payload for row in rows}
    return [
        ScoredResult(id=result.id, score=result.score, payload=payloads[str(result.id)])
        for result in scored
        if str(result.id) in payloads
    ]
