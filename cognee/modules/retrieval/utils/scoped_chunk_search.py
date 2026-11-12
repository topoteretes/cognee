"""Chunk search restricted to a typed graph scope.

A chunk search scoped by ``node_type`` other than ``NodeSet`` has two steps, the
same two ``brute_force_triplet_search`` takes for a typed scope:

1. The graph decides what is in scope. ``get_nodeset_subgraph`` projects the named
   nodes of ``node_type`` and their one-hop neighbours; the ``DocumentChunk`` nodes
   among them are the scope. For a document that is its chunks (``is_part_of``),
   for an entity the chunks that mention it (``contains``).
2. The vector store scores exactly those chunk ids against the query
   (``score_by_ids``). An adapter without it is refused: the alternative is an
   unlimited scan of the whole collection on every request.

The vector store never learns what a document or an entity is: it only ever gets
chunk ids. The ``node_name`` vector filter means node-set membership and is never
applied to a typed scope.
"""

from cognee.exceptions import CogneeConfigurationError
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

    Raises ``CollectionNotFoundError`` like ``search`` does when there are no chunks,
    and ``CogneeConfigurationError`` when the adapter cannot score by id.
    """
    chunk_ids = await scoped_chunk_ids(
        graph_engine, node_type, node_name, node_name_filter_operator
    )
    if not chunk_ids:
        return []

    if not engine_supports_id_scoring(vector_engine):
        # Without id scoring the only way to score the scope is an unlimited scan of
        # the whole collection filtered here, which costs the collection's size per
        # request, and adapters that cap an unlimited read would silently drop
        # in-scope chunks. Refuse instead; the adapter's fix is to add score_by_ids.
        adapter = type(vector_engine).__name__
        raise CogneeConfigurationError(
            message=(
                f"A node_name scope on node_type={node_type.__name__} needs a vector adapter "
                f"that implements score_by_ids; {adapter} does not."
            ),
            name="ScopedChunkSearchUnsupported",
            remediation=(
                f"Implement VectorDBInterface.score_by_ids on {adapter}, or scope by node set "
                "(node_type=NodeSet)."
            ),
        )

    query_vector = (await vector_engine.embedding_engine.embed_text([query]))[0]
    scored = await vector_engine.score_by_ids(CHUNK_COLLECTION, chunk_ids, query_vector)
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
