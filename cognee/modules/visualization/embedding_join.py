"""Join graph nodes to their stored embeddings — the plumbing behind the semantic map.

``visualize_graph()`` forwards only nodes/links; no embeddings ride along. This
module fetches the embedding for each node by looking it up in the vector store,
reusing the fact that a graph node id is stored verbatim as the vector-row id
(both sides use ``str(data_point.id)``).

Strategy per node type:
  * Group node ids by type and derive the collection ``f"{Type}_{field}"`` from
    the type's indexed field.
  * One batched ``retrieve(..., include_vector=True)`` per collection — never per
    node. On adapters that don't support ``include_vector`` (everything except
    LanceDB), fall back to re-embedding the indexed field in one batch (the stored
    vector is ``embed(indexed_field)``, so the re-embedded vector matches).

Bounding lives in :func:`select_nodes`: the orchestrator samples the graph down
to ``SEMANTIC_NODE_CAP`` once, and the fetch, projection, clustering, and
de-overlap passes all operate on that same subset.

Any vector-engine failure is logged and yields a partial/empty dict — the classic
render must never break because the semantic tab couldn't fetch vectors.
"""

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

from cognee.shared.logging_utils import get_logger

logger = get_logger("embedding_join")

# type_name -> indexed field. The vector collection is ``f"{type_name}_{field}"``.
# Mirrors each DataPoint subclass's ``metadata["index_fields"]``; this fallback
# covers the node types the graph visualization actually surfaces.
DEFAULT_INDEX_FIELDS: Dict[str, str] = {
    "Entity": "name",
    "EntityType": "name",
    "TextSummary": "text",
    "DocumentChunk": "text",
    "TextDocument": "name",
}

# Max nodes the semantic map renders. Bounds the vector fetch AND the O(n²)
# layout/neighbor passes downstream — everything runs on the same sample.
SEMANTIC_NODE_CAP = 2000

# Seed for the deterministic over-cap sample. Fixed so a given graph always
# samples the same nodes across runs (snapshot tests depend on this).
SAMPLE_SEED = 42


def select_nodes(nodes: List[Dict[str, Any]], cap: int = SEMANTIC_NODE_CAP) -> List[Dict[str, Any]]:
    """Sort nodes by id and, if over ``cap``, take a deterministic seeded sample.

    The single bounding step for the semantic map: the embedding fetch and every
    downstream pass (projection, clustering, de-overlap) run on this subset only.
    """
    ordered = sorted(nodes, key=lambda n: str(n["id"]))
    if len(ordered) <= cap:
        return ordered
    rng = random.Random(SAMPLE_SEED)
    picked = sorted(rng.sample(range(len(ordered)), cap))
    logger.warning(
        "select_nodes: %d nodes exceeds cap %d; the semantic map shows a deterministic sample",
        len(ordered),
        cap,
    )
    return [ordered[i] for i in picked]


async def _reembed(
    vector_engine, type_nodes: List[Dict[str, Any]], field: str
) -> Dict[str, List[float]]:
    """Fallback: re-embed each node's indexed field in one batch."""
    texts: List[str] = []
    ids: List[str] = []
    for node in type_nodes:
        value = node.get(field)
        if value is None:
            value = node.get("name")
        if value is None:
            continue
        texts.append(str(value))
        ids.append(str(node["id"]))
    if not texts:
        return {}
    vectors = await vector_engine.embedding_engine.embed_text(texts)
    return {nid: list(vec) for nid, vec in zip(ids, vectors)}


async def _fetch_for_collection(
    vector_engine, collection: str, type_nodes: List[Dict[str, Any]], field: str
) -> Dict[str, List[float]]:
    """One batched retrieve for a collection, with re-embed fallback."""
    ids = [str(node["id"]) for node in type_nodes]
    try:
        results = await vector_engine.retrieve(collection, ids, include_vector=True)
    except TypeError:
        # Adapter's retrieve() doesn't accept include_vector -> unsupported.
        return await _reembed(vector_engine, type_nodes, field)

    found: Dict[str, List[float]] = {}
    for result in results:
        payload = result.payload if isinstance(result.payload, dict) else None
        vector = payload.get("vector") if payload else None
        if vector is not None:
            found[str(result.id)] = list(vector)

    if not found and results:
        # Adapter accepted the flag but returned no vectors -> re-embed.
        return await _reembed(vector_engine, type_nodes, field)
    return found


async def fetch_node_embeddings(
    nodes: List[Dict[str, Any]],
    vector_engine=None,
    index_fields: Optional[Dict[str, str]] = None,
) -> Dict[str, List[float]]:
    """Return ``{node_id: vector}`` for as many nodes as the vector store can supply.

    Args:
        nodes: renderer-facing node dicts (each carries ``id`` and ``type``),
            already bounded by :func:`select_nodes`.
        vector_engine: injected engine (defaults to ``await get_vector_engine_async()``).
            This is the single mocking seam for tests.
        index_fields: type -> indexed-field override (defaults to
            ``DEFAULT_INDEX_FIELDS``).

    Missing vectors are simply absent from the dict; the layout handles them.
    """
    fields = index_fields or DEFAULT_INDEX_FIELDS

    if vector_engine is None:
        from cognee.infrastructure.databases.vector import get_vector_engine_async

        vector_engine = await get_vector_engine_async()

    by_type: Dict[Optional[str], List[Dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        by_type[node.get("type")].append(node)

    embeddings: Dict[str, List[float]] = {}
    hit_collections = 0
    missing_collections: List[str] = []
    unmapped_types: List[str] = []
    for type_name, type_nodes in by_type.items():
        field = fields.get(type_name)
        if not field:
            if type_name is not None:
                unmapped_types.append(type_name)
            continue
        collection = f"{type_name}_{field}"
        try:
            if not await vector_engine.has_collection(collection):
                missing_collections.append(collection)
                continue
            found = await _fetch_for_collection(vector_engine, collection, type_nodes, field)
        except Exception as exc:  # never let a vector-store failure break the render
            logger.warning("fetch_node_embeddings: fetch failed for %s: %s", collection, exc)
            continue
        if found:
            hit_collections += 1
        embeddings.update(found)

    # Join hit-rate: turn a silent empty map into a diagnosable one. A zero
    # resolution over non-empty input almost always means an id/collection-name
    # mismatch — surface which collections were missing and which types were
    # unmapped instead of rendering blank with no signal.
    total = len(nodes)
    logger.info(
        "fetch_node_embeddings: resolved %d/%d node embeddings across %d collection(s)",
        len(embeddings),
        total,
        hit_collections,
    )
    if total and not embeddings:
        logger.warning(
            "fetch_node_embeddings: no embeddings resolved — the semantic map will be empty. "
            "Missing collections: %s. Unmapped node types: %s.",
            missing_collections or "none",
            unmapped_types or "none",
        )

    return embeddings
