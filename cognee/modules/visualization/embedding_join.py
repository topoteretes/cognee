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

# Seed for the deterministic over-cap sample. Fixed so a given graph always
# samples the same nodes across runs (snapshot tests depend on this).
SAMPLE_SEED = 42


def _select_nodes(nodes: List[Dict[str, Any]], cap: int) -> List[Dict[str, Any]]:
    """Sort nodes by id and, if over ``cap``, take a deterministic seeded sample."""
    ordered = sorted(nodes, key=lambda n: str(n["id"]))
    if len(ordered) <= cap:
        return ordered
    rng = random.Random(SAMPLE_SEED)
    picked = sorted(rng.sample(range(len(ordered)), cap))
    logger.warning(
        "fetch_node_embeddings: %d nodes exceeds cap %d; sampling %d deterministically",
        len(ordered),
        cap,
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
    cap: int = 2000,
    vector_engine=None,
    index_fields: Optional[Dict[str, str]] = None,
) -> Dict[str, List[float]]:
    """Return ``{node_id: vector}`` for as many nodes as the vector store can supply.

    Args:
        nodes: renderer-facing node dicts (each carries ``id`` and ``type``).
        cap: max nodes to fetch; over-cap graphs are sampled deterministically.
        vector_engine: injected engine (defaults to ``get_vector_engine()``). This
            is the single mocking seam for tests.
        index_fields: type -> indexed-field override (defaults to
            ``DEFAULT_INDEX_FIELDS``).

    Missing vectors are simply absent from the dict; the layout handles them.
    """
    fields = index_fields or DEFAULT_INDEX_FIELDS

    if vector_engine is None:
        from cognee.infrastructure.databases.vector import get_vector_engine

        vector_engine = get_vector_engine()

    selected = _select_nodes(nodes, cap)

    by_type: Dict[Optional[str], List[Dict[str, Any]]] = defaultdict(list)
    for node in selected:
        by_type[node.get("type")].append(node)

    embeddings: Dict[str, List[float]] = {}
    for type_name, type_nodes in by_type.items():
        field = fields.get(type_name)
        if not field:
            continue
        collection = f"{type_name}_{field}"
        try:
            if not await vector_engine.has_collection(collection):
                continue
            found = await _fetch_for_collection(vector_engine, collection, type_nodes, field)
        except Exception as exc:  # never let a vector-store failure break the render
            logger.warning("fetch_node_embeddings: fetch failed for %s: %s", collection, exc)
            continue
        for nid, vector in found.items():
            if nid not in embeddings:
                embeddings[nid] = vector  # first hit wins across collections
            else:
                logger.debug("node %s matched multiple collections; keeping first", nid)

    return embeddings
