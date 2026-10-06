"""Widen a temporal rerank's candidate pool with what the window itself points at.

Hybrid's candidates are chosen by vector similarity, and embeddings barely encode
dates: on a table of 445 contract rows the nine approved on 2 October 2026 rarely
land in the twenty candidates for "approved on 2 October 2026". The graph knows
them anyway: every one has an edge into that day's ``Timestamp`` node. So once
the window is known, the nodes attached to the in-window timestamps are added to
the candidates (after hybrid's own, which keep their relevance order) before the
anchors are computed; the existing rerank then moves them to the front.

Only chunks are added (document chunks and DLT rows): what counts as one is
decided by the vector store, not by type names, so an attached id found in a
chunk collection joins and anything else (entities, schema tables, columns) is
ignored. Entities stay with the entity lane: an entity attached to a window is
usually a hub with edges into many times ("the space race", "cold war"), and
promoting those displaces the entities the question is about.
"""

from typing import Any

from cognee.modules.retrieval.hybrid.results import payload, payload_matches_node_filter, result_id


def attached_node_ids(timestamps: list[dict], nodes_edges: tuple[list, list]) -> list[str]:
    """Ids of the nodes with an edge into one of ``timestamps``, in timestamp order.

    ``nodes_edges`` is ``get_neighborhood(timestamp ids, depth=1)``; only edges
    whose target is one of the timestamps count, so a timestamp's other
    neighbours (and the timestamps themselves) are never candidates.
    """
    order = {str(ts["id"]): index for index, ts in enumerate(timestamps)}
    _nodes, edges = nodes_edges
    attached: dict[str, int] = {}
    for source, target, _relationship, _properties in edges:
        rank = order.get(str(target))
        if rank is None or str(source) in order:
            continue
        current = attached.get(str(source))
        if current is None or rank < current:
            attached[str(source)] = rank
    return sorted(attached, key=lambda node_id: (attached[node_id], node_id))


async def retrieve_in_collections(
    vector_engine: Any,
    collections: tuple[str, ...],
    node_ids: list[str],
    node_name: list[str] | None,
    node_name_filter_operator: str,
) -> list[Any]:
    """The vector rows for ``node_ids`` across ``collections``, in ``node_ids`` order.

    A collection the dataset does not have raises in the engine the same way the
    lanes' searches do; callers pass the collections the search already gated.
    """
    if not node_ids:
        return []
    by_id: dict[str, Any] = {}
    for collection in collections:
        for hit in await vector_engine.retrieve(collection, node_ids):
            hit_id = result_id(hit)
            if hit_id and hit_id not in by_id:
                if payload_matches_node_filter(payload(hit), node_name, node_name_filter_operator):
                    by_id[hit_id] = hit
    return [by_id[node_id] for node_id in node_ids if node_id in by_id]
