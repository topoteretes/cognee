import asyncio
from typing import Any

from cognee.infrastructure.databases.graph import get_graph_engine


# region get_entities_with_neighborhood helper functions
async def get_all_entity_nodes(graph_engine):
    """Retrieve all nodes of type Entity from the graph."""
    nodes, _ = await graph_engine.get_filtered_graph_data([{"type": ["Entity"]}])
    return nodes


async def get_entity_neighborhood(
    node_id: str, props: dict[str, Any], graph_engine
) -> dict[str, Any]:
    """Fetch and format data for a single entity node.

    Keeps the node's full stored properties (not a hand-picked subset) -
    build_entity() needs them all to rebuild the Entity without dropping
    feedback_weight, importance_weight, belongs_to_set, and every other field
    this pipeline has no opinion about.
    """
    edges_with_endpoints = await get_edges_with_endpoints(graph_engine, node_id)

    entity_types, edges, filtered_neighbors = format_edges_with_endpoints(
        node_id, edges_with_endpoints
    )
    entity_props = dict(props)
    if "id" not in entity_props:
        entity_props["id"] = str(node_id)
    return {
        "properties": entity_props,
        "edges": edges,
        "neighbors": filtered_neighbors,
        "entity_types": entity_types,
    }


async def get_edges_with_endpoints(graph_engine, node_id):
    """Incident edges as (source, edge, target), including edge properties.

    Wraps graph_engine.get_connections(); get_edges() never returns edge_text.
    """
    return await graph_engine.get_connections(node_id)


def _is_outgoing(edge_info: dict[str, Any], node_id: str, node_in_source_slot: bool) -> bool:
    """Whether this edge points away from node_id.

    Read from the edge's own source_node_id rather than the triple's slot
    order, because the two disagree by backend. Ladybug's get_connections
    matches undirected (``MATCH (n)-[r:EDGE]-(m)`` with ``n`` pinned to the
    queried node), so it returns the queried node in the source slot for
    incoming edges too; Neo4j, Neptune, Turso and the Postgres adapters
    preserve the real direction. get_graph_from_model stamps source_node_id
    on every edge cognee writes, so it is the one answer that holds
    everywhere - same fallback delete_chunks_incremental uses.

    Edges written without that property (an adapter-level write, an older
    graph) fall back to slot order, which is what this code did before.
    """
    edge_source_id = edge_info.get("source_node_id")
    if edge_source_id is None:
        return node_in_source_slot
    return str(edge_source_id) == str(node_id)


def format_edges_with_endpoints(
    node_id: str,
    edges_with_endpoints: list[Any],
    node_fields: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, str | None]]], list[dict[str, Any]]]:
    """Split (source, edge, target) triples into EntityType neighbors, edges, and other neighbors.

    node_id can be on either side of the edge, so the neighbor is the other
    endpoint. Unlike get_edges() (which never carries edge properties on any
    backend), the edge dict here includes edge_text when the edge has one.

    A membership is an outgoing is_a edge from this node to an EntityType, at
    most once per type id. Node type alone is not enough: any other edge that
    happens to land on an EntityType node (or an is_a edge pointing the other
    way) is not a typing statement, and treating it as one makes this pipeline
    write back an is_a edge cognify never asserted.

    An entity can have more than one such type - e.g. classified differently
    across separate ingestions of the same (name-deduped) entity - so
    entity_types is a list, not a single value that would silently drop all
    but the last one found. It is deduped by type id: two distinct edges to
    the same type are one membership, not two, or the entity is counted twice
    in that type's member list and inflates the total member count the type
    summary is required to state.

    Two distinct edges can also connect this node to the SAME neighbor (e.g.
    "works_at" and "visited" both linking the same pair) - edges maps each
    neighbor id to a list of every edge found, not a single dict, so a later
    edge to an already-seen neighbor never overwrites an earlier one.
    filtered_neighbors lists each distinct neighbor once regardless of how
    many edges connect to it; build_node_neighborhood_prompt is what expands
    a multi-edge neighbor back into one line per edge.
    """
    if node_fields is None:
        node_fields = {"id", "name", "description", "text", "type"}

    entity_types: list[dict[str, Any]] = []
    edges: dict[str, list[dict[str, str | None]]] = {}
    filtered_neighbors: list[dict[str, Any]] = []
    seen_neighbor_ids: set[str] = set()
    seen_entity_type_ids: set[str] = set()

    for triple in edges_with_endpoints:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue

        source, edge_info, target = triple
        # Slot order finds the NEIGHBOR on every backend - whichever endpoint
        # is not this node - but it does not give the edge's direction: the
        # default backend matches undirected and always returns the queried
        # node in the source slot, so the two questions need two answers.
        node_in_source_slot = str(source.get("id")) == str(node_id)
        neighbor = target if node_in_source_slot else source
        neighbor_id = str(neighbor.get("id", ""))
        relationship_name = str(edge_info.get("relationship_name") or "related to")
        is_outgoing = _is_outgoing(edge_info, node_id, node_in_source_slot)

        edges.setdefault(neighbor_id, []).append(
            {
                "relationship_name": relationship_name,
                "edge_text": str(edge_info["edge_text"]) if edge_info.get("edge_text") else None,
            }
        )

        if (
            neighbor.get("type") == "EntityType"
            and relationship_name == "is_a"
            and is_outgoing
            and neighbor_id not in seen_entity_type_ids
        ):
            seen_entity_type_ids.add(neighbor_id)
            entity_types.append(neighbor)

        if neighbor_id not in seen_neighbor_ids:
            seen_neighbor_ids.add(neighbor_id)
            filtered_neighbor = {k: v for k, v in neighbor.items() if k in node_fields}
            if len(filtered_neighbor) > 1:
                filtered_neighbors.append(filtered_neighbor)

    return entity_types, edges, filtered_neighbors


# endregion


async def get_entities_with_neighborhood(args) -> list[dict[str, Any]]:
    """Iterate through all Entity nodes and fetch their edges and neighbor nodes."""
    graph_engine = await get_graph_engine()
    entity_nodes = await get_all_entity_nodes(graph_engine)

    get_entity_neighborhood_tasks = (
        get_entity_neighborhood(node_id, props, graph_engine) for node_id, props in entity_nodes
    )

    return await asyncio.gather(*get_entity_neighborhood_tasks)
