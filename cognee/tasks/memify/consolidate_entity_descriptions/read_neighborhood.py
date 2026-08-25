import asyncio
from typing import Any, Dict, List, Optional, Set

from cognee.infrastructure.databases.graph import get_graph_engine


# region get_entities_with_neighborhood helper functions
async def get_all_entity_nodes(graph_engine):
    """Retrieve all nodes of type Entity from the graph."""
    nodes, _ = await graph_engine.get_filtered_graph_data([{"type": ["Entity"]}])
    return nodes


async def get_entity_neighborhood(
    node_id: str, props: Dict[str, Any], graph_engine
) -> Dict[str, Any]:
    """Fetch and format data for a single entity node.

    Keeps the node's full stored properties (not a hand-picked subset) -
    build_entity() needs them all to rebuild the Entity without dropping
    feedback_weight, importance_weight, belongs_to_set, and every other field
    this pipeline has no opinion about.
    """
    connections = await graph_engine.get_connections(node_id)

    entity_types, edges, filtered_neighbors = format_connections(node_id, connections)
    entity_props = dict(props)
    if "id" not in entity_props:
        entity_props["id"] = str(node_id)
    return {
        "properties": entity_props,
        "edges": edges,
        "neighbors": filtered_neighbors,
        "entity_types": entity_types,
    }


def format_connections(
    node_id: str,
    connections: List[Any],
    node_fields: Optional[Set[str]] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Optional[str]]], List[Dict[str, Any]]]:
    """Split get_connections() triples into EntityType neighbors, edge info, and other neighbors.

    get_connections(node_id) returns (source, edge, target) triples where node_id
    can be on either side of the edge, so the neighbor is whichever side does not
    match node_id. Unlike get_edges() (which never carries edge properties on any
    backend), the edge dict here includes edge_text when the edge has one.

    An entity can have more than one EntityType neighbor - e.g. classified
    differently across separate ingestions of the same (name-deduped) entity -
    so entity_types is a list, not a single value that would silently drop all
    but the last one found.
    """
    if node_fields is None:
        node_fields = {"id", "name", "description", "text", "type"}

    entity_types: List[Dict[str, Any]] = []
    edges: Dict[str, Dict[str, Optional[str]]] = {}
    filtered_neighbors: List[Dict[str, Any]] = []

    for connection in connections:
        if not isinstance(connection, (list, tuple)) or len(connection) != 3:
            continue

        source, edge_info, target = connection
        neighbor = target if str(source.get("id")) == str(node_id) else source
        neighbor_id = str(neighbor.get("id", ""))

        edges[neighbor_id] = {
            "relationship_name": str(edge_info.get("relationship_name") or "related to"),
            "edge_text": str(edge_info["edge_text"]) if edge_info.get("edge_text") else None,
        }

        if neighbor.get("type") == "EntityType":
            entity_types.append(neighbor)

        filtered_neighbor = {k: v for k, v in neighbor.items() if k in node_fields}
        if len(filtered_neighbor) > 1:
            filtered_neighbors.append(filtered_neighbor)

    return entity_types, edges, filtered_neighbors


# endregion


async def get_entities_with_neighborhood(args) -> List[Dict[str, Any]]:
    """Iterate through all Entity nodes and fetch their edges and neighbor nodes."""
    graph_engine = await get_graph_engine()
    entity_nodes = await get_all_entity_nodes(graph_engine)

    get_entity_neighborhood_tasks = (
        get_entity_neighborhood(node_id, props, graph_engine) for node_id, props in entity_nodes
    )

    return await asyncio.gather(*get_entity_neighborhood_tasks)
