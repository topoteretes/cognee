from uuid import UUID
from typing import Dict, List, Tuple

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.modules.engine.utils import generate_node_name
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text
from cognee.tests.utils.get_contains_edge_text import get_contains_edge_text


def format_relationship(
    relationship: Tuple[UUID, UUID, str, Dict],
    node: Dict,
    graph_edges_by_key: Dict[Tuple[str, str, str], Dict],
):
    edge_properties = graph_edges_by_key.get(
        (str(relationship[0]), str(relationship[1]), relationship[2]),
        {},
    )
    relationship_name = get_edge_retrieval_text(
        edge_properties.get("edge_text"),
        edge_properties.get("relationship_name") or relationship[2],
    )

    if relationship[2] == "contains":
        if not relationship_name or relationship_name == "contains":
            relationship_name = get_contains_edge_text(
                generate_node_name(node["name"]),
                node["description"],
            )

    return {str(EdgeType.id_for(relationship_name)): relationship_name}


async def assert_edges_vector_index_present(
    relationships: List[Tuple[UUID, UUID, str, Dict]], convert_to_new_format: bool = True
):
    vector_engine = await get_vector_engine_async()

    graph_engine = await get_graph_engine()
    nodes, graph_edges = await graph_engine.get_graph_data()

    nodes_by_id = {str(node[0]): node[1] for node in nodes}
    graph_edges_by_key = {
        (str(edge[0]), str(edge[1]), edge[2]): edge[3] or {} for edge in graph_edges
    }

    query_edge_ids = {}
    for relationship in relationships:
        query_edge_ids = {
            **query_edge_ids,
            **(
                format_relationship(
                    relationship,
                    nodes_by_id[str(relationship[1])],
                    graph_edges_by_key,
                )
                if convert_to_new_format
                else {str(EdgeType.id_for(relationship[2])): relationship[2]}
            ),
        }

    vector_items = await vector_engine.retrieve(
        "EdgeType_relationship_name", list(query_edge_ids.keys())
    )

    vector_items_by_id = {str(vector_item.id): vector_item for vector_item in vector_items}

    for relationship_id, relationship_name in query_edge_ids.items():
        assert relationship_id in vector_items_by_id, (
            f"Relationship '{relationship_name}' not found in vector store."
        )

        vector_relationship = vector_items_by_id[relationship_id]
        assert vector_relationship.payload["text"] == relationship_name, (
            f"Vectorized edge '{vector_relationship.payload['text']}' does not match the relationship text '{relationship_name}'."
        )
