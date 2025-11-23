from uuid import UUID
from typing import Dict, List, Tuple

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.engine.utils import generate_edge_id, generate_node_name
from cognee.tests.utils.get_contains_edge_text import get_contains_edge_text


def format_relationship(relationship: Tuple[UUID, UUID, str, Dict], node: Dict):
    if relationship[2] == "contains":
        relationship_name = get_contains_edge_text(
            generate_node_name(node["name"]),
            node["description"],
        )

        return {
            str(generate_edge_id(relationship_name)): relationship_name,
        }

    return {str(generate_edge_id(relationship[2])): relationship[2]}


async def assert_edges_vector_index_present(
    relationships: List[Tuple[UUID, UUID, str, Dict]], convert_to_new_format: bool = True
):
    vector_engine = get_vector_engine()

    graph_engine = await get_graph_engine()
    nodes, _ = await graph_engine.get_graph_data()

    nodes_by_id = {str(node[0]): node[1] for node in nodes}

    query_edge_ids = {}
    for relationship in relationships:
        query_edge_ids = {
            **query_edge_ids,
            **(
                format_relationship(relationship, nodes_by_id[str(relationship[1])])
                if convert_to_new_format
                else {str(generate_edge_id(relationship[2])): relationship[2]}
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
