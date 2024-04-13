from uuid import uuid4
from typing import List, Tuple, Dict, TypedDict
from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.databases.vector import DataPoint

class GraphLike(TypedDict):
    nodes: List
    edges: List

async def add_cognitive_layer_graphs(graph_client, parent_node_id: str, layer_graphs: List[Tuple[str, GraphLike]]):
    vector_client = infrastructure_config.get_config("vector_engine")

    for (layer_id, layer_graph) in layer_graphs:
        graph_nodes = []
        graph_edges = []
        graph_entity_types: Dict[str, Tuple] = {}

        for node in layer_graph.nodes:
            node_id = generate_node_id(node.id)

            if node.entity_type not in graph_entity_types:
                entity_type_node_id = generate_node_id(node.entity_type)

                entity_type_node = (
                    entity_type_node_id,
                    dict(
                        label = node.entity_type.lower().capitalize(),
                        entity_type = node.entity_type.lower().capitalize(),
                    )
                )

                graph_nodes.append(entity_type_node)

                # Add relationship between document and entity type: "Document contains Person"
                graph_edges.append((
                    parent_node_id,
                    entity_type_node_id,
                    dict(relationship_name = "contains"),
                ))

                graph_entity_types[node.entity_type] = entity_type_node

            graph_nodes.append((
                node_id,
                dict(
                    label = node.entity_name,
                    entity_name = node.entity_name,
                    entity_type = node.entity_type.lower().capitalize(),
                )
            ))

            # Add relationship between entity type and entity itself: "Jake is Person"
            graph_edges.append((
                node_id,
                graph_entity_types[node.entity_type][0],
                dict(relationship_name = "is"),
            ))

            graph_edges.append((
                layer_id,
                node_id,
                dict(relationship_name = "decomposed_as"),
            ))

        # Add relationship that came from graphs.
        for edge in layer_graph.edges:
            graph_edges.append((
                generate_node_id(edge.source_node_id),
                generate_node_id(edge.target_node_id),
                dict(relationship_name = edge.relationship_name),
            ))

        await graph_client.add_nodes(graph_nodes)

        await graph_client.add_edges(graph_edges)

        try:
            await vector_client.create_collection(layer_id)
        except Exception:
            # It's ok if the collection already exists.
            pass

        data_points = [
            DataPoint(
                id = str(uuid4()),
                payload = dict(
                    value = node_data["entity_name"]
                ),
                embed_field = "value"
            ) for (node_id, node_data) in graph_nodes if "entity_name" in node_data
        ]

        await vector_client.create_data_points(layer_id, data_points)


def generate_node_id(node_id: str) -> str:
    return f"COGNITIVE_LAYER_NODE-{node_id.upper().replace(' ', '_')}"
