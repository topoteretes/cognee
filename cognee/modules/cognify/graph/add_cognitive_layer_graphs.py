from datetime import datetime
from uuid import uuid4
from typing import List, Tuple, TypedDict
from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.databases.vector import DataPoint
from cognee.shared.data_models import KnowledgeGraph
from cognee.utils import extract_pos_tags, extract_named_entities, extract_sentiment_vader

class GraphLike(TypedDict):
    nodes: List
    edges: List


async def add_cognitive_layer_graphs(
    graph_client,
    chunk_collection: str,
    chunk_id: str,
    layer_graphs: List[Tuple[str, GraphLike]],
):
    vector_client = infrastructure_config.get_config("vector_engine")

    for (layer_id, layer_graph) in layer_graphs:
        graph_nodes = []
        graph_edges = []

        if not isinstance(layer_graph, KnowledgeGraph):
            layer_graph = KnowledgeGraph.parse_obj(layer_graph)

        for node in layer_graph.nodes:
            node_id = generate_proposition_node_id(node.id)

            entity_type_node_id = generate_type_node_id(node.entity_type)
            entity_type_node = await graph_client.extract_node(entity_type_node_id)

            if not entity_type_node:
                node_name = node.entity_type.lower().capitalize()
              
                entity_type_node = (
                    entity_type_node_id,
                    dict(
                        id = entity_type_node_id,
                        name = node_name,
                        entity_type = node_name,
                        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                )

                graph_nodes.append(entity_type_node)

                # Add relationship between document and entity type: "Document contains Person"
                graph_edges.append((
                    layer_id,
                    entity_type_node_id,
                    "contains",
                    dict(relationship_name = "contains"),
                ))

            pos_tags = extract_pos_tags(node.entity_description)
            named_entities = extract_named_entities(node.entity_description)
            sentiment = extract_sentiment_vader(node.entity_description)

            graph_nodes.append((
                node_id,
                dict(
                    id = node_id,
                    layer_id = layer_id,
                    chunk_id = chunk_id,
                    chunk_collection = chunk_collection,
                    name = node.entity_name,
                    entity_type = node.entity_type.lower().capitalize(),
                    description = node.entity_description,
                    pos_tags = pos_tags,
                    sentiment = sentiment,
                    named_entities = named_entities,
                    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
            ))

            # Add relationship between entity type and entity itself: "Jake is Person"
            graph_edges.append((
                node_id,
                entity_type_node_id,
                "is",
                dict(relationship_name = "is"),
            ))

            graph_edges.append((
                layer_id,
                node_id,
                "contains",
                dict(relationship_name = "contains"),
            ))

        # Add relationship that came from graphs.
        for edge in layer_graph.edges:
            graph_edges.append((
                generate_proposition_node_id(edge.source_node_id),
                generate_proposition_node_id(edge.target_node_id),
                edge.relationship_name,
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
                    value = node_data["name"],
                    references = dict(
                        node_id = node_id,
                        cognitive_layer = layer_id,
                    ),
                ),
                embed_field = "value"
            ) for (node_id, node_data) in graph_nodes
        ]

        await vector_client.create_data_points(layer_id, data_points)


def generate_proposition_node_id(node_id: str) -> str:
    return f"PROPOSITION_NODE__{node_id.upper().replace(' ', '_')}".replace("'", "")

def generate_type_node_id(node_id: str) -> str:
    return f"PROPOSITION_TYPE_NODE__{node_id.upper().replace(' ', '_')}".replace("'", "")