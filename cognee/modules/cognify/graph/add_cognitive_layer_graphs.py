from datetime import datetime
from uuid import uuid4
from typing import List, Tuple, TypedDict
from pydantic import BaseModel
from cognee.infrastructure.databases.vector import DataPoint
# from cognee.utils import extract_pos_tags, extract_named_entities, extract_sentiment_vader
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.vector.config import get_vectordb_config
graph_config = get_graph_config()
vectordb_config = get_vectordb_config()
class GraphLike(TypedDict):
    nodes: List
    edges: List


async def add_cognitive_layer_graphs(
    graph_client,
    chunk_collection: str,
    chunk_id: str,
    layer_graphs: List[Tuple[str, GraphLike]],
):
    vector_client = vectordb_config.vector_engine
    graph_model = graph_config.graph_model

    for (layer_id, layer_graph) in layer_graphs:
        graph_nodes = []
        graph_edges = []

        if not isinstance(layer_graph, graph_model):
            layer_graph = graph_model.parse_obj(layer_graph)

        for node in layer_graph.nodes:
            node_id = generate_node_id(node.id)

            type_node_id = generate_node_id(node.type)
            type_node = await graph_client.extract_node(type_node_id)

            if not type_node:
                node_name = node.type.lower().capitalize()
              
                type_node = (
                    type_node_id,
                    dict(
                        id = type_node_id,
                        name = node_name,
                        type = node_name,
                        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                )

                graph_nodes.append(type_node)

                # Add relationship between document and entity type: "Document contains Person"
                graph_edges.append((
                    layer_id,
                    type_node_id,
                    "contains",
                    dict(relationship_name = "contains"),
                ))

            # pos_tags = extract_pos_tags(node.description)
            # named_entities = extract_named_entities(node.description)
            # sentiment = extract_sentiment_vader(node.description)

            id, type, name, description, *node_properties = node

            print("Node properties: ", node_properties)

            node_properties = dict(node_properties)

            graph_nodes.append((
                node_id,
                dict(
                    id = node_id,
                    layer_id = layer_id,
                    chunk_id = chunk_id,
                    chunk_collection = chunk_collection,
                    name = node.name,
                    type = node.type.lower().capitalize(),
                    description = node.description,
                    # pos_tags = pos_tags,
                    # sentiment = sentiment,
                    # named_entities = named_entities,
                    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **node_properties,
                )
            ))

            # Add relationship between entity type and entity itself: "Jake is Person"
            graph_edges.append((
                node_id,
                type_node_id,
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
                generate_node_id(edge.source_node_id),
                generate_node_id(edge.target_node_id),
                edge.relationship_name,
                dict(relationship_name = edge.relationship_name),
            ))

        await graph_client.add_nodes(graph_nodes)

        await graph_client.add_edges(graph_edges)

        class References(BaseModel):
            node_id: str
            cognitive_layer: str

        class PayloadSchema(BaseModel):
            value: str
            references: References

        try:
            await vector_client.create_collection(layer_id, payload_schema = PayloadSchema)
        except Exception:
            # It's ok if the collection already exists.
            pass

        data_points = [
            DataPoint[PayloadSchema](
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


def generate_node_id(node_id: str) -> str:
    return node_id.upper().replace(' ', '_').replace("'", "")
