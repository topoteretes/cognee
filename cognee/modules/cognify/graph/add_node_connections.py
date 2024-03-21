from networkx import Graph
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognee.shared.data_models import GraphDBType


async def extract_node_descriptions(node):
    descriptions = []

    for node_id, attributes in node:
        if "description" in attributes and "unique_id" in attributes:
            descriptions.append({
                "node_id": attributes["unique_id"],
                "description": attributes["description"],
                "layer_uuid": attributes["layer_uuid"],
                "layer_decomposition_uuid": attributes["layer_decomposition_uuid"]
            })

    return descriptions


async def group_nodes_by_layer(node_descriptions):
    grouped_data = {}

    for item in node_descriptions:
        uuid = item["layer_decomposition_uuid"]

        if uuid not in grouped_data:
            grouped_data[uuid] = []

        grouped_data[uuid].append(item)

    return grouped_data

def connect_nodes_in_graph(graph: Graph, relationship_dict: dict) -> Graph:
    """
    For each relationship in relationship_dict, check if both nodes exist in the graph based on node attributes.
    If they do, create a connection (edge) between them.

    :param graph: A NetworkX graph object
    :param relationship_dict: A dictionary containing relationships between nodes
    """
    for id, relationships in relationship_dict.items():
        for relationship in relationships:
            searched_node_attr_id = relationship["searched_node_id"]
            score_attr_id = relationship["original_id_for_search"]
            score = relationship["score"]

            # Initialize node keys for both searched_node and score_node
            searched_node_key, score_node_key = None, None

            # Find nodes in the graph that match the searched_node_id and score_id from their attributes
            for node, attrs in graph.nodes(data = True):
                if "unique_id" in attrs:  # Ensure there is an "id" attribute
                    if attrs["unique_id"] == searched_node_attr_id:
                        searched_node_key = node
                    elif attrs["unique_id"] == score_attr_id:
                        score_node_key = node

                # If both nodes are found, no need to continue checking other nodes
                if searched_node_key and score_node_key:
                    break

            # Check if both nodes were found in the graph
            if searched_node_key is not None and score_node_key is not None:
                # If both nodes exist, create an edge between them
                # You can customize the edge attributes as needed, here we use "score" as an attribute
                graph.add_edge(
                    searched_node_key,
                    score_node_key,
                    weight = score,
                    score_metadata = relationship.get("score_metadata")
                )

    return graph


def graph_ready_output(results):
    relationship_dict = {}

    for result in results:
        layer_id = result["layer_id"]
        layer_nodes = result["layer_nodes"]

        # Ensure there's a list to collect related items for this uuid
        if layer_id not in relationship_dict:
            relationship_dict[layer_id] = []

        for node in layer_nodes:  # Iterate over the list of ScoredPoint lists
            for score_point in node["score_points"]:
                # Append a new dictionary to the list associated with the uuid
                relationship_dict[layer_id].append({
                    "collection_id": layer_id,
                    "searched_node_id": node["id"],
                    "score": score_point.score,
                    "score_metadata": score_point.payload,
                    "original_id_for_search": score_point.id,
                })

    return relationship_dict
