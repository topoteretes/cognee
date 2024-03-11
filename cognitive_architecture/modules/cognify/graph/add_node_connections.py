from cognitive_architecture.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognitive_architecture.shared.data_models import GraphDBType


async def extract_node_descriptions(data):
    descriptions = []
    for node_id, attributes in data:
        if 'description' in attributes and 'unique_id' in attributes:
            descriptions.append({'node_id': attributes['unique_id'], 'description': attributes['description'], 'layer_uuid': attributes['layer_uuid'], 'layer_decomposition_uuid': attributes['layer_decomposition_uuid'] })
    return descriptions




async def add_node_connection(graph_client, vector_database_client, data):

    graph = graph_client.graph
    node_descriptions = data


    grouped_data = {}

    # Iterate through each dictionary in the list
    for item in node_descriptions:
        # Get the layer_decomposition_uuid of the current dictionary
        uuid = item['layer_decomposition_uuid']

        # Check if this uuid is already a key in the grouped_data dictionary
        if uuid not in grouped_data:
            # If not, initialize a new list for this uuid
            grouped_data[uuid] = []

        # Append the current dictionary to the list corresponding to its uuid
        grouped_data[uuid].append(item)

    return grouped_data


def connect_nodes_in_graph(graph, relationship_dict):
    """
    For each relationship in relationship_dict, check if both nodes exist in the graph based on node attributes.
    If they do, create a connection (edge) between them.

    :param graph: A NetworkX graph object
    :param relationship_dict: A dictionary containing relationships between nodes
    """
    for id, relationships in relationship_dict.items():
        for relationship in relationships:
            searched_node_attr_id = relationship['searched_node_id']
            print(searched_node_attr_id)
            score_attr_id = relationship['original_id_for_search']
            score = relationship['score']

            # Initialize node keys for both searched_node and score_node
            searched_node_key, score_node_key = None, None

            # Find nodes in the graph that match the searched_node_id and score_id from their attributes
            for node, attrs in graph.nodes(data=True):
                if 'unique_id' in attrs:  # Ensure there is an 'id' attribute
                    if attrs['unique_id'] == searched_node_attr_id:
                        searched_node_key = node
                    elif attrs['unique_id'] == score_attr_id:
                        score_node_key = node

                # If both nodes are found, no need to continue checking other nodes
                if searched_node_key and score_node_key:
                    break

            # Check if both nodes were found in the graph
            if searched_node_key is not None and score_node_key is not None:
                print(searched_node_key)
                print(score_node_key)
                # If both nodes exist, create an edge between them
                # You can customize the edge attributes as needed, here we use 'score' as an attribute
                graph.add_edge(searched_node_key, score_node_key, weight=score,
                               score_metadata=relationship.get('score_metadata'))

    return graph
def graph_ready_output(results):
    relationship_dict = {}

    for result_tuple in results:

        uuid, scored_points_list, desc, node_id = result_tuple
        # Unpack the tuple

        # Ensure there's a list to collect related items for this uuid
        if uuid not in relationship_dict:
            relationship_dict[uuid] = []

        for scored_points in scored_points_list:  # Iterate over the list of ScoredPoint lists
            for scored_point in scored_points:  # Iterate over each ScoredPoint object
                if scored_point.score > 0.9:  # Check the score condition
                    # Append a new dictionary to the list associated with the uuid
                    relationship_dict[uuid].append({
                        'collection_name_uuid': uuid,
                        'searched_node_id': scored_point.id,
                        'score': scored_point.score,
                        'score_metadata': scored_point.payload,
                        'original_id_for_search': node_id,
                    })
    return relationship_dict




if __name__ == '__main__':
    graph_client = get_graph_client(GraphDBType.NETWORKX)
    add_node_connection(graph_client, None, None)



    # db = get_vector_database()