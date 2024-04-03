from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognee.shared.data_models import GraphDBType


# async def extract_node_descriptions(node):
#     descriptions = []
#
#     if G.has_node(node_id):
#         # Get the attributes of the node
#         attributes = G.nodes[node_id]
#
#
#     for node_id, attributes in node:
#         if "description" in attributes and "unique_id" in attributes:
#             descriptions.append({
#                 "node_id": attributes["unique_id"],
#                 "description": attributes["description"],
#                 "layer_uuid": attributes["layer_uuid"],
#                 "layer_decomposition_uuid": attributes["layer_decomposition_uuid"]
#             })
#
#     return descriptions


async def group_nodes_by_layer(node_descriptions):
    grouped_data = {}

    for item in node_descriptions:
        uuid = item["layer_decomposition_uuid"]

        if uuid not in grouped_data:
            grouped_data[uuid] = []

        grouped_data[uuid].append(item)

    print("GROUPED DATA", grouped_data)

    return grouped_data


async def get_node_by_unique_id(graph, unique_id):
    # Iterate through all nodes and their attributes in the graph
    for node, attrs in graph.nodes(data=True):
        # Check if the current node's attributes contain the unique_id we're looking for
        if attrs.get('unique_id') == unique_id:
            return node  # Return the node (identifier) if found
    return None

async def connect_nodes_in_graph(graph, relationship_dict, score_threshold=0.9):
    if not graph or not relationship_dict:
        return graph

    for _, relationships in relationship_dict.items():
        for relationship in relationships:

            if relationship['score'] > score_threshold:

                # For NetworkX
                if infrastructure_config.get_config()["graph_engine"] == GraphDBType.NETWORKX:

                    searched_node_id_found = await get_node_by_unique_id(graph.graph, relationship['searched_node_id'])


                    original_id_for_search_found = await get_node_by_unique_id(graph.graph, relationship['original_id_for_search'])


                    if searched_node_id_found and original_id_for_search_found:
                        print("NETWORKX adding edgedsadas", searched_node_id_found, original_id_for_search_found)
                        await graph.add_edge(
                            relationship['searched_node_id'],
                            relationship['original_id_for_search'],
                            weight=relationship['score'],
                            score_metadata=relationship.get('score_metadata', {})
                        )
                # For Neo4j
                elif infrastructure_config.get_config()["graph_engine"] == GraphDBType.NEO4J:
                    # Neo4j specific logic to add an edge
                    # This is just a placeholder, replace it with actual Neo4j logic
                    await graph.query(f"""MATCH (a), (b) WHERE a.id = {relationship['searched_node_id']} AND b.id = {relationship['original_id_for_search']}
                              CREATE (a)-[:CONNECTED {{weight:{relationship['score']}}}]->(b)""")
    return graph

# async def connect_nodes_in_graph(graph=None, relationship_dict: dict=None, score_treshold:float=None):
#     """
#     For each relationship in relationship_dict, check if both nodes exist in the graph based on node attributes.
#     If they do, create a connection (edge) between them.
#
#     :param graph: A NetworkX graph object
#     :param relationship_dict: A dictionary containing relationships between nodes
#     """
#     if score_treshold is None:
#         score_treshold = 0.9
#     for id, relationships in relationship_dict.items():
#         for relationship in relationships:
#             searched_node_attr_id = relationship["searched_node_id"]
#             score_attr_id = relationship["original_id_for_search"]
#             score = relationship["score"]
#
#             if score> score_treshold:
#                 # Initialize node keys for both searched_node and score_node
#                 searched_node_key, score_node_key = None, None
#
#                 # Find nodes in the graph that match the searched_node_id and score_id from their attributes
#
#                 if infrastructure_config.get_config()["graph_engine"] == GraphDBType.NETWORKX:
#                     searched_node_key = await graph.extract_node(node_string_id=searched_node_attr_id)
#                     print("NETWORKX SEARCHED NODE KEY", searched_node_key)
#                     score_node_key = await graph.extract_node(node_string_id=searched_node_attr_id)
#                     print("NETWORKX SCORE NODE KEY", score_node_key)
#
#                 elif infrastructure_config.get_config()["graph_engine"] == GraphDBType.NEO4J:
#                     searched_node_key = await graph.extract_node(node_string_id=searched_node_attr_id)
#                     print("NEO4J SEARCHED NODE KEY", searched_node_key)
#                     score_node_key = await graph.extract_node(node_string_id=searched_node_attr_id)
#                     print("NEO4J SCORE NODE KEY", score_node_key)
#
#
#                 # for node, attrs in graph.nodes(data = True):
#                 #     if "unique_id" in attrs:  # Ensure there is an "id" attribute
#                 #         if attrs["unique_id"] == searched_node_attr_id:
#                 #             searched_node_key = node
#                 #         elif attrs["unique_id"] == score_attr_id:
#                 #             score_node_key = node
#
#                     # If both nodes are found, no need to continue checking other nodes
#                     if searched_node_key and score_node_key:
#                         break
#
#                 # Check if both nodes were found in the graph
#                 if searched_node_key is not None and score_node_key is not None:
#                     # print(f"Connecting {searched_node_key} to {score_node_key}")
#                     # If both nodes exist, create an edge between them
#                     # You can customize the edge attributes as needed, here we use "score" as an attribute
#                     graph.add_edge(
#                         searched_node_key,
#                         score_node_key,
#                         weight = score,
#                         score_metadata = relationship.get("score_metadata")
#                     )
#             else:
#                 pass
#
#     return graph


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


if __name__ == "__main__":

    async def main():
        graph_client = get_graph_client(GraphDBType.NETWORKX)

        await graph_client.load_graph_from_file()


        graph = graph_client.graph

        # for nodes, attr in graph.nodes(data=True):
        #     if 'd0bd0f6a-09e5-4308-89f6-400d66895126' in nodes:
        #         print(nodes)

        #
        # relationships = {'SuaGeKyKWKWyaSeiqWeWaSyuSKqieSamiyah': [{'collection_id': 'SuaGeKyKWKWyaSeiqWeWaSyuSKqieSamiyah', 'searched_node_id': 'd0bd0f6a-09e5-4308-89f6-400d66895126', 'score': 1.0, 'score_metadata': {'text': 'Pravilnik o izmenama i dopunama Pravilnika o sadržini, načinu i postupku izrade i način vršenja kontrole tehničke dokumentacije prema klasi i nameni objekata'}, 'original_id_for_search': '2801f7b5-55bf-499b-9843-97d48f8e067a'}, {'collection_id': 'SuaGeKyKWKWyaSeiqWeWaSyuSKqieSamiyah', 'searched_node_id': 'd0bd0f6a-09e5-4308-89f6-400d66895126', 'score': 0.1648828387260437, 'score_metadata': {'text': 'Zakon o planiranju i izgradnji'}, 'original_id_for_search': '57966b55-33e2-4eae-a7fa-2f0237643bbe'}, {'collection_id': 'SuaGeKyKWKWyaSeiqWeWaSyuSKqieSamiyah', 'searched_node_id': 'd0bd0f6a-09e5-4308-89f6-400d66895126', 'score': 0.12986786663532257, 'score_metadata': {'text': 'Službeni glasnik RS, broj 77/2015'}, 'original_id_for_search': '0f626d48-4441-43c1-9060-ea7e54f6d8e2'}, {'collection_id': 'SuaGeKyKWKWyaSeiqWeWaSyuSKqieSamiyah', 'searched_node_id': 'c9b9a460-c64a-4e2e-a4d6-aa5b3769274b', 'score': 1.0, 'score_metadata': {'text': 'Službeni glasnik RS, broj 77/2015'}, 'original_id_for_search': '0f626d48-4441-43c1-9060-ea7e54f6d8e2'}, {'collection_id': 'SuaGeKyKWKWyaSeiqWeWaSyuSKqieSamiyah', 'searched_node_id': 'c9b9a460-c64a-4e2e-a4d6-aa5b3769274b', 'score': 0.07603412866592407, 'score_metadata': {'text': 'Prof. dr Zorana Mihajlović'}, 'original_id_for_search': '5d064a62-3cd6-4895-9f60-1a0d8bc299e8'}, {'collection_id': 'SuaGeKyKWKWyaSeiqWeWaSyuSKqieSamiyah', 'searched_node_id': 'c9b9a460-c64a-4e2e-a4d6-aa5b3769274b', 'score': 0.07226034998893738, 'score_metadata': {'text': 'Ministar građevinarstva, saobraćaja i infrastrukture'}, 'original_id_for_search': 'f5d052ca-c4a0-490e-a3ac-d8ad522dea83'}, {'collection_id': 'SuaGeKyKWKWyaSeiqWeWaSyuSKqieSamiyah', 'searched_node_id': 'bbd6d2d6-e673-4b59-a50c-516972a9d0de', 'score': 0.5, 'score_metadata': {'text': 'Pravilnik o izmenama i dopunama Pravilnika o sadržini, načinu i postupku izrade i način vršenja kontrole tehničke dokumentacije prema klasi i nameni objekata'}, 'original_id_for_search': '2801f7b5-55bf-499b-9843-97d48f8e067a'}]}
        #
        # connect_nodes_in_graph(graph, relationships)

        from cognee.utils import render_graph

        graph_url = await render_graph(graph)

        print(graph_url)

    import asyncio
    asyncio.run(main())
