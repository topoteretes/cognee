import uuid

from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognee.shared.data_models import GraphDBType



async def group_nodes_by_layer(node_descriptions):
    """ Group nodes by layer decomposition uuid """
    grouped_data = {}

    for item in node_descriptions:
        uuid = item["layer_decomposition_uuid"]
        if uuid not in grouped_data:
            grouped_data[uuid] = []
        grouped_data[uuid].append(item)

    print("GROUPED DATA", grouped_data)

    return grouped_data


async def get_node_by_unique_id(graph, unique_id):
    """ Get a node by its unique_id"""
    # Iterate through all nodes and their attributes in the graph
    for node, attrs in graph.nodes(data=True):
        # Check if the current node's attributes contain the unique_id we're looking for
        if attrs.get('unique_id') == unique_id:
            return node  # Return the node (identifier) if found
    return None

async def connect_nodes_in_graph(graph, relationship_dict, score_threshold=0.9):
    """ Connect nodes in the graph based on the relationship_dict and score_threshold"""
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
                        await graph.add_edge(
                            searched_node_id_found,
                            original_id_for_search_found,
                            weight=relationship['score'],
                            score_metadata=relationship.get('score_metadata', {}),
                            id = f""" SEMANTIC_CONNECTION_{searched_node_id_found}_{original_id_for_search_found}_{str(uuid.uuid4())}"""
                        )

                # For Neo4j
                elif infrastructure_config.get_config()["graph_engine"] == GraphDBType.NEO4J:
                    # Neo4j specific logic to add an edge
                    # This is just a placeholder, replace it with actual Neo4j logic
                    print("query is ", f"""MATCH (a), (b) WHERE a.unique_id = '{relationship['searched_node_id']}' AND b.unique_id = '{relationship['original_id_for_search']}' CREATE (a)-[:CONNECTED {{weight:{relationship['score']}}}]->(b)""")
                    result = await graph.query(f"""MATCH (a), (b) WHERE a.unique_id = '{relationship['searched_node_id']}' AND b.unique_id = '{relationship['original_id_for_search']}'
                              CREATE (a)-[:SEMANTIC_CONNECTION {{weight:{relationship['score']}}}]->(b)""")
                    await graph.close()


def graph_ready_output(results):
    """ Generate a dictionary of relationships from the results of the graph search."""
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
        graph_client = await get_graph_client(GraphDBType.NEO4J)
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
