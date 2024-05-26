from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognee.infrastructure.databases.graph.config import get_graph_config
graph_config = get_graph_config()
from cognee.infrastructure.databases.vector.config import get_vectordb_config
vector_config = get_vectordb_config()

async def search_similarity(query: str, graph):
    graph_db_type = graph_config.graph_engine

    graph_client = await get_graph_client(graph_db_type)

    layer_nodes = await graph_client.get_layer_nodes()

    unique_layer_uuids = set(node["layer_id"] for node in layer_nodes)
    print("unique_layer_uuids", unique_layer_uuids)


    graph_nodes = []

    for layer_id in unique_layer_uuids:
        vector_engine = vector_config.vector_engine

        results = await vector_engine.search(layer_id, query_text = query, limit = 10)
        print("results", results)
        print("len_rs", len(results))

        if len(results) > 0:
            graph_nodes.extend([
                dict(
                    layer_id = result.payload["references"]["cognitive_layer"],
                    node_id = result.payload["references"]["node_id"],
                    score = result.score,
                ) for result in results if result.score > 0.3
            ])

    if len(graph_nodes) == 0:
        return []


    return graph_nodes



    # for graph_node_data in graph_nodes:
    #     if graph_node_data['score'] >0.8:
    #         graph_node = await graph_client.extract_node(graph_node_data["node_id"])
    #
    #         if "chunk_collection" not in graph_node and "chunk_id" not in graph_node:
    #             continue
    #
    #         vector_point = await vector_engine.retrieve(
    #             graph_node["chunk_collection"],
    #             graph_node["chunk_id"],
    #         )
    #
    #         print("vector_point", vector_point.payload["text"])
    #
    #         relevant_context.append(vector_point.payload["text"])
    #
    # print(relevant_context)
    #
    # return deduplicate(relevant_context)
