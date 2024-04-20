from dsp.utils import deduplicate
from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client


async def search_similarity(query: str, graph):
    graph_db_type = infrastructure_config.get_config()["graph_engine"]

    graph_client = await get_graph_client(graph_db_type)

    layer_nodes = await graph_client.get_layer_nodes()

    unique_layer_uuids = set(node["layer_id"] for node in layer_nodes)

    graph_nodes = []

    for layer_id in unique_layer_uuids:
        vector_engine = infrastructure_config.get_config()["vector_engine"]

        results = await vector_engine.search(layer_id, query_text = query, limit = 10)

        if len(results) > 0:
            graph_nodes.extend([
                dict(
                    layer_id = result.payload["references"]["cognitive_layer"],
                    node_id = result.payload["references"]["node_id"],
                    score = result.score,
                ) for result in results if result.score > 0.5
            ])

    if len(graph_nodes) == 0:
        return []

    relevant_context = []

    for graph_node_data in graph_nodes:
        graph_node = await graph_client.extract_node(graph_node_data["node_id"])

        if "chunk_collection" not in graph_node and "chunk_id" not in graph_node:
            continue

        vector_point = await vector_engine.retrieve(graph_node["chunk_collection"], graph_node["chunk_id"])

        relevant_context.append(vector_point.payload["text"])

    return deduplicate(relevant_context)
