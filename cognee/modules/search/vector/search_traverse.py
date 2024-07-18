from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine

async def search_traverse(query: str, graph): # graph must be there in order to be compatible with generic call
    graph_engine = await get_graph_engine()
    vector_engine = get_vector_engine()

    results = await vector_engine.search("classification", query_text = query, limit = 10)

    rules = []

    if len(results) > 0:
        for result in results:
            graph_node_id = result.id

            edges = await graph_engine.get_edges(graph_node_id)

            for edge in edges:
                rules.append(f"{edge[0]} {edge[2]['relationship_name']} {edge[1]}")

    return rules
