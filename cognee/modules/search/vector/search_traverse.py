import asyncio
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine

async def search_traverse(query: str):
    node_id = query
    rules = set()

    graph_engine = await get_graph_engine()
    vector_engine = get_vector_engine()

    exact_node = await graph_engine.extract_node(node_id)

    if exact_node is not None and "uuid" in exact_node:
        edges = await graph_engine.get_edges(exact_node["uuid"])

        for edge in edges:
            rules.add(f"{edge[0]} {edge[2]['relationship_name']} {edge[1]}")
    else:
        results = await asyncio.gather(
            vector_engine.search("entities", query_text = query, limit = 10),
            vector_engine.search("classification", query_text = query, limit = 10),
        )
        results = [*results[0], *results[1]]
        relevant_results = [result for result in results if result.score < 0.5][:5]

        if len(relevant_results) > 0:
            for result in relevant_results:
                graph_node_id = result.id

                edges = await graph_engine.get_edges(graph_node_id)

                for edge in edges:
                    rules.add(f"{edge[0]} {edge[2]['relationship_name']} {edge[1]}")

    return list(rules)
