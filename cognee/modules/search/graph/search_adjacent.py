import asyncio
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine

async def search_adjacent(query: str) -> list[(str, str)]:
    """
    Find the neighbours of a given node in the graph and return their ids and descriptions.

    Parameters:
    - query (str): The query string to filter nodes by.

    Returns:
    - list[(str, str)]: A list containing the unique identifiers and names of the neighbours of the given node.
    """
    node_id = query

    if node_id is None:
        return {}

    graph_engine = await get_graph_engine()

    exact_node = await graph_engine.extract_node(node_id)

    if exact_node is not None and "uuid" in exact_node:
        neighbours = await graph_engine.get_neighbours(exact_node["uuid"])
    else:
        vector_engine = get_vector_engine()
        results = await asyncio.gather(
            vector_engine.search("entities", query_text = query, limit = 10),
            vector_engine.search("classification", query_text = query, limit = 10),
        )
        results = [*results[0], *results[1]]
        relevant_results = [result for result in results if result.score < 0.5][:5]

        if len(relevant_results) == 0:
            return []

        node_neighbours = await asyncio.gather(*[graph_engine.get_neighbours(result.id) for result in relevant_results])
        neighbours = []
        for neighbour_ids in node_neighbours:
            neighbours.extend(neighbour_ids)

    return neighbours
