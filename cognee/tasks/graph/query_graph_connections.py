import asyncio
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine


async def query_graph_connections(query: str, exploration_levels=1) -> list[(str, str, str)]:
    """
    Find the neighbours of a given node in the graph and return formed sentences.

    Parameters:
    - query (str): The query string to filter nodes by.
    - exploration_levels (int): The number of jumps through edges to perform.

    Returns:
    - list[(str, str, str)]: A list containing the source and destination nodes and relationship.
    """
    if query is None:
        return []

    node_id = query

    graph_engine = await get_graph_engine()

    exact_node = await graph_engine.extract_node(node_id)

    if exact_node is not None and "id" in exact_node:
        node_connections = await graph_engine.get_connections(str(exact_node["id"]))
    else:
        vector_engine = get_vector_engine()
        results = await asyncio.gather(
            vector_engine.search("entity_name", query_text=query, limit=5),
            vector_engine.search("entity_type_name", query_text=query, limit=5),
        )
        results = [*results[0], *results[1]]
        relevant_results = [result for result in results if result.score < 0.5][:5]

        if len(relevant_results) == 0:
            return []

        node_connections_results = await asyncio.gather(
            *[graph_engine.get_connections(result.id) for result in relevant_results]
        )

        node_connections = []
        for neighbours in node_connections_results:
            node_connections.extend(neighbours)

    unique_node_connections_map = {}
    unique_node_connections = []
    for node_connection in node_connections:
        if "id" not in node_connection[0] or "id" not in node_connection[2]:
            continue

        unique_id = f"{node_connection[0]['id']} {node_connection[1]['relationship_name']} {node_connection[2]['id']}"

        if unique_id not in unique_node_connections_map:
            unique_node_connections_map[unique_id] = True

            unique_node_connections.append(node_connection)

    return unique_node_connections
