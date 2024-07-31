
from cognee.infrastructure.databases.graph import get_graph_engine, get_graph_config

async def search_cypher(query: str):
    """
    Use a Cypher query to search the graph and return the results.
    """
    graph_config = get_graph_config()

    if graph_config.graph_database_provider == "neo4j":
        graph_engine = await get_graph_engine()
        result = await graph_engine.graph().run(query)
        return result
    else:
        raise ValueError("Unsupported search type for the used graph engine.")
