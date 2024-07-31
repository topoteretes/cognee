from cognee.infrastructure.databases.vector import get_vector_engine

async def search_similarity(query: str) -> list[str, str]:
    """
    Parameters:
    - query (str): The query string to filter nodes by.

    Returns:
    - list(chunk): A list of objects providing information about the chunks related to query.
    """
    vector_engine = get_vector_engine()

    similar_results = await vector_engine.search("chunks", query, limit = 5)

    results = [result.payload for result in similar_results]

    return results
