from cognee.infrastructure.databases.vector import get_vector_engine

async def query_chunks(query: str) -> list[str]:
    """
    Parameters:
    - query (str): The query string to filter nodes by.

    Returns:
    - list(chunk): A list of objects providing information about the chunks related to query.
    """
    vector_engine = get_vector_engine()

    found_chunks = await vector_engine.search("chunks", query, limit = 5)

    chunks = [result.payload for result in found_chunks]

    return chunks
