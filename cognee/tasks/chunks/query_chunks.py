from cognee.infrastructure.databases.vector import get_vector_engine


async def query_chunks(query: str) -> list[dict]:
    """

    Queries the vector database to retrieve chunks related to the given query string.

    Parameters:
    - query (str): The query string to filter nodes by.

    Returns:
    - list(dict): A list of objects providing information about the chunks related to query.

    Notes:
        - The function uses the `search` method of the vector engine to find matches.
        - Limits the results to the top 5 matching chunks to balance performance and relevance.
        - Ensure that the vector database is properly initialized and contains the "document_chunk_text" collection.
    """
    vector_engine = get_vector_engine()

    found_chunks = await vector_engine.search("document_chunk_text", query, limit=5)

    chunks = [result.payload for result in found_chunks]

    return chunks
