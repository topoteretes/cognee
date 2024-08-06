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

    results = [
        parse_payload(result.payload) for result in similar_results
    ]

    return results


def parse_payload(payload: dict) -> dict:
    return {
        "text": payload["text"],
        "chunk_id": payload["chunk_id"],
        "document_id": payload["document_id"],
    }
