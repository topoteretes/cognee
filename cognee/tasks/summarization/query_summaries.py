from cognee.infrastructure.databases.vector import get_vector_engine


async def query_summaries(query: str) -> list:
    """
    Parameters:
    - query (str): The query string to filter summaries by.

    Returns:
    - list[str, UUID]: A list of objects providing information about the summaries related to query.
    """
    vector_engine = get_vector_engine()

    summaries_results = await vector_engine.search("text_summary_text", query, limit=5)

    summaries = [summary.payload for summary in summaries_results]

    return summaries
