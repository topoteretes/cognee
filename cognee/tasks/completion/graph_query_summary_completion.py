from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.tasks.completion.graph_query_completion import (
    graph_query_completion,
    retrieved_edges_to_string,
)


async def retrieved_edges_to_summary(retrieved_edges: list) -> str:
    """
    Converts a list of retrieved graph edges into a summary without redundancies.

    """
    edges_string = await retrieved_edges_to_string(retrieved_edges)
    system_prompt = read_query_prompt("summarize_search_results.txt")
    llm_client = get_llm_client()
    summarized_context = await llm_client.acreate_structured_output(
        text_input=edges_string,
        system_prompt=system_prompt,
        response_model=str,
    )
    return summarized_context


async def graph_query_summary_completion(query: str) -> list:
    return await graph_query_completion(query, edges_to_string_converter=retrieved_edges_to_summary)
