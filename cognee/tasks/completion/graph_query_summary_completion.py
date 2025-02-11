from cognee.infrastructure.engine import ExtendableDataPoint
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.tasks.completion.exceptions import NoRelevantDataFound
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.modules.retrieval.brute_force_triplet_search import brute_force_triplet_search
from cognee.tasks.completion.graph_query_completion import retrieved_edges_to_string


async def retrieved_edges_to_summary(retrieved_edges: list) -> str:
    """
    Converts a list of retrieved graph edges into a summary without redundancies.

    """
    edges_string = retrieved_edges_to_string(retrieved_edges)
    system_prompt = read_query_prompt("summarize_search_results.txt")
    llm_client = get_llm_client()
    summarized_context = await llm_client.acreate_structured_output(
        text_input=edges_string,
        system_prompt=system_prompt,
        response_model=str,
    )
    return summarized_context


async def graph_query_summary_completion(query: str) -> list:
    """
    Executes a query on the graph database, summarizes the results and retrieves a relevant completion based on the found data.

    Parameters:
    - query (str): The query string to compute.

    Returns:
    - list: Answer to the query.

    Notes:
    - The `brute_force_triplet_search` is used to retrieve relevant graph data.
    - Prompts are dynamically rendered and provided to the LLM for contextual understanding.
    - Ensure that the LLM client and graph database are properly configured and accessible.
    """

    subclasses = get_all_subclasses(ExtendableDataPoint)

    vector_index_collections = []

    for subclass in subclasses:
        index_fields = subclass.model_fields["metadata"].default.get("index_fields", [])
        for field_name in index_fields:
            vector_index_collections.append(f"{subclass.__name__}_{field_name}")

    found_triplets = await brute_force_triplet_search(
        query, top_k=5, collections=vector_index_collections or None
    )

    if len(found_triplets) == 0:
        raise NoRelevantDataFound

    args = {
        "question": query,
        "context": await retrieved_edges_to_summary(found_triplets),
    }
    user_prompt = render_prompt("graph_context_for_question.txt", args)
    system_prompt = read_query_prompt("answer_simple_question_restricted.txt")

    llm_client = get_llm_client()
    computed_answer = await llm_client.acreate_structured_output(
        text_input=user_prompt,
        system_prompt=system_prompt,
        response_model=str,
    )

    return [computed_answer]
