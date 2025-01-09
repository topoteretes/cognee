from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.tasks.completion.exceptions import NoRelevantDataFound
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.modules.retrieval.brute_force_triplet_search import brute_force_triplet_search


def retrieved_edges_to_string(retrieved_edges: list) -> str:
    edge_strings = []
    for edge in retrieved_edges:
        node1_string = edge.node1.attributes.get("text") or edge.node1.attributes.get("name")
        node2_string = edge.node2.attributes.get("text") or edge.node2.attributes.get("name")
        edge_string = edge.attributes["relationship_type"]
        edge_str = f"{node1_string} -- {edge_string} -- {node2_string}"
        edge_strings.append(edge_str)
    return "\n---\n".join(edge_strings)


async def graph_query_completion(query: str) -> list:
    """
    Parameters:
    - query (str): The query string to compute.

    Returns:
    - list: Answer to the query.
    """
    found_triplets = await brute_force_triplet_search(query, top_k=5)

    if len(found_triplets) == 0:
        raise NoRelevantDataFound

    args = {
        "question": query,
        "context": retrieved_edges_to_string(found_triplets),
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
