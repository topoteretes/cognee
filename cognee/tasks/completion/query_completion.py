from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.tasks.completion.exceptions import NoRelevantDataFound
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt


async def query_completion(query: str) -> list:
    """
    Parameters:
    - query (str): The query string to compute.

    Returns:
    - list: Answer to the query.
    """
    vector_engine = get_vector_engine()

    found_chunks = await vector_engine.search("document_chunk_text", query, limit=1)

    if len(found_chunks) == 0:
        raise NoRelevantDataFound

    args = {
        "question": query,
        "context": found_chunks[0].payload["text"],
    }
    user_prompt = render_prompt("context_for_question.txt", args)
    system_prompt = read_query_prompt("answer_simple_question.txt")

    llm_client = get_llm_client()
    computed_answer = await llm_client.acreate_structured_output(
        text_input=user_prompt,
        system_prompt=system_prompt,
        response_model=str,
    )

    return [computed_answer]
