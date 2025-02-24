import json
import os
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.tasks.completion.exceptions import NoRelevantDataFound
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt


async def query_completion(query: str, save_context_path: str = None) -> list:
    """

    Executes a query against a vector database and computes a relevant response using an LLM.

    Parameters:
    - query (str): The query string to compute.
    - save_context_path (str): The path to save the context.

    Returns:
    - list: Answer to the query.

    Notes:
    - Limits the search to the top 1 matching chunk for simplicity and relevance.
    - Ensure that the vector database and LLM client are properly configured and accessible.
    - The response model used for the LLM output is expected to be a string.

    """
    vector_engine = get_vector_engine()

    found_chunks = await vector_engine.search("DocumentChunk_text", query, limit=1)

    if len(found_chunks) == 0:
        raise NoRelevantDataFound

    # Get context and optionally dump it
    context = found_chunks[0].payload["text"]
    if save_context_path:
        try:
            os.makedirs(os.path.dirname(save_context_path), exist_ok=True)
            with open(save_context_path, "w", encoding="utf-8") as f:
                json.dump(context, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.error(f"Failed to save context to {save_context_path}: {str(e)}")
            # Continue execution as context saving is optional
    args = {
        "question": query,
        "context": context,
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
