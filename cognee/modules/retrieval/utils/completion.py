from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt


async def generate_completion(
    query: str,
    context: str,
    user_prompt_path: str,
    system_prompt_path: str,
) -> str:
    """Generates a completion using LLM with given context and prompts."""
    args = {"question": query, "context": context}
    user_prompt = render_prompt(user_prompt_path, args)
    system_prompt = read_query_prompt(system_prompt_path)

    llm_client = get_llm_client()
    return await llm_client.acreate_structured_output(
        text_input=user_prompt,
        system_prompt=system_prompt,
        response_model=str,
    )


async def summarize_text(
    text: str,
    prompt_path: str = "summarize_search_results.txt",
) -> str:
    """Summarizes text using LLM with the specified prompt."""
    system_prompt = read_query_prompt(prompt_path)
    llm_client = get_llm_client()

    return await llm_client.acreate_structured_output(
        text_input=text,
        system_prompt=system_prompt,
        response_model=str,
    )
