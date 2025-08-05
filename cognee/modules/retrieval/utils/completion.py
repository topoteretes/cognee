from cognee.infrastructure.llm.LLMAdapter import LLMAdapter


async def generate_completion(
    query: str,
    context: str,
    user_prompt_path: str,
    system_prompt_path: str,
) -> str:
    """Generates a completion using LLM with given context and prompts."""
    args = {"question": query, "context": context}
    user_prompt = LLMAdapter.render_prompt(user_prompt_path, args)
    system_prompt = LLMAdapter.read_query_prompt(system_prompt_path)

    return await LLMAdapter.acreate_structured_output(
        text_input=user_prompt,
        system_prompt=system_prompt,
        response_model=str,
    )


async def summarize_text(
    text: str,
    prompt_path: str = "summarize_search_results.txt",
) -> str:
    """Summarizes text using LLM with the specified prompt."""
    system_prompt = LLMAdapter.read_query_prompt(prompt_path)

    return await LLMAdapter.acreate_structured_output(
        text_input=text,
        system_prompt=system_prompt,
        response_model=str,
    )
