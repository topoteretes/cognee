from cognee.infrastructure.llm.LLMGateway import LLMGateway


async def generate_completion(
    query: str,
    context: str,
    user_prompt_path: str,
    system_prompt_path: str,
    user_prompt: str = None,
    system_prompt: str = None,
    only_context: bool = False,
) -> str:
    """Generates a completion using LLM with given context and prompts."""
    args = {"question": query, "context": context}
    user_prompt = LLMGateway.render_prompt(user_prompt if user_prompt else user_prompt_path, args)
    system_prompt = LLMGateway.read_query_prompt(
        system_prompt if system_prompt else system_prompt_path
    )

    if only_context:
        return context
    else:
        return await LLMGateway.acreate_structured_output(
            text_input=user_prompt,
            system_prompt=system_prompt,
            response_model=str,
        )


async def summarize_text(
    text: str,
    prompt_path: str = "summarize_search_results.txt",
) -> str:
    """Summarizes text using LLM with the specified prompt."""
    system_prompt = LLMGateway.read_query_prompt(prompt_path)

    return await LLMGateway.acreate_structured_output(
        text_input=text,
        system_prompt=system_prompt,
        response_model=str,
    )
