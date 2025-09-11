from typing import Optional
from cognee.infrastructure.llm.LLMGateway import LLMGateway


async def generate_completion(
    query: str,
    context: str,
    user_prompt_path: str,
    system_prompt_path: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Generates a completion using LLM with given context and prompts."""
    args = {"question": query, "context": context}
    user_prompt = LLMGateway.render_prompt(user_prompt_path, args)
    system_prompt = (
        system_prompt if system_prompt else LLMGateway.read_query_prompt(system_prompt_path)
    )

    return await LLMGateway.acreate_structured_output(
        text_input=user_prompt,
        system_prompt=system_prompt,
        response_model=str,
    )


async def summarize_text(
    text: str,
    system_prompt_path: str = "summarize_search_results.txt",
    system_prompt: str = None,
) -> str:
    """Summarizes text using LLM with the specified prompt."""
    system_prompt = (
        system_prompt if system_prompt else LLMGateway.read_query_prompt(system_prompt_path)
    )

    return await LLMGateway.acreate_structured_output(
        text_input=text,
        system_prompt=system_prompt,
        response_model=str,
    )
