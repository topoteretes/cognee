from typing import Optional, Type, Any
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt, read_query_prompt


async def generate_structured_completion(
    query: str,
    context: str,
    user_prompt_path: str,
    system_prompt_path: str,
    system_prompt: Optional[str] = None,
    conversation_history: Optional[str] = None,
    response_model: Type = str,
) -> Any:
    """Generates a structured completion using LLM with given context and prompts."""
    args = {"question": query, "context": context}
    user_prompt = render_prompt(user_prompt_path, args)
    system_prompt = system_prompt if system_prompt else read_query_prompt(system_prompt_path)

    if conversation_history:
        #:TODO: I would separate the history and put it into the system prompt but we have to test what works best with longer convos
        system_prompt = conversation_history + "\nTASK:" + system_prompt

    return await LLMGateway.acreate_structured_output(
        text_input=user_prompt,
        system_prompt=system_prompt,
        response_model=response_model,
    )


async def generate_completion(
    query: str,
    context: str,
    user_prompt_path: str,
    system_prompt_path: str,
    system_prompt: Optional[str] = None,
    conversation_history: Optional[str] = None,
) -> str:
    """Generates a completion using LLM with given context and prompts."""
    return await generate_structured_completion(
        query=query,
        context=context,
        user_prompt_path=user_prompt_path,
        system_prompt_path=system_prompt_path,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        response_model=str,
    )


async def summarize_text(
    text: str,
    system_prompt_path: str = "summarize_search_results.txt",
    system_prompt: str = None,
) -> str:
    """Summarizes text using LLM with the specified prompt."""
    system_prompt = system_prompt if system_prompt else read_query_prompt(system_prompt_path)

    return await LLMGateway.acreate_structured_output(
        text_input=text,
        system_prompt=system_prompt,
        response_model=str,
    )
