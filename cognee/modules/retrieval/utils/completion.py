import asyncio
from typing import Optional, Type, Any, List
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt, read_query_prompt


async def generate_completion(
    query: str,
    context: str,
    user_prompt_path: str,
    system_prompt_path: str,
    system_prompt: Optional[str] = None,
    conversation_history: Optional[str] = None,
    response_model: Type = str,
) -> Any:
    """Generates a completion using LLM with given context and prompts."""
    args = {"question": query, "context": context}
    user_prompt = render_prompt(user_prompt_path, args)
    system_prompt = system_prompt if system_prompt else read_query_prompt(system_prompt_path)

    if conversation_history:
        system_prompt = conversation_history + "\nTASK:" + system_prompt

    return await LLMGateway.acreate_structured_output(
        text_input=user_prompt,
        system_prompt=system_prompt,
        response_model=response_model,
    )


async def generate_completion_batch(
    query_batch: List[str],
    context: List[str],
    user_prompt_path: str,
    system_prompt_path: str,
    system_prompt: Optional[str] = None,
    conversation_history: Optional[str] = "",
    response_model: Type = str,
) -> List[Any]:
    """Generates completions for a batch of queries in parallel."""
    return await asyncio.gather(
        *[
            generate_completion(
                query=q,
                context=c,
                user_prompt_path=user_prompt_path,
                system_prompt_path=system_prompt_path,
                system_prompt=system_prompt,
                conversation_history=conversation_history,
                response_model=response_model,
            )
            for q, c in zip(query_batch, context)
        ]
    )


async def summarize_and_generate_completion(
    context: str,
    query: str,
    user_prompt_path: str,
    system_prompt_path: str,
    system_prompt: Optional[str] = None,
    conversation_history: Optional[str] = None,
    response_model: Type = str,
) -> tuple:
    """Summarizes context and generates completion in parallel. Returns (context_summary, completion)."""
    return await asyncio.gather(
        summarize_text(context),
        generate_completion(
            query=query,
            context=context,
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            response_model=response_model,
        ),
    )


async def batch_llm_completion(
    user_prompts: List[str],
    system_prompt: str,
    response_model: Type = str,
) -> List[Any]:
    """Run a batch of pre-built prompts through the LLM in parallel."""
    return list(
        await asyncio.gather(
            *[
                LLMGateway.acreate_structured_output(
                    text_input=prompt, system_prompt=system_prompt, response_model=response_model
                )
                for prompt in user_prompts
            ]
        )
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
