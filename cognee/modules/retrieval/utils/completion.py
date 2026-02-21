import asyncio
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type

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


async def generate_session_completion_with_optional_summary(
    *,
    query: str,
    context: str,
    conversation_history: str,
    user_prompt_path: str,
    system_prompt_path: str,
    system_prompt: Optional[str] = None,
    response_model: Type = str,
    summarize_context: bool = False,
    run_feedback_detection: bool = False,
) -> Tuple[Any, str, Optional["FeedbackDetectionResult"]]:
    """
    Run LLM completion (and optionally summarization) for the session-manager flow.
    Returns (completion, context_to_store, feedback_result).
    When summarize_context is True, context_to_store is the summarized context; otherwise "".
    When run_feedback_detection is True, runs feedback detection in parallel; feedback_result
    is the detection result, otherwise None.
    """
    from cognee.infrastructure.session.feedback_detection import detect_feedback
    from cognee.infrastructure.session.feedback_models import FeedbackDetectionResult

    if summarize_context:
        if run_feedback_detection:
            context_summary, completion, feedback_result = await asyncio.gather(
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
                detect_feedback(query),
            )
            return (completion, context_summary, feedback_result)
        context_summary, completion = await asyncio.gather(
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
        return (completion, context_summary, None)

    if run_feedback_detection:
        completion, feedback_result = await asyncio.gather(
            generate_completion(
                query=query,
                context=context,
                user_prompt_path=user_prompt_path,
                system_prompt_path=system_prompt_path,
                system_prompt=system_prompt,
                conversation_history=conversation_history,
                response_model=response_model,
            ),
            detect_feedback(query),
        )
        return (completion, "", feedback_result)
    completion = await generate_completion(
        query=query,
        context=context,
        user_prompt_path=user_prompt_path,
        system_prompt_path=system_prompt_path,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        response_model=response_model,
    )
    return (completion, "", None)


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
