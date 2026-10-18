import asyncio
from typing import Any, NamedTuple

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.pipeline_stage import pipeline_stage
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.infrastructure.llm.streaming.token_sink import answer_scope
from cognee.modules.observability import COGNEE_RESULT_SUMMARY, new_span


class SessionPrompt(NamedTuple):
    """The session layer of a completion: the conversation history and the guidance
    block (session guidance, or the durable preference block when sessionless). Either
    may be empty; ``SessionPrompt()`` is the sessionless, preference-free layer.
    """

    history: str = ""
    guidance: str = ""


def build_completion_prompts(
    *,
    query: str,
    context: Any,
    user_prompt_path: str,
    system_prompt_path: str,
    system_prompt: str | None = None,
    session: SessionPrompt | None = None,
) -> tuple[str, str]:
    """Assemble the exact ``(user_prompt, system_prompt)`` pair a completion sends.

    The system prompt is the retriever's task template, nothing else. The user prompt is
    the session history, then the rendered question and context, then the guidance block
    last; empty layers are skipped. The order is measured, not chosen by taste: see
    "Completion prompt layout" in CLAUDE.md before moving anything.

    Pure and side-effect free. ``generate_completion`` builds its prompts here, and so
    does the ``only_context`` preview, so a caller asking what the LLM *would* receive
    gets the real strings instead of a reconstruction that drifts as templates change.
    """
    rendered = render_prompt(user_prompt_path, {"question": query, "context": context})
    resolved_system_prompt = (
        system_prompt if system_prompt else read_query_prompt(system_prompt_path)
    )
    if resolved_system_prompt is None:
        # read_query_prompt logs and returns None on a missing file. Surfacing that here
        # keeps the Tuple[str, str] contract honest and stops a typo in a prompt path
        # from masquerading as "this retriever has no prompt template" downstream.
        raise FileNotFoundError(f"System prompt template {system_prompt_path!r} could not be read.")

    session = session or SessionPrompt()
    user_prompt = "\n\n".join(filter(None, (session.history, rendered, session.guidance)))
    return user_prompt, resolved_system_prompt


async def generate_completion(
    query: str,
    context: str,
    user_prompt_path: str,
    system_prompt_path: str,
    system_prompt: str | None = None,
    session: SessionPrompt | None = None,
    response_model: type = str,
) -> Any:
    """Generates a completion using LLM with given context and prompts."""
    user_prompt, system_prompt = build_completion_prompts(
        query=query,
        context=context,
        user_prompt_path=user_prompt_path,
        system_prompt_path=system_prompt_path,
        system_prompt=system_prompt,
        session=session,
    )

    with pipeline_stage("query"), new_span("cognee.llm.completion") as span:
        span.set_attribute("cognee.llm.prompt_path", system_prompt_path)
        span.set_attribute("cognee.llm.context_length", len(context))
        span.set_attribute("cognee.llm.query_length", len(query))
        result = await LLMGateway.acreate_structured_output(
            text_input=user_prompt,
            system_prompt=system_prompt,
            response_model=response_model,
        )
        if isinstance(result, str):
            span.set_attribute("cognee.llm.response_length", len(result))
        span.set_attribute(COGNEE_RESULT_SUMMARY, "LLM completion generated")
        return result


async def generate_answer(
    query: str,
    context: str,
    user_prompt_path: str,
    system_prompt_path: str,
    system_prompt: str | None = None,
    session: SessionPrompt | None = None,
    response_model: type = str,
) -> Any:
    """The one completion a listening client may watch.

    Identical to :func:`generate_completion` except that a client streaming this
    request receives this call's tokens as they are produced. That is the whole
    difference, and it is why the choice is a function name rather than a flag:
    every other completion in a request — turn analysis, summarisation, subquery
    answers, agentic steps — calls ``generate_completion`` and therefore can
    never take the stream, without any of them knowing streaming exists.

    Callers do not need a sink, a session, or the feature enabled. With none of
    those the call behaves exactly like ``generate_completion``.
    """
    # A structured response_model never reaches the adapters' streaming path, so
    # say so here rather than announcing a stream that emits nothing. This is the
    # natural home for the check: generate_answer is the only place that holds
    # both the response model and the decision to stream.
    async with answer_scope(stage="generating", can_stream=response_model is str):
        return await generate_completion(
            query=query,
            context=context,
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
            system_prompt=system_prompt,
            session=session,
            response_model=response_model,
        )


async def generate_completion_batch(
    query_batch: list[str],
    context: list[str],
    user_prompt_path: str,
    system_prompt_path: str,
    system_prompt: str | None = None,
    session: SessionPrompt | None = None,
    response_model: type = str,
) -> list[Any]:
    """Generates completions for a batch of queries in parallel."""
    return await asyncio.gather(
        *[
            generate_completion(
                query=q,
                context=c,
                user_prompt_path=user_prompt_path,
                system_prompt_path=system_prompt_path,
                system_prompt=system_prompt,
                session=session,
                response_model=response_model,
            )
            for q, c in zip(query_batch, context)
        ]
    )


async def generate_session_completion_with_optional_summary(
    *,
    query: str,
    context: str,
    session: SessionPrompt | None,
    user_prompt_path: str,
    system_prompt_path: str,
    system_prompt: str | None = None,
    response_model: type = str,
    summarize_context: bool = False,
) -> tuple[Any, str, Any]:
    """
    Run LLM completion (and optionally summarization) for the session-manager flow.
    Returns (completion, context_to_store, feedback_result).
    When summarize_context is True, context_to_store is the summarized context; otherwise "".
    Feedback analysis runs before retrieval/generation in SessionManager.prepare_session_turn.
    """
    if summarize_context:
        context_summary, completion = await asyncio.gather(
            summarize_text(context),
            generate_completion(
                query=query,
                context=context,
                user_prompt_path=user_prompt_path,
                system_prompt_path=system_prompt_path,
                system_prompt=system_prompt,
                session=session,
                response_model=response_model,
            ),
        )
        return (completion, context_summary, None)

    completion = await generate_completion(
        query=query,
        context=context,
        user_prompt_path=user_prompt_path,
        system_prompt_path=system_prompt_path,
        system_prompt=system_prompt,
        session=session,
        response_model=response_model,
    )
    return (completion, "", None)


async def batch_llm_completion(
    user_prompts: list[str],
    system_prompt: str,
    response_model: type = str,
) -> list[Any]:
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
    system_prompt: str | None = None,
) -> str:
    """Summarizes text using LLM with the specified prompt."""
    system_prompt = system_prompt if system_prompt else read_query_prompt(system_prompt_path)

    with new_span("cognee.llm.summarize") as span:
        span.set_attribute("cognee.llm.input_length", len(text))
        result = await LLMGateway.acreate_structured_output(
            text_input=text,
            system_prompt=system_prompt,
            response_model=str,
        )
        if isinstance(result, str):
            span.set_attribute("cognee.llm.response_length", len(result))
        span.set_attribute(COGNEE_RESULT_SUMMARY, "Text summarized")
        return result
