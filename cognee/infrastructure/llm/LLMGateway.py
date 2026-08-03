import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar
from weakref import WeakKeyDictionary

from pydantic import BaseModel

from cognee.infrastructure.llm import get_llm_config
from cognee.infrastructure.llm.config import get_llm_context_config
from cognee.infrastructure.llm.retry_config import raise_if_quota_error
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.types import (
    TranscriptionReturnType,
)

T = TypeVar("T", bound="BaseModel | str")


def _inject_agent_memory(text_input: str) -> str:
    from cognee.modules.agent_memory import get_current_agent_memory_context

    context = get_current_agent_memory_context()
    if context is None or not context.memory_context:
        return text_input

    return f"Additional Memory Context:\n{context.memory_context}\n\nOriginal Input:\n{text_input}"


async def _record_session_usage_after(
    coro: Coroutine,
    *,
    text_input: str,
) -> T:
    """Run the LLM coroutine, then (best-effort) accumulate usage onto
    any active session tracker. Failures never propagate — usage
    accounting is strictly auxiliary.
    """
    result = await coro
    try:
        from cognee.modules.session_lifecycle.usage_tracking import record_llm_call

        if isinstance(result, BaseModel):
            output_repr = result.model_dump_json()
        else:
            output_repr = str(result)
        model = get_llm_context_config().llm_model
        await record_llm_call(
            input_text=text_input,
            output_text=output_repr,
            model=model,
        )
    except Exception:
        pass
    return result


async def _fail_fast_on_quota(coro: Coroutine) -> T:
    """Convert provider quota/billing exhaustion into ``LLMQuotaExceededError``.

    Runs at the single choke point every structured-output call flows through,
    so it is provider- and framework-agnostic.
    """
    try:
        return await coro
    except Exception as error:
        raise_if_quota_error(error)
        raise


# Per-event-loop concurrency semaphores. Keyed by a WEAK reference to the loop
# object (not id(loop)): CPython reuses object addresses, so a fresh loop could get a
# GC'd loop's id and be handed a semaphore bound to the dead loop (RuntimeError on
# await) — a real hazard for the asyncio.run-per-call test pattern. A WeakKeyDictionary
# keys on identity and drops entries when the loop is collected, so no stale reuse.
# Recreated when the configured limit changes.
_concurrency_semaphores: "WeakKeyDictionary[asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]]" = (
    WeakKeyDictionary()
)


def _concurrency_semaphore(limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    cached = _concurrency_semaphores.get(loop)
    if cached is None or cached[0] != limit:
        semaphore = asyncio.Semaphore(limit)
        _concurrency_semaphores[loop] = (limit, semaphore)
        return semaphore
    return cached[1]


async def _limit_concurrency(coro: Coroutine) -> T:
    """Bound global in-flight structured-output concurrency (CLO-409 Phase 0b).

    ``acreate_structured_output`` is the single choke point every structured-output
    call flows through, so acquiring one semaphore here caps the whole engine's
    concurrent LLM fan-out — bounding the burst that lets a cognify run overshoot a
    LiteLLM key's async budget cap. A no-op when ``llm_max_concurrent_requests <= 0``
    (OSS default), so behavior is unchanged unless a deployment opts in.
    """
    limit = get_llm_config().llm_max_concurrent_requests
    if limit <= 0:
        return await coro
    async with _concurrency_semaphore(limit):
        return await coro


class LLMGateway:
    """
    Class handles selection of structured output frameworks and LLM functions.
    Class used as a namespace for LLM related functions, should not be instantiated, all methods are static.
    """

    @staticmethod
    def acreate_structured_output(
        text_input: str,
        system_prompt: str,
        response_model: type[T],
        **kwargs: Any,
    ) -> Coroutine[Any, Any, T]:
        text_input = _inject_agent_memory(text_input)
        llm_config = get_llm_config()
        if llm_config.structured_output_framework.upper() == "BAML":
            from cognee.infrastructure.llm.structured_output_framework.baml.baml_src.extraction import (
                acreate_structured_output,
            )

            inner = acreate_structured_output(
                text_input=text_input,
                system_prompt=system_prompt,
                response_model=response_model,
            )
        elif llm_config.structured_output_framework.upper() == "LITELLM_NATIVE":
            from cognee.infrastructure.llm.structured_output_framework.litellm_native.get_native_client import (
                get_native_client,
            )

            llm_client = get_native_client()
            inner = llm_client.acreate_structured_output(
                text_input=text_input,
                system_prompt=system_prompt,
                response_model=response_model,
                **kwargs,
            )
        else:
            from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
                get_llm_client,
            )

            llm_client = get_llm_client()
            inner = llm_client.acreate_structured_output(
                text_input=text_input,
                system_prompt=system_prompt,
                response_model=response_model,
                **kwargs,
            )

        # Wrap so usage is recorded against any active session tracker (no-op when no
        # tracker is installed), then bound global concurrency at this single choke
        # point (no-op unless llm_max_concurrent_requests is set).
        return _limit_concurrency(
            _fail_fast_on_quota(_record_session_usage_after(inner, text_input=text_input))
        )

    @staticmethod
    def create_transcript(input, **kwargs) -> Coroutine[Any, Any, TranscriptionReturnType | None]:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
            get_llm_client,
        )

        llm_client = get_llm_client()
        return llm_client.create_transcript(input=input, **kwargs)

    @staticmethod
    def transcribe_image(input: str) -> Coroutine[Any, Any, Any]:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
            get_llm_client,
        )

        llm_client = get_llm_client()
        return llm_client.transcribe_image(input=input)
