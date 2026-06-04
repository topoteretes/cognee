import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

from pydantic import BaseModel

from cognee.infrastructure.llm import get_llm_config
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


async def _with_call_timeout(coro: Coroutine) -> T:
    """Bound a single LLM structured-output call so a non-responsive endpoint
    cannot hang cognify/search/remember indefinitely.

    A non-positive configured timeout disables the bound (back-compat escape hatch).
    """
    # Defensive getattr: callers (and tests) may supply a partial/mock config
    # that predates this field; fall back to the LLMConfig default (300s).
    timeout = getattr(get_llm_config(), "llm_call_timeout_seconds", 300)
    if not timeout or timeout <= 0:
        return await coro
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError as error:
        raise TimeoutError(
            f"LLM call did not complete within {timeout}s. The endpoint accepted the "
            "connection but never responded. Check that your LLM endpoint is reachable "
            "and responsive, or raise LLM_CALL_TIMEOUT_SECONDS."
        ) from error


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
        model = get_llm_config().llm_model
        await record_llm_call(
            input_text=text_input,
            output_text=output_repr,
            model=model,
        )
    except Exception:
        pass
    return result


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

        # Bound the call so an unresponsive endpoint cannot hang forever, then
        # wrap so usage is recorded against any active session tracker.
        # The usage wrapper is a no-op when no tracker is installed.
        return _record_session_usage_after(_with_call_timeout(inner), text_input=text_input)

    @staticmethod
    def create_transcript(input) -> Coroutine[Any, Any, TranscriptionReturnType | None]:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
            get_llm_client,
        )

        llm_client = get_llm_client()
        return llm_client.create_transcript(input=input)

    @staticmethod
    def transcribe_image(input: str) -> Coroutine[Any, Any, Any]:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
            get_llm_client,
        )

        llm_client = get_llm_client()
        return llm_client.transcribe_image(input=input)
