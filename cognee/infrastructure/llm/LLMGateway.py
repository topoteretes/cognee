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


async def _record_session_usage_after(
    coro: Coroutine,
    *,
    text_input: str,
) -> T:
    """Run the LLM coroutine, then (best-effort) accumulate usage onto
    any active session tracker. Failures never propagate — usage
    accounting is strictly auxiliary.
    """
    from cognee.modules.session_lifecycle.usage_tracking import (
        pop_last_llm_usage,
        record_llm_call,
    )

    try:
        result = await coro
    except Exception:
        # Discard any usage an adapter captured before raising, so it can't
        # be misattributed to the next unrelated call in this context.
        pop_last_llm_usage()
        raise

    try:
        usage = pop_last_llm_usage()
        if isinstance(result, BaseModel):
            output_repr = result.model_dump_json()
        else:
            output_repr = str(result)
        model = get_llm_config().llm_model
        await record_llm_call(
            input_text=text_input,
            output_text=output_repr,
            model=model,
            tokens_in_override=usage[0] if usage is not None else None,
            tokens_out_override=usage[1] if usage is not None else None,
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

        # Wrap so usage is recorded against any active session tracker.
        # No-op when no tracker is installed.
        return _record_session_usage_after(inner, text_input=text_input)

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
