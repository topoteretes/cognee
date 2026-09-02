from collections.abc import Coroutine
from typing import Any, Optional, TypeVar

from pydantic import BaseModel

from cognee.infrastructure.llm import get_llm_config
from cognee.infrastructure.llm.config import get_llm_context_config
from cognee.infrastructure.llm.exceptions import raise_if_budget_exhausted
from cognee.infrastructure.llm.retry_config import raise_if_quota_error
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.types import (
    TranscriptionReturnType,
)
from cognee.shared.logging_utils import get_logger

T = TypeVar("T", bound="BaseModel | str")


def _inject_agent_memory(text_input: str) -> str:
    from cognee.modules.agent_memory import get_current_agent_memory_context

    context = get_current_agent_memory_context()
    if context is None or not context.memory_context:
        return text_input

    return f"Additional Memory Context:\n{context.memory_context}\n\nOriginal Input:\n{text_input}"


def _exact_usage_from_result(result: Any) -> tuple[Optional[int], Optional[int]]:
    """Real prompt/completion token counts from the raw provider response —
    (None, None) otherwise, so the caller falls back to its char-based
    estimate.

    The structured-output adapters attach the raw litellm response to the
    parsed model as ``_raw_response``: litellm_native (the default) does it
    explicitly (``_attach_raw_response``), and instructor does it internally
    (``instructor.processing.response.process_response``). That raw response
    carries ``.usage`` with the provider-billed counts, including hidden
    reasoning tokens no text estimate can see.

    Returns (None, None) for the plain-string path (returns a bare ``str``,
    nothing to attach to) and for BAML, which doesn't go through litellm.
    """
    usage = getattr(getattr(result, "_raw_response", None), "usage", None)
    if usage is None:
        return None, None
    tokens_in = getattr(usage, "prompt_tokens", None)
    tokens_out = getattr(usage, "completion_tokens", None)
    if tokens_in is None or tokens_out is None:
        return None, None
    return int(tokens_in), int(tokens_out)


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
        tokens_in, tokens_out = _exact_usage_from_result(result)
        await record_llm_call(
            input_text=text_input,
            output_text=output_repr,
            model=model,
            tokens_in_override=tokens_in,
            tokens_out_override=tokens_out,
        )
    except Exception:
        pass
    return result


async def _fail_fast_on_quota(coro: Coroutine) -> T:
    """Convert provider quota/billing exhaustion into an actionable error.

    Runs at the single choke point every structured-output call flows through,
    so it is provider- and framework-agnostic.

    Budget exhaustion is checked first, and here rather than only in the
    adapters: each instructor adapter catches ``InstructorRetryException`` in a
    clause that precedes its own budget handler, and the plain-text path
    (``response_model is str``) bypasses those handlers altogether. Classifying
    at this choke point gives every adapter and both paths the same 402.
    """
    try:
        return await coro
    except Exception as error:
        raise_if_budget_exhausted(error)
        raise_if_quota_error(error)
        raise


logger = get_logger("LLMGateway")


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

        # Wrap so usage is recorded against any active session tracker.
        # No-op when no tracker is installed.
        return _fail_fast_on_quota(_record_session_usage_after(inner, text_input=text_input))

    @staticmethod
    def supports_answer_streaming() -> bool:
        """Whether the configured client can stream a plain-text answer.

        Resolved through the same framework/provider dispatch a real call takes,
        because that dispatch is what decides which adapter answers — a hook
        present on one combination is absent on every other. Never raises: a
        deployment whose client cannot even be constructed simply does not
        stream, which is the safe direction.
        """
        llm_config = get_llm_config()
        framework = llm_config.structured_output_framework.upper()
        try:
            if framework == "BAML":
                # The BAML path bypasses the adapters entirely.
                return False
            if framework == "LITELLM_NATIVE":
                from cognee.infrastructure.llm.structured_output_framework.litellm_native.get_native_client import (
                    get_native_client,
                )

                return bool(getattr(get_native_client(), "supports_answer_streaming", False))
            from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
                get_llm_client,
            )

            return bool(getattr(get_llm_client(), "supports_answer_streaming", False))
        except Exception:  # noqa: BLE001 - a capability probe must not break a request
            logger.debug("Could not resolve answer-streaming support", exc_info=True)
            return False

    @staticmethod
    def create_transcript(input, **kwargs) -> Coroutine[Any, Any, TranscriptionReturnType | None]:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
            get_llm_client,
        )

        llm_client = get_llm_client()
        return llm_client.create_transcript(input=input, **kwargs)

    @staticmethod
    def transcribe_image(
        input: str,
        prompt: str | None = None,
        max_completion_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> Coroutine[Any, Any, Any]:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
            get_llm_client,
        )

        llm_client = get_llm_client()
        return llm_client.transcribe_image(
            input=input,
            prompt=prompt,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
        )
