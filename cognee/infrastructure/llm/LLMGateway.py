import logging
from collections.abc import Coroutine
from typing import Any, TypeVar

import litellm
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_delay,
    wait_exponential_jitter,
)

from cognee.infrastructure.llm import get_llm_config
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.types import (
    TranscriptionReturnType,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger()

T = TypeVar("T", bound="BaseModel | str")


def _inject_agent_memory(text_input: str) -> str:
    from cognee.modules.agent_memory import get_current_agent_memory_context

    context = get_current_agent_memory_context()
    if context is None or not context.memory_context:
        return text_input

    return f"Additional Memory Context:\n{context.memory_context}\n\nOriginal Input:\n{text_input}"


@retry(
    stop=stop_after_delay(128),
    wait=wait_exponential_jitter(8, 128),
    retry=retry_if_not_exception_type(
        (litellm.exceptions.NotFoundError, litellm.exceptions.AuthenticationError)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def direct_str_completion(text_input: str, system_prompt: str, **kwargs: Any) -> str:
    """Call litellm directly for plain-text completions, bypassing instructor.

    Instructor wraps the call in JSON/tool-call schemas that local
    llama.cpp-compatible servers don't honour, causing repeated
    InstructorRetryException and tenacity retry sleeps.
    """
    from cognee.shared.rate_limiting import llm_rate_limiter_context_manager

    llm_config = get_llm_config()
    merged_kwargs = {**(llm_config.llm_args or {}), **kwargs}

    async with llm_rate_limiter_context_manager():
        resp = await litellm.acompletion(
            model=llm_config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_input},
            ],
            api_key=llm_config.llm_api_key,
            api_base=llm_config.llm_endpoint,
            api_version=llm_config.llm_api_version,
            **merged_kwargs,
        )
    return resp.choices[0].message.content or ""


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

        # response_model=str is a plain-text completion — instructor adds
        # JSON/tool-call schemas that local servers don't honour, causing
        # repeated InstructorRetryException, so call litellm directly instead.
        # BAML handles str natively via its own configured client, so only
        # short-circuit the instructor path here.
        if response_model is str and llm_config.structured_output_framework.upper() != "BAML":
            return _record_session_usage_after(
                direct_str_completion(text_input, system_prompt, **kwargs),
                text_input=text_input,
            )

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
