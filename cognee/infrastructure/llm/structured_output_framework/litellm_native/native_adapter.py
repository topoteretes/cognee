"""Universal LiteLLM-native adapter for structured output.

Implements ``NativeLiteLLMAdapter`` — a single adapter that works with every
provider LiteLLM supports, using LiteLLM's own ``response_format`` to obtain
validated Pydantic objects **without** the ``instructor`` library.

Two paths, chosen per model via ``litellm.supports_response_schema``:

* *Schema-native* (OpenAI, Azure, Gemini, Mistral, Bedrock, …): the Pydantic
  class is passed as ``response_format`` and the JSON response is validated.
* *JSON-object fallback* (Ollama, llama.cpp, custom endpoints): asks for a JSON
  object, injects the schema into the prompt, validates, and on failure feeds
  the validation error back so the model can self-correct.

Rate-limit, auth, and budget errors propagate immediately; content-policy
violations fall back to the configured fallback model. This file never imports
``instructor``.
"""

import json
import logging
from typing import Any, TypeVar

import litellm
from litellm.exceptions import ContentPolicyViolationError
from pydantic import BaseModel, ValidationError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    wait_exponential_jitter,
)

from cognee.infrastructure.llm.exceptions import (
    ContentPolicyFilterError,
    LLMPaymentRequiredError,
    is_budget_exhausted_error,
)
from cognee.infrastructure.llm.retry_config import llm_retry_stop_condition
from cognee.modules.observability.get_observe import get_observe
from cognee.shared.logging_utils import get_logger
from cognee.shared.rate_limiting import llm_rate_limiter_context_manager

T = TypeVar("T", bound="BaseModel | str")

logger = get_logger()
observe = get_observe()

# Max self-correction attempts when a json-object provider returns JSON that
# fails validation. Separate from the tenacity retry (transient HTTP errors).
_MAX_VALIDATION_RETRIES: int = 3


def _supports_native_schema(model_name: str) -> bool:
    """Whether *model_name* can enforce a Pydantic schema via ``response_format``.

    Delegates to LiteLLM's own capability table so routing stays correct as
    providers gain support. Unknown models (and any lookup error) default to the
    json-object fallback, which works everywhere.
    """
    try:
        return bool(litellm.supports_response_schema(model=model_name))
    except Exception:
        return False


def _enrich_llm_span(model: str, name: str) -> None:
    """Set LLM attributes on the current OTEL span, if tracing is enabled.

    Mirrors the helper in ``generic_llm_api/adapter.py``.
    """
    from cognee.modules.observability.trace_context import is_tracing_enabled

    if not is_tracing_enabled():
        return

    try:
        from opentelemetry import trace as otel_trace  # ty:ignore[unresolved-import]

        from cognee.modules.observability.tracing import COGNEE_LLM_MODEL, COGNEE_LLM_PROVIDER

        current_span = otel_trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute(COGNEE_LLM_MODEL, model)
            current_span.set_attribute(COGNEE_LLM_PROVIDER, name)
    except Exception:
        pass


class NativeLiteLLMAdapter:
    """Structured output via LiteLLM's native ``response_format`` (no instructor).

    One class handles every provider. The connection params for a given call are
    passed through the private helpers rather than stored per call, so a single
    (cached) instance is safe to share across concurrent calls — including the
    content-policy fallback path, which uses different model/key/endpoint values.

    Instance variables:
        - model, api_key, endpoint, api_version, max_completion_tokens,
          fallback_model, fallback_api_key, fallback_endpoint, llm_args, name
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        max_completion_tokens: int,
        name: str = "LiteLLM-Native",
        endpoint: str | None = None,
        api_version: str | None = None,
        fallback_model: str | None = None,
        fallback_api_key: str | None = None,
        fallback_endpoint: str | None = None,
        llm_args: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.api_key = api_key
        self.api_version = api_version
        self.endpoint = endpoint
        self.max_completion_tokens = max_completion_tokens
        self.fallback_model = fallback_model
        self.fallback_api_key = fallback_api_key
        self.fallback_endpoint = fallback_endpoint
        self.llm_args: dict[str, Any] = llm_args or {}

    async def _acreate_str_output(
        self,
        text_input: str,
        system_prompt: str,
        *,
        model: str,
        api_key: str | None,
        endpoint: str | None,
        api_version: str | None,
        **merged_kwargs: Any,
    ) -> str:
        """Plain-text completion without any schema (mirrors GenericAPIAdapter)."""
        async with llm_rate_limiter_context_manager():
            response = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_input},
                ],
                api_key=api_key,
                api_base=endpoint,
                api_version=api_version,
                **merged_kwargs,
            )
        return response.choices[0].message.content or ""

    async def _acreate_schema_native(
        self,
        text_input: str,
        system_prompt: str,
        response_model: type[BaseModel],
        *,
        model: str,
        api_key: str | None,
        endpoint: str | None,
        api_version: str | None,
        **merged_kwargs: Any,
    ) -> BaseModel:
        """Pass the Pydantic model as ``response_format`` and validate the JSON."""
        async with llm_rate_limiter_context_manager():
            response = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_input},
                ],
                response_format=response_model,
                api_key=api_key,
                api_base=endpoint,
                api_version=api_version,
                **merged_kwargs,
            )
        raw_content = response.choices[0].message.content or "{}"
        return response_model.model_validate_json(raw_content)

    async def _acreate_json_fallback(
        self,
        text_input: str,
        system_prompt: str,
        response_model: type[BaseModel],
        *,
        model: str,
        api_key: str | None,
        endpoint: str | None,
        api_version: str | None,
        **merged_kwargs: Any,
    ) -> BaseModel:
        """Ask for a JSON object with the schema injected into the prompt.

        On a validation failure we retry up to ``_MAX_VALIDATION_RETRIES`` times,
        feeding the error back so the model can self-correct.
        """
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        augmented_system_prompt = (
            f"{system_prompt}\n\n"
            f"You MUST respond with valid JSON conforming to this schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Do not include any text outside the JSON object."
        )

        last_error: Exception | None = None
        for attempt in range(_MAX_VALIDATION_RETRIES):
            user_content = text_input
            if last_error is not None:
                user_content = (
                    f"{text_input}\n\n"
                    f"Your previous response failed validation with this error:\n"
                    f"{last_error}\n\n"
                    f"Please fix the JSON and try again."
                )

            async with llm_rate_limiter_context_manager():
                response = await litellm.acompletion(
                    model=model,
                    messages=[
                        {"role": "system", "content": augmented_system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                    api_key=api_key,
                    api_base=endpoint,
                    api_version=api_version,
                    **merged_kwargs,
                )

            raw_content = response.choices[0].message.content or "{}"
            try:
                return response_model.model_validate_json(raw_content)
            except (ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "litellm_native validation retry",
                    attempt=attempt + 1,
                    max_retries=_MAX_VALIDATION_RETRIES,
                    error=str(exc),
                )

        # All self-correction attempts exhausted — surface the last error.
        raise last_error  # type: ignore[misc]

    async def _acreate_structured(
        self,
        text_input: str,
        system_prompt: str,
        response_model: type[BaseModel],
        *,
        model: str,
        api_key: str | None,
        endpoint: str | None,
        api_version: str | None,
        **merged_kwargs: Any,
    ) -> BaseModel:
        """Route to the schema-native or json-object path based on the model."""
        if _supports_native_schema(model):
            return await self._acreate_schema_native(
                text_input,
                system_prompt,
                response_model,
                model=model,
                api_key=api_key,
                endpoint=endpoint,
                api_version=api_version,
                **merged_kwargs,
            )
        return await self._acreate_json_fallback(
            text_input,
            system_prompt,
            response_model,
            model=model,
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version,
            **merged_kwargs,
        )

    @observe(as_type="generation")
    @retry(
        stop=llm_retry_stop_condition,
        wait=wait_exponential_jitter(8, 128),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.NotFoundError,
                litellm.exceptions.AuthenticationError,
                # Rate-limit and budget errors are non-retryable — retrying just
                # burns the remaining quota.
                litellm.exceptions.RateLimitError,
                LLMPaymentRequiredError,
            )
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def acreate_structured_output(
        self,
        text_input: str,
        system_prompt: str,
        response_model: type[T],
        **kwargs: Any,
    ) -> T:
        """Return a validated instance of *response_model* (or a plain ``str``).

        Rate-limit / auth errors raise immediately; a budget/402 error is
        surfaced as ``LLMPaymentRequiredError`` (also non-retryable); a
        content-policy violation retries once on the fallback model before
        raising ``ContentPolicyFilterError``.
        """
        merged_kwargs = {**self.llm_args, **kwargs}

        # A plain string needs no schema — skip structured output entirely.
        if response_model is str:
            return await self._acreate_str_output(
                text_input,
                system_prompt,
                model=self.model,
                api_key=self.api_key,
                endpoint=self.endpoint,
                api_version=self.api_version,
                **merged_kwargs,
            )

        try:
            result = await self._acreate_structured(
                text_input,
                system_prompt,
                response_model,
                model=self.model,
                api_key=self.api_key,
                endpoint=self.endpoint,
                api_version=self.api_version,
                **merged_kwargs,
            )
            _enrich_llm_span(self.model, self.name)
            return result

        except ContentPolicyViolationError as error:
            # Try the fallback model before giving up, if one is configured.
            if not (self.fallback_model and self.fallback_api_key):
                raise ContentPolicyFilterError(
                    "The provided input contains content that is not aligned "
                    f"with our content policy: {text_input}"
                ) from error

            logger.warning(
                "Primary model hit content policy; trying fallback",
                primary_model=self.model,
                fallback_model=self.fallback_model,
            )
            try:
                return await self._acreate_structured(
                    text_input,
                    system_prompt,
                    response_model,
                    model=self.fallback_model,
                    api_key=self.fallback_api_key,
                    endpoint=self.fallback_endpoint,
                    api_version=self.api_version,
                    **merged_kwargs,
                )
            except ContentPolicyViolationError as fallback_error:
                raise ContentPolicyFilterError(
                    "The provided input contains content that is not aligned "
                    f"with our content policy: {text_input}"
                ) from fallback_error

        except Exception as error:
            # Surface quota/budget exhaustion as an actionable, non-retryable
            # error, matching the instructor adapters.
            if is_budget_exhausted_error(error):
                raise LLMPaymentRequiredError() from error
            raise
