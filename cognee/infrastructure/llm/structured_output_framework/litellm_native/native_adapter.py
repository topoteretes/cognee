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

Retry/error handling mirrors the instructor adapters: transient errors (incl.
rate limits) retry with backoff, while auth and quota/budget exhaustion (mapped
to ``LLMPaymentRequiredError``) are terminal; content-policy violations fall back
to the configured fallback model. This file never imports ``instructor``.
"""

import asyncio
import json
import logging
import re
from typing import Any, cast

import litellm
from litellm.exceptions import BadRequestError, ContentPolicyViolationError
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
from cognee.infrastructure.llm.streaming.stream_completion import stream_text_completion
from cognee.infrastructure.llm.streaming.token_sink import get_active_token_sink
from cognee.shared.rate_limiting import llm_rate_limiter_context_manager

logger = get_logger()
observe = get_observe()

# Models on the prompted-JSON path routinely wrap their answer in a markdown
# fence (```json ... ```). That is not a malformed answer -- the JSON inside is
# valid -- but model_validate_json() sees a backtick at column 1 and rejects it,
# and the self-correction retry does not help because a model that fences once
# fences again. Strip a fence that wraps the entire payload before parsing.
# Deliberately anchored to the whole string: JSON that merely *contains*
# backticks in a value is left alone.
_JSON_FENCE_RE = re.compile(r"\A\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*\Z", re.DOTALL)


def _strip_json_fence(text: str) -> str:
    """Return ``text`` with a wrapping markdown code fence removed, if present."""
    match = _JSON_FENCE_RE.match(text)
    return match.group(1) if match else text


# Max self-correction attempts when a json-object provider returns JSON that
# fails validation. Separate from the tenacity retry (transient HTTP errors).
_MAX_VALIDATION_RETRIES: int = 3

# (llm_model, response_model.__name__) pairs whose schema the provider's strict
# mode has rejected. Once demoted, calls go straight to the non-strict payload,
# so the failed strict request is paid once per process, not per call.
_NONSTRICT_DEMOTIONS: set[tuple[str, str]] = set()


def clear_nonstrict_demotions() -> None:
    _NONSTRICT_DEMOTIONS.clear()


def _nonstrict_response_format(response_model: type[BaseModel]) -> dict:
    """Non-strict ``json_schema`` payload: the raw schema travels as guidance.

    Without ``strict: true`` the provider does not enforce its restricted
    schema subset, so constructs strict mode rejects (``oneOf``/``discriminator``
    from discriminated unions, free-form dict fields, ``format``) are accepted.
    Conformance is still checked app-side by validating against the original
    Pydantic model.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": response_model.model_json_schema(),
            "strict": False,
        },
    }


def _attach_raw_response(instance: BaseModel, response) -> BaseModel:
    """Attach the raw litellm response as ``_raw_response``, like instructor does.

    The LLMGateway usage recorder reads ``result._raw_response.usage`` for the
    provider-billed token counts (which include hidden reasoning tokens no
    text-based estimate can see). ``object.__setattr__`` bypasses pydantic's
    field validation for the non-field attribute.
    """
    object.__setattr__(instance, "_raw_response", response)
    return instance


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

        from cognee.context_global_variables import current_pipeline_stage
        from cognee.modules.observability.tracing import (
            COGNEE_LLM_MODEL,
            COGNEE_LLM_PROVIDER,
            COGNEE_PIPELINE_STAGE,
        )

        current_span = otel_trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute(COGNEE_LLM_MODEL, model)
            current_span.set_attribute(COGNEE_LLM_PROVIDER, name)
            stage = current_pipeline_stage.get()
            if stage:
                current_span.set_attribute(COGNEE_PIPELINE_STAGE, stage)
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

    # The default framework's answer path, so this is the one that decides
    # whether an out-of-the-box install can stream at all.
    supports_answer_streaming = True

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
        """Plain-text completion without any schema (mirrors GenericAPIAdapter).

        This is the default framework's answer path, so it is the one that has to
        stream for streaming to reach anybody: STRUCTURED_OUTPUT_FRAMEWORK
        defaults to litellm_native, which routes here regardless of provider.
        """
        sink = get_active_token_sink()
        if sink is not None:
            return await stream_text_completion(
                sink=sink,
                model=model,
                system_prompt=system_prompt,
                text_input=text_input,
                api_key=api_key,
                endpoint=endpoint,
                api_version=api_version,
                adapter_name="litellm_native",
                **merged_kwargs,
            )

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
        if not response.choices:
            raise ValueError("litellm_native returned no choices for a plain-text completion")
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
        strict: bool = True,
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
                response_format=(
                    response_model if strict else _nonstrict_response_format(response_model)
                ),
                api_key=api_key,
                api_base=endpoint,
                api_version=api_version,
                **merged_kwargs,
            )
        raw_content = response.choices[0].message.content or "{}"
        return _attach_raw_response(response_model.model_validate_json(raw_content), response)

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

            raw_content = _strip_json_fence(response.choices[0].message.content or "{}")
            try:
                return _attach_raw_response(
                    response_model.model_validate_json(raw_content), response
                )
            except (ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "litellm_native validation retry %s/%s: %s",
                    attempt + 1,
                    _MAX_VALIDATION_RETRIES,
                    exc,
                )

        # All self-correction attempts exhausted — surface the last error.
        raise (
            last_error
            if last_error is not None
            else RuntimeError("litellm_native json fallback produced no result")
        )

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
            demotion_key = (model, response_model.__name__)
            attempts = [False] if demotion_key in _NONSTRICT_DEMOTIONS else [True, False]
            for strict in attempts:
                try:
                    return await self._acreate_schema_native(
                        text_input,
                        system_prompt,
                        response_model,
                        model=model,
                        api_key=api_key,
                        endpoint=endpoint,
                        api_version=api_version,
                        strict=strict,
                        **merged_kwargs,
                    )
                except ValidationError as error:
                    # Output that fails app-side validation must not bubble into
                    # the tenacity retry: that re-sends the same prompt blindly
                    # (no error feedback) under a 240s stop floor. The
                    # prompted-JSON path below has the self-correcting retry
                    # loop, so route there instead.
                    logger.warning(
                        "litellm_native: %s native output failed validation for %s; "
                        "using json fallback (%s)",
                        model,
                        response_model.__name__,
                        error,
                    )
                    break
                except BadRequestError as error:
                    # Strict schema-native mode rejects Pydantic models whose
                    # JSON schema it cannot enforce — e.g. a free-form dict
                    # field, which OpenAI 400s with "'additionalProperties' is
                    # required to be supplied and to be false". Non-strict mode
                    # accepts those schemas as guidance, so demote and retry
                    # once before giving up on the native path entirely.
                    if "schema" not in str(error).lower():
                        raise
                    if strict:
                        _NONSTRICT_DEMOTIONS.add(demotion_key)
                        logger.warning(
                            "litellm_native: %s rejected the strict schema for %s; "
                            "demoting to non-strict json_schema for this process (%s)",
                            model,
                            response_model.__name__,
                            error,
                        )
                    else:
                        logger.warning(
                            "litellm_native: %s rejected the non-strict schema for "
                            "%s; retrying via json fallback (%s)",
                            model,
                            response_model.__name__,
                            error,
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
                # Quota/billing exhaustion is terminal (#3643); transient rate
                # limits still retry with backoff, matching the instructor adapters.
                LLMPaymentRequiredError,
                # A cancelled task must propagate, not be retried.
                asyncio.CancelledError,
            )
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def acreate_structured_output(
        self,
        text_input: str,
        system_prompt: str,
        response_model: type[BaseModel | str],
        **kwargs: Any,
    ) -> BaseModel | str:
        """Return a validated instance of *response_model* (or a plain ``str``).

        Auth errors and quota/budget exhaustion (surfaced as
        ``LLMPaymentRequiredError``) are terminal; transient errors incl. rate
        limits retry with backoff. A content-policy violation retries once on the
        fallback model before raising ``ContentPolicyFilterError``.
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

        structured_model = cast(type[BaseModel], response_model)
        try:
            result = await self._acreate_structured(
                text_input,
                system_prompt,
                structured_model,
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
                "Primary model %s hit content policy; trying fallback %s",
                self.model,
                self.fallback_model,
            )
            try:
                return await self._acreate_structured(
                    text_input,
                    system_prompt,
                    structured_model,
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
