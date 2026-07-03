"""Universal LiteLLM-native adapter for structured output.

This module implements ``NativeLiteLLMAdapter`` — a single adapter that works
with *every* provider LiteLLM supports, using LiteLLM's own ``response_format``
parameter to obtain validated Pydantic objects **without** the ``instructor``
library.

Design decisions
~~~~~~~~~~~~~~~~
* **One adapter for all providers.**  Unlike the ``litellm_instructor`` framework
  which ships a separate adapter per provider, LiteLLM already abstracts provider
  differences when ``response_format`` is a Pydantic model.  There is no reason
  to duplicate that abstraction.

* **Two code paths inside one adapter.**
  - *Schema-native* providers (OpenAI, Azure, Groq, Anthropic, Gemini, Mistral)
    accept a Pydantic class directly via ``response_format=<model>``.  LiteLLM
    serialises the JSON Schema and instructs the provider to constrain output.
  - *JSON-only* providers (Ollama, Llama.cpp, custom endpoints) only honour
    ``response_format={"type": "json_object"}``.  We add the schema to the
    system prompt and manually parse + validate the JSON response.

* **Self-correcting retries.**  When JSON parsing or Pydantic validation fails,
  we inject the error message into the next request so the model can fix itself.
  Rate-limit and authentication errors are raised immediately — retrying them
  would be futile.

* **Zero instructor dependency.**  This file must never ``import instructor``.
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

from cognee.infrastructure.llm.exceptions import ContentPolicyFilterError
from cognee.infrastructure.llm.retry_config import llm_retry_stop_condition
from cognee.infrastructure.llm.structured_output_framework.litellm_native.llm_native_interface import (
    LLMNativeInterface,
)
from cognee.modules.observability.get_observe import get_observe
from cognee.shared.logging_utils import get_logger
from cognee.shared.rate_limiting import llm_rate_limiter_context_manager

T = TypeVar("T", bound="BaseModel | str")

logger = get_logger()
observe = get_observe()

# ---------------------------------------------------------------------------
# Provider capability classification
# ---------------------------------------------------------------------------

# Providers whose LiteLLM integration supports passing a Pydantic model
# directly as ``response_format``.  When the user-configured provider string
# (lowercased) starts with any of these prefixes, we use schema-native mode.
_SCHEMA_NATIVE_PROVIDER_PREFIXES: frozenset[str] = frozenset(
    {
        "openai",
        "azure",
        "groq",
        "anthropic",
        "gemini",
        "vertex_ai",
        "mistral",
    }
)

# Maximum number of self-correction retries when the model returns JSON that
# does not pass Pydantic validation.  Kept separate from the tenacity retry
# (which handles transient HTTP errors) — this is an *application-level* loop.
_MAX_VALIDATION_RETRIES: int = 3


def _supports_native_schema(model_name: str) -> bool:
    """Decide whether *model_name* supports ``response_format=<PydanticModel>``.

    LiteLLM model strings use a ``provider/model`` convention (e.g.
    ``groq/llama-3.3-70b-versatile``).  We check whether the prefix before the
    first ``/`` belongs to the set of schema-capable providers.

    Args:
        model_name: The model identifier as stored in ``LLMConfig.llm_model``.

    Returns:
        ``True`` when the provider supports passing a Pydantic class directly
        to ``response_format``, ``False`` when we must fall back to JSON-object
        mode with manual validation.
    """
    # Extract provider prefix.  Some model names omit the provider slash
    # (e.g. "gpt-5-mini") — those default to OpenAI, which is schema-native.
    provider_prefix = model_name.split("/")[0].lower() if "/" in model_name else "openai"
    return provider_prefix in _SCHEMA_NATIVE_PROVIDER_PREFIXES


def _enrich_llm_span(model: str, name: str) -> None:
    """Set LLM attributes on the current OTEL span, if tracing is enabled.

    Mirrors the same helper in ``generic_llm_api/adapter.py`` to keep
    observability consistent across frameworks.
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


class NativeLiteLLMAdapter(LLMNativeInterface):
    """Adapter that uses LiteLLM's native ``response_format`` for structured output.

    This replaces the ``instructor`` library by leveraging LiteLLM's built-in
    support for Pydantic response schemas (or falling back to JSON-object mode
    for providers that lack schema support).

    Public methods:
        - acreate_structured_output

    Instance variables:
        - model, api_key, endpoint, api_version, max_completion_tokens,
          streaming, fallback_model, fallback_api_key, fallback_endpoint,
          llm_args, name
    """

    MAX_RETRIES = 2

    def __init__(
        self,
        api_key: str,
        model: str,
        max_completion_tokens: int,
        name: str = "LiteLLM-Native",
        endpoint: str | None = None,
        api_version: str | None = None,
        instructor_mode: str | None = None,  # accepted for interface compat; unused
        streaming: bool = False,
        fallback_model: str | None = None,
        fallback_api_key: str | None = None,
        fallback_endpoint: str | None = None,
        llm_args: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the adapter.

        Args:
            api_key: Provider API key (e.g. Groq, OpenAI).
            model: LiteLLM model string (e.g. ``"groq/llama-3.3-70b-versatile"``).
            max_completion_tokens: Maximum tokens the model may generate.
            name: Human-readable adapter name for logging/tracing.
            endpoint: Optional custom API base URL.
            api_version: Optional API version (e.g. for Azure).
            instructor_mode: Ignored — kept for interface compatibility with
                ``get_llm_client`` cache-key construction.
            streaming: Whether to enable streaming (not yet used).
            fallback_model: Secondary model tried on content-policy errors.
            fallback_api_key: API key for the fallback model.
            fallback_endpoint: Endpoint for the fallback model.
            llm_args: Extra kwargs merged into every LiteLLM call.
        """
        self.name = name
        self.model = model
        self.api_key = api_key
        self.api_version = api_version
        self.endpoint = endpoint
        self.max_completion_tokens = max_completion_tokens
        self.streaming = streaming
        self.fallback_model = fallback_model
        self.fallback_api_key = fallback_api_key
        self.fallback_endpoint = fallback_endpoint
        self.llm_args: dict[str, Any] = llm_args or {}

    # ------------------------------------------------------------------
    # Plain-text fallback (when response_model is str)
    # ------------------------------------------------------------------

    async def _acreate_str_output(
        self,
        text_input: str,
        system_prompt: str,
        **merged_kwargs: Any,
    ) -> str:
        """Plain-text completion without any JSON schema constraints.

        Matches the ``acreate_str_output`` pattern in GenericAPIAdapter.
        """
        async with llm_rate_limiter_context_manager():
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_input},
                ],
                api_key=self.api_key,
                api_base=self.endpoint,
                api_version=self.api_version,
                **merged_kwargs,
            )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Schema-native completion (provider supports response_format=Model)
    # ------------------------------------------------------------------

    async def _acreate_schema_native(
        self,
        text_input: str,
        system_prompt: str,
        response_model: type[BaseModel],
        **merged_kwargs: Any,
    ) -> BaseModel:
        """Call LiteLLM with ``response_format`` set to a Pydantic class.

        LiteLLM serialises the class's JSON Schema and sends it to the
        provider, which constrains its output accordingly.  The response
        is already validated JSON; we parse it through Pydantic for safety.

        Args:
            text_input: User content.
            system_prompt: System instructions.
            response_model: The Pydantic model the provider should emit.
            **merged_kwargs: Forwarded to ``litellm.acompletion``.

        Returns:
            Validated Pydantic instance.
        """
        async with llm_rate_limiter_context_manager():
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_input},
                ],
                response_format=response_model,
                api_key=self.api_key,
                api_base=self.endpoint,
                api_version=self.api_version,
                **merged_kwargs,
            )
        raw_content = response.choices[0].message.content or "{}"
        # Even though the provider constrains output, we still validate
        # through Pydantic to catch edge-case schema violations.
        return response_model.model_validate_json(raw_content)

    # ------------------------------------------------------------------
    # JSON-object fallback (provider only supports {"type": "json_object"})
    # ------------------------------------------------------------------

    async def _acreate_json_fallback(
        self,
        text_input: str,
        system_prompt: str,
        response_model: type[BaseModel],
        **merged_kwargs: Any,
    ) -> BaseModel:
        """Call LiteLLM with ``response_format={"type": "json_object"}``.

        The provider is told to emit JSON but is *not* given a schema
        constraint.  Instead, we inject the Pydantic schema into the system
        prompt so the model knows what shape to produce.  If validation
        fails, we retry up to ``_MAX_VALIDATION_RETRIES`` times, feeding
        the validation error back so the model can self-correct.

        Args:
            text_input: User content.
            system_prompt: System instructions.
            response_model: The Pydantic model describing expected output.
            **merged_kwargs: Forwarded to ``litellm.acompletion``.

        Returns:
            Validated Pydantic instance.

        Raises:
            ValidationError: After all self-correction retries are exhausted.
        """
        schema_json = json.dumps(
            response_model.model_json_schema(), indent=2
        )
        # Augment the system prompt with the expected JSON schema so the
        # model knows exactly what to produce, even without native
        # schema enforcement.
        augmented_system_prompt = (
            f"{system_prompt}\n\n"
            f"You MUST respond with valid JSON conforming to this schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Do not include any text outside the JSON object."
        )

        last_error: Exception | None = None

        for attempt in range(_MAX_VALIDATION_RETRIES):
            # On retries after a validation failure, append the error to
            # the user message so the model can self-correct.
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
                    model=self.model,
                    messages=[
                        {"role": "system", "content": augmented_system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                    api_key=self.api_key,
                    api_base=self.endpoint,
                    api_version=self.api_version,
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

        # All retries exhausted — raise the last validation error.
        raise last_error  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @observe(as_type="generation")
    @retry(
        stop=llm_retry_stop_condition,
        wait=wait_exponential_jitter(8, 128),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.NotFoundError,
                litellm.exceptions.AuthenticationError,
                # Rate-limit errors should propagate immediately — retrying
                # would just burn through the remaining quota faster.
                litellm.exceptions.RateLimitError,
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
        """Produce validated structured output from an LLM call.

        Routes to the appropriate code path based on the provider's
        capability:
        - ``str`` → plain-text completion (no schema)
        - Schema-native provider → ``response_format=<model>``
        - JSON-only provider → ``response_format={"type": "json_object"}``
          with manual validation and self-correcting retries

        Args:
            text_input: The user-supplied content to send to the model.
            system_prompt: System-level instructions that guide the model.
            response_model: A Pydantic ``BaseModel`` subclass describing the
                expected JSON shape, or the literal type ``str`` for plain text.
            **kwargs: Extra keyword arguments forwarded to the underlying LLM
                completion call (e.g. ``temperature``, ``max_tokens``).

        Returns:
            A validated instance of *response_model*, or a plain ``str``.

        Raises:
            litellm.exceptions.RateLimitError: On quota exhaustion (immediate).
            litellm.exceptions.AuthenticationError: On auth failure (immediate).
            ContentPolicyFilterError: When content policy is violated and no
                fallback is configured.
            ValidationError: After self-correction retries are exhausted
                (JSON-fallback mode only).
        """
        merged_kwargs = {**self.llm_args, **kwargs}

        # Plain string needs no schema — skip structured output entirely.
        if response_model is str:
            return await self._acreate_str_output(
                text_input, system_prompt, **merged_kwargs
            )

        try:
            result = await self._dispatch_structured_call(
                text_input=text_input,
                system_prompt=system_prompt,
                response_model=response_model,
                model=self.model,
                api_key=self.api_key,
                endpoint=self.endpoint,
                api_version=self.api_version,
                **merged_kwargs,
            )
            _enrich_llm_span(self.model, self.name)
            return result

        except ContentPolicyViolationError as error:
            # If a fallback model is configured, try it before giving up.
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
                return await self._dispatch_structured_call(
                    text_input=text_input,
                    system_prompt=system_prompt,
                    response_model=response_model,
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

    # ------------------------------------------------------------------
    # Internal dispatcher
    # ------------------------------------------------------------------

    async def _dispatch_structured_call(
        self,
        *,
        text_input: str,
        system_prompt: str,
        response_model: type[BaseModel],
        model: str,
        api_key: str | None,
        endpoint: str | None,
        api_version: str | None,
        **merged_kwargs: Any,
    ) -> BaseModel:
        """Route to schema-native or JSON-fallback based on the model name.

        This private helper exists so both primary and fallback calls can
        share the same routing logic without duplication.
        """
        # Temporarily swap connection params for the duration of this call
        # (needed for fallback).  We save/restore to stay thread-safe in
        # the common single-call case.
        saved = (self.model, self.api_key, self.endpoint, self.api_version)
        self.model = model
        self.api_key = api_key or ""
        self.endpoint = endpoint
        self.api_version = api_version

        try:
            if _supports_native_schema(model):
                return await self._acreate_schema_native(
                    text_input, system_prompt, response_model, **merged_kwargs
                )
            else:
                return await self._acreate_json_fallback(
                    text_input, system_prompt, response_model, **merged_kwargs
                )
        finally:
            self.model, self.api_key, self.endpoint, self.api_version = saved
