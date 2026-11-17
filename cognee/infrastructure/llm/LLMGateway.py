import time
from collections.abc import Coroutine
from datetime import datetime, timezone
from typing import Any, TypeVar

from pydantic import BaseModel

from cognee.infrastructure.llm import get_llm_config
from cognee.infrastructure.llm.config import (
    get_llm_context_config,
    resolve_structured_output_framework,
)
from cognee.infrastructure.llm.exceptions import raise_if_budget_exhausted
from cognee.infrastructure.llm.retry_config import raise_if_quota_error
from cognee.infrastructure.llm.types import TranscriptionReturnType
from cognee.modules.observability.llm_calls import capture_enabled, capture_llm_call
from cognee.shared.logging_utils import get_logger

T = TypeVar("T", bound="BaseModel | str")


def _inject_agent_memory(text_input: str) -> str:
    from cognee.modules.agent_memory import get_current_agent_memory_context

    context = get_current_agent_memory_context()
    if context is None or not context.memory_context:
        return text_input

    return f"Additional Memory Context:\n{context.memory_context}\n\nOriginal Input:\n{text_input}"


def _exact_usage_from_result(result: Any) -> tuple[int | None, int | None]:
    """Real prompt/completion token counts from the raw provider response —
    (None, None) otherwise, so the caller falls back to its char-based
    estimate.

    The structured-output adapters attach the raw litellm response to the
    parsed model as ``_raw_response``: litellm_native (the default) does it
    explicitly (``_attach_raw_response``), and the legacy framework does it
    internally while post-processing the response. That raw response carries
    ``.usage`` with the provider-billed counts, including hidden reasoning
    tokens no text estimate can see.

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
        logger.debug("Ignoring exception in _record_session_usage_after", exc_info=True)
    return result


async def _capture_llm_call_after(
    coro: Coroutine,
    *,
    text_input: str,
    system_prompt: str,
    response_model: Any,
    framework: str,
) -> T:
    """Run the LLM coroutine and record it for LLM call capture, failed calls included.

    A no-op wrapper while capture is off. Capture never raises, and an error
    from the call is re-raised unchanged after it is recorded.
    """
    if not capture_enabled():
        return await coro

    model = get_llm_context_config().llm_model
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()

    def record(
        output: Any,
        status: str,
        error_class: str | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
    ) -> None:
        capture_llm_call(
            started_at=started_at,
            duration_ms=(time.perf_counter() - started) * 1000,
            model=model,
            framework=framework,
            response_type=getattr(response_model, "__name__", str(response_model)),
            system_prompt=system_prompt,
            text_input=text_input,
            output=output,
            status=status,
            error_class=error_class,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    try:
        result = await coro
    except Exception as error:
        record(None, "error", error_class=type(error).__name__)
        raise

    try:
        tokens_in, tokens_out = _exact_usage_from_result(result)
        record(result, "ok", tokens_in=tokens_in, tokens_out=tokens_out)
    except Exception:
        logger.warning("LLM call capture failed", exc_info=True)

    return result


async def _fail_fast_on_quota(coro: Coroutine) -> T:
    """Convert provider quota/billing exhaustion into an actionable error.

    Runs at the single choke point every structured-output call flows through,
    so it is provider- and framework-agnostic.

    Budget exhaustion is checked first, and here rather than only in the
    adapters: the legacy adapters catch their framework's retry-exhausted
    wrapper in a clause that precedes their own budget handler, and the
    plain-text path (``response_model is str``) bypasses those handlers
    altogether. Classifying at this choke point gives every adapter and both
    paths the same 402.
    """
    try:
        return await coro
    except Exception as error:
        raise_if_budget_exhausted(error)
        raise_if_quota_error(error)
        raise


logger = get_logger("LLMGateway")


class LLMGateway:
    """Single entry point for cognee's LLM calls: structured output, audio and image.

    Graph extraction, completions, session analysis and the media loaders call these
    static methods instead of a provider SDK, so provider choice is configuration
    (``LLM_PROVIDER``, ``LLM_MODEL``, ``LLM_API_KEY``, ``LLM_ENDPOINT``), and code
    that calls the gateway, including your own tasks and retrievers, runs on whichever
    provider is configured. ``STRUCTURED_OUTPUT_FRAMEWORK`` picks the adapter:
    ``litellm_native`` (default; litellm's schema-native ``response_format`` with a
    prompted-JSON, self-correcting fallback for models without schema support),
    ``instructor`` (legacy, needs the ``instructor`` extra) or ``baml`` (needs the
    ``baml`` extra; its structured-output calls read the ``BAML_LLM_*`` settings
    instead of ``LLM_*``). Every structured-output call, whatever the framework, also gets
    the shared handling described on ``acreate_structured_output``.

    A namespace: do not instantiate it.
    """

    @staticmethod
    def acreate_structured_output(
        text_input: str,
        system_prompt: str,
        response_model: type[T],
        **kwargs: Any,
    ) -> Coroutine[Any, Any, T]:
        """Ask the configured LLM for a validated ``response_model`` instance (or plain text).

        Pass a Pydantic model to get it back filled and validated (this is how
        cognify gets a ``KnowledgeGraph``), or ``str`` to get the answer as text
        without a schema. The call is dispatched to the framework
        ``STRUCTURED_OUTPUT_FRAMEWORK`` selects; on the default framework the adapter
        applies the LLM rate limiter, retries transient failures with backoff and
        retries once on the configured fallback model after a content-policy refusal.

        Every framework gets the same wrapping here:

        - Inside a function decorated with ``@agent_memory``, the memory it retrieved
          is prepended to ``text_input``, so the agent's LLM calls see it without
          passing it along.
        - Token usage is added to the active operation and session usage trackers,
          with the provider's exact counts when the adapter attaches the raw litellm
          response to a structured result (litellm_native does) and a text-length
          estimate otherwise (plain-text answers, BAML).
        - Budget exhaustion raises ``LLMPaymentRequiredError`` (HTTP 402) and other
          quota or billing exhaustion ``LLMQuotaExceededError``, on every framework
          and on the plain-text path, so callers can tell an exhausted account from
          a transient failure; ``improve()`` stops its run on the former.
        - With ``LLM_CALL_CAPTURE_ENABLED=true`` the call is recorded, failures
          included, with its model, duration, tokens and pipeline stage, run and
          chunk; prompts and outputs only with ``LLM_CALL_CAPTURE_INCLUDE_CONTENT``.

        Usage accounting and capture never change the result or the error.

        Args:
            text_input: The user content of the request.
            system_prompt: The system instructions.
            response_model: The Pydantic model to fill, or ``str`` for plain text.
            **kwargs: Extra completion arguments for the adapter (on the default
                framework, litellm arguments merged over the configured ones);
                ignored by the ``baml`` framework.

        Returns:
            A coroutine; await it for the validated model instance or the text.

        Raises:
            LLMAPIKeyNotSetError: On the ``litellm_native`` and ``instructor``
                frameworks, when the call is made without an API key for a provider
                that needs one.
            InstructorNotInstalledError: ``instructor`` framework selected without
                the ``instructor`` extra.
        """
        text_input = _inject_agent_memory(text_input)
        llm_config = get_llm_config()
        framework = resolve_structured_output_framework(llm_config)
        if framework == "baml":
            from cognee.infrastructure.llm.structured_output_framework.baml.baml_src.extraction import (
                acreate_structured_output,
            )

            inner = acreate_structured_output(
                text_input=text_input,
                system_prompt=system_prompt,
                response_model=response_model,
            )
        elif framework == "litellm_native":
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
        return _capture_llm_call_after(
            _fail_fast_on_quota(_record_session_usage_after(inner, text_input=text_input)),
            text_input=text_input,
            system_prompt=system_prompt,
            response_model=response_model,
            framework=framework,
        )

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
        except Exception:  # a capability probe must not break a request
            logger.debug("Could not resolve answer-streaming support", exc_info=True)
            return False

    @staticmethod
    def _transcription_client():
        """Client for the audio/image transcription entry points.

        Transcription is a plain litellm call on every framework, so only the
        ``instructor`` framework builds the legacy client; ``litellm_native``
        and BAML (which has no transcription of its own) use the native client,
        which never needs the optional ``instructor`` package.
        """
        llm_config = get_llm_config()
        if resolve_structured_output_framework(llm_config) == "instructor":
            from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
                get_llm_client,
            )

            return get_llm_client()

        from cognee.infrastructure.llm.structured_output_framework.litellm_native.get_native_client import (
            get_native_client,
        )

        return get_native_client()

    @staticmethod
    def create_transcript(input, **kwargs) -> Coroutine[Any, Any, TranscriptionReturnType | None]:
        """Transcribe an audio file with the configured transcription model.

        The audio and video loaders call this during ``add()``, so recordings become
        text that cognify chunks and extracts like any document. Uses
        ``TRANSCRIPTION_MODEL`` (default ``whisper-1``) with the LLM provider's
        credentials. Only the ``instructor`` framework routes this through its
        legacy client; ``litellm_native`` and ``baml`` use the native litellm
        client, so they need no optional extra for transcription.

        Args:
            input: Path of the audio file (a local or storage path).
            **kwargs: Passed to litellm's transcription call, e.g.
                ``response_format="verbose_json"`` for segments.

        Returns:
            A coroutine; await it for a ``TranscriptionReturnType`` whose ``text``
            holds the transcript and ``payload`` the provider response.
        """
        llm_client = LLMGateway._transcription_client()
        return llm_client.create_transcript(input=input, **kwargs)

    @staticmethod
    def transcribe_image(
        input: str,
        prompt: str | None = None,
        max_completion_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> Coroutine[Any, Any, Any]:
        """Describe an image file with the configured multimodal model.

        The image loader calls this during ``add()``, so images become text that
        cognify chunks and extracts like any document. Uses
        ``IMAGE_TRANSCRIBE_MODEL``, or ``LLM_MODEL`` when that is empty, with the LLM
        provider's credentials, through the same client rules as
        ``create_transcript``.

        Args:
            input: Path of the image file; its extension must map to an image type.
            prompt: What to ask about the image; the default framework falls back to
                a plain caption request.
            max_completion_tokens: Output cap; the default framework falls back to 300.
            reasoning_effort: On the default framework, passed to models that support
                it and dropped for the rest.

        Returns:
            A coroutine; await it for the provider response, with the description at
            ``choices[0].message.content``.
        """
        llm_client = LLMGateway._transcription_client()
        return llm_client.transcribe_image(
            input=input,
            prompt=prompt,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
        )
