"""Per-session token / cost tracking via a ContextVar scope.

Call sites that know the active session_id wrap their work in
``track_session_usage(session_id, user_id)``. Inside that scope,
``LLMGateway.acreate_structured_output`` (and any other caller that
opts in) calls ``record_llm_call`` after each LLM completion. The
tracker accumulates into the ``SessionRecord`` row.

Each LLM adapter calls ``capture_llm_usage(result)`` before returning,
which extracts exact token counts from the instructor result's
``_raw_response`` attribute (set by instructor on every parsed model).
``_record_session_usage_after`` in LLMGateway then reads these via
``pop_last_llm_usage()`` and passes them as overrides to ``record_llm_call``,
bypassing the char/4 heuristic entirely.
"""

from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, Optional
from uuid import UUID as UUIDType

from cognee.shared.logging_utils import get_logger

logger = get_logger("session_usage")


# (session_id, user_id) when active, else None.
_active_session: ContextVar[Optional[tuple[str, UUIDType]]] = ContextVar(
    "cognee_session_usage_target", default=None
)

# (operation_id, user_id) when a non-session API operation is active, else None.
# Session scope takes priority — when both are set, tokens go to the session.
_active_operation: ContextVar[Optional[tuple[str, UUIDType]]] = ContextVar(
    "cognee_operation_usage_target", default=None
)

# Exact (tokens_in, tokens_out) written by adapters after each LLM call and
# consumed once by _record_session_usage_after in LLMGateway.
_last_llm_usage: ContextVar[Optional[tuple[int, int]]] = ContextVar(
    "cognee_last_llm_usage", default=None
)


def set_last_llm_usage(tokens_in: int, tokens_out: int) -> None:
    _last_llm_usage.set((tokens_in, tokens_out))


def pop_last_llm_usage() -> Optional[tuple[int, int]]:
    """Return and clear the usage written by the most recent adapter call."""
    val = _last_llm_usage.get()
    _last_llm_usage.set(None)
    return val


def capture_llm_usage(result: Any) -> None:
    """Extract exact token counts from an instructor result and store them.

    Instructor attaches the raw provider response as ``result._raw_response``
    on every parsed Pydantic model.  For litellm-backed adapters the raw
    response carries ``usage.prompt_tokens / completion_tokens``; for the
    direct Anthropic SDK adapter it carries ``usage.input_tokens /
    output_tokens``.  When the adapter returns the raw litellm response
    directly (Mistral), ``usage`` sits on ``result`` itself.

    Falls back silently if usage information is absent.
    """
    try:
        raw = getattr(result, "_raw_response", result)
        usage = getattr(raw, "usage", None)
        if usage is None:
            return
        # LiteLLM / OpenAI field names
        tin = getattr(usage, "prompt_tokens", None)
        tout = getattr(usage, "completion_tokens", None)
        # Anthropic direct-SDK field names
        if tin is None:
            tin = getattr(usage, "input_tokens", None)
        if tout is None:
            tout = getattr(usage, "output_tokens", None)
        if tin is not None and tout is not None:
            set_last_llm_usage(int(tin), int(tout))
    except Exception:
        pass


@asynccontextmanager
async def track_session_usage(session_id: str, user_id: UUIDType):
    """Bind a session as the target for LLM-usage accumulation inside this scope."""
    if not session_id or user_id is None:
        yield
        return
    token = _active_session.set((session_id, user_id))
    try:
        yield
    finally:
        _active_session.reset(token)


def _estimate_tokens(text: str) -> int:
    """Very rough char-based estimate. Good enough for dashboard aggregates."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# Minimal per-model pricing table. Conservative and incomplete —
# unrecognized models cost $0.
# USD per 1M tokens: (input, output)
_PRICING_PER_M_TOKENS = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-haiku": (0.25, 1.25),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
}


# Longest prefix first so specific models (e.g. ``gpt-4o-mini``) win
# over their more general family (``gpt-4o``). Computed once at import.
_PRICING_SORTED = sorted(_PRICING_PER_M_TOKENS.items(), key=lambda kv: -len(kv[0]))


def _estimate_cost_usd(model: Optional[str], tokens_in: int, tokens_out: int) -> float:
    if not model:
        return 0.0
    # Normalize: strip provider prefix ("openai/gpt-4o" → "gpt-4o"), drop date suffix.
    key = model.split("/")[-1].lower()
    for base, (pin, pout) in _PRICING_SORTED:
        if key.startswith(base):
            return (tokens_in / 1_000_000) * pin + (tokens_out / 1_000_000) * pout
    return 0.0


async def record_llm_call(
    *,
    input_text: str,
    output_text: str,
    model: Optional[str] = None,
    tokens_in_override: Optional[int] = None,
    tokens_out_override: Optional[int] = None,
) -> None:
    """Accumulate this LLM call's usage into the active session or operation.

    Session scope takes priority over operation scope — when both are set
    (e.g. recall inside a cognify operation), tokens go to the session.

    Pass ``tokens_in_override`` / ``tokens_out_override`` when the
    caller has exact counts from ``response.usage``; otherwise the
    char-based estimate is used.
    """
    tokens_in = (
        tokens_in_override if tokens_in_override is not None else _estimate_tokens(input_text)
    )
    tokens_out = (
        tokens_out_override if tokens_out_override is not None else _estimate_tokens(output_text)
    )
    cost = _estimate_cost_usd(model, tokens_in, tokens_out)

    session_target = _active_session.get()
    if session_target is not None:
        session_id, user_id = session_target
        try:
            from cognee.modules.session_lifecycle.metrics import accumulate_usage

            await accumulate_usage(
                session_id=session_id,
                user_id=user_id,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
                model=model,
            )
        except Exception as exc:
            logger.debug("record_llm_call: session accumulate failed (%s)", exc)
        return

    op_target = _active_operation.get()
    if op_target is not None:
        operation_id, user_id = op_target
        try:
            from cognee.modules.session_lifecycle.metrics import accumulate_operation_usage

            await accumulate_operation_usage(
                operation_id=operation_id,
                user_id=user_id,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
                model=model,
            )
        except Exception as exc:
            logger.debug("record_llm_call: operation accumulate failed (%s)", exc)


class OperationOutcome:
    """Mutable success flag yielded by ``track_operation_usage``.

    An exception propagating out of the ``async with`` block marks it
    failed automatically. Call ``mark_failed()`` when the wrapped call
    instead returns an error sentinel (e.g. ``PipelineRunErrored``, or a
    result object with ``status == "errored"``) so the recorded status
    reflects what actually happened.
    """

    __slots__ = ("success",)

    def __init__(self) -> None:
        self.success = True

    def mark_failed(self) -> None:
        self.success = False


@asynccontextmanager
async def track_operation_usage(
    operation_id: str,
    user_id: UUIDType,
    operation_type: str,
    *,
    dataset_id: Optional[UUIDType] = None,
    background: bool = False,
):
    """Activate operation-level token accounting for the duration of the scope.

    Creates (or upserts) an ``OperationRecord`` row, sets ``_active_operation``
    so every ``LLMGateway.acreate_structured_output`` call within this scope
    writes to it, then marks the operation completed (or failed) on exit.

    Yields an ``OperationOutcome``. Call ``outcome.mark_failed()`` inside the
    scope when the wrapped call returns an error sentinel instead of raising
    — pipeline calls like ``cognify``/``memify``/``improve``/``remember`` can
    fail this way, and an exception is the only signal ``__aexit__`` sees on
    its own.

    When ``background=True`` and the scope exits successfully,
    ``mark_operation_ended`` is suppressed — the caller is returning early
    while an ``asyncio.create_task`` continues the work. The child task
    inherits a copy of the ContextVar (Python copies context on
    ``create_task``), so background LLM calls still accumulate. The record
    stays ``running`` and is inferred ``abandoned`` after the
    ``SESSION_ABANDON_AFTER_SECONDS`` threshold at read time. If the scope
    instead exits via an exception (e.g. a failure before the background
    task was ever created), the operation is always marked ``failed``
    immediately regardless of ``background`` — there's no background task
    left to finish it later.
    """
    from cognee.modules.session_lifecycle.metrics import (
        SessionStatus,
        ensure_and_touch_operation,
        mark_operation_ended,
    )

    try:
        await ensure_and_touch_operation(
            operation_id=operation_id,
            user_id=user_id,
            operation_type=operation_type,
            dataset_id=dataset_id,
        )
    except Exception as exc:
        logger.debug("track_operation_usage: ensure failed (%s), tracking disabled", exc)
        yield OperationOutcome()
        return

    token = _active_operation.set((operation_id, user_id))
    outcome = OperationOutcome()
    try:
        yield outcome
    except Exception:
        outcome.success = False
        raise
    finally:
        _active_operation.reset(token)
        if not (background and outcome.success):
            try:
                await mark_operation_ended(
                    operation_id=operation_id,
                    user_id=user_id,
                    status=SessionStatus.COMPLETED if outcome.success else SessionStatus.FAILED,
                )
            except Exception as exc:
                logger.debug("track_operation_usage: mark_ended failed (%s)", exc)
