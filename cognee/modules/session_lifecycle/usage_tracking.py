"""Per-session token / cost tracking via a ContextVar scope.

Call sites that know the active session_id wrap their work in
``track_session_usage(session_id, user_id)``. Inside that scope,
``LLMGateway.acreate_structured_output`` (and any other caller that
opts in) calls ``record_llm_call`` after each LLM completion. The
tracker accumulates into the ``SessionRecord`` row.

Token counts are approximate — we don't currently extract
``response.usage`` from the litellm/instructor client (requires
changes deeper in the stack). A ~chars/4 heuristic is close enough
for the dashboard's "are we spending?" question without plumbing
upstream.
"""

from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional
from uuid import UUID as UUIDType

from cognee.shared.logging_utils import get_logger

logger = get_logger("session_usage")


# (session_id, user_id) when active, else None.
_active_session: ContextVar[Optional[tuple[str, UUIDType]]] = ContextVar(
    "cognee_session_usage_target", default=None
)


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


# ── Activity log (opt-in; inert unless a sink is registered) ─────────────────
#
# A parallel, operation-scoped accumulator alongside the session tracker above,
# for consumers (e.g. the Cognee Cloud control plane) that want a per-operation
# activity log — one record per add/cognify/improve/remember/forget/recall/
# search run, carrying who/what/when + token usage.
#
# Inert by default: with no sink registered, ``track_operation_usage`` is a
# passthrough and ``record_llm_call`` skips the operation branch, so open-source
# users incur zero overhead and nothing is stored anywhere. A deployment opts in
# by calling ``register_activity_sink`` once at startup; the sink receives each
# finished ``ActivityEvent`` and is responsible for persisting it.


@dataclass
class ActivityEvent:
    """One completed user-facing operation, handed to the registered sink."""

    operation: str
    user_id: Optional[UUIDType] = None
    tenant_id: Optional[UUIDType] = None
    dataset_id: Optional[UUIDType] = None
    dataset_name: Optional[str] = None
    session_id: Optional[str] = None
    origin: str = "api"
    pipeline_run_id: Optional[UUIDType] = None
    tokens_in: int = 0
    tokens_out: int = 0
    status: str = "completed"
    error: Optional[str] = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None


ActivitySink = Callable[[ActivityEvent], Awaitable[None]]

_activity_sink: Optional[ActivitySink] = None

_active_operation: ContextVar[Optional[ActivityEvent]] = ContextVar(
    "cognee_activity_operation", default=None
)


def register_activity_sink(sink: Optional[ActivitySink]) -> None:
    """Install (or clear, with ``None``) the process-wide activity-log sink.

    Only one sink is supported; the last registration wins. Deployments that
    persist an activity log call this once at startup; open-source leaves it
    unset, keeping the whole mechanism inert.
    """
    global _activity_sink
    _activity_sink = sink


@asynccontextmanager
async def track_operation_usage(operation: str, **context):
    """Accumulate token usage for one user-facing operation, then emit it to the
    registered activity sink on exit. No-op when no sink is registered.

    Any LLM call made inside this scope adds its tokens to the event via
    ``record_llm_call``. The finished event (with status/error/duration) is
    handed to the sink; sink failures are swallowed so activity logging can
    never break the operation it observes.
    """
    if _activity_sink is None:
        # Open-source default: nothing consumes activity — add zero overhead.
        yield
        return

    event = ActivityEvent(operation=operation, **context)
    token = _active_operation.set(event)
    try:
        yield
    except Exception as exc:
        event.status = "errored"
        event.error = str(exc)
        raise
    finally:
        _active_operation.reset(token)
        event.ended_at = datetime.now(timezone.utc)
        sink = _activity_sink
        if sink is not None:
            try:
                await sink(event)
            except Exception as exc:
                logger.debug("activity sink failed (%s)", exc)


def begin_operation(operation: str, **context) -> Optional[object]:
    """Open an operation scope without a context manager (for async generators
    like ``run_tasks`` where wrapping in ``async with`` is awkward).

    Returns a reset token to hand to ``end_operation``, or ``None`` when no sink
    is registered (inert — the caller's ``end_operation``/``mark_*`` become
    no-ops).
    """
    if _activity_sink is None:
        return None
    event = ActivityEvent(operation=operation, **context)
    return _active_operation.set(event)


def mark_active_operation_errored(error: str) -> None:
    """Mark the active operation (if any) errored, before ``end_operation``."""
    event = _active_operation.get()
    if event is not None:
        event.status = "errored"
        event.error = error


async def end_operation(token) -> None:
    """Close a scope opened by ``begin_operation``: emit the event, then reset.

    ``token is None`` (no sink) is a no-op. Sink failures are swallowed so
    activity logging can never break the operation it observes.
    """
    if token is None:
        return
    event = _active_operation.get()
    _active_operation.reset(token)
    if event is None:
        return
    event.ended_at = datetime.now(timezone.utc)
    sink = _activity_sink
    if sink is not None:
        try:
            await sink(event)
        except Exception as exc:
            logger.debug("activity sink failed (%s)", exc)


# Exact per-call token usage, reported by an adapter right after a completion
# so the gateway's usage hook can use real counts instead of the char estimate.
# One-shot: consumed (and cleared) by the very next record_llm_call.
_last_call_usage: ContextVar[Optional[tuple[int, int]]] = ContextVar(
    "cognee_last_llm_usage", default=None
)


def report_llm_usage(prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> None:
    """Report exact token counts for the just-completed LLM call.

    Adapters call this immediately after a completion (from ``response.usage``)
    so the gateway records real tokens rather than the char estimate. Inert if
    either count is missing (falls back to the estimate).
    """
    if prompt_tokens is None or completion_tokens is None:
        return
    # Accumulate within a public call: one acreate_structured_output may make
    # several internal completions (e.g. the JSON-fallback validation retries),
    # and the gateway consumes the total exactly once. Cleared by consume.
    prev = _last_call_usage.get()
    if prev is None:
        _last_call_usage.set((int(prompt_tokens), int(completion_tokens)))
    else:
        _last_call_usage.set((prev[0] + int(prompt_tokens), prev[1] + int(completion_tokens)))


def consume_last_usage() -> Optional[tuple[int, int]]:
    """Return and clear the last reported ``(prompt, completion)`` token counts.

    Cleared on read so a later call that reports nothing can't reuse stale
    counts.
    """
    usage = _last_call_usage.get()
    if usage is not None:
        _last_call_usage.set(None)
    return usage


def _estimate_tokens(text: str) -> int:
    """Very rough char-based estimate. Good enough for dashboard aggregates."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# Rough per-model pricing for cost estimates. Longest-prefix match (see below),
# so a model id need only start with a key; unknown models cost $0 and callers
# warn. USD per 1M tokens (input, output) at each provider's base tier; verified
# against the official pricing pages July 2026.
_PRICING_PER_M_TOKENS = {
    # OpenAI — https://developers.openai.com/api/docs/pricing
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.5-pro": (30.00, 180.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-pro": (30.00, 180.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o3": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
    # Anthropic — https://platform.claude.com/docs/en/about-claude/pricing
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus-4-5": (5.00, 25.00),
    "claude-opus-4": (15.00, 75.00),  # Opus 4.0 / 4.1
    "claude-sonnet-5": (3.00, 15.00),  # $2/$10 introductory through 2026-08-31
    "claude-sonnet-4": (3.00, 15.00),  # Sonnet 4.0 / 4.5 / 4.6
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-haiku": (0.25, 1.25),
    # Google Gemini — https://ai.google.dev/gemini-api/docs/pricing
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-pro": (2.00, 12.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3-flash": (0.50, 3.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
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


def estimate_cost_usd(model: Optional[str], tokens_in: int, tokens_out: int) -> float:
    """Estimate USD cost for a model using Cognee's rough pricing table.

    Unrecognized models cost $0 — callers that surface the number should say so.
    """
    return _estimate_cost_usd(model, tokens_in, tokens_out)


async def record_llm_call(
    *,
    input_text: str,
    output_text: str,
    model: Optional[str] = None,
    tokens_in_override: Optional[int] = None,
    tokens_out_override: Optional[int] = None,
) -> None:
    """If there's an active session, accumulate this call's usage into it.

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

    # Activity log (operation scope): accumulate independently of any session,
    # so an operation's per-run tokens are captured even when no session is active.
    operation = _active_operation.get()
    if operation is not None:
        operation.tokens_in += tokens_in
        operation.tokens_out += tokens_out

    # Session usage accumulation (unchanged): only when a session is active.
    target = _active_session.get()
    if target is None:
        return
    session_id, user_id = target
    cost = _estimate_cost_usd(model, tokens_in, tokens_out)

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
        logger.debug("record_llm_call: accumulate failed (%s)", exc)
