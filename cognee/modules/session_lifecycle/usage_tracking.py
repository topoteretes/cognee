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
from typing import Optional
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


def _estimate_tokens(text: str) -> int:
    """Very rough char-based estimate. Good enough for dashboard aggregates."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# Rough per-model pricing table for the dry-run estimator and session cost
# tracking. Unrecognized models cost $0 (callers that surface the number warn).
# Matching is longest-prefix-first (see below), so a model id need only start
# with one of these keys; date suffixes and newer point releases fall back to
# their family. Prices are USD per 1M tokens: (input, output), using each
# provider's base short-context tier. Verified against the official pricing
# pages July 2026 — update as providers change rates.
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
    target = _active_session.get()
    if target is None:
        return
    session_id, user_id = target

    tokens_in = (
        tokens_in_override if tokens_in_override is not None else _estimate_tokens(input_text)
    )
    tokens_out = (
        tokens_out_override if tokens_out_override is not None else _estimate_tokens(output_text)
    )
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
