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
