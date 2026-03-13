from __future__ import annotations

from contextlib import asynccontextmanager, nullcontext
from typing import AsyncIterator, Optional

import tiktoken
from aiolimiter import AsyncLimiter

from cognee.infrastructure.llm.config import LLMConfig, get_llm_config

# ---------------------------------------------------------------------------
# Module-level cached state — lazily populated on first use
# ---------------------------------------------------------------------------
_llm_request_limiter: Optional[AsyncLimiter] = None
_llm_token_limiter: Optional[AsyncLimiter] = None
_embedding_request_limiter: Optional[AsyncLimiter] = None
_embedding_token_limiter: Optional[AsyncLimiter] = None
_llm_config_cache: Optional[LLMConfig] = None
_embedding_config_cache: Optional[LLMConfig] = None

_tiktoken_encoding: Optional[tiktoken.Encoding] = None


def _get_tiktoken_encoding() -> tiktoken.Encoding:
    """Return (and lazily create) a shared tiktoken encoding."""
    global _tiktoken_encoding
    if _tiktoken_encoding is None:
        _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_encoding


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(*texts: str) -> int:
    """Estimate the number of input tokens across one or more text strings.

    Uses tiktoken's cl100k_base encoding for accurate counting.
    Accepts varargs so callers avoid throwaway string concatenation::

        estimate_tokens(text_input, system_prompt)

    Note: this estimates *input* tokens only; output tokens cannot be
    predicted before the LLM call.
    """
    enc = _get_tiktoken_encoding()
    total = sum(len(enc.encode(t)) for t in texts if t)
    return max(1, total)


# ---------------------------------------------------------------------------
# Lazy limiter initialisation helpers
# ---------------------------------------------------------------------------


def _get_llm_limiters():
    """Return ``(config, request_limiter, token_limiter | None)`` for LLM calls.

    Creates the limiters on first invocation, then returns cached instances.
    """
    global _llm_config_cache, _llm_request_limiter, _llm_token_limiter

    if _llm_config_cache is None:
        cfg = get_llm_config()
        _llm_config_cache = cfg
        _llm_request_limiter = AsyncLimiter(
            cfg.llm_rate_limit_requests, cfg.llm_rate_limit_interval
        )
        _llm_token_limiter = (
            AsyncLimiter(cfg.llm_rate_limit_tokens, cfg.llm_rate_limit_interval)
            if cfg.llm_rate_limit_tokens > 0
            else None
        )

    return _llm_config_cache, _llm_request_limiter, _llm_token_limiter


def _get_embedding_limiters():
    """Return ``(config, request_limiter, token_limiter | None)`` for embedding calls.

    Creates the limiters on first invocation, then returns cached instances.
    """
    global _embedding_config_cache, _embedding_request_limiter, _embedding_token_limiter

    if _embedding_config_cache is None:
        cfg = get_llm_config()
        _embedding_config_cache = cfg
        _embedding_request_limiter = AsyncLimiter(
            cfg.embedding_rate_limit_requests, cfg.embedding_rate_limit_interval
        )
        _embedding_token_limiter = (
            AsyncLimiter(cfg.embedding_rate_limit_tokens, cfg.embedding_rate_limit_interval)
            if cfg.embedding_rate_limit_tokens > 0
            else None
        )

    return _embedding_config_cache, _embedding_request_limiter, _embedding_token_limiter


def reset_rate_limiters() -> None:
    """Discard cached limiters so the next call picks up fresh config.

    Useful after ``get_llm_config.cache_clear()`` or in tests.
    """
    global _llm_config_cache, _llm_request_limiter, _llm_token_limiter
    global _embedding_config_cache, _embedding_request_limiter, _embedding_token_limiter
    _llm_config_cache = None
    _llm_request_limiter = None
    _llm_token_limiter = None
    _embedding_config_cache = None
    _embedding_request_limiter = None
    _embedding_token_limiter = None


# ---------------------------------------------------------------------------
# Combined request + token gate
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _combined_limiter(
    request_limiter: AsyncLimiter,
    token_limiter: Optional[AsyncLimiter],
    max_tokens: int,
    tokens: int,
) -> AsyncIterator[None]:
    """Acquire both request and token capacity."""
    async with request_limiter:
        if token_limiter is not None and tokens > 0:
            # Cap to max_tokens to avoid ValueError (single large request gets
            # through after waiting, but still respects the rate over time).
            capped = min(tokens, max_tokens)
            await token_limiter.acquire(capped)
        yield


# ---------------------------------------------------------------------------
# Public context-manager factories
# ---------------------------------------------------------------------------


def llm_rate_limiter_context_manager(tokens: int = 0):
    """Rate limiter for LLM calls.

    Args:
        tokens: Estimated token count for this request. When > 0 and token
                limiting is configured, enforces tokens-per-interval budget.

    Returns an async context manager (either the real gate or ``nullcontext``).
    """
    cfg, req_limiter, tok_limiter = _get_llm_limiters()
    if not cfg.llm_rate_limit_enabled:
        return nullcontext()
    return _combined_limiter(
        request_limiter=req_limiter,
        token_limiter=tok_limiter,
        max_tokens=cfg.llm_rate_limit_tokens,
        tokens=tokens,
    )


def embedding_rate_limiter_context_manager(tokens: int = 0):
    """Rate limiter for embedding calls.

    Args:
        tokens: Estimated token count for this request. When > 0 and token
                limiting is configured, enforces tokens-per-interval budget.

    Returns an async context manager (either the real gate or ``nullcontext``).
    """
    cfg, req_limiter, tok_limiter = _get_embedding_limiters()
    if not cfg.embedding_rate_limit_enabled:
        return nullcontext()
    return _combined_limiter(
        request_limiter=req_limiter,
        token_limiter=tok_limiter,
        max_tokens=cfg.embedding_rate_limit_tokens,
        tokens=tokens,
    )
