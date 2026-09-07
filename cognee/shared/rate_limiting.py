"""Rate-limiting context managers for LLM and embedding dispatch.

This module is the pacing seam: adapters wrap each dispatch attempt with
``llm_rate_limiter_context_manager()`` (inside their tenacity retry) or
``embedding_rate_limiter_context_manager()``. Overload detection and reaction
policy live in ``cognee.infrastructure.llm.overload_policy``; configuration
is read at use-time, never snapshotted at import.
"""

import logging
from contextlib import asynccontextmanager, nullcontext
from typing import Any, Iterable, Union

from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

# Limiters are built lazily on first use so their budgets bind the resolved
# configuration (LLMConfig gives local inference servers a smaller default
# LLM_RATE_LIMIT_REQUESTS) and so importing this module stays dependency-light.
_llm_rate_limiter: "AsyncLimiter | None" = None
_embedding_rate_limiter: "AsyncLimiter | None" = None
_embedding_token_rate_limiter: "AsyncLimiter | None" = None


def _get_llm_rate_limiter() -> AsyncLimiter:
    global _llm_rate_limiter
    if _llm_rate_limiter is None:
        # NOTE: Import inside function to avoid a circular import at module load.
        from cognee.infrastructure.llm.config import get_llm_config

        llm_config = get_llm_config()
        _llm_rate_limiter = AsyncLimiter(
            llm_config.llm_rate_limit_requests, llm_config.llm_rate_limit_interval
        )
    return _llm_rate_limiter


@asynccontextmanager
async def _governed_llm_dispatch():
    """Gate one LLM dispatch attempt.

    Everything runs unbounded (the fast path). The RPM limiter paces when
    LLM_RATE_LIMIT_ENABLED is set, or while the overload policy reports an
    active episode — evidence being a rate-limit error, an HTTP 503/529, or
    a timeout, with the episode lapsing back to the configured state after a
    quiet cooldown. Adapters enter this context manager INSIDE their tenacity
    retry, so retried attempts are paced as well.
    """
    # NOTE: Imports inside function to avoid circular imports at module load.
    from cognee.infrastructure.llm.config import get_llm_config
    from cognee.infrastructure.llm.overload_policy import llm_overload_policy

    llm_config = get_llm_config()
    pace = llm_config.llm_rate_limit_enabled or llm_overload_policy.is_paced()
    async with _get_llm_rate_limiter() if pace else nullcontext():
        try:
            yield
        except Exception as error:
            if llm_config.auto_rate_limit:
                llm_overload_policy.on_error(error)
            raise


def llm_rate_limiter_context_manager():
    return _governed_llm_dispatch()


def embedding_rate_limiter_context_manager():
    global _embedding_rate_limiter
    # NOTE: Import inside function to avoid a circular import at module load
    # (embedding engines, which EmbeddingConfig is reached through, import
    # this module).
    from cognee.infrastructure.databases.vector.embeddings.config import (
        get_embedding_config,
    )

    embedding_config = get_embedding_config()
    if not embedding_config.embedding_rate_limit_enabled:
        #  Return a no-op context manager if rate limiting is disabled
        return nullcontext()

    if _embedding_rate_limiter is None:
        _embedding_rate_limiter = AsyncLimiter(
            embedding_config.embedding_rate_limit_requests,
            embedding_config.embedding_rate_limit_interval,
        )
    return _embedding_rate_limiter


def count_embedding_tokens(tokenizer: Any, texts: Union[str, Iterable[str]]) -> int:
    """Size one embedding dispatch in tokens, for EMBEDDING_RATE_LIMIT_TOKENS.

    Returns 0 — "no token budget applies to this dispatch" — whenever the budget
    is not in force. The configuration is read before the tokenizer runs, so the
    default path (no budget configured) never pays for tokenization. A tokenizer
    that cannot size the input also yields 0: a pacing knob must never be able to
    fail a dispatch.
    """
    # NOTE: Import inside function to avoid a circular import at module load
    # (embedding engines, which EmbeddingConfig is reached through, import
    # this module).
    from cognee.infrastructure.databases.vector.embeddings.config import (
        get_embedding_config,
    )

    embedding_config = get_embedding_config()
    if not (
        embedding_config.embedding_rate_limit_enabled
        and embedding_config.embedding_rate_limit_tokens > 0
    ):
        return 0

    if tokenizer is None:
        return 0

    if isinstance(texts, str):
        texts = [texts]

    try:
        return sum(tokenizer.count_tokens(text) for text in texts)
    except Exception as error:
        logger.warning(
            "Could not count embedding tokens for rate limiting (%s: %s). "
            "Pacing this dispatch by request count only.",
            type(error).__name__,
            error,
        )
        return 0


async def consume_embedding_token_budget(tokenizer: Any, texts: Union[str, Iterable[str]]) -> None:
    """Pace one embedding dispatch against EMBEDDING_RATE_LIMIT_TOKENS.

    Called from inside ``embedding_rate_limiter_context_manager()``, which paces
    requests per interval; this adds the tokens-per-interval budget on the same
    interval. It is a no-op unless the operator sets a token budget, so the
    shipped default (0 = disabled) is unchanged.
    """
    global _embedding_token_rate_limiter
    # NOTE: Import inside function to avoid a circular import at module load.
    from cognee.infrastructure.databases.vector.embeddings.config import (
        get_embedding_config,
    )

    token_count = count_embedding_tokens(tokenizer, texts)
    if token_count <= 0:
        return

    embedding_config = get_embedding_config()
    token_budget = embedding_config.embedding_rate_limit_tokens
    if _embedding_token_rate_limiter is None:
        _embedding_token_rate_limiter = AsyncLimiter(
            token_budget, embedding_config.embedding_rate_limit_interval
        )

    # A single dispatch larger than the whole budget can never be admitted, so
    # spend one full interval's worth on it rather than raising ValueError out
    # of the limiter.
    await _embedding_token_rate_limiter.acquire(min(token_count, token_budget))
