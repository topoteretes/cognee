"""Rate-limiting context managers for LLM and embedding dispatch.

This module is the pacing seam: adapters wrap each dispatch attempt with
``llm_rate_limiter_context_manager()`` (inside their tenacity retry) or
``embedding_rate_limiter_context_manager()``. Overload detection and reaction
policy live in ``cognee.infrastructure.llm.overload_policy``; configuration
is read at use-time, never snapshotted at import.
"""

from contextlib import asynccontextmanager, nullcontext

from aiolimiter import AsyncLimiter

# Limiters are built lazily on first use so their budgets bind the resolved
# configuration (LLMConfig gives local inference servers a smaller default
# LLM_RATE_LIMIT_REQUESTS) and so importing this module stays dependency-light.
_llm_rate_limiter: "AsyncLimiter | None" = None
_embedding_rate_limiter: "AsyncLimiter | None" = None


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
    # NOTE: Import inside function to avoid a circular import at module load.
    # On this branch the embedding rate-limit knobs still live on LLMConfig
    # (dev's move to EmbeddingConfig is not part of this release).
    from cognee.infrastructure.llm.config import get_llm_config

    llm_config = get_llm_config()
    if not llm_config.embedding_rate_limit_enabled:
        #  Return a no-op context manager if rate limiting is disabled
        return nullcontext()

    if _embedding_rate_limiter is None:
        _embedding_rate_limiter = AsyncLimiter(
            llm_config.embedding_rate_limit_requests,
            llm_config.embedding_rate_limit_interval,
        )
    return _embedding_rate_limiter
