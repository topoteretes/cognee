"""Rate-limiting and concurrency-capping context managers for LLM and
embedding dispatch.

This module is the pacing seam: adapters wrap each dispatch attempt with
``llm_rate_limiter_context_manager()`` (inside their tenacity retry) or
``embedding_rate_limiter_context_manager()``. Overload detection and reaction
policy live in ``cognee.infrastructure.llm.overload_policy``; configuration
is read at use-time, never snapshotted at import.
"""

import asyncio
from contextlib import asynccontextmanager, nullcontext

from aiolimiter import AsyncLimiter

# Limiters are built lazily on first use so their budgets bind the resolved
# configuration (LLMConfig gives local inference servers a smaller default
# LLM_RATE_LIMIT_REQUESTS) and so importing this module stays dependency-light.
_llm_rate_limiter: "AsyncLimiter | None" = None
_embedding_rate_limiter: "AsyncLimiter | None" = None

# In-flight caps are asyncio.Semaphores, which are loop-bound and cognee runs
# several event loops (and worker subprocesses) per process, so each cap is
# keyed by (loop id, model). The model is the one an LLM call will actually
# hit (base model, or the model the active pipeline stage is routed to),
# because the cap protects a serving backend: two stages routed to the same
# model share that model's slots. The governing cap comes from the *context*
# config, so a stage routed to a local server gets LOCAL_DEFAULT_MAX_
# CONCURRENT_REQUESTS on its own (stage_config applies it per stage). Caps are
# per-process, the same scope as DATASET_QUEUE_MAX_CONCURRENT. ``loop`` entries
# die with their loop; the dict holds one small tuple per (loop, model), which
# is bounded in practice.
_concurrency_state: "dict[tuple, tuple]" = {}


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


def _llm_concurrency_limit():
    """Return ``(cap, model)`` governing the current call, or ``(0, "")`` when
    the cap is off.

    Reads the *context* config, so pipeline-stage routing decides both which
    backend is hit and which cap governs it: a stage routed to a local server
    carries LOCAL_DEFAULT_MAX_CONCURRENT_REQUESTS on its own config (via
    stage_config), while cloud stages keep the global setting.
    """
    # NOTE: Imports inside function to avoid circular imports at module load.
    from cognee.infrastructure.llm.config import get_llm_context_config

    config = get_llm_context_config()
    cap = getattr(config, "llm_max_concurrent_requests", 0)
    model = getattr(config, "llm_model", "") or ""
    return cap, model


def _get_semaphore(key, cap):
    entry = _concurrency_state.get(key)
    if entry is None or entry[0] != cap:
        entry = (cap, asyncio.Semaphore(cap))
        _concurrency_state[key] = entry
    return entry[1]


@asynccontextmanager
async def _acquire_llm_slots():
    """Hold one in-flight slot for a dispatch attempt.

    Pacing (the RPM bucket) happens BEFORE acquiring a slot so a task never
    idles on the token bucket while occupying a scarce backend slot.
    """
    cap, model = _llm_concurrency_limit()
    if cap <= 0:
        yield
        return

    key = (id(asyncio.get_running_loop()), model)
    semaphore = _get_semaphore(key, cap)
    await semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


@asynccontextmanager
async def _governed_llm_dispatch():
    """Gate one LLM dispatch attempt.

    Everything runs unbounded (the fast path). The RPM limiter paces when
    LLM_RATE_LIMIT_ENABLED is set, or while the overload policy reports an
    active episode — evidence being a rate-limit error, an HTTP 503/529, or
    a timeout, with the episode lapsing back to the configured state after a
    quiet cooldown. The concurrency cap (LLM_MAX_CONCURRENT_REQUESTS) bounds
    in-flight requests per backend model and applies independently of the RPM
    pacing. Adapters enter this context manager INSIDE their tenacity retry,
    so retried attempts are paced and slotted as well.
    """
    # NOTE: Imports inside function to avoid circular imports at module load.
    from cognee.infrastructure.llm.config import get_llm_config
    from cognee.infrastructure.llm.overload_policy import llm_overload_policy

    llm_config = get_llm_config()
    pace = llm_config.llm_rate_limit_enabled or llm_overload_policy.is_paced()
    async with (
        _get_llm_rate_limiter() if pace else nullcontext(),
        _acquire_llm_slots(),
    ):
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
