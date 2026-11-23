"""Rate-limiting and concurrency-capping context managers for LLM and
embedding dispatch.

This module is the pacing seam: adapters wrap each dispatch attempt with
``llm_rate_limiter_context_manager()`` (inside their tenacity retry) or
``embedding_rate_limiter_context_manager()``. Overload detection and reaction
policy live in ``cognee.shared.overload_policy``; each side's policy instance
sits with its config. Configuration is read at use-time, never snapshotted at
import.
"""

import asyncio
from contextlib import asynccontextmanager

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
# config, the one pipeline-stage routing selects. Caps are
# per-process, the same scope as DATASET_QUEUE_MAX_CONCURRENT. Entries are
# never pruned: the dict keeps one (cap, semaphore) per (loop, model) for the
# life of the process, and a semaphore that ever had a waiter keeps its closed
# loop alive.
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


def _get_embedding_rate_limiter() -> AsyncLimiter:
    global _embedding_rate_limiter
    if _embedding_rate_limiter is None:
        # NOTE: Import inside function to avoid a circular import at module load.
        from cognee.infrastructure.databases.vector.embeddings.config import (
            get_embedding_config,
        )

        embedding_config = get_embedding_config()
        _embedding_rate_limiter = AsyncLimiter(
            embedding_config.embedding_rate_limit_requests,
            embedding_config.embedding_rate_limit_interval,
        )
    return _embedding_rate_limiter


def _llm_concurrency_limit():
    """Return ``(cap, model)`` governing the current call, or ``(0, "")`` when
    the cap is off.

    Reads the *context* config, so pipeline-stage routing decides which
    backend is hit and therefore which model's slots the call takes.
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
async def _paced_slot(should_pace, get_limiter, semaphore):
    """Pace (when ``should_pace()``) and then hold one slot of ``semaphore``.

    Pacing comes before the slot, so a task never waits on the token bucket
    while holding a scarce slot. Pacing is decided again once the slot is held:
    throttling can begin while a request waits for its slot, and such a
    request gives the slot back, takes a token, and queues again instead of
    reaching the provider unpaced. An attempt takes at most one token.
    ``semaphore`` is None when no in-flight cap applies.
    """
    paced = False
    while True:
        if not paced and should_pace():
            async with get_limiter():
                pass
            paced = True
        if semaphore is None:
            break
        await semaphore.acquire()
        if paced or not should_pace():
            break
        semaphore.release()
    try:
        yield
    finally:
        if semaphore is not None:
            semaphore.release()


def _llm_slot_semaphore() -> asyncio.Semaphore | None:
    """The in-flight slots of the model this call hits, or None when the cap is off."""
    cap, model = _llm_concurrency_limit()
    if cap <= 0:
        return None
    key = (id(asyncio.get_running_loop()), model)
    return _get_semaphore(key, cap)


def _embedding_slot_semaphore(cap: int) -> asyncio.Semaphore:
    """The process-wide embedding request slots of this event loop.

    One cap for every embedding caller on this event loop, not per model or per
    context (UNR-069).
    """
    key = ("embedding_requests", id(asyncio.get_running_loop()))
    return _get_semaphore(key, cap)


@asynccontextmanager
async def embedding_batch_slot():
    """Hold one of the process-wide slots for an indexing write batch.

    Sized by EMBEDDING_MAX_CONCURRENT_REQUESTS like the request slots, so every
    request slot can be busy while other batches are being written. A batch takes
    this slot before any request slot, and requests never take batch slots, so the
    two cannot deadlock.
    """
    # NOTE: Import inside function to avoid a circular import at module load.
    from cognee.infrastructure.databases.vector.embeddings.config import (
        get_embedding_config,
    )

    cap = get_embedding_config().embedding_max_concurrent_requests
    key = ("embedding_batches", id(asyncio.get_running_loop()))
    async with _get_semaphore(key, cap):
        yield


@asynccontextmanager
async def _governed_llm_dispatch():
    """Gate one LLM dispatch attempt.

    Everything runs unbounded (the fast path). The RPM limiter paces when
    LLM_RATE_LIMIT_ENABLED is set, or while the overload policy reports an
    active throttling period — evidence being a rate-limit error, an HTTP 503/529,
    or a timeout, with the period lapsing back to the configured state after a
    quiet cooldown. The concurrency cap (LLM_MAX_CONCURRENT_REQUESTS) bounds
    in-flight requests per backend model and applies independently of the RPM
    pacing. Adapters enter this context manager INSIDE their tenacity retry,
    so retried attempts are paced and slotted as well.
    """
    # NOTE: Imports inside function to avoid circular imports at module load.
    from cognee.infrastructure.llm.config import get_llm_config, llm_overload_policy

    llm_config = get_llm_config()

    def should_pace() -> bool:
        return llm_config.llm_rate_limit_enabled or llm_overload_policy.is_paced()

    async with _paced_slot(should_pace, _get_llm_rate_limiter, _llm_slot_semaphore()):
        try:
            yield
        except Exception as error:
            if llm_config.auto_rate_limit:
                llm_overload_policy.on_error(error)
            raise


def llm_rate_limiter_context_manager():
    return _governed_llm_dispatch()


@asynccontextmanager
async def _governed_embedding_dispatch():
    """Gate one embedding request attempt, the embedding twin of the LLM seam.

    The RPM limiter paces when EMBEDDING_RATE_LIMIT_ENABLED is set, or while the
    embedding overload policy reports a throttling period (its own, never the LLM one).
    Then the attempt holds one slot of the process-wide cap
    EMBEDDING_MAX_CONCURRENT_REQUESTS while in flight. Engines enter this INSIDE
    their retry, so retried attempts are paced and slotted as well.
    """
    # NOTE: Imports inside function to avoid a circular import at module load
    # (embedding engines, which EmbeddingConfig is reached through, import
    # this module).
    from cognee.infrastructure.databases.vector.embeddings.config import (
        embedding_overload_policy,
        get_embedding_config,
    )
    from cognee.infrastructure.llm.config import get_llm_config

    embedding_config = get_embedding_config()

    def should_pace() -> bool:
        return embedding_config.embedding_rate_limit_enabled or embedding_overload_policy.is_paced()

    async with _paced_slot(
        should_pace,
        _get_embedding_rate_limiter,
        _embedding_slot_semaphore(embedding_config.embedding_max_concurrent_requests),
    ):
        try:
            yield
        except Exception as error:
            # AUTO_RATE_LIMIT is one switch for both sides; each throttles on its own.
            if get_llm_config().auto_rate_limit:
                embedding_overload_policy.on_error(error)
            raise


def embedding_rate_limiter_context_manager():
    return _governed_embedding_dispatch()
