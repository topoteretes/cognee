"""Graceful shutdown for Cognee.

Ensures all database adapters, embedding clients, executor threads, and
pending async close tasks are drained before the process exits.  Without
this, ``asyncio.run()`` returns while non-daemon ``ThreadPoolExecutor``
threads (Ladybug adapter) and unclosed ``httpx.AsyncClient`` instances
(``AsyncOpenAI`` in the embedding engine) keep the process alive
indefinitely.

Usage — at the end of any script or CLI command::

    await cognee.shutdown()          # inside an async context
    # or
    cognee.shutdown_sync()           # from sync code (spawns a thread)
    # or
    cognee.shutdown_sync(force=True) # calls os._exit as a last resort
"""

import asyncio
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Timeout (seconds) for draining pending async close tasks.
_DRAIN_TIMEOUT: float = 5.0
# Timeout (seconds) for the hard os._exit fallback.
_FORCE_EXIT_TIMEOUT: float = 10.0


async def shutdown(*, drain_timeout: float = _DRAIN_TIMEOUT) -> None:
    """Gracefully release all Cognee resources.

    1. Clear the ``closing_lru_cache`` instances for graph and vector
       engines, which triggers their ``close()`` methods.
    2. Drain all pending async close tasks tracked by ``_PENDING_CLOSE_TASKS``.
    3. Close the cached embedding engine's HTTP client (if applicable).
    """
    # --- 1. Clear database adapter caches (triggers async close) ----------
    _clear_database_caches()

    # --- 2. Drain pending close tasks from closing_lru_cache --------------
    await _drain_pending_close_tasks(timeout=drain_timeout)

    # --- 3. Close embedding engine HTTP client ----------------------------
    _close_embedding_engine()

    logger.debug("Cognee shutdown complete.")


def _clear_database_caches() -> None:
    """Clear closing_lru_cache instances for graph and vector engines."""
    try:
        from cognee.infrastructure.databases.graph.get_graph_engine import (
            _create_graph_engine,
        )

        if hasattr(_create_graph_engine, "cache_clear"):
            _create_graph_engine.cache_clear()
    except Exception:
        logger.debug("Could not clear graph engine cache", exc_info=True)

    try:
        from cognee.infrastructure.databases.vector.create_vector_engine import (
            _create_vector_engine,
        )

        if hasattr(_create_vector_engine, "cache_clear"):
            _create_vector_engine.cache_clear()
    except Exception:
        logger.debug("Could not clear vector engine cache", exc_info=True)


async def _drain_pending_close_tasks(timeout: float) -> None:
    """Wait for all fire-and-forget close tasks to finish."""
    try:
        from cognee.infrastructure.databases.utils.closing_lru_cache import (
            _PENDING_CLOSE_TASKS,
        )

        if _PENDING_CLOSE_TASKS:
            logger.debug("Draining %d pending close task(s)...", len(_PENDING_CLOSE_TASKS))
            pending = list(_PENDING_CLOSE_TASKS)
            done, still_pending = await asyncio.wait(pending, timeout=timeout)
            if still_pending:
                logger.warning(
                    "%d close task(s) did not finish within %.1fs — cancelling.",
                    len(still_pending),
                    timeout,
                )
                for task in still_pending:
                    task.cancel()
    except Exception:
        logger.debug("Error draining pending close tasks", exc_info=True)


def _close_embedding_engine() -> None:
    """Close the AsyncOpenAI client held by the cached embedding engine."""
    try:
        from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
            create_embedding_engine,
        )

        # The lru_cache doesn't expose cached values directly, but we can
        # access the engine through the vector engine handle.
        from cognee.infrastructure.databases.vector import get_vector_engine

        try:
            engine = get_vector_engine()
            emb = engine.embedding_engine
            if hasattr(emb, "close"):
                emb.close()
            elif hasattr(emb, "_client") and hasattr(emb._client, "close"):
                # Synchronously close the httpx client inside AsyncOpenAI
                try:
                    emb._client.close()
                except Exception:
                    pass
        except Exception:
            pass

        # Also clear the embedding engine lru_cache itself
        if hasattr(create_embedding_engine, "cache_clear"):
            create_embedding_engine.cache_clear()
    except Exception:
        logger.debug("Could not close embedding engine", exc_info=True)


def shutdown_sync(
    *,
    drain_timeout: float = _DRAIN_TIMEOUT,
    force: bool = False,
    force_timeout: float = _FORCE_EXIT_TIMEOUT,
) -> None:
    """Synchronous wrapper for :func:`shutdown`.

    Parameters
    ----------
    force : bool
        If True, start a watchdog thread that calls ``os._exit(0)`` after
        *force_timeout* seconds if the process hasn't exited yet.  This
        is the nuclear option for environments (like CLI scripts) where
        a clean exit is more important than resource cleanup.
    """
    if force:
        _start_force_exit_watchdog(force_timeout)

    # If there's a running event loop (unlikely after asyncio.run()),
    # schedule the shutdown.  Otherwise spin up a temporary loop.
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(shutdown(drain_timeout=drain_timeout))
    except RuntimeError:
        try:
            asyncio.run(shutdown(drain_timeout=drain_timeout))
        except Exception:
            logger.debug("Sync shutdown failed", exc_info=True)


def _start_force_exit_watchdog(timeout: float) -> None:
    """Spawn a daemon thread that calls ``os._exit(0)`` after *timeout*."""

    def _watchdog():
        import time

        time.sleep(timeout)
        logger.warning("Cognee process did not exit within %.1fs — forcing exit.", timeout)
        os._exit(0)

    t = threading.Thread(target=_watchdog, daemon=True)
    t.start()
