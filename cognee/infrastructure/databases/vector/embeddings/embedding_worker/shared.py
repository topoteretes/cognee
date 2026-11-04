"""Transports shared per event loop and worker endpoint.

Engines with the same worker settings share one transport (one broker
connection), one request budget (``COGNEE_EMBEDDING_WORKER_MAX_CONCURRENT_REQUESTS``)
and one readiness answer, so concurrent callers neither open a connection each
nor pile more requests onto the worker fleet than the operator sized it for.

Connections and semaphores belong to one event loop, so the cache is per loop. A
loop's transports close when it shuts down: ``asyncio.run`` (and every runner
built on it, uvicorn included) finalizes the loop's async generators, which runs
the parked closer below. Adapted from the GLiNER remote adapter's cache
(``cognee/tasks/graph/gliner_demo/remote/adapter.py``), whose tests patch its
module state, so the two are kept apart.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass

from cognee.shared.logging_utils import get_logger

from .errors import EmbeddingWorkerConfigError
from .protocol import EmbeddingWorkerTransport, WorkerInfo
from .settings import EmbeddingWorkerSettings

logger = get_logger("embedding_worker")


@dataclass
class SharedWorker:
    transport: EmbeddingWorkerTransport
    budget: asyncio.Semaphore
    # The readiness probe in flight, if any: callers that need one while it runs
    # share its outcome, a failure included, instead of each probing in turn.
    probe: asyncio.Future[None] | None = None
    ready_until: float = 0.0
    # The worker's model info from the last readiness probe; None before the
    # first probe, or when the worker predates info (see info_unsupported).
    info: WorkerInfo | None = None
    info_unsupported: bool = False


@dataclass
class _LoopWorkers:
    workers: dict[tuple, SharedWorker]
    closer: AsyncGenerator[None, None]


# The values reference their loop, so weak keys would never be released; the
# cache holds the loops strongly and lets go of them itself, see _loop_workers.
_shared: dict[asyncio.AbstractEventLoop, _LoopWorkers] = {}


def load_transport_module(settings: EmbeddingWorkerSettings) -> None:
    """Import the module of the transport ``settings.transport`` names.

    The engine calls it when it is built, so a missing aio-pika is an
    :class:`EmbeddingWorkerConfigError` then rather than on the first embedding.
    """
    settings.require_amqp_endpoint()
    if settings.transport == "amqp":
        from . import amqp  # raises EmbeddingWorkerConfigError without aio-pika

        return
    raise EmbeddingWorkerConfigError(f"Unknown embedding worker transport {settings.transport!r}.")


def build_transport(settings: EmbeddingWorkerSettings) -> EmbeddingWorkerTransport:
    """Create the transport ``settings.transport`` names."""
    endpoint = settings.require_amqp_endpoint()
    if settings.transport == "amqp":
        from .amqp import AmqpEmbeddingTransport

        return AmqpEmbeddingTransport(
            endpoint,
            queue=settings.amqp_queue,
            connect_timeout=settings.connect_timeout_secs,
            request_timeout=settings.request_timeout_secs,
        )
    raise EmbeddingWorkerConfigError(f"Unknown embedding worker transport {settings.transport!r}.")


def _shared_key(settings: EmbeddingWorkerSettings) -> tuple:
    return (
        settings.transport,
        settings.endpoint,
        settings.amqp_queue,
        settings.connect_timeout_secs,
        settings.request_timeout_secs,
        settings.max_concurrent_requests,
    )


async def _close_workers(workers: Iterable[SharedWorker]) -> None:
    for worker in list(workers):
        try:
            await worker.transport.aclose()
        except Exception:  # closing is best effort; it must not mask the caller's outcome
            logger.debug("Closing an embedding worker transport failed", exc_info=True)


async def _close_at_loop_end(
    loop: asyncio.AbstractEventLoop, workers: dict[tuple, SharedWorker]
) -> AsyncGenerator[None, None]:
    """Parked on its loop until ``shutdown_asyncgens`` finalizes it."""
    try:
        yield
    finally:
        entry = _shared.get(loop)
        if entry is not None and entry.workers is workers:
            del _shared[loop]
        await _close_workers(workers.values())
        workers.clear()


async def _loop_workers() -> dict[tuple, SharedWorker]:
    loop = asyncio.get_running_loop()
    # A loop closed without finalizing its generators never ran its closer.
    for stale in [other for other in _shared if other.is_closed()]:
        del _shared[stale]
    entry = _shared.get(loop)
    if entry is None:
        workers: dict[tuple, SharedWorker] = {}
        closer = _close_at_loop_end(loop, workers)
        await closer.asend(None)  # registers it with the loop's shutdown_asyncgens
        entry = _shared[loop] = _LoopWorkers(workers, closer)
    return entry.workers


async def shared_worker(settings: EmbeddingWorkerSettings) -> SharedWorker:
    workers = await _loop_workers()
    key = _shared_key(settings)
    worker = workers.get(key)
    if worker is None:
        worker = SharedWorker(
            transport=build_transport(settings),
            budget=asyncio.Semaphore(settings.max_concurrent_requests),
        )
        workers[key] = worker
    return worker


async def close_embedding_worker_transports() -> None:
    """Close the transports this event loop opened now (tests, orderly shutdown).

    Not required for correctness: they also close when the loop shuts down.
    """
    entry = _shared.pop(asyncio.get_running_loop(), None)
    if entry is not None:
        await _close_workers(entry.workers.values())
        entry.workers.clear()
        await entry.closer.aclose()
