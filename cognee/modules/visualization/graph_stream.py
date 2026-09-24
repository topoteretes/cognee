"""Server-sent-events transport for the streamed ``GET /v1/visualize/json``.

A graph of 20k nodes does not fit one response: at the Mindmap's measured
6.67 MB per 1000 nodes with full properties it is over 100 MB. The streamed
read sends the SDK-786 chunks in compact form, then the graph-wide fields, so
neither the pod nor the client ever holds the whole graph with its properties.

Events, in order: ``meta`` (seeds, seed source, bounds), one ``chunk`` per
store chunk (compact ``nodes`` and ``links``; every link's endpoints were
already sent), ``summary`` (``importance`` and ``label_priority`` per node, plus
``entity_type`` where it corrects a type a chunk sent as "Entity", in events of
at most ``chunk_size`` nodes, the first also carrying ``color_maps.node_set``), ``done`` (totals). A failure after the 200 is one
``error`` event carrying the status the JSON path would have returned.

Four properties are deliberate and easy to break:

* **Nothing is sent until the first chunk is read or the read failed.** Seed
  resolution and the store's membership query run before the headers, so a
  failure there keeps its real status code instead of becoming a 200 with an
  error event, the same trade ``recall_stream`` makes.
* **The read never waits for the client.** It runs in a task of its own,
  because the dataset's database context it holds, and that context's
  ``DatasetQueue`` slot, belong to the task that entered them. It reads to the
  end into an unbounded queue of encoded frames and leaves the context, so the
  slot is held for as long as the store takes, about 2 s at 20k nodes, never
  for as long as a slow client takes. Holding it across a slow client let a
  handful of them stall every dataset operation in the process.
* **Memory is capped by streams, not by clients.** An in-flight stream holds
  its compact graph until the client has taken it (5.8 MB measured at 20k
  nodes, more with dense links). At most ``MAX_STREAMS_IN_FLIGHT`` run at once
  and the next is refused with a 503, and a stream the client has not finished
  within ``STREAM_LIFETIME_SECONDS`` is dropped, so a stalled client, or a
  body the server never iterated, cannot hold either for long.
* **A disconnect cancels the read.** Unlike a recall, which detaches to persist
  its answer, a graph read has nothing left worth finishing, and cancelling it
  closes the chunk's session and returns its connection to the pool.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

from cognee.api.sse import KEEPALIVE_COMMENT, encode_sse
from cognee.exceptions import CogneeApiError
from cognee.shared.logging_utils import get_logger

from .exceptions import GraphStreamCapacityError
from .preprocessor import (
    COMPACT_PROPERTY_KEYS,
    CompactGraphAccumulator,
    compact_chunk,
)
from .subgraph_data import iter_seed_neighborhood, resolve_seed_node_ids

logger = get_logger("visualization.graph_stream")

# Nodes per chunk event. At about 250 bytes per compact node that is ~0.25 MB
# of nodes; a hub's links can add as much again.
STREAM_CHUNK_SIZE = 1000

# The same interval recall uses, below common proxy idle timeouts.
KEEPALIVE_SECONDS = 15.0

# Streams whose frames this process may hold at once. Refused beyond it rather
# than queued: a waiter would need a timeout around Semaphore.acquire, and on
# Python 3.10 and 3.11 asyncio.wait_for can drop a permit granted just as it
# times out.
MAX_STREAMS_IN_FLIGHT = 8

# How long one stream may keep its permit and its frames. A 20k-node graph is
# a few MB; a client that has not taken it in this long has stalled.
STREAM_LIFETIME_SECONDS = 300.0

# Keeps a strong reference to running reads so they are not collected mid-read.
_STREAM_TASKS: set = set()

_permits: asyncio.Semaphore | None = None


def _stream_permits() -> asyncio.Semaphore:
    """The process-wide cap on streams in flight, created on first use."""
    global _permits
    if _permits is None:
        _permits = asyncio.Semaphore(MAX_STREAMS_IN_FLIGHT)
    return _permits


async def stream_graph_events(
    graph_engine: Any,
    *,
    query: str | None,
    seed_node_ids: list[str] | None,
    neighborhood_depth: int,
    seed_top_k: int,
    max_nodes: int,
    chunk_size: int = STREAM_CHUNK_SIZE,
) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
    """The streamed graph as ``(event, data)`` pairs, read chunk by chunk."""
    seeds, source = await resolve_seed_node_ids(
        graph_engine, seed_node_ids=seed_node_ids, query=query, seed_top_k=seed_top_k
    )
    yield (
        "meta",
        {
            "seeds": seeds,
            "seed_source": source,
            "depth": neighborhood_depth,
            "max_nodes": max_nodes,
            "chunk_size": chunk_size,
        },
    )

    accumulator = CompactGraphAccumulator()
    chunks = 0
    if seeds:
        async for nodes_data, edges_data in iter_seed_neighborhood(
            graph_engine,
            seeds,
            neighborhood_depth,
            max_nodes,
            chunk_size=chunk_size,
            property_keys=COMPACT_PROPERTY_KEYS,
        ):
            nodes, links = compact_chunk(nodes_data, edges_data)
            accumulator.add(nodes, links)
            yield "chunk", {"index": chunks, "nodes": nodes, "links": links}
            chunks += 1

    # One summary event per chunk_size nodes: at 20k nodes a single one is
    # about 1.5 MB, three times the size of a chunk.
    summary = accumulator.summary()
    entries = list(summary["nodes"].items())
    for start in range(0, max(len(entries), 1), chunk_size):
        part: dict[str, Any] = {"nodes": dict(entries[start : start + chunk_size])}
        if start == 0:
            part["color_maps"] = summary["color_maps"]
        yield "summary", part
    yield (
        "done",
        {"nodes": accumulator.node_count, "links": accumulator.link_count, "chunks": chunks},
    )


_END = object()
_EXPIRED = object()


async def _produce(
    events: AsyncGenerator[tuple[str, Any], None], queue: asyncio.Queue, stats: dict[str, Any]
) -> None:
    """Run the read to its end, queueing each event as an encoded SSE frame.

    Nothing here waits for the client, so the read, and the dataset slot its
    context holds, last only as long as the store takes. Encoding here, not in
    the response body, turns a value JSON cannot encode into a failed read,
    which reaches the client as an error event instead of cutting the stream.
    """
    # aclosing unwinds the read's own context managers (dataset context, chunk
    # session) in this task, where they were entered, on cancel as on failure.
    async with aclosing(events):
        async for event, data in events:
            if event in ("meta", "done"):
                stats[event] = data
            queue.put_nowait((event, encode_sse(event, data)))
            # Nothing else here awaits, and an adapter that yields chunks from
            # memory (the interface default does) would otherwise encode the
            # whole graph without letting the event loop run.
            await asyncio.sleep(0)
    queue.put_nowait(_END)


def _error_payload(error: BaseException) -> dict[str, Any]:
    """The status the JSON path would have returned, carried as data."""
    status = getattr(error, "status_code", None)
    if isinstance(error, CogneeApiError) and isinstance(status, int):
        return {"message": str(getattr(error, "message", None) or error), "status": status}
    return {"message": "Failed to build the visualization payload", "status": 409}


class _Failure:
    def __init__(self, error: BaseException) -> None:
        self.error = error


class GraphStream:
    """A streamed graph read, relaying its queued frames to the response body."""

    def __init__(self, task: asyncio.Task, queue: asyncio.Queue, stats: dict[str, Any]) -> None:
        self._task = task
        self._queue = queue
        self._stats = stats
        self._getter: asyncio.Future | None = None
        self._buffered: list = []
        self._started = time.monotonic()
        self._first_chunk_ms = 0.0
        self._holds_permit = True
        self._lifetime = asyncio.get_running_loop().call_later(
            STREAM_LIFETIME_SECONDS, self._expire
        )

    async def _next(self, timeout: float | None) -> Any:
        """The next queued item, a ``_Failure``, or ``None`` if ``timeout`` passed.

        Everything the read queued is delivered before its failure. The pending
        get survives a timeout, so a frame is never lost to a keepalive.
        """
        while True:
            if self._getter is None:
                self._getter = asyncio.ensure_future(self._queue.get())
            done, _ = await asyncio.wait(
                {self._getter, self._task}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if self._getter in done:
                item, self._getter = self._getter.result(), None
                return item
            if not done:
                return None
            # The read ended with nothing queued. A read that finished queued
            # _END first, so only a failed or cancelled read gets here for good.
            if self._queue.empty():
                if self._task.cancelled():
                    return _Failure(asyncio.CancelledError())
                error = self._task.exception()
                if error is not None:
                    return _Failure(error)
                return _END

    async def _start(self) -> None:
        """Buffer frames up to the first chunk, or raise the read's failure."""
        while True:
            item = await self._next(timeout=None)
            if isinstance(item, _Failure):
                raise item.error
            self._buffered.append(item)
            # _EXPIRED here means the first chunk took the whole lifetime; the
            # body then reports it as its error event.
            if item is _END or item is _EXPIRED or item[0] in ("chunk", "done"):
                self._first_chunk_ms = (time.monotonic() - self._started) * 1000
                return

    async def frames(self) -> AsyncGenerator[str, None]:
        """The SSE body."""
        sent_bytes = 0
        cancelled = False
        try:
            while True:
                if self._buffered:
                    item = self._buffered.pop(0)
                else:
                    item = await self._next(timeout=KEEPALIVE_SECONDS)
                if item is None:
                    yield KEEPALIVE_COMMENT
                    continue
                if item is _END:
                    return
                if item is _EXPIRED:
                    yield encode_sse(
                        "error",
                        {
                            "message": "The graph stream was not read in time and was closed.",
                            "status": 408,
                        },
                    )
                    return
                if isinstance(item, _Failure):
                    logger.error(
                        "Streamed visualization failed after the response started",
                        exc_info=item.error,
                    )
                    yield encode_sse("error", _error_payload(item.error))
                    return
                _, frame = item
                sent_bytes += len(frame)
                yield frame
        except (asyncio.CancelledError, GeneratorExit):
            cancelled = True
            raise
        finally:
            await self.close()
            meta = self._stats.get("meta", {})
            totals = self._stats.get("done", {})
            logger.info(
                "Streamed visualization: seeds=%d source=%s depth=%s max_nodes=%s "
                "nodes=%s links=%s chunks=%s first_chunk_ms=%.0f total_ms=%.0f "
                "bytes=%d cancelled=%s",
                len(meta.get("seeds", [])),
                meta.get("seed_source"),
                meta.get("depth"),
                meta.get("max_nodes"),
                totals.get("nodes"),
                totals.get("links"),
                totals.get("chunks"),
                self._first_chunk_ms,
                (time.monotonic() - self._started) * 1000,
                sent_bytes,
                cancelled,
            )

    def _drop(self) -> None:
        """Stop the read, free the queued frames and give the permit back."""
        self._lifetime.cancel()
        if self._getter is not None:
            self._getter.cancel()
            self._getter = None
        if not self._task.done():
            self._task.cancel()
        while not self._queue.empty():
            self._queue.get_nowait()
        self._buffered.clear()
        if self._holds_permit:
            self._holds_permit = False
            _stream_permits().release()

    def _expire(self) -> None:
        """The lifetime ran out: drop everything, and tell frames() if it resumes."""
        logger.warning(
            "Streamed visualization dropped: not read within %.0f s", STREAM_LIFETIME_SECONDS
        )
        self._drop()
        self._queue.put_nowait(_EXPIRED)

    async def close(self) -> None:
        """Cancel the read if it is still running and release what the stream holds."""
        self._drop()
        # return_exceptions: the read's own failure was already reported.
        await asyncio.gather(self._task, return_exceptions=True)


async def begin_graph_stream(events: AsyncGenerator[tuple[str, Any], None]) -> GraphStream:
    """Start the read and wait for its first chunk, or its end, or its failure.

    Raises ``GraphStreamCapacityError`` (503) when ``MAX_STREAMS_IN_FLIGHT``
    streams are already open, before anything is read, and otherwise whatever
    the read raised before its first chunk, unchanged, so the route's handlers
    give it the status code the JSON path would.
    """
    permits = _stream_permits()
    if permits.locked():
        await events.aclose()
        raise GraphStreamCapacityError()
    # Not locked, so this returns at once and never parks on the event loop.
    await permits.acquire()

    queue: asyncio.Queue = asyncio.Queue()
    stats: dict[str, Any] = {}
    task = asyncio.create_task(_produce(events, queue, stats))
    _STREAM_TASKS.add(task)
    task.add_done_callback(_STREAM_TASKS.discard)

    stream = GraphStream(task, queue, stats)
    try:
        await stream._start()
    except BaseException:
        # A failure, or the client gone before the headers were sent.
        await stream.close()
        raise
    return stream
