"""
Coalescing wrapper around an :class:`EmbeddingEngine`.

Accumulates concurrent ``embed_text`` calls into a shared queue and dispatches
them to the underlying engine as a single batched request, either when:

* the total number of pending strings reaches the inner engine's
  ``get_batch_size()``, or
* the oldest item in the queue has been waiting longer than
  ``flush_timeout_seconds``.

Designed for cognee's single-event-loop usage (one shared engine instance
per process — see ``create_embedding_engine``'s ``@lru_cache``). Internal
state is protected by an ``asyncio.Lock``; callers running on a different
loop are not supported.

Note on recursive context-window splits: the inner engine (e.g.
``LiteLLMEmbeddingEngine``) recurses on ``ContextWindowExceededError`` by
calling ``self.embed_text`` on the split halves. Because ``self`` there
refers to the inner engine — not this wrapper — those recursive calls bypass
the queue, which is intentional: a merged batch that exceeded the context
window should be re-split immediately, not re-queued.
"""

from __future__ import annotations

import asyncio

from cognee.shared.logging_utils import get_logger

from .EmbeddingEngine import EmbeddingEngine

logger = get_logger("AccumulatingEmbeddingEngine")


_PendingItem = tuple[list[str], asyncio.Future]


class AccumulatingEmbeddingEngine:
    """
    Transparent wrapper that batches concurrent ``embed_text`` calls.

    Public surface mirrors :class:`EmbeddingEngine`; any attribute not
    defined here is forwarded to the wrapped engine so callers can keep
    using ``tokenizer``, ``get_vector_size()``, etc. without changes.
    """

    def __init__(
        self,
        inner: EmbeddingEngine,
        flush_timeout_seconds: float = 0.1,
    ) -> None:
        if flush_timeout_seconds <= 0:
            raise ValueError("flush_timeout_seconds must be positive")

        max_batch_size = inner.get_batch_size()
        if not max_batch_size or max_batch_size <= 0:
            raise ValueError(
                f"Inner engine returned non-positive batch size ({max_batch_size!r}); "
                "AccumulatingEmbeddingEngine needs a real batch size to coalesce."
            )

        self._inner = inner
        self._max_batch_size = max_batch_size
        self._flush_timeout = flush_timeout_seconds

        self._lock = asyncio.Lock()
        self._queue: list[_PendingItem] = []
        self._pending_strings = 0
        self._oldest_enqueued_at: float | None = None
        self._timer_task: asyncio.Task | None = None
        # Keep references to in-flight dispatch tasks so they aren't garbage
        # collected mid-flight (which would cancel the underlying API call).
        self._inflight: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def embed_text(self, text: list[str]) -> list[list[float]]:
        if not text:
            return []

        # Oversized request: pass through directly. Splitting it into the
        # accumulator would just delay it and the inner engine already
        # knows how to handle large inputs. Still validate the response
        # length so the bypass path can't silently return a wrong-sized
        # result while the coalesced path raises.
        if len(text) >= self._max_batch_size:
            vectors = await self._inner.embed_text(text)
            if len(vectors) != len(text):
                raise RuntimeError(
                    f"Embedding engine returned {len(vectors)} vectors for {len(text)} inputs."
                )
            return vectors

        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[list[float]]] = loop.create_future()

        items_to_dispatch: list[_PendingItem] | None = None
        async with self._lock:
            # If appending would overflow the batch, flush the current queue
            # first so this request starts a fresh accumulation window.
            if self._queue and self._pending_strings + len(text) > self._max_batch_size:
                items_to_dispatch = self._drain_locked()

            self._queue.append((text, future))
            self._pending_strings += len(text)
            if self._oldest_enqueued_at is None:
                self._oldest_enqueued_at = loop.time()

            if self._pending_strings >= self._max_batch_size:
                # Batch is full — flush immediately rather than wait.
                immediate = self._drain_locked()
                if items_to_dispatch is None:
                    items_to_dispatch = immediate
                else:
                    items_to_dispatch.extend(immediate)
            else:
                self._ensure_timer_locked()

        if items_to_dispatch:
            # Dispatch outside the lock so we don't serialize the API call.
            self._start_dispatch(items_to_dispatch)

        return await future

    def get_vector_size(self) -> int:
        return self._inner.get_vector_size()

    def get_batch_size(self) -> int:
        return self._max_batch_size

    # ------------------------------------------------------------------ #
    # Transparent forwarding for everything else (tokenizer, etc.)
    # ------------------------------------------------------------------ #

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _drain_locked(self) -> list[_PendingItem]:
        """Atomically remove and return all queued items. Caller holds lock."""
        items = self._queue
        self._queue = []
        self._pending_strings = 0
        self._oldest_enqueued_at = None
        if self._timer_task is not None and not self._timer_task.done():
            # Cancellation is delivered asynchronously, so a freshly-started
            # timer may still run one more loop iteration before observing
            # the cancel. That's harmless: the next iteration takes the lock
            # and sees an empty queue (or one re-armed under a fresh timer).
            self._timer_task.cancel()
        self._timer_task = None
        return items

    def _ensure_timer_locked(self) -> None:
        """Start the flush timer if one isn't already running. Caller holds lock."""
        if self._timer_task is None or self._timer_task.done():
            self._timer_task = asyncio.create_task(self._timer_loop())

    async def _timer_loop(self) -> None:
        """
        Sleep until the oldest queued item is at least ``flush_timeout``
        seconds old, then flush. Re-arms itself if it wakes up early
        (e.g. because items were drained by a batch-full flush and a new
        item was enqueued after).
        """
        while True:
            async with self._lock:
                if not self._queue or self._oldest_enqueued_at is None:
                    # Nothing to do — let the next embed_text restart us.
                    self._timer_task = None
                    return
                age = asyncio.get_running_loop().time() - self._oldest_enqueued_at
                remaining = self._flush_timeout - age
                if remaining <= 0:
                    items = self._drain_locked()
                    # We just nulled self._timer_task in _drain_locked;
                    # dispatch and exit.
                    if items:
                        self._start_dispatch(items)
                    return
            await asyncio.sleep(remaining)

    def _start_dispatch(self, items: list[_PendingItem]) -> None:
        """Spawn a dispatch task and retain a reference until it finishes."""
        task = asyncio.create_task(self._dispatch(items))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def _dispatch(self, items: list[_PendingItem]) -> None:
        """Merge queued requests into a single API call and split the result."""
        # If every waiter has been cancelled before we even started, skip the
        # API call entirely.
        if all(future.done() for _, future in items):
            return

        merged: list[str] = []
        sizes: list[int] = []
        for texts, _ in items:
            merged.extend(texts)
            sizes.append(len(texts))

        try:
            vectors = await self._inner.embed_text(merged)
        except BaseException as exc:  # noqa: BLE001 — propagate to every waiter
            for _, future in items:
                if not future.done():
                    future.set_exception(exc)
            return

        if len(vectors) != len(merged):
            # Underlying engine returned a mismatched count. Fail loudly
            # so callers don't silently get the wrong vectors.
            err = RuntimeError(
                f"Embedding engine returned {len(vectors)} vectors for "
                f"{len(merged)} inputs; cannot split across {len(items)} callers."
            )
            logger.error("%s", err)
            for _, future in items:
                if not future.done():
                    future.set_exception(err)
            return

        idx = 0
        for (_, future), size in zip(items, sizes):
            slice_ = vectors[idx : idx + size]
            idx += size
            if not future.done():
                future.set_result(slice_)
