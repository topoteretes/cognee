"""Per-channel ingestion buffer + cognify trigger for the Slack bot (issue #3609).

This sits on top of the Commit-2 :class:`ChatMemory` adapter and decides *when*
to turn buffered messages into a knowledge graph. It does not call cognee
directly — it orchestrates the adapter's ``ingest`` / ``flush`` / ``answer``.

Trigger policy:

1. **Size threshold** — flush (cognify) a channel once ``cognify_batch_size``
   messages have accumulated for it. This is the primary trigger; batching keeps
   cost sane (a full cognify per message would be prohibitively expensive).
2. **On-demand before answer** — :meth:`answer` flushes any pending messages for
   the channel first, so a question always reflects everything said so far.

The per-channel pending counter is only a *trigger heuristic*. cognee's
``cognify`` reprocesses the whole (incrementally-loaded) dataset, so even if a
flush is skipped or fails, a later flush still picks up any un-cognified
messages — nothing is permanently missed.

Accumulation is guarded by an ``asyncio.Lock``; the slow adapter awaits
(``ingest`` / ``flush``) happen outside the lock so they never serialize.
"""

from __future__ import annotations

import asyncio

from .memory_adapter import Answer, ChatMemory, ConversationRef

# Default number of buffered messages that triggers a cognify for a channel.
DEFAULT_COGNIFY_BATCH_SIZE = 10


class IngestionBuffer:
    """Buffers ingested messages per channel and cognifies once a batch fills."""

    def __init__(
        self,
        memory: ChatMemory,
        *,
        batch_size: int = DEFAULT_COGNIFY_BATCH_SIZE,
    ):
        self._memory = memory
        self._batch_size = max(1, batch_size)

        self._lock = asyncio.Lock()
        # channel_id -> count of messages ingested but not yet cognified.
        self._pending: dict[str, int] = {}
        # channel_id -> lock serializing cognify for that channel (see flush()).
        self._flush_locks: dict[str, asyncio.Lock] = {}

    def pending_count(self, channel_id: str) -> int:
        """Number of buffered-but-not-yet-cognified messages for a channel."""
        return self._pending.get(channel_id, 0)

    async def add_message(
        self,
        ref: ConversationRef,
        *,
        ts: str,
        text: str,
        permalink: str,
        author: str,
    ) -> None:
        """Ingest a message into its channel and flush if a trigger fires."""
        # Ingest (the slow cognee.add) happens outside the lock.
        await self._memory.ingest(ref, ts=ts, text=text, permalink=permalink, author=author)

        async with self._lock:
            channel_id = ref.channel_id
            self._pending[channel_id] = self._pending.get(channel_id, 0) + 1
            should_flush = self._pending[channel_id] >= self._batch_size

        if should_flush:
            await self.flush(ref)

    async def flush(self, ref: ConversationRef) -> None:
        """Cognify the channel's dataset if it has pending messages; else no-op.

        Empty-buffer flush never calls the adapter (clean no-op). The counter is
        reset under the lock, then cognify runs under a per-channel lock so a
        size-triggered flush and an on-demand (answer-time) flush never run
        cognee.cognify on the same dataset concurrently — overlapping cognify on
        one dataset can corrupt/deadlock the underlying store.
        """
        channel_id = ref.channel_id
        async with self._lock:
            if self._pending.get(channel_id, 0) == 0:
                return
            self._pending[channel_id] = 0
            flush_lock = self._flush_locks.get(channel_id)
            if flush_lock is None:
                flush_lock = self._flush_locks[channel_id] = asyncio.Lock()

        async with flush_lock:
            await self._memory.flush(ref)

    async def answer(self, ref: ConversationRef, *, query: str) -> Answer:
        """Flush pending messages for the channel, then answer the query."""
        await self.flush(ref)
        return await self._memory.answer(ref, query=query)

    async def forget(self, ref: ConversationRef) -> None:
        """Drop the channel's buffered state and delete its memory via the adapter."""
        async with self._lock:
            self._pending.pop(ref.channel_id, None)
        await self._memory.forget(ref)
