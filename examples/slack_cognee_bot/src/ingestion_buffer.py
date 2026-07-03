"""Per-channel ingestion buffer + cognify trigger for the Slack bot (issue #3609).

This sits on top of the Commit-2 :class:`ChatMemory` adapter and decides *when*
to turn buffered messages into a knowledge graph. It does not call cognee
directly — it orchestrates the adapter's ``ingest`` / ``flush`` / ``answer``.

Trigger policy (exactly the triggers the Phase 3 plan specified):

1. **Size threshold** — flush (cognify) a channel once ``cognify_batch_size``
   messages have accumulated for it. This is the primary trigger.
2. **Time interval** — optionally flush a channel once ``flush_interval_seconds``
   has elapsed since its batch started accumulating. Disabled by default
   (``None``). Implemented as a passive elapsed-time check evaluated on activity
   with an injectable clock, so it is deterministic and needs no free-running
   background task for an example app.
3. **On-demand before answer** — :meth:`answer` flushes any pending messages for
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
import time
from collections.abc import Callable

from src.config import IngestionSettings, load_ingestion_settings
from src.memory_adapter import Answer, ChatMemory, ConversationRef


class IngestionBuffer:
    """Buffers ingested messages per channel and triggers cognify per the policy."""

    def __init__(
        self,
        memory: ChatMemory,
        *,
        settings: IngestionSettings | None = None,
        time_fn: Callable[[], float] = time.monotonic,
    ):
        settings = settings or load_ingestion_settings()
        self._memory = memory
        self._batch_size = max(1, settings.cognify_batch_size)
        self._flush_interval = settings.flush_interval_seconds
        self._time_fn = time_fn

        self._lock = asyncio.Lock()
        # channel_id -> count of messages ingested but not yet cognified.
        self._pending: dict[str, int] = {}
        # channel_id -> monotonic time the current batch started accumulating.
        self._opened_at: dict[str, float] = {}

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
            if self._pending.get(channel_id, 0) == 0:
                self._opened_at[channel_id] = self._time_fn()
            self._pending[channel_id] = self._pending.get(channel_id, 0) + 1
            should_flush = self._should_flush_locked(channel_id)

        if should_flush:
            await self.flush(ref)

    def _should_flush_locked(self, channel_id: str) -> bool:
        """Decide whether ``channel_id`` should flush. Caller must hold the lock."""
        count = self._pending.get(channel_id, 0)
        if count == 0:
            return False
        if count >= self._batch_size:
            return True
        if self._flush_interval is not None:
            opened_at = self._opened_at.get(channel_id)
            if opened_at is not None and self._time_fn() - opened_at >= self._flush_interval:
                return True
        return False

    async def flush(self, ref: ConversationRef) -> None:
        """Cognify the channel's dataset if it has pending messages; else no-op.

        Empty-buffer flush never calls the adapter (clean no-op). The counter is
        reset under the lock before the (idempotent) cognify runs.
        """
        channel_id = ref.channel_id
        async with self._lock:
            if self._pending.get(channel_id, 0) == 0:
                return
            self._pending[channel_id] = 0
            self._opened_at.pop(channel_id, None)

        await self._memory.flush(ref)

    async def answer(self, ref: ConversationRef, *, query: str) -> Answer:
        """Flush pending messages for the channel, then answer the query."""
        await self.flush(ref)
        return await self._memory.answer(ref, query=query)
