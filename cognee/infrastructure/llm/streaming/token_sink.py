"""Side-channel for streaming answer tokens out of a deep LLM call.

The call chain from an HTTP handler down to litellm is single-value at every
hop: ``recall()`` materialises ``list[RecallResponse]``, ``SearchResultPayload``
is a pydantic model whose ``completion`` field has a serializer that stringifies
anything it does not recognise, and the retriever returns one finished string
that ``commit_turn`` then persists. An async generator cannot cross any of that.

So tokens travel sideways instead. The adapter pushes each delta into a sink
while still returning the complete string, and every existing signature stays
exactly as it was — retrieval, the session lock, ``add_qa``, usage accounting and
serialisation all run byte-for-byte as before. Streaming is therefore never
load-bearing: if nothing is listening, or the request takes a path that cannot
stream, the caller still gets the identical payload with zero events.

Two context variables, and both are load-bearing:

* ``requested_token_sink`` is set once at request entry, so every task the
  request spawns inherits it.
* ``active_token_sink`` is promoted only around the answer call itself. This is
  what keeps the *analysis* lane out of the stream: the live cloud path runs
  ``asyncio.gather(analyze_turn, _retrieve_and_answer)``, two concurrent LLM
  calls, and ``gather`` snapshots the context per task — so a variable set inside
  the answer task is structurally invisible to its sibling. A single
  request-scoped flag would interleave ``SessionTurnAnalysis`` output into the
  user's tokens.

``try_claim`` handles the other fan-out: ``search_in_datasets_context`` launches
one task per authorised dataset, each running its own answer call. The first to
claim wins and the rest simply do not stream, so a multi-dataset recall emits one
coherent answer instead of several interleaved ones — and still returns the
complete list at the end.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional


@dataclass
class StreamEvent:
    """One event on the sink's queue.

    ``reset`` tells a consumer to discard what it has rendered so far: a retried
    LLM call restarts generation from scratch, and without it the client would
    show the answer twice.
    """

    type: str  # "stage" | "delta" | "reset" | "answer_done" | "error"
    text: Optional[str] = None
    stage: Optional[str] = None
    error: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


_SENTINEL = object()


class TokenSink:
    """Fan-out channel for answer tokens. Never affects the returned value."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self._claimed = False
        self._emitted = False
        self._detached = False
        self._closed = False

    # -- producer side -------------------------------------------------

    def try_claim(self) -> bool:
        """Claim the sink for one producer. Later callers get ``False``.

        Not a lock: claiming happens inside a single synchronous block with no
        await, so it cannot interleave.
        """
        if self._claimed:
            return False
        self._claimed = True
        return True

    def begin_attempt(self) -> None:
        """Mark the start of an LLM attempt, resetting any partial output.

        Tenacity retries the whole call, so a second attempt re-streams from the
        beginning. Emitting ``reset`` first is what stops the consumer
        concatenating two copies of the answer.
        """
        if self._emitted:
            self._put(StreamEvent(type="reset"))
            self._emitted = False

    def put_delta(self, text: str) -> None:
        if not text or self._detached:
            return
        self._emitted = True
        self._put(StreamEvent(type="delta", text=text))

    def mark_stage(self, stage: str) -> None:
        self._put(StreamEvent(type="stage", stage=stage))

    def answer_done(self) -> None:
        """The last token has been generated; work after this is persistence."""
        self._put(StreamEvent(type="answer_done"))

    def fail(self, error: BaseException) -> None:
        self._put(StreamEvent(type="error", error=str(error)))
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put_nowait(_SENTINEL)

    def detach(self) -> None:
        """The consumer is gone; stop buffering but let the producer finish.

        Deliberately *not* cancellation. The LLM call keeps running and the
        session write still lands, so a user who closes the tab does not lose the
        turn — they simply stop seeing it arrive.
        """
        self._detached = True

    def _put(self, event: StreamEvent) -> None:
        if self._detached or self._closed:
            return
        # Unbounded and non-blocking: a slow consumer must never apply
        # backpressure to the LLM call it is watching.
        self._queue.put_nowait(event)

    # -- consumer side -------------------------------------------------

    async def __aiter__(self) -> AsyncIterator[StreamEvent]:
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                return
            yield item


requested_token_sink: ContextVar[Optional[TokenSink]] = ContextVar(
    "requested_token_sink", default=None
)
active_token_sink: ContextVar[Optional[TokenSink]] = ContextVar("active_token_sink", default=None)


def get_active_token_sink() -> Optional[TokenSink]:
    """The sink the *current task* may stream into, if any.

    Returns ``None`` everywhere except inside :func:`stream_answer_tokens`, which
    is what keeps every other LLM call in the request out of the user's stream.
    """
    return active_token_sink.get()


@asynccontextmanager
async def stream_answer_tokens(stage: Optional[str] = None) -> AsyncIterator[None]:
    """Promote the requested sink to active for this task only.

    Wrap the answer-generating call and nothing else. Entering also marks the
    stage, which gives the retrieval→generation boundary for free — useful for
    telling how much of the wait is retrieval that streaming cannot help with.
    """
    sink = requested_token_sink.get()
    if sink is None or not sink.try_claim():
        yield
        return

    if stage:
        sink.mark_stage(stage)

    token = active_token_sink.set(sink)
    try:
        yield
    finally:
        active_token_sink.reset(token)
