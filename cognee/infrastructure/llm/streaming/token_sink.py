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

**Ownership is decided by the first delta, not by entering the block.** Several
lanes may be promoted at once — ``search_in_datasets_context`` launches one
answer call per authorised dataset — and which of them *can* stream is unknown
until one actually does: a structured ``response_model`` never streams, and not
every adapter has a plain-text streaming path. Claiming on entry would let a lane
that turns out not to stream lock out one that would have. So promotion is free,
the first lane to emit becomes the owner, and later lanes' deltas are dropped.
The request emits one coherent answer and still returns the complete list.

**Who closes the sink.** The creator does — whoever set ``requested_token_sink``
owns terminating it, because only they know when the whole request is finished.
:func:`answer_scope` closes early *when its answer streamed*, so a
consumer sees the answer end without waiting for persistence; every other exit
leaves the sink open because a later call in the same request may still stream.
A consumer that iterates a sink nobody closes waits forever, which is why the
transport in ``api/v1/recall/recall_stream.py`` closes in a ``finally``.

**Deltas are a preview, not the payload.** Two things make the streamed text a
strict prefix of what the caller finally receives, and a consumer must treat the
returned value as authoritative rather than concatenating deltas and stopping:

* ``append_references`` runs *after* the answer call, so with
  ``include_references=True`` the citations are appended to the completion and
  never appear as deltas.
* Only the concurrent session path promotes a sink today. A sessionless
  ``recall()``, ``SESSION_SEARCH_MODE=sequential``, or a retriever outside the
  concurrent-eligible set answers through the sequential runner and streams
  nothing at all — the request still succeeds and returns the identical payload.
  The sequential path is deliberately excluded for now because it gathers
  ``summarize_text`` alongside the answer, and a shared hook would leak the
  summariser's tokens into the user's stream — the same interleaving problem the
  two-ContextVar split exists to prevent.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import AsyncIterator, Literal, Optional

from cognee.shared.logging_utils import get_logger

logger = get_logger("token_sink")

EventType = Literal["stage", "delta", "reset", "answer_done", "error"]

# Deltas are a preview and `final` is authoritative, so buffering is bounded:
# a sink nobody drains (a transport that died before iterating, a background
# task that inherited the ContextVar) would otherwise retain one object per
# token for the life of the request.
MAX_BUFFERED_EVENTS = 2048


@dataclass(slots=True)
class StreamEvent:
    """One event on the sink's queue.

    ``reset`` tells a consumer to discard what it has rendered so far: a retried
    LLM call restarts generation from scratch, and without it the client would
    show the answer twice.
    """

    type: EventType
    text: Optional[str] = None
    stage: Optional[str] = None
    error: Optional[str] = None
    # Set on ``error`` only. The status line was already committed as 200 when
    # the first frame went out, so this is the one place a caller can still
    # learn that the failure was, say, 402 rather than a generic server error.
    status: Optional[int] = None


_SENTINEL = object()

# Identifies one promoted answer call, so the sink can tell the lane that owns
# the stream from a concurrent lane whose deltas must be dropped.
_active_producer: ContextVar[Optional[object]] = ContextVar("active_producer", default=None)


class TokenSink:
    """Fan-out channel for answer tokens. Never affects the returned value."""

    def __init__(self, max_buffered_events: Optional[int] = None) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        # Read at construction, not bound as a default, so the cap stays one
        # knob rather than a value frozen at import.
        self._max_buffered = (
            MAX_BUFFERED_EVENTS if max_buffered_events is None else max_buffered_events
        )
        self._owner: Optional[object] = None
        self._emitted = False
        self._detached = False
        self._closed = False
        self._dropped = False
        self._last_stage: Optional[str] = None
        self._iterator: Optional[AsyncIterator[StreamEvent]] = None

    # -- producer side -------------------------------------------------

    def owns(self, producer: Optional[object]) -> bool:
        """Whether this producer is the one whose deltas reach the consumer."""
        return producer is not None and self._owner is producer

    @property
    def is_closed(self) -> bool:
        """Whether the stream has already been terminated."""
        return self._closed

    @property
    def dropped_events(self) -> bool:
        """Whether the buffer overflowed and preview frames were discarded.

        A consumer that fell behind has silent gaps in what it rendered, so it
        needs telling to stop trusting the preview and wait for the payload.
        """
        return self._dropped

    @property
    def emitted_any(self) -> bool:
        """Whether any delta was produced (whether or not it was delivered)."""
        return self._emitted

    def begin_attempt(self) -> None:
        """Mark the start of an LLM attempt, resetting any partial output.

        Tenacity retries the whole call, so a second attempt re-streams from the
        beginning. Emitting ``reset`` first is what stops the consumer
        concatenating two copies of the answer.

        Only the owning lane may reset. Every promoted lane calls this on its way
        into the adapter, so without the ownership check a second dataset's
        answer call would tell the consumer to discard the first one's answer
        mid-stream — and its own deltas are then dropped, so nothing replaces it.
        """
        if self._emitted and self.owns(_active_producer.get()):
            self._put(StreamEvent(type="reset"))

    def put_delta(self, text: str) -> None:
        """Offer one token to the stream.

        The first producer to get here owns the sink for the rest of the request;
        deltas from any other lane are dropped rather than interleaved.
        """
        if not text:
            return
        producer = _active_producer.get()
        if producer is not None:
            # None means the caller is not inside a promotion (a direct producer,
            # or a test). Claiming for None would make `owns()` false for every
            # lane, so nothing would ever terminate the stream.
            if self._owner is None:
                self._owner = producer
            elif self._owner is not producer:
                return
        # Set before the detached check: a consumer that left mid-answer does not
        # change the fact that this producer streamed, and treating it as "never
        # streamed" would hand ownership to another lane.
        self._emitted = True
        self._put(StreamEvent(type="delta", text=text))

    def mark_stage(self, stage: str) -> None:
        """Announce a phase change, skipping a repeat of the current phase.

        Promotion happens before it is known whether this lane will stream, so
        several lanes can mark the same stage for one answer; a consumer timing
        the retrieval→generation boundary must not see it twice.
        """
        if stage == self._last_stage:
            return
        self._last_stage = stage
        self._put(StreamEvent(type="stage", stage=stage))

    def answer_done(self) -> None:
        """The last token has been generated; work after this is persistence."""
        self._put(StreamEvent(type="answer_done"))

    def fail(self, message: str, status: Optional[int] = None) -> None:
        """Report a failure to the consumer, then terminate the stream.

        Takes an already-safe message rather than an exception: provider errors
        embed the rendered prompt (and therefore the retrieved graph context),
        endpoints and request bodies, none of which may reach an HTTP client.

        ``status`` is the HTTP status the same failure would have produced on the
        JSON path. It travels in the frame because it can no longer travel in the
        status line: by the time an answer call fails, the response is committed
        to 200. A client that only reads the status code cannot be helped, but one
        that reads the frame can still tell a 402 from a 500.
        """
        self._put(StreamEvent(type="error", error=message, status=status))
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

    # Dropping one of these is worse than the memory it costs: a lost `error`
    # ends the stream cleanly on a failed request, and a lost `reset` lets the
    # consumer concatenate two copies of a retried answer — the exact failures
    # those events exist to prevent.
    _UNDROPPABLE = frozenset({"error", "reset", "answer_done", "stage"})

    def _put(self, event: StreamEvent) -> None:
        if self._detached or self._closed:
            return
        if event.type not in self._UNDROPPABLE and self._queue.qsize() >= self._max_buffered:
            # Drop rather than block: a slow or absent consumer must never apply
            # backpressure to the LLM call it is watching. Dropping preview
            # frames is consistent with `final` being the authoritative answer.
            if not self._dropped:
                self._dropped = True
                logger.warning(
                    "Token sink buffer full (%d events) — dropping stream events; "
                    "the returned answer is unaffected",
                    self._max_buffered,
                )
            return
        self._queue.put_nowait(event)

    # -- consumer side -------------------------------------------------

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        """Iterate the event stream. Single-consumer by construction.

        Two iterators over one queue would split the events between them and
        only one would ever see the sentinel, so the second consumer would hang
        on a stream that looked half-delivered.
        """
        if self._iterator is not None:
            raise RuntimeError("TokenSink supports a single consumer")
        self._iterator = self._iterate()
        return self._iterator

    async def _iterate(self) -> AsyncIterator[StreamEvent]:
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

    Returns ``None`` everywhere except inside :func:`answer_scope`, which
    is what keeps every other LLM call in the request out of the user's stream.
    """
    return active_token_sink.get()


@asynccontextmanager
async def answer_scope(
    stage: Optional[str] = None, *, can_stream: bool = True
) -> AsyncIterator[None]:
    """Mark this task as the one completion a listening client may watch.

    A scope, not a verb: entering it usually promotes nothing. It yields without
    promoting when no sink was requested, when the feature switch is off, when the
    configured adapter cannot stream at all, when the caller declares this call
    cannot (a structured ``response_model``), or when an earlier lane already
    closed the sink — so the caller cannot tell from the name whether tokens will
    flow, and must not care.

    Do not call this directly from a request or session path. The one production
    caller is :func:`cognee.modules.retrieval.utils.completion.generate_answer`;
    choosing that function over ``generate_completion`` *is* the decision to
    stream, and keeping the decision there is what stops unrelated layers from
    having to import this module.

    Every precondition is checked here, in one place, so a request cannot promote a sink
    that the adapter below will then refuse to stream into.

    On the way out it ends the stream **when this lane actually streamed**:
    ``answer_done`` + ``close`` on success, a sanitised ``error`` on failure.
    Any other exit leaves the sink open, because a later call in the same request
    may still stream; terminating it for good is the creator's job (see the
    module docstring).
    """
    from cognee.infrastructure.llm.config import get_llm_context_config
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    sink = requested_token_sink.get()
    if sink is None or not get_llm_context_config().llm_answer_streaming:
        yield
        return
    if not LLMGateway.supports_answer_streaming():
        # Bedrock, Ollama, llama.cpp, MCP-sampling and the BAML framework answer
        # without ever reaching the streaming path. Promoting for them announces
        # a stream that produces no tokens — a `stage` event and then silence —
        # so the request stays exactly as it is with the flag off.
        yield
        return
    if sink.is_closed:
        # An earlier lane already finished the answer. Promoting anyway would
        # send this call down the streaming path for output nobody can receive.
        yield
        return
    if not can_stream:
        # The caller knows something the sink cannot see — in practice a
        # structured ``response_model``, which both adapters route past the
        # plain-text door that reaches stream_text_completion
        # (native_adapter.py, generic_llm_api/adapter.py: ``if response_model is
        # str``). Promoting would emit ``stage: generating`` and then nothing at
        # all, which is strictly worse than never announcing a stream: the
        # consumer waits for tokens that are not coming, and — because that
        # stage frame is the sink's first event — the transport commits HTTP 200
        # before the answer call has even run.
        yield
        return

    producer = object()
    sink_token = active_token_sink.set(sink)
    producer_token = _active_producer.set(producer)
    if stage:
        sink.mark_stage(stage)
    try:
        yield
    except (asyncio.CancelledError, GeneratorExit):
        # Not a failure: the request was torn down (client disconnect, timeout,
        # shutdown). str(CancelledError()) is "", so reporting it would render a
        # blank error — and it would defeat detach(), whose entire purpose is
        # that a vanished consumer does not become a visible failure.
        raise
    except BaseException as error:
        # The consumer has already been handed a 200 and part of an answer, so
        # the failure has to reach it as an event. The detail stays server-side:
        # provider errors embed the rendered prompt and connection details.
        logger.error("Answer streaming failed: %s", error, exc_info=True)
        if sink.owns(producer):
            sink.fail(
                f"{type(error).__name__} during answer generation",
                status=getattr(error, "status_code", None),
            )
        raise
    else:
        if sink.owns(producer) and sink.emitted_any:
            sink.answer_done()
            sink.close()
    finally:
        _active_producer.reset(producer_token)
        active_token_sink.reset(sink_token)
