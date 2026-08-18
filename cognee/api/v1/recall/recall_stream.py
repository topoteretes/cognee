"""Server-sent-events transport for ``POST /v1/recall``.

The engine (``infrastructure/llm/streaming/token_sink``) already pushes answer
tokens sideways out of the LLM call while the call chain keeps returning one
finished value. This module is the other end: it installs a sink for the
request, runs the ordinary ``recall()`` inside that context, relays what the sink
emits as SSE, and sends the *same* payload a non-streaming request would have
received as a final event.

**Nothing is sent until the recall has either produced output or failed.** Once
the first byte goes out the status line is fixed at 200 and a failure can only be
described in the body, so a permission denial or an invalid ``scope`` would
arrive as a 200 carrying an opaque error — strictly worse than the JSON path,
which maps them to 403 and 422. Waiting for the first event costs the retrieval
phase's head start (the JSON path waits for all of it anyway) and buys back the
whole error taxonomy: :func:`begin_recall_stream` simply raises what ``recall()``
raised, and the route's existing handlers turn it into the same response they
always did. Failures *after* that point are unavoidable as events, and carry the
status the JSON path would have used so a client can still tell them apart.

Three more properties are deliberate and easy to break:

* **Deltas are a preview; ``final`` is authoritative.** They legitimately differ:
  ``include_references`` appends citations after the answer call, and a
  sessionless recall, ``SESSION_SEARCH_MODE=sequential`` or ``only_context``
  stream nothing at all. Zero deltas is a supported outcome. ``final`` is
  validated against the same model the JSON route declares, so the two
  transports cannot drift into returning different shapes.
* **The recall task is awaited, not backgrounded.** ``commit_turn`` writes the
  Q&A inside the session turn lock, and the driver awaits the whole call before
  emitting ``final`` — so the write still happens inside the lock exactly as in
  the blocking path. Backgrounding it here would release the lock early and
  reintroduce the turn-overwrite race.
* **A disconnect detaches, it does not cancel.** The user closed the tab; the
  answer is still worth finishing and persisting, and cancelling would leave
  whatever the LLM already charged for unaccounted. Starlette cancels the
  response generator on disconnect, so that arrives here as ``CancelledError``
  and is handled where the generator unwinds — not by polling
  ``request.is_disconnected()``, which reads the same ASGI receive channel
  Starlette's own disconnect listener is already consuming.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from fastapi.encoders import jsonable_encoder
from pydantic import TypeAdapter

from cognee.api.sse import (
    KEEPALIVE_COMMENT,
    KeepaliveReader,
    encode_sse,
    keepalive_until,
)
from cognee.api.v1.recall.recall import RecallResponse
from cognee.infrastructure.llm.streaming.token_sink import (
    StreamEvent,
    TokenSink,
    requested_token_sink,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("recall_stream")

# Proxies and load balancers drop idle connections, and both the retrieval and
# the persistence phases can run for seconds without producing an event.
KEEPALIVE_SECONDS = 15.0

# asyncio.create_task() keeps only a weak reference; without a strong one the
# recall can be garbage-collected mid-flight. Same idiom as remember.py.
_STREAM_TASKS: set = set()

# The same model the route declares, so `final` is filtered and validated exactly
# as FastAPI would have done — returning a Response bypasses that machinery.
_RESULTS_ADAPTER = TypeAdapter(list[RecallResponse])


def _encode_stream_event(event: StreamEvent) -> Optional[str]:
    if event.type == "delta":
        return encode_sse("delta", {"text": event.text or ""})
    if event.type == "stage":
        return encode_sse("stage", {"stage": event.stage})
    if event.type == "reset":
        # The LLM call was retried and re-streams from the beginning; the client
        # must discard what it has rendered or it shows the answer twice.
        return encode_sse("reset", {})
    if event.type == "answer_done":
        return encode_sse("answer_done", {})
    if event.type == "error":
        return encode_sse("error", {"message": event.error or "streaming failed"})
    return None


async def _run_and_close(run_recall: Callable[[], Awaitable[Any]], sink: TokenSink) -> Any:
    """Run the recall, and make sure the relay loop always terminates.

    The engine closes the sink when an answer streams, but a recall that never
    reaches a streaming answer call — no session, a non-streaming retriever, an
    early error — would otherwise leave the consumer waiting on a sentinel that
    never comes. The creator of the sink owns terminating it; this is that.
    """
    try:
        return await run_recall()
    finally:
        sink.close()


class RecallStream:
    """A recall that has already started and is known not to have failed yet."""

    def __init__(
        self,
        task: asyncio.Task,
        sink: TokenSink,
        iterator: AsyncIterator[StreamEvent],
        first_event: Optional[StreamEvent],
    ) -> None:
        self._task = task
        self._sink = sink
        self._reader = KeepaliveReader(iterator, KEEPALIVE_SECONDS)
        self._first_event = first_event
        self._errored = False
        self._exhausted = first_event is None

    async def frames(self) -> AsyncIterator[str]:
        """The SSE body: relayed events, then the authoritative payload."""
        try:
            async for frame in self._relay():
                yield frame
        except (asyncio.CancelledError, GeneratorExit):
            # Starlette cancels this generator when the client disconnects. Stop
            # buffering, but leave the recall running so the turn is still
            # answered and persisted.
            self._sink.detach()
            await self._reader.aclose()
            raise

    async def _relay(self) -> AsyncIterator[str]:
        # The event consumed while deciding the status code is replayed here, so
        # deciding costs nothing but the wait.
        if self._first_event is not None:
            frame = _encode_stream_event(self._first_event)
            if frame:
                if self._first_event.type == "error":
                    self._errored = True
                yield frame

        while not self._exhausted:
            try:
                got_event, event = await self._reader.next_or_keepalive()
            except StopAsyncIteration:
                break
            if not got_event:
                yield KEEPALIVE_COMMENT
                continue
            frame = _encode_stream_event(event)
            if frame:
                if event.type == "error":
                    self._errored = True
                yield frame

        if self._sink.dropped_events:
            # The consumer fell behind the model and preview frames were dropped,
            # so what it has rendered has silent gaps. `final` is authoritative
            # and is still coming; this tells it to stop trusting the preview.
            yield encode_sse("reset", {})

        # The answer is generated but persistence is not finished, and the sink
        # closed with the last token — so this phase produces no events of its
        # own and would otherwise sit silent past a proxy's idle timeout, losing
        # `final` altogether.
        async for pending in keepalive_until(self._task, KEEPALIVE_SECONDS):
            if pending is None:
                yield KEEPALIVE_COMMENT
                continue
            try:
                results = pending.result()
            except (asyncio.CancelledError, GeneratorExit):
                raise
            except Exception as error:  # noqa: BLE001 - the client already has a 200
                logger.error("Streaming recall failed: %s", error, exc_info=True)
                if not self._errored:
                    # Only if the engine has not already reported it: a second
                    # `error` frame would arrive after a client that treats the
                    # first as terminal has torn its reader down.
                    yield encode_sse(
                        "error",
                        {"message": "An error occurred during recall.", "status": 409},
                    )
                return
            if self._errored:
                # An inner layer swallowed the failure and recall still returned
                # a payload. `final` after `error` contradicts error's terminal
                # meaning, so the payload is dropped and the error stands.
                logger.warning("Streaming recall reported an error but still returned a payload")
                return
            yield encode_sse("final", {"results": jsonable_encoder(_validate(results))})


def _validate(results: Any) -> Any:
    """Apply the route's response model, as FastAPI would for a JSON response.

    Returning a ``Response`` skips ``serialize_response`` entirely, so without
    this the streamed payload could carry fields the JSON payload filters out —
    the two transports diverging in exactly the place the tests compare them.
    """
    try:
        return _RESULTS_ADAPTER.validate_python(results)
    except Exception:  # noqa: BLE001 - a preview must not fail on a shape mismatch
        logger.warning("Streamed recall payload did not match the response model", exc_info=True)
        return results


async def begin_recall_stream(run_recall: Callable[[], Awaitable[Any]]) -> RecallStream:
    """Start the recall and wait until it produces output or fails.

    Raises whatever ``recall()`` raised, unchanged, so the caller's existing
    error handling maps it to the same status code the JSON path would return.
    No keepalive is possible here — nothing has been sent yet — but this waits no
    longer than the JSON path would have for the same request.
    """
    sink = TokenSink()
    token = requested_token_sink.set(sink)
    try:
        # Created inside the ContextVar scope so every task recall spawns — the
        # per-dataset searches, the concurrent turn's two lanes — inherits it.
        task = asyncio.create_task(_run_and_close(run_recall, sink))
        _STREAM_TASKS.add(task)
        task.add_done_callback(_STREAM_TASKS.discard)
    finally:
        requested_token_sink.reset(token)

    iterator = sink.__aiter__()
    first_event: Optional[StreamEvent] = None
    try:
        first_event = await iterator.__anext__()
    except StopAsyncIteration:
        # Closed without emitting anything: either the recall failed, or it
        # answered without streaming. Surface a failure now, while a status code
        # is still possible; a successful payload becomes `final` in the relay.
        if task.done() and task.exception() is not None:
            raise task.exception()
    return RecallStream(task, sink, iterator, first_event)
