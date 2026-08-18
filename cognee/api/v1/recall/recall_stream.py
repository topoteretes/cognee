"""Server-sent-events transport for ``POST /v1/recall``.

The engine (``infrastructure/llm/streaming/token_sink``) already pushes answer
tokens sideways out of the LLM call while the call chain keeps returning one
finished value. This module is the other end: it installs a sink for the
request, runs the ordinary ``recall()`` inside that context, and relays what the
sink emits to the client as SSE — then sends the *same* JSON payload a
non-streaming request would have received as a final event.

Three properties are deliberate and easy to break:

* **Deltas are a preview; ``final`` is authoritative.** A consumer must render
  the concatenated deltas but keep the ``final`` payload as the answer. They can
  differ: ``include_references=True`` appends citations *after* the answer call,
  and some paths (a sessionless recall, ``SESSION_SEARCH_MODE=sequential``,
  ``only_context``) legitimately stream nothing at all and still return a normal
  payload. "No deltas" is a supported outcome, never an error.
* **The recall task is awaited, not backgrounded.** ``commit_turn`` writes the
  Q&A inside the session turn lock, and the driver awaits the whole call before
  emitting ``final`` — so the write still happens inside the lock exactly as in
  the blocking path. Backgrounding it here would release the lock early and
  reintroduce the turn-overwrite race.
* **A disconnect detaches, it does not cancel.** The user closed the tab; the
  answer is still worth finishing and persisting. Cancelling would lose the turn
  *and* leave whatever the LLM already charged for unaccounted.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from fastapi.encoders import jsonable_encoder

from cognee.infrastructure.llm.streaming.token_sink import (
    StreamEvent,
    TokenSink,
    requested_token_sink,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("recall_stream")

SSE_MEDIA_TYPE = "text/event-stream"

# Proxies and load balancers drop idle connections, and retrieval can run for
# seconds before the first token exists. A comment line keeps the socket warm
# without being an event the client has to know about.
KEEPALIVE_SECONDS = 15.0

# asyncio.create_task() keeps only a weak reference; without a strong one the
# recall can be garbage-collected mid-flight. Same idiom as remember.py.
_STREAM_TASKS: set = set()


def wants_event_stream(accept: Optional[str], stream_flag: Optional[bool]) -> bool:
    """Whether this request asked for SSE.

    Streaming requires an *explicit* ``Accept: text/event-stream``. ``fetch``
    sends ``*/*`` unless told otherwise, so every existing caller — the frontend,
    the Slack integration, the SDK's own HTTP client — keeps receiving JSON with
    no change on their side. ``stream: false`` in the body is an override for a
    client that cannot control its Accept header.
    """
    if stream_flag is False:
        return False
    if not accept:
        return False
    return any(part.split(";")[0].strip() == SSE_MEDIA_TYPE for part in accept.split(","))


def encode_sse(event_type: str, data: Any) -> str:
    """One SSE frame. ``data`` is JSON so newlines in tokens cannot split it."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n"


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


async def _next_event(iterator, timeout: float, pending: list):
    """Next sink event, or ``None`` on a keepalive tick.

    ``asyncio.wait`` rather than ``wait_for``: a timeout must not cancel the
    pending queue read. Cancelling an async generator mid-``__anext__`` would
    leave it unusable, and the event it was about to deliver would be lost.
    """
    if not pending:
        pending.append(asyncio.ensure_future(iterator.__anext__()))
    done, _ = await asyncio.wait(pending, timeout=timeout)
    if not done:
        return None
    task = pending.pop()
    return task.result()  # re-raises StopAsyncIteration when the sink closed


async def stream_recall(
    run_recall: Callable[[], Awaitable[Any]],
    *,
    is_disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
) -> AsyncIterator[str]:
    """Drive one recall and yield its SSE frames.

    ``run_recall`` is the exact call the non-streaming path makes, so the two
    paths cannot drift: whatever it returns becomes the ``final`` payload.
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

    yield encode_sse("stage", {"stage": "retrieving"})

    iterator = sink.__aiter__()
    pending: list = []
    detached = False
    try:
        while True:
            try:
                event = await _next_event(iterator, KEEPALIVE_SECONDS, pending)
            except StopAsyncIteration:
                break
            if event is None:
                if is_disconnected is not None and await is_disconnected():
                    # Stop buffering, but let the answer finish and persist.
                    sink.detach()
                    detached = True
                    break
                yield ": keepalive\n\n"
                continue
            frame = _encode_stream_event(event)
            if frame:
                yield frame
    finally:
        for leftover in pending:
            leftover.cancel()

    if detached:
        return

    try:
        results = await task
    except Exception as error:  # noqa: BLE001 - the client already has a 200
        # The status line went out with the first frame, so a failure can only
        # reach the client as an event. It is logged where the handler's own
        # error path would have logged it.
        logger.error("Streaming recall failed: %s", error, exc_info=True)
        yield encode_sse("error", {"message": "An error occurred during recall."})
        return

    yield encode_sse("final", {"results": jsonable_encoder(results)})


async def _run_and_close(run_recall: Callable[[], Awaitable[Any]], sink: TokenSink) -> Any:
    """Run the recall, and make sure the relay loop always terminates.

    The engine closes the sink when an answer streams, but a recall that never
    reaches a streaming answer call — no session, a non-streaming retriever, an
    early error — would otherwise leave the consumer waiting on a sentinel that
    never comes.
    """
    try:
        return await run_recall()
    finally:
        sink.close()
