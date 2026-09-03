"""Server-sent-events plumbing shared by the streaming endpoints.

Frame encoding, ``Accept`` negotiation and the keepalive-aware read loop are the
same for any endpoint that streams; only what is being streamed differs. They
live here rather than beside the first endpoint that needed them so the next one
does not import from ``api/v1/recall/`` or fork a second copy of the parsing.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Optional

SSE_MEDIA_TYPE = "text/event-stream"
JSON_MEDIA_TYPE = "application/json"


def sse_headers() -> dict:
    """A fresh copy per response, so a per-request header cannot leak globally."""
    return dict(_SSE_HEADERS)


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    # nginx buffers proxied responses by default, which holds every token until
    # the answer is finished — exactly what streaming exists to avoid.
    "X-Accel-Buffering": "no",
    # Deliberately no `Connection: keep-alive`: it is a hop-by-hop header an ASGI
    # app must not set, HTTP/2 forbids it outright, and the server owns the
    # connection anyway.
}


def _quality(accept: str, media_type: str) -> float:
    """The client's q-value for ``media_type``, 0.0 if it does not accept it.

    RFC 7231 §5.3.2 ranks by *specificity*, not by the highest match: an exact
    type beats ``type/*``, which beats ``*/*``. Taking the maximum instead would
    make ``application/json;q=0.1, */*`` score JSON at 1.0, so a client could
    never deprioritise a type while keeping a wildcard fallback.
    """
    type_ = media_type.partition("/")[0]
    # Most specific first; the first tier that matches wins.
    by_specificity: dict[str, float] = {}
    for part in accept.split(","):
        segments = part.split(";")
        candidate = segments[0].strip().lower()
        if candidate not in (media_type, f"{type_}/*", "*/*"):
            continue
        quality = 1.0
        for parameter in segments[1:]:
            name, _, value = parameter.partition("=")
            if name.strip().lower() == "q":
                try:
                    quality = float(value.strip())
                except ValueError:
                    quality = 0.0
        by_specificity[candidate] = max(by_specificity.get(candidate, 0.0), quality)
    for tier in (media_type, f"{type_}/*", "*/*"):
        if tier in by_specificity:
            return by_specificity[tier]
    return 0.0


def wants_event_stream(accept: Optional[str], stream_flag: Optional[bool] = None) -> bool:
    """Whether this request asked for SSE.

    ``stream`` in the body decides outright when present, for clients that cannot
    set their own headers. Otherwise the client must prefer SSE *over* JSON:
    streaming is opt-in, so anything ambiguous stays on the JSON path.

    A tie goes to JSON, which is what makes ``Accept: */*`` — sent by ``fetch``,
    ``httpx`` and ``requests`` unless told otherwise — and the MCP
    Streamable-HTTP header ``application/json, text/event-stream`` behave as they
    did before streaming existed. Only a client that names SSE alone, or ranks it
    above JSON, is switched.
    """
    if stream_flag is not None:
        return stream_flag
    if not accept:
        return False
    # A wildcard is not a request for SSE. `text/*` matches text/event-stream but
    # can never match application/json, so scoring it would hand a stream to any
    # client that merely normalised its Accept header.
    if not any(part.split(";")[0].strip().lower() == SSE_MEDIA_TYPE for part in accept.split(",")):
        return False
    sse = _quality(accept, SSE_MEDIA_TYPE)
    if sse <= 0.0:
        return False
    return sse > _quality(accept, JSON_MEDIA_TYPE)


def encode_sse(event_type: str, data: Any) -> str:
    """One SSE frame. ``data`` is JSON so newlines in tokens cannot split it."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n"


KEEPALIVE_COMMENT = ": keepalive\n\n"


class KeepaliveReader:
    """Reads an async iterator, reporting a keepalive tick when it goes quiet.

    ``asyncio.wait`` rather than ``wait_for``: a timeout must not cancel the
    pending read. Cancelling mid-``__anext__`` leaves the iterator unusable and
    loses the event it was about to deliver, so the pending read is kept across
    ticks and only ever consumed by the caller.
    """

    def __init__(self, iterator: AsyncIterator[Any], interval: float) -> None:
        self._iterator = iterator
        self._interval = interval
        self._pending: Optional[asyncio.Future] = None

    async def next_or_keepalive(self) -> tuple[bool, Any]:
        """``(got_event, event)``. ``(False, None)`` means the interval elapsed."""
        if self._pending is None:
            self._pending = asyncio.ensure_future(self._iterator.__anext__())
        done, _ = await asyncio.wait({self._pending}, timeout=self._interval)
        if not done:
            return False, None
        finished, self._pending = self._pending, None
        return True, finished.result()  # re-raises StopAsyncIteration when closed

    async def aclose(self) -> None:
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None


async def keepalive_until(awaitable, interval: float) -> AsyncIterator[Any]:
    """Yield keepalive comments while ``awaitable`` runs; finally yield its result.

    The last item is the result; everything before it is a keepalive frame. Used
    for the phases that produce no events of their own — a stream that goes quiet
    for longer than the proxy's idle timeout is torn down before it can finish.
    """
    task = asyncio.ensure_future(awaitable)
    while True:
        done, _ = await asyncio.wait({task}, timeout=interval)
        if done:
            yield task
            return
        yield None
