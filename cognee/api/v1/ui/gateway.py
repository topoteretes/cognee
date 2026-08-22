"""
ASGI reverse proxy that serves cognee's frontend, backend and MCP server behind one port.

Each service runs as its own process assuming it owns the root path of its own origin. The
gateway matches a request against the longest configured prefix, strips it, and forwards to
that upstream — so the upstreams keep routing on the paths they already know. Responses
stream through unbuffered, and WebSockets are proxied too.

A stripped prefix is invisible to the upstream, so anything it says about its own paths
comes back unprefixed and has to be corrected on the way out: redirect targets here, and
the callback URL MCP advertises when opening an SSE stream. The backend is told its prefix
directly through uvicorn's --root-path instead.
"""

import asyncio
import threading
from typing import List, Optional, Tuple

import httpx
import uvicorn
import websockets

from cognee.shared.logging_utils import get_logger

logger = get_logger()

# Headers that describe a single transport hop and must not be forwarded to the next one.
# Content-Length is dropped too: the response is re-chunked as it streams through.
HOP_BY_HOP_HEADERS = frozenset(
    [
        b"connection",
        b"content-length",
        b"keep-alive",
        b"proxy-authenticate",
        b"proxy-authorization",
        b"te",
        b"trailer",
        b"transfer-encoding",
        b"upgrade",
    ]
)


# Headers the gateway's own server regenerates for its own hop. Relaying the upstream's
# copy as well emits each one twice, and Date is a singleton field that may not repeat.
ORIGIN_HEADERS = frozenset([b"date", b"server"])


def normalize_prefix(prefix: str) -> str:
    """
    Normalize a path prefix to "/" or to "/name" with no trailing slash.
    """
    return "/" + prefix.strip("/")


def _filter_headers(
    headers: List[Tuple[bytes, bytes]], drop: frozenset = HOP_BY_HOP_HEADERS
) -> List[Tuple[bytes, bytes]]:
    return [(name, value) for name, value in headers if name.lower() not in drop]


class SubpathGateway:
    """
    Route requests to upstream services by path prefix.

    Args:
        routes: (path_prefix, upstream_base_url) pairs, for example
            [("/backend", "http://localhost:8000"), ("/", "http://localhost:3000")].
            Order does not matter — matching is always longest-prefix-first.
    """

    def __init__(self, routes: List[Tuple[str, str]]):
        normalized = [
            (normalize_prefix(prefix), upstream.rstrip("/")) for prefix, upstream in routes
        ]
        # Longest prefix first so "/backend" wins over a "/" catch-all.
        self.routes = sorted(normalized, key=lambda route: len(route[0]), reverse=True)
        self._client: Optional[httpx.AsyncClient] = None

    def match(self, path: str) -> Optional[Tuple[str, str, str]]:
        """
        Return (prefix, upstream, remaining_path) for the first matching route.

        The remaining path is what the upstream will see: the request path with the prefix
        removed. A request for the prefix itself ("/backend") maps to the upstream root.
        """
        for prefix, upstream in self.routes:
            if prefix == "/":
                return prefix, upstream, path
            if path == prefix:
                return prefix, upstream, "/"
            if path.startswith(prefix + "/"):
                return prefix, upstream, path[len(prefix) :]

        return None

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            await self._handle_lifespan(receive, send)
        elif scope["type"] == "http":
            await self._proxy_http(scope, receive, send)
        elif scope["type"] == "websocket":
            await self._proxy_websocket(scope, receive, send)

    async def _handle_lifespan(self, receive, send):
        while True:
            message = await receive()

            if message["type"] == "lifespan.startup":
                # One client for the whole gateway: connection pooling matters here, and
                # redirects must reach the browser rather than being followed internally.
                self._client = httpx.AsyncClient(timeout=None, follow_redirects=False)
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                if self._client is not None:
                    await self._client.aclose()
                    self._client = None
                await send({"type": "lifespan.shutdown.complete"})
                return

    def _build_forwarded_headers(self, scope, prefix: str) -> List[Tuple[bytes, bytes]]:
        headers = _filter_headers(scope["headers"])

        scheme = scope.get("scheme", "http")
        # Websocket schemes describe the client hop; the forwarded header describes the
        # original request scheme, for which http/https is what upstreams expect.
        forwarded_proto = {"ws": "http", "wss": "https"}.get(scheme, scheme)
        host = dict(scope["headers"]).get(b"host", b"")
        client = scope.get("client")

        headers.append((b"x-forwarded-proto", forwarded_proto.encode()))
        if host:
            headers.append((b"x-forwarded-host", host))
        if client:
            headers.append((b"x-forwarded-for", str(client[0]).encode()))
        if prefix != "/":
            headers.append((b"x-forwarded-prefix", prefix.encode()))

        return headers

    def _rewrite_location(self, location: bytes, prefix: str) -> bytes:
        """
        Put the prefix back on a root-relative redirect target.

        The upstream answers in its own path space, so a "Location: /login" would send the
        browser out of the prefix and into whichever service owns the gateway root.
        Absolute URLs are left alone — they already name their destination.
        """
        if prefix == "/" or not location.startswith(b"/") or location.startswith(b"//"):
            return location

        return prefix.encode() + location

    async def _proxy_http(self, scope, receive, send):
        match = self.match(scope["path"])

        if match is None or self._client is None:
            await self._send_plain_response(send, 502, b"No upstream configured for this path.")
            return

        prefix, upstream, upstream_path = match
        url = upstream + upstream_path
        if scope.get("query_string"):
            url += "?" + scope["query_string"].decode("latin-1")

        request = self._client.build_request(
            scope["method"],
            url,
            headers=self._build_forwarded_headers(scope, prefix),
            content=_request_body_stream(receive),
        )

        try:
            response = await self._client.send(request, stream=True)
        except httpx.HTTPError as error:
            logger.error(f"Gateway could not reach {upstream}: {error}")
            await self._send_plain_response(send, 502, b"Upstream service is unavailable.")
            return

        try:
            headers = []
            for name, value in _filter_headers(
                response.headers.raw, HOP_BY_HOP_HEADERS | ORIGIN_HEADERS
            ):
                if name.lower() == b"location":
                    value = self._rewrite_location(value, prefix)
                headers.append((name, value))

            await send(
                {
                    "type": "http.response.start",
                    "status": response.status_code,
                    "headers": headers,
                }
            )

            # aiter_raw keeps the upstream's own framing, so streamed responses (SSE, large
            # downloads) reach the client as they arrive instead of being buffered whole.
            body = response.aiter_raw()

            if prefix != "/" and _is_event_stream(response.headers):
                body = _rewrite_sse_endpoint_event(body, prefix)

            async for chunk in body:
                await send({"type": "http.response.body", "body": chunk, "more_body": True})

            await send({"type": "http.response.body", "body": b"", "more_body": False})
        finally:
            await response.aclose()

    async def _send_plain_response(self, send, status: int, body: bytes):
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"text/plain; charset=utf-8")],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    async def _proxy_websocket(self, scope, receive, send):
        # The connect message has to be consumed before the gateway answers at all, so it
        # is taken here rather than after dialling — the failure paths below reply too.
        await receive()

        match = self.match(scope["path"])

        if match is None:
            await send({"type": "websocket.close", "code": 1008})
            return

        prefix, upstream, upstream_path = match
        url = upstream.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
        url += upstream_path
        if scope.get("query_string"):
            url += "?" + scope["query_string"].decode("latin-1")

        # The handshake headers the client library sets itself would collide with the ones
        # copied from the incoming request.
        forwarded = [
            (name.decode("latin-1"), value.decode("latin-1"))
            for name, value in self._build_forwarded_headers(scope, prefix)
            if name.lower() not in (b"host", b"sec-websocket-key", b"sec-websocket-version")
        ]
        subprotocols = scope.get("subprotocols") or None

        try:
            upstream_socket = await websockets.connect(
                url,
                additional_headers=forwarded,
                subprotocols=subprotocols,
                open_timeout=10,
            )
        except Exception as error:
            logger.debug(f"Gateway could not open a WebSocket to {url}: {error}")
            await send({"type": "websocket.close", "code": 1011})
            return

        await send({"type": "websocket.accept", "subprotocol": upstream_socket.subprotocol})

        # When one side goes away the other pump is mid-await on a connection that will
        # never deliver another frame, and raises. That is the ordinary way a proxied
        # socket ends — a closed tab, or Next.js cycling its hot-reload socket — so it is
        # logged rather than propagated.
        async def relay(direction, pump):
            try:
                await pump()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.debug(f"WebSocket {direction} ended: {type(error).__name__}: {error}")

        async def client_to_upstream():
            while True:
                message = await receive()

                if message["type"] == "websocket.disconnect":
                    await upstream_socket.close()
                    return
                if message.get("text") is not None:
                    await upstream_socket.send(message["text"])
                elif message.get("bytes") is not None:
                    await upstream_socket.send(message["bytes"])

        async def upstream_to_client():
            async for message in upstream_socket:
                if isinstance(message, str):
                    await send({"type": "websocket.send", "text": message})
                else:
                    await send({"type": "websocket.send", "bytes": message})

        pumps = [
            asyncio.create_task(relay("client->upstream", client_to_upstream)),
            asyncio.create_task(relay("upstream->client", upstream_to_client)),
        ]

        try:
            # Either direction closing ends the connection; the other pump is then dead
            # weight waiting on a socket that will never produce another frame.
            done, pending = await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            # Consume the finished task's result so a failure can never surface later as
            # an unretrieved task exception.
            for task in done:
                task.result()
        finally:
            await upstream_socket.close()
            try:
                await send({"type": "websocket.close", "code": 1000})
            except Exception:
                # The client is already gone — nothing left to tell it.
                pass


def _is_event_stream(headers) -> bool:
    return headers.get("content-type", "").split(";")[0].strip() == "text/event-stream"


async def _rewrite_sse_endpoint_event(chunks, prefix: str):
    """
    Put the prefix back on the callback path an SSE "endpoint" event advertises.

    This is the same correction applied to a root-relative Location header, in a response
    body instead: the upstream names a path in its own root space, and the client is about
    to request it against the gateway, where that path belongs to another service. MCP's
    SSE transport opens exactly this way — it answers the stream with the URL the client
    must post messages to, and unprefixed that URL lands on whatever owns the gateway root.

    Only the one event is touched, and only until it has been seen; everything else streams
    through untouched.
    """
    prefixed = prefix.encode()
    pending = b""
    last_event = None
    rewritten = False

    async for chunk in chunks:
        if rewritten:
            yield chunk
            continue

        pending += chunk
        out = b""

        # Rewriting is only safe on a whole line, so a partial trailing one stays buffered.
        while b"\n" in pending:
            line, _, pending = pending.partition(b"\n")

            if line.startswith(b"event:"):
                last_event = line[len(b"event:") :].strip()
            elif line.startswith(b"data:") and last_event == b"endpoint":
                path = line[len(b"data:") :].strip()
                # A protocol-relative or absolute URL already names its destination.
                if path.startswith(b"/") and not path.startswith(b"//"):
                    line = b"data: " + prefixed + path
                    rewritten = True

            out += line + b"\n"

        if out:
            yield out

    if pending:
        yield pending


async def _request_body_stream(receive):
    """
    Yield the request body as it arrives, so uploads are not buffered in the gateway.
    """
    while True:
        message = await receive()

        if message["type"] == "http.disconnect":
            return

        body = message.get("body", b"")
        if body:
            yield body
        if not message.get("more_body", False):
            return


def start_gateway(
    gateway_port: int,
    routes: List[Tuple[str, str]],
    host: str = "0.0.0.0",
) -> threading.Thread:
    """
    Serve the gateway on gateway_port in a background thread.

    start_ui() is synchronous and must return the frontend process to its caller, so the
    gateway cannot own the main thread. The thread is a daemon: it goes away with the
    process the CLI already supervises and shuts down.

    Args:
        gateway_port: Port to bind the gateway on
        routes: (path_prefix, upstream_base_url) pairs
        host: Interface to bind (default: all, since this is the public listener)

    Returns:
        Thread running the server
    """
    app = SubpathGateway(routes)
    config = uvicorn.Config(app, host=host, port=gateway_port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True, name="cognee-gateway")
    thread.start()

    for prefix, upstream in app.routes:
        logger.info(f"✓ Gateway route {prefix} -> {upstream}")

    return thread
