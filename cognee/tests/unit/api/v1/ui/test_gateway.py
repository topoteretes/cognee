"""Tests for the single-port gateway in cognee.api.v1.ui.gateway and its start_ui wiring."""

import asyncio
import gc
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from cognee.api.v1.ui.gateway import (
    HOP_BY_HOP_HEADERS,
    ORIGIN_HEADERS,
    SubpathGateway,
    _filter_headers,
    normalize_prefix,
)


async def call_gateway(gateway, path, method="GET", query=b"", headers=None, body=b""):
    """Drive the gateway's ASGI interface once and collect what it sends back."""
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": query,
        "scheme": "http",
        "headers": headers if headers is not None else [(b"host", b"cognee.example")],
        "client": ("203.0.113.7", 51234),
    }

    incoming = [{"type": "http.request", "body": body, "more_body": False}]
    sent = []

    async def receive():
        return incoming.pop(0)

    async def send(message):
        sent.append(message)

    await gateway(scope, receive, send)

    start = next(m for m in sent if m["type"] == "http.response.start")
    payload = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")

    return start, payload, sent


def attach_upstream(gateway, upstream_app):
    """Point the gateway at an in-process ASGI app instead of a real socket."""
    gateway._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=upstream_app),
        timeout=None,
        follow_redirects=False,
    )


class TestNormalizePrefix:
    """Prefixes are accepted in whatever shape the caller wrote them."""

    def test_variants_collapse_to_one_form(self):
        assert normalize_prefix("/backend") == "/backend"
        assert normalize_prefix("backend") == "/backend"
        assert normalize_prefix("/backend/") == "/backend"
        assert normalize_prefix("backend/") == "/backend"
        assert normalize_prefix("/") == "/"
        assert normalize_prefix("") == "/"


class TestHeaderFiltering:
    """Headers scoped to a single connection must not be relayed to the next one."""

    def test_hop_by_hop_headers_are_dropped(self):
        headers = [
            (b"host", b"cognee.example"),
            (b"connection", b"keep-alive"),
            (b"keep-alive", b"timeout=5"),
            (b"transfer-encoding", b"chunked"),
            (b"content-length", b"12"),
            (b"upgrade", b"websocket"),
            (b"te", b"trailers"),
            (b"trailer", b"Expires"),
            (b"proxy-authenticate", b"Basic"),
            (b"proxy-authorization", b"Basic hop"),
            (b"authorization", b"Bearer token"),
            (b"cookie", b"auth_token=abc"),
        ]

        assert _filter_headers(headers) == [
            (b"host", b"cognee.example"),
            (b"authorization", b"Bearer token"),
            (b"cookie", b"auth_token=abc"),
        ]

    def test_origin_headers_are_dropped_from_responses(self):
        # The gateway's own server emits Date and Server for its hop; relaying the
        # upstream's copies too would send each twice, and Date may not repeat.
        headers = [
            (b"date", b"Mon, 17 Aug 2026 11:11:38 GMT"),
            (b"server", b"uvicorn"),
            (b"content-type", b"application/json"),
        ]

        assert _filter_headers(headers, HOP_BY_HOP_HEADERS | ORIGIN_HEADERS) == [
            (b"content-type", b"application/json"),
        ]


class TestRouteMatching:
    """The gateway picks an upstream by longest prefix and strips that prefix."""

    def setup_method(self):
        self.gateway = SubpathGateway(
            [
                ("/", "http://localhost:3000"),
                ("/backend", "http://localhost:8000"),
                ("/mcp", "http://localhost:8001"),
            ]
        )

    def test_longest_prefix_wins_over_root_catch_all(self):
        prefix, upstream, path = self.gateway.match("/backend/api/v1/datasets")

        assert prefix == "/backend"
        assert upstream == "http://localhost:8000"
        assert path == "/api/v1/datasets"

    def test_bare_prefix_maps_to_upstream_root(self):
        assert self.gateway.match("/backend")[2] == "/"

    def test_prefix_only_matches_on_a_segment_boundary(self):
        # "/backendish" is a frontend route, not a mistyped backend one.
        prefix, upstream, path = self.gateway.match("/backendish")

        assert prefix == "/"
        assert upstream == "http://localhost:3000"
        assert path == "/backendish"

    def test_unmatched_paths_fall_through_to_the_frontend(self):
        for path in ["/", "/dashboard", "/_next/static/chunk.js"]:
            assert self.gateway.match(path)[1] == "http://localhost:3000"

    def test_no_match_without_a_root_route(self):
        gateway = SubpathGateway([("/backend", "http://localhost:8000")])

        assert gateway.match("/dashboard") is None


class TestHttpProxying:
    """Requests reach the upstream unchanged apart from the stripped prefix."""

    def setup_method(self):
        self.seen = {}

        async def upstream(scope, receive, send):
            body = b""
            while True:
                message = await receive()
                body += message.get("body", b"")
                if not message.get("more_body", False):
                    break

            self.seen["path"] = scope["path"]
            self.seen["method"] = scope["method"]
            self.seen["query_string"] = scope["query_string"]
            self.seen["headers"] = dict(scope["headers"])
            self.seen["body"] = body

            await send(
                {
                    "type": "http.response.start",
                    "status": 201,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": b'{"ok":true}'})

        self.gateway = SubpathGateway(
            [("/", "http://frontend"), ("/backend", "http://backend")],
        )
        attach_upstream(self.gateway, upstream)

    @pytest.mark.asyncio
    async def test_prefix_is_stripped_before_the_upstream_sees_it(self):
        start, payload, _ = await call_gateway(self.gateway, "/backend/api/v1/datasets")

        assert self.seen["path"] == "/api/v1/datasets"
        assert start["status"] == 201
        assert payload == b'{"ok":true}'

    @pytest.mark.asyncio
    async def test_method_query_string_and_body_survive_the_hop(self):
        await call_gateway(
            self.gateway,
            "/backend/api/v1/search",
            method="POST",
            query=b"top_k=5",
            body=b'{"query":"hello"}',
        )

        assert self.seen["method"] == "POST"
        assert self.seen["query_string"] == b"top_k=5"
        assert self.seen["body"] == b'{"query":"hello"}'

    @pytest.mark.asyncio
    async def test_forwarded_headers_describe_the_original_request(self):
        await call_gateway(self.gateway, "/backend/health")

        headers = self.seen["headers"]
        assert headers[b"x-forwarded-proto"] == b"http"
        assert headers[b"x-forwarded-host"] == b"cognee.example"
        assert headers[b"x-forwarded-for"] == b"203.0.113.7"
        assert headers[b"x-forwarded-prefix"] == b"/backend"

    @pytest.mark.asyncio
    async def test_no_forwarded_prefix_for_the_root_route(self):
        await call_gateway(self.gateway, "/dashboard")

        assert b"x-forwarded-prefix" not in self.seen["headers"]

    @pytest.mark.asyncio
    async def test_hop_by_hop_headers_are_not_forwarded(self):
        await call_gateway(
            self.gateway,
            "/backend/health",
            headers=[
                (b"host", b"cognee.example"),
                (b"transfer-encoding", b"chunked"),
                (b"proxy-authorization", b"Basic hop"),
                (b"te", b"trailers"),
                (b"authorization", b"Bearer token"),
            ],
        )

        headers = self.seen["headers"]
        # These describe the client's hop to the gateway, not the gateway's to the upstream.
        # transfer-encoding and connection are not asserted here: httpx sets its own for
        # the outgoing hop, so what arrives is its framing rather than a copied header.
        assert b"proxy-authorization" not in headers
        assert b"te" not in headers
        # End-to-end headers must still get through.
        assert headers[b"authorization"] == b"Bearer token"

    @pytest.mark.asyncio
    async def test_unreachable_upstream_answers_502(self):
        gateway = SubpathGateway([("/backend", "http://127.0.0.1:1")])
        gateway._client = httpx.AsyncClient(timeout=1, follow_redirects=False)

        start, payload, _ = await call_gateway(gateway, "/backend/health")

        assert start["status"] == 502
        assert b"unavailable" in payload

    @pytest.mark.asyncio
    async def test_unroutable_path_answers_502(self):
        gateway = SubpathGateway([("/backend", "http://backend")])
        attach_upstream(gateway, lambda scope, receive, send: None)

        start, _, _ = await call_gateway(gateway, "/nowhere")

        assert start["status"] == 502


class ChunkStream(httpx.AsyncByteStream):
    """A response body that arrives in distinct chunks, the way an SSE endpoint answers."""

    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


class ChunkedTransport(httpx.AsyncBaseTransport):
    def __init__(self, chunks):
        self.chunks = chunks

    async def handle_async_request(self, request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ChunkStream(self.chunks),
        )


class TestStreaming:
    """Chunks reach the client as they are produced, not buffered into one response.

    Server-sent events are how the MCP server and the cognify progress endpoint report,
    so a gateway that waited for the last chunk would stall them indefinitely.
    """

    @pytest.mark.asyncio
    async def test_chunks_are_relayed_individually(self):
        chunks = [b"data: 0\n\n", b"data: 1\n\n", b"data: 2\n\n"]
        gateway = SubpathGateway([("/mcp", "http://mcp")])
        gateway._client = httpx.AsyncClient(
            transport=ChunkedTransport(chunks), timeout=None, follow_redirects=False
        )

        _, payload, sent = await call_gateway(gateway, "/mcp/sse")

        assert payload == b"".join(chunks)
        body_messages = [m for m in sent if m["type"] == "http.response.body" and m.get("body")]
        assert len(body_messages) == len(chunks), "streamed chunks were buffered into one response"


class TestSseEndpointRewriting:
    """MCP's SSE handshake advertises the path the client must post messages to.

    It is written relative to the server root, so behind a prefix it names a path the
    gateway routes to a different service entirely — the same defect as a root-relative
    Location header, in a body instead of a header.
    """

    def _sse_gateway(self, chunks):
        gateway = SubpathGateway([("/", "http://frontend"), ("/mcp", "http://mcp")])
        gateway._client = httpx.AsyncClient(
            transport=ChunkedTransport(chunks), timeout=None, follow_redirects=False
        )
        return gateway

    @pytest.mark.asyncio
    async def test_endpoint_path_gains_the_prefix(self):
        gateway = self._sse_gateway(
            [b"event: endpoint\ndata: /messages/?session_id=abc123\n\n"],
        )

        _, payload, _ = await call_gateway(gateway, "/mcp/sse")

        assert b"data: /mcp/messages/?session_id=abc123" in payload

    @pytest.mark.asyncio
    async def test_endpoint_split_across_chunks_is_still_rewritten(self):
        # The rewrite must not depend on the frame arriving in one piece.
        gateway = self._sse_gateway(
            [b"event: endp", b"oint\ndata: /messa", b"ges/?session_id=abc123\n\n"],
        )

        _, payload, _ = await call_gateway(gateway, "/mcp/sse")

        assert b"data: /mcp/messages/?session_id=abc123" in payload

    @pytest.mark.asyncio
    async def test_later_data_frames_are_left_alone(self):
        # Only the handshake names a path; message payloads must stream through verbatim.
        gateway = self._sse_gateway(
            [
                b"event: endpoint\ndata: /messages/?session_id=abc\n\n",
                b'event: message\ndata: {"jsonrpc":"2.0","result":"/not/a/path"}\n\n',
            ],
        )

        _, payload, _ = await call_gateway(gateway, "/mcp/sse")

        assert b'data: {"jsonrpc":"2.0","result":"/not/a/path"}' in payload

    @pytest.mark.asyncio
    async def test_absolute_endpoint_is_left_alone(self):
        gateway = self._sse_gateway(
            [b"event: endpoint\ndata: https://mcp.example/messages/\n\n"],
        )

        _, payload, _ = await call_gateway(gateway, "/mcp/sse")

        assert b"data: https://mcp.example/messages/" in payload

    @pytest.mark.asyncio
    async def test_root_route_is_never_rewritten(self):
        gateway = self._sse_gateway([b"event: endpoint\ndata: /messages/\n\n"])

        _, payload, _ = await call_gateway(gateway, "/stream")

        assert b"data: /messages/" in payload
        assert b"/mcp/messages" not in payload

    @pytest.mark.asyncio
    async def test_non_sse_bodies_are_not_inspected(self):
        async def upstream(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": b'{"data": "/messages/"}'})

        gateway = SubpathGateway([("/mcp", "http://mcp")])
        attach_upstream(gateway, upstream)

        _, payload, _ = await call_gateway(gateway, "/mcp/info")

        assert payload == b'{"data": "/messages/"}'


class TestWebSocketTeardown:
    """A socket dying mid-relay is how proxied WebSockets normally end, not a failure.

    Next.js cycles its hot-reload socket constantly, so this is the common case rather
    than an edge one, and it must not surface as an unhandled task exception.
    """

    @pytest.mark.asyncio
    async def test_client_vanishing_mid_stream_is_handled(self):
        import websockets

        gateway = SubpathGateway([("/backend", "http://backend")])

        class Upstream:
            subprotocol = None

            def __aiter__(self):
                return self

            async def __anext__(self):
                return "frame"

            async def close(self):
                pass

        sent = []

        async def send(message):
            sent.append(message)
            # The browser is gone; uvicorn raises on any further send.
            if message["type"] == "websocket.send":
                raise RuntimeError("ClientDisconnected")

        async def receive():
            await asyncio.sleep(0.05)
            return {"type": "websocket.disconnect", "code": 1001}

        scope = {
            "type": "websocket",
            "path": "/backend/ws",
            "query_string": b"",
            "scheme": "ws",
            "headers": [(b"host", b"cognee.example")],
            "client": ("203.0.113.7", 51234),
            "subprotocols": [],
        }

        # "Task exception was never retrieved" is reported to the loop handler when the
        # abandoned task is collected, so that is what this has to watch.
        unretrieved = []
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, context: unretrieved.append(context)
        )

        with patch.object(websockets, "connect", return_value=_awaitable(Upstream())):
            # Must return rather than raise, and must not leave a task exception behind.
            await asyncio.wait_for(gateway(scope, receive, send), timeout=5)

        gc.collect()
        await asyncio.sleep(0)

        assert any(m["type"] == "websocket.accept" for m in sent)
        assert not unretrieved, f"pump left an unhandled exception: {unretrieved}"


def _awaitable(value):
    async def _coro():
        return value

    return _coro()


class TestRedirectHandling:
    """Redirects reach the browser, and root-relative ones stay inside the prefix."""

    def setup_method(self):
        async def upstream(scope, receive, send):
            location = scope["path"].encode().replace(b"/redirect", b"") or b"/login"
            await send(
                {
                    "type": "http.response.start",
                    "status": 307,
                    "headers": [(b"location", location)],
                }
            )
            await send({"type": "http.response.body", "body": b""})

        self.gateway = SubpathGateway([("/", "http://frontend"), ("/backend", "http://backend")])
        attach_upstream(self.gateway, upstream)

    def _location(self, start):
        return dict(start["headers"])[b"location"]

    @pytest.mark.asyncio
    async def test_root_relative_location_is_rewritten_under_the_prefix(self):
        start, _, _ = await call_gateway(self.gateway, "/backend/redirect/login")

        # 307 rather than a followed redirect: the browser must see it, not the gateway.
        assert start["status"] == 307
        assert self._location(start) == b"/backend/login"

    @pytest.mark.asyncio
    async def test_root_route_locations_are_left_alone(self):
        start, _, _ = await call_gateway(self.gateway, "/redirect/login")

        assert self._location(start) == b"/login"

    @pytest.mark.asyncio
    async def test_absolute_location_is_left_alone(self):
        async def upstream(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 302,
                    "headers": [(b"location", b"https://auth.example/callback")],
                }
            )
            await send({"type": "http.response.body", "body": b""})

        gateway = SubpathGateway([("/backend", "http://backend")])
        attach_upstream(gateway, upstream)

        start, _, _ = await call_gateway(gateway, "/backend/login")

        assert self._location(start) == b"https://auth.example/callback"


class TestStartUiGatewayWiring:
    """start_ui only changes how it launches the services when a gateway is asked for."""

    def _run_start_ui(self, **kwargs):
        """Run start_ui far enough to launch the backend, then let it bail on the frontend."""
        from cognee.api.v1.ui.ui import start_ui

        with (
            patch("cognee.api.v1.ui.ui._check_required_ports", return_value=(True, [])),
            patch("cognee.api.v1.ui.ui.find_frontend_path", return_value=None),
            patch("cognee.api.v1.ui.ui.prompt_user_for_download", return_value=False),
            patch("cognee.api.v1.ui.ui.time.sleep"),
            patch("cognee.api.v1.ui.ui._stream_process_output"),
            patch("cognee.api.v1.ui.gateway.start_gateway") as mock_gateway,
            patch("subprocess.Popen") as mock_popen,
        ):
            mock_popen.return_value.poll.return_value = None
            start_ui(
                pid_callback=lambda pid: None,
                start_backend=True,
                start_mcp=False,
                **kwargs,
            )

            backend_command = mock_popen.call_args_list[0][0][0]

        return backend_command, mock_gateway

    def test_backend_learns_its_prefix_through_root_path(self):
        backend_command, _ = self._run_start_ui(gateway_port=9000)

        assert "--root-path" in backend_command
        assert backend_command[backend_command.index("--root-path") + 1] == "/backend"

    def test_custom_backend_path_is_passed_through(self):
        backend_command, _ = self._run_start_ui(gateway_port=9000, backend_path="api/")

        assert backend_command[backend_command.index("--root-path") + 1] == "/api"

    def test_default_launch_is_unchanged(self):
        backend_command, mock_gateway = self._run_start_ui()

        assert "--root-path" not in backend_command
        mock_gateway.assert_not_called()

    def test_gateway_port_joins_the_port_preflight(self):
        from cognee.api.v1.ui.ui import start_ui

        with (
            patch("cognee.api.v1.ui.ui._check_required_ports", return_value=(True, [])) as ports,
            patch("cognee.api.v1.ui.ui.find_frontend_path", return_value=None),
            patch("cognee.api.v1.ui.ui.prompt_user_for_download", return_value=False),
        ):
            start_ui(pid_callback=lambda pid: None, gateway_port=9000)

        assert (9000, "Gateway") in ports.call_args[0][0]


class TestStartUiFrontendEnvironment:
    """In gateway mode the UI talks to the API as a same-origin path."""

    def _run(self, **kwargs):
        """Run start_ui through the frontend launch, returning its env and the gateway mock."""
        from cognee.api.v1.ui.ui import start_ui

        with (
            patch("cognee.api.v1.ui.ui._check_required_ports", return_value=(True, [])),
            # find_frontend_path returns a Path, not a str — a str stand-in would hide a
            # filesystem path being mistaken for the URL prefix of the same name.
            patch("cognee.api.v1.ui.ui.find_frontend_path", return_value=Path("/tmp/frontend")),
            patch("cognee.api.v1.ui.ui.check_node_npm", return_value=(True, "ok")),
            patch("cognee.api.v1.ui.ui.install_frontend_dependencies", return_value=True),
            patch(
                "cognee.api.v1.ui.ui.get_nvm_sh_path", return_value=MagicMock(exists=lambda: False)
            ),
            patch("cognee.api.v1.ui.ui._stream_process_output"),
            patch("cognee.api.v1.ui.ui.time.sleep"),
            patch("cognee.api.v1.ui.gateway.start_gateway") as mock_gateway,
            patch("subprocess.Popen") as mock_popen,
        ):
            mock_popen.return_value.poll.return_value = None
            start_ui(
                pid_callback=lambda pid: None,
                start_backend=True,
                start_mcp=False,
                open_browser=False,
                **kwargs,
            )

            frontend_call = mock_popen.call_args_list[-1]

        return frontend_call[1]["env"], mock_gateway

    def test_api_url_is_relative_so_the_browser_stays_on_one_origin(self):
        env, _ = self._run(gateway_port=9000)

        assert env["NEXT_PUBLIC_LOCAL_API_URL"] == "/backend"

    def test_api_url_is_untouched_without_a_gateway(self):
        env, _ = self._run()

        assert "NEXT_PUBLIC_LOCAL_API_URL" not in env

    def test_gateway_routes_the_url_prefix_not_the_frontend_directory(self):
        """start_ui also has a frontend *directory* local; the gateway must not get it."""
        _, mock_gateway = self._run(gateway_port=9000)

        routes = mock_gateway.call_args[0][1]
        prefixes = [prefix for prefix, _ in routes]

        assert "/" in prefixes
        assert all(isinstance(prefix, str) for prefix, _ in routes)
        assert not any("tmp" in prefix for prefix in prefixes)
