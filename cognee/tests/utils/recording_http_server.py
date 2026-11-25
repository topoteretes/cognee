"""A local HTTP server that records every request, for asserting a code path makes none."""

import http.server
import threading
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def recording_http_server(
    body: bytes = b"{}", content_type: str = "application/json"
) -> Iterator[tuple[str, list[str]]]:
    """Serve ``body`` on 127.0.0.1; yields the base URL and the list of request paths seen."""
    requests: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(body)

        do_POST = do_GET

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", requests
    finally:
        server.shutdown()
        server.server_close()
