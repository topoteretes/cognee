"""Real HTTP transport checks against a loopback proxy, with no external services."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from tenacity import wait_none

from cognee.shared import telemetry_sender
from cognee.shared.telemetry_sender import TelemetrySender


@pytest.mark.parametrize("status", [200, 401, 413, 429, 503])
def test_real_http_batch_delivery_and_rejection(status, monkeypatch):
    bodies = []

    class Proxy(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            bodies.append(json.loads(body))
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_):
            return

    monkeypatch.setattr(telemetry_sender, "FLUSH_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(telemetry_sender, "wait_exponential_jitter", lambda **_: wait_none())
    with ThreadingHTTPServer(("127.0.0.1", 0), Proxy) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        sender = TelemetrySender(f"http://127.0.0.1:{server.server_port}", 0.5)
        try:
            with sender._condition:
                for n in range(10):
                    sender.enqueue({"n": n})
            deadline = time.monotonic() + 3
            expected = 4 if status == 503 else 1
            while sender.snapshot()["exported_events"] + sender.snapshot()["dropped_http"] < 10:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert len(bodies) == expected
            assert all(body == [{"n": n} for n in range(10)] for body in bodies)
            assert sender.snapshot()["exported_events"] == (10 if status == 200 else 0)
        finally:
            sender.close()
            server.shutdown()
            thread.join()
