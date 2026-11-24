"""Real git/libcurl must use the pin and refuse redirects to a new destination."""

import ipaddress
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cognee.tasks.code_graph.resolve_repo import _credential_env, _pinned_http_env


def test_git_uses_pinned_address_and_refuses_redirects():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/private")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://pinning.invalid:{server.server_port}/repo.git"
        # Loopback is a test fixture only. Production supplies validate_outbound_url's
        # public addresses; pinning.invalid has no usable public DNS resolution.
        env = _pinned_http_env(url, [ipaddress.ip_address("127.0.0.1")], None)
        result = subprocess.run(
            ["git", "ls-remote", url],
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode != 0
        assert len(requests) == 1
        assert "302" in result.stderr
        assert "port 1" not in result.stderr  # No connection to the redirect target.
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_pinning_preserves_scoped_credentials():
    url = "https://github.com/org/repo.git"
    auth = _credential_env("test-token", url)
    result = _pinned_http_env(url, [ipaddress.ip_address("140.82.112.3")], auth)
    assert result["GIT_CONFIG_KEY_0"] == auth["GIT_CONFIG_KEY_0"]
    assert result["GIT_CONFIG_VALUE_0"] == auth["GIT_CONFIG_VALUE_0"]
    values = {
        result[f"GIT_CONFIG_KEY_{i}"]: result[f"GIT_CONFIG_VALUE_{i}"]
        for i in range(int(result["GIT_CONFIG_COUNT"]))
    }
    assert values["http.curloptResolve"] == "github.com:443:140.82.112.3"
    assert values["http.followRedirects"] == "false"
    assert values["http.sslVerify"] == "true"
