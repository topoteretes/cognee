"""T9 — Frontend container smoke test.

Builds the cognee-frontend Docker image, starts it alongside the
backend, and verifies that:
  1. The frontend serves HTTP on port 3000.
  2. The served page contains expected Next.js/React markup.
  3. The frontend can reach the backend (NEXT_PUBLIC_BACKEND_API_URL).

This is intentionally a smoke test, not a full UI suite. It proves
the image boots and the frontend↔backend wiring works.

Ref: https://github.com/topoteretes/cognee/issues/3367
"""

import httpx
import pytest

pytestmark = pytest.mark.deployment


class TestFrontendServes:
    """Frontend image builds and serves on :3000."""

    def test_returns_200(self, frontend_container):
        resp = httpx.get("http://localhost:13000", timeout=10)
        assert resp.status_code == 200

    def test_serves_html(self, frontend_container):
        resp = httpx.get("http://localhost:13000", timeout=10)
        content_type = resp.headers.get("content-type", "")
        assert "text/html" in content_type

    def test_contains_nextjs_markup(self, frontend_container):
        resp = httpx.get("http://localhost:13000", timeout=10)
        body = resp.text.lower()
        assert "<div" in body
        assert "__next" in body or "next" in body


class TestFrontendBackendConnectivity:
    """Frontend can reach the backend through the Docker network."""

    def test_backend_health_from_host(self, backend_container):
        """Sanity check — backend is reachable from the test runner."""
        resp = httpx.get("http://localhost:18000/health", timeout=10)
        assert resp.status_code == 200

    def test_frontend_can_reach_backend(self, frontend_container, backend_container):
        """Verify the frontend's env wiring by curling the backend from
        inside the frontend container.

        The frontend itself doesn't proxy API calls server-side (it's a
        client-rendered SPA), so we exec into the container to confirm
        DNS resolution and network connectivity to the backend service.
        """
        import subprocess

        result = subprocess.run(
            [
                "docker",
                "exec",
                frontend_container,
                "wget",
                "-q",
                "-O",
                "-",
                f"http://{backend_container}:8000/health",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, f"Frontend cannot reach backend: {result.stderr}"
        assert "status" in result.stdout
