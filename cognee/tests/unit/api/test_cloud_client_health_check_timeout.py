"""CloudClient._health_check accepts an explicit timeout, used by
``_serve_cloud`` to fail fast when Auth0 / management is degraded.

Before this change, ``_health_check`` was a single line that always used
the session's default 5-minute ``ClientTimeout``. That meant a single
``/health`` call at SDK startup could block up to 30 seconds when the
service was unreachable (sock_connect=30s) — directly causing the
"``cognee.serve()`` hangs on Auth0 outage" symptom in issue #3249.

This test exercises the new ``timeout`` parameter and the new
``HEALTH_CHECK_TIMEOUT`` class attribute, and locks in the default
behaviour for callers that don't pass an explicit timeout.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import aiohttp

from cognee.api.v1.serve.cloud_client import CloudClient


class _FakeResponse:
    """Minimal aiohttp.ClientResponse stand-in for a /health call."""

    def __init__(self, status: int = 200):
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Captures the ``timeout`` kwarg each call was made with."""

    def __init__(self, status: int = 200):
        self.calls: list[dict] = []
        self._status = status

    def get(self, url, **kwargs):
        self.calls.append({"url": url, "kwargs": kwargs})
        return _FakeResponse(self._status)

    async def close(self):
        pass


def test_health_check_default_uses_class_timeout_when_not_specified():
    client = CloudClient(service_url="https://example.test", api_key="k")
    fake = _FakeSession(status=200)

    async def run():
        with patch.object(client, "_get_session", return_value=fake):
            return await client._health_check()

    result = asyncio.run(run())

    assert result is True
    assert len(fake.calls) == 1
    # No explicit timeout → no `timeout` kwarg → preserves prior
    # behaviour (aiohttp's session default kicks in).
    assert "timeout" not in fake.calls[0]["kwargs"]
    assert fake.calls[0]["url"] == "https://example.test/health"


def test_health_check_with_explicit_timeout_uses_health_check_class_default():
    client = CloudClient(service_url="https://example.test", api_key="k")
    fake = _FakeSession(status=200)

    async def run():
        with patch.object(client, "_get_session", return_value=fake):
            # Mirrors the call site in _serve_cloud.
            return await client._health_check(timeout=client.HEALTH_CHECK_TIMEOUT)

    result = asyncio.run(run())

    assert result is True
    passed = fake.calls[0]["kwargs"]["timeout"]
    # 5s total / 2s sock_connect — short enough to surface Auth0/management
    # outages quickly without flaking on a healthy service.
    assert passed.total == 5
    assert passed.sock_connect == 2


def test_health_check_returns_false_on_non_2xx_status():
    client = CloudClient(service_url="https://example.test", api_key="k")
    fake = _FakeSession(status=500)

    async def run():
        with patch.object(client, "_get_session", return_value=fake):
            return await client._health_check(timeout=client.HEALTH_CHECK_TIMEOUT)

    assert asyncio.run(run()) is False


def test_health_check_returns_false_when_session_get_raises():
    """A network error (timeout, connection refused) must not propagate — the
    fast-path caller relies on False to fall through to token refresh."""

    client = CloudClient(service_url="https://example.test", api_key="k")

    class _BrokenSession:
        def get(self, *_args, **_kwargs):
            raise aiohttp.ClientError("connection refused")

        async def close(self):
            pass

    async def run():
        with patch.object(client, "_get_session", return_value=_BrokenSession()):
            return await client._health_check(timeout=client.HEALTH_CHECK_TIMEOUT)

    assert asyncio.run(run()) is False
