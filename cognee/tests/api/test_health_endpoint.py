"""Tests for the health endpoint routing and timeout behavior (CLO-318)."""

import asyncio

import pytest

from cognee.api.v1.health import health as health_mod
from cognee.api.v1.health.health import (
    ComponentHealth,
    HealthChecker,
    HealthStatus,
    _check_failure_detail,
    _health_check_timeout,
)


def _healthy(provider):
    async def _check():
        return ComponentHealth(
            status=HealthStatus.HEALTHY,
            provider=provider,
            response_time_ms=1,
            details="ok",
        )

    return _check


class TestVersionedHealthRoute:
    """/api/v1/health must exist (it 404'd before CLO-318); /health is retained."""

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient

        from cognee.api.client import app

        return TestClient(app)

    def test_versioned_health_route_exists(self, client):
        resp = client.get("/api/v1/health")
        # The whole point of CLO-318 AC#2: no longer a 404. 200 (ready) or 503
        # (degraded) are both valid — either proves the route is wired.
        assert resp.status_code != 404
        assert resp.status_code in (200, 503)
        assert "status" in resp.json()

    def test_legacy_health_route_retained(self, client):
        resp = client.get("/health")
        assert resp.status_code in (200, 503)


class TestHealthCheckTimeout:
    @pytest.mark.asyncio
    async def test_slow_component_times_out_instead_of_hanging(self, monkeypatch):
        monkeypatch.setenv("HEALTH_CHECK_TIMEOUT_SECONDS", "0.05")
        checker = HealthChecker()

        async def _hang():
            await asyncio.sleep(10)

        monkeypatch.setattr(checker, "check_relational_db", _hang)
        monkeypatch.setattr(checker, "check_vector_db", _healthy("lancedb"))
        monkeypatch.setattr(checker, "check_graph_db", _healthy("ladybug"))
        monkeypatch.setattr(checker, "check_file_storage", _healthy("local"))

        # Outer guard: if the per-check timeout regressed, this would hang.
        result = await asyncio.wait_for(checker.get_health_status(), timeout=3)

        rel = result.components["relational_db"]
        assert rel.status == HealthStatus.UNHEALTHY
        assert "timed out" in rel.details.lower()
        # A wedged critical component makes the whole tenant report unhealthy (503).
        assert result.status == HealthStatus.UNHEALTHY
        # Unaffected components are still reported healthy.
        assert result.components["vector_db"].status == HealthStatus.HEALTHY


class TestHealthCheckTimeoutConfig:
    def test_defaults_when_unset_or_invalid(self, monkeypatch):
        default = health_mod.DEFAULT_HEALTH_CHECK_TIMEOUT_SECONDS

        monkeypatch.delenv("HEALTH_CHECK_TIMEOUT_SECONDS", raising=False)
        assert _health_check_timeout() == default

        monkeypatch.setenv("HEALTH_CHECK_TIMEOUT_SECONDS", "not-a-number")
        assert _health_check_timeout() == default

        monkeypatch.setenv("HEALTH_CHECK_TIMEOUT_SECONDS", "0")
        assert _health_check_timeout() == default

        monkeypatch.setenv("HEALTH_CHECK_TIMEOUT_SECONDS", "-3")
        assert _health_check_timeout() == default

    def test_honors_valid_override(self, monkeypatch):
        monkeypatch.setenv("HEALTH_CHECK_TIMEOUT_SECONDS", "2.5")
        assert _health_check_timeout() == 2.5


class TestCheckFailureDetail:
    def test_timeout_message(self):
        detail = _check_failure_detail(asyncio.TimeoutError(), 5.0)
        assert "timed out" in detail.lower()
        assert "5" in detail

    def test_generic_failure_message(self):
        assert _check_failure_detail(RuntimeError("boom"), 5.0) == "Health check failed: boom"
