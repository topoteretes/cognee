"""Unit tests for telemetry identity helpers."""

from collections.abc import Coroutine
from typing import Any

from cognee.shared import utils
from cognee.shared.utils import _get_api_key_tracking_id, send_telemetry


def test_api_key_tracking_id_uses_full_key_not_visible_tail(monkeypatch):
    """Keys sharing the same visible tail should still produce different IDs."""
    monkeypatch.setenv("LLM_API_KEY", "provider-prefix-first-12345")
    first = _get_api_key_tracking_id()

    monkeypatch.setenv("LLM_API_KEY", "provider-prefix-second-12345")
    second = _get_api_key_tracking_id()

    assert first.startswith("ak_")
    assert second.startswith("ak_")
    assert first != second
    assert "12345" not in first
    assert "12345" not in second


def test_api_key_tracking_id_supports_deployment_salt(monkeypatch):
    """Deployments can isolate analytics IDs by setting a telemetry salt."""
    monkeypatch.setenv("LLM_API_KEY", "provider-prefix-shared-key")
    monkeypatch.setenv("TELEMETRY_API_KEY_TRACKING_SALT", "first-salt")
    first = _get_api_key_tracking_id()

    monkeypatch.setenv("TELEMETRY_API_KEY_TRACKING_SALT", "second-salt")
    second = _get_api_key_tracking_id()

    assert first != second


def test_send_telemetry_includes_api_key_tracking_id(monkeypatch):
    """Telemetry payloads expose a pseudonymous API-key ID for analytics grouping."""
    payloads: list[dict[str, Any]] = []

    def capture_payload(payload: dict[str, Any]) -> Coroutine[Any, Any, None]:
        payloads.append(payload)

        async def noop() -> None:
            return None

        return noop()

    class CapturingLoop:
        def create_task(self, coroutine: Coroutine[Any, Any, None]) -> None:
            coroutine.close()

    monkeypatch.setenv("ENV", "prod")
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "provider-prefix-secret-12345")
    monkeypatch.setattr(utils, "get_anonymous_id", lambda: "anonymous-test-id")
    monkeypatch.setattr(utils, "get_persistent_id", lambda: "persistent-test-id")
    monkeypatch.setattr(utils, "_send_telemetry_request", capture_payload)
    monkeypatch.setattr(utils.asyncio, "get_running_loop", lambda: CapturingLoop())

    send_telemetry("test_event", "user-123", {"url": "https://example.test/raw"})

    assert len(payloads) == 1
    payload = payloads[0]
    tracking_id = payload["properties"]["api_key_tracking_id"]

    assert tracking_id.startswith("ak_")
    assert payload["user_properties"]["api_key_tracking_id"] == tracking_id
    assert payload["properties"]["api_key_hash"] == tracking_id
    assert payload["user_properties"]["api_key_hash"] == tracking_id
    assert "12345" not in str(payload)
    assert "provider-prefix-secret" not in str(payload)
