"""Unit tests for telemetry identity helpers."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cognee.shared import utils
from cognee.shared.utils import CLOUD_TELEMETRY_EVENTS, _get_api_key_tracking_id, send_telemetry


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


def test_api_key_tracking_id_is_derived_once_per_key(monkeypatch):
    """PBKDF2 runs once per key, not on every telemetry event."""
    monkeypatch.setenv("LLM_API_KEY", "provider-prefix-cached-key")
    utils._derive_api_key_tracking_id.cache_clear()

    first = _get_api_key_tracking_id()

    assert _get_api_key_tracking_id() == first
    assert utils._derive_api_key_tracking_id.cache_info().hits == 1


def test_send_telemetry_includes_api_key_tracking_id(monkeypatch):
    """Telemetry payloads expose a pseudonymous API-key ID for analytics grouping."""
    payloads: list[dict[str, Any]] = []

    monkeypatch.setenv("ENV", "prod")
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "provider-prefix-secret-12345")
    monkeypatch.setattr(utils, "get_anonymous_id", lambda: "anonymous-test-id")
    monkeypatch.setattr(utils, "get_persistent_id", lambda: "persistent-test-id")
    monkeypatch.setattr(utils._telemetry_sender, "enqueue", payloads.append)

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


def test_send_telemetry_includes_origin_from_env(monkeypatch):
    """telemetry_origin defaults to "sdk" when TELEMETRY_ORIGIN is unset and is
    honored (e.g. "cloud") when set, so events can be segmented by origin."""
    payloads: list[dict[str, Any]] = []

    monkeypatch.setenv("ENV", "prod")
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setattr(utils, "get_anonymous_id", lambda: "anonymous-test-id")
    monkeypatch.setattr(utils, "get_persistent_id", lambda: "persistent-test-id")
    monkeypatch.setattr(utils._telemetry_sender, "enqueue", payloads.append)

    # Unset -> defaults to "sdk".
    monkeypatch.delenv("TELEMETRY_ORIGIN", raising=False)
    send_telemetry("test_event", "user-123")
    assert payloads[-1]["properties"]["telemetry_origin"] == "sdk"

    # Set -> honored (the managed cloud sets this to "cloud").
    monkeypatch.setenv("TELEMETRY_ORIGIN", "cloud")
    send_telemetry("Pipeline Run Started", "user-123")
    assert payloads[-1]["properties"]["telemetry_origin"] == "cloud"


def test_cloud_origin_sends_only_allowlisted_events(monkeypatch):
    sent: list[str] = []
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setattr(utils._telemetry_sender, "enqueue", lambda p: sent.append(p["event_name"]))

    monkeypatch.setenv("TELEMETRY_ORIGIN", "cloud")
    send_telemetry("Task Started", "user-123")
    send_telemetry("Pipeline Run Started", "user-123")
    monkeypatch.setenv("TELEMETRY_ORIGIN", "sdk")
    send_telemetry("Task Started", "user-123")

    assert sent == ["Pipeline Run Started", "Task Started"]


def test_cloud_allowlist_names_only_events_cognee_emits():
    package = Path(utils.__file__).resolve().parents[1]
    sources = [
        path.read_text()
        for path in package.rglob("*.py")
        if "tests" not in path.parts and path != Path(utils.__file__).resolve()
    ]

    missing = {
        name for name in CLOUD_TELEMETRY_EVENTS if not any(f'"{name}"' in s for s in sources)
    }

    assert not missing, f"Allowlisted events nothing emits: {missing}"


def test_event_properties_cannot_override_the_origin(monkeypatch):
    payloads: list[dict[str, Any]] = []
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setenv("TELEMETRY_ORIGIN", "cloud")
    monkeypatch.setattr(utils._telemetry_sender, "enqueue", payloads.append)

    send_telemetry("cognee.recall", "user-123", {"telemetry_origin": "sdk"})

    assert payloads[0]["properties"]["telemetry_origin"] == "cloud"


def test_send_telemetry_stamps_event_time(monkeypatch):
    """Batching delays delivery, so the event carries when it happened (SDK-1082)."""
    payloads: list[dict[str, Any]] = []
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setattr(utils._telemetry_sender, "enqueue", payloads.append)

    before = datetime.now(timezone.utc)
    send_telemetry("test_event", "user-123")
    event_time = datetime.fromisoformat(payloads[0]["event_time"])

    assert event_time.tzinfo == timezone.utc
    assert before <= event_time <= datetime.now(timezone.utc)
