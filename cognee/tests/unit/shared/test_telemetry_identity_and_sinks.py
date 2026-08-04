"""Unit tests for telemetry identity resolution and sink selection."""

import uuid
from collections.abc import Coroutine
from typing import Any

from cognee.shared import utils
from cognee.shared.utils import (
    _is_internal_task_event,
    _resolve_identity,
    _telemetry_sinks,
    send_telemetry,
)


class FakeUser:
    """Stands in for the User model, which defines no ``__str__``.

    ``str()`` on it yields an object repr — the exact failure this suite guards.
    """

    def __init__(self, user_id, tenant_id=None):
        self.id = user_id
        self.tenant_id = tenant_id


def test_resolve_identity_reads_id_and_tenant_off_a_user_model():
    user_id, tenant_id = uuid.uuid4(), uuid.uuid4()

    resolved_user, resolved_tenant = _resolve_identity(FakeUser(user_id, tenant_id))

    assert resolved_user == str(user_id)
    assert resolved_tenant == str(tenant_id)


def test_resolve_identity_never_returns_an_object_repr():
    """A User forwarded whole must not be stringified into ``<... object at 0x...>``."""
    user = FakeUser(uuid.uuid4(), uuid.uuid4())
    assert "object at 0x" in str(user)

    resolved_user, _ = _resolve_identity(user)

    assert "object at 0x" not in resolved_user


def test_resolve_identity_passes_through_sentinels_and_bare_uuids():
    user_id = uuid.uuid4()

    assert _resolve_identity("sdk") == ("sdk", None)
    assert _resolve_identity(user_id) == (str(user_id), None)
    assert _resolve_identity(None) == ("None", None)


def test_resolve_identity_reports_no_tenant_for_untenanted_users():
    assert _resolve_identity(FakeUser(uuid.uuid4(), None))[1] is None


def test_telemetry_sinks_defaults_to_http(monkeypatch):
    monkeypatch.delenv("TELEMETRY_SINK", raising=False)
    assert _telemetry_sinks() == ["http"]


def test_telemetry_sinks_parses_a_list(monkeypatch):
    """Adding a destination must be configuration, not code."""
    monkeypatch.setenv("TELEMETRY_SINK", " Postgres , http ")
    assert _telemetry_sinks() == ["postgres", "http"]


def test_internal_per_task_events_are_recognised():
    # Names are built as f"{task_type} Task Started" in run_tasks_base, so match
    # on the suffix rather than enumerating task types.
    assert _is_internal_task_event("Coroutine Task Started")
    assert _is_internal_task_event("Async Generator Task Completed")
    assert _is_internal_task_event("Some Future Type Task Errored")
    assert not _is_internal_task_event("cognee.remember")
    assert not _is_internal_task_event("Add API Endpoint Invoked")


def _capture_http(monkeypatch) -> list[dict[str, Any]]:
    """Capture HTTP-sink payloads without touching the network."""
    payloads: list[dict[str, Any]] = []

    def capture(payload: dict[str, Any]) -> Coroutine[Any, Any, None]:
        payloads.append(payload)

        async def noop() -> None:
            return None

        return noop()

    class CapturingLoop:
        def create_task(self, coroutine: Coroutine[Any, Any, None]) -> None:
            coroutine.close()

    monkeypatch.setenv("ENV", "prod")
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setattr(utils, "get_anonymous_id", lambda: "anonymous-test-id")
    monkeypatch.setattr(utils, "get_persistent_id", lambda: "persistent-test-id")
    monkeypatch.setattr(utils, "_send_telemetry_request", capture)
    monkeypatch.setattr(utils.asyncio, "get_running_loop", lambda: CapturingLoop())
    return payloads


def test_payload_carries_tenant_id(monkeypatch):
    payloads = _capture_http(monkeypatch)
    monkeypatch.setenv("TELEMETRY_SINK", "http")
    user_id, tenant_id = uuid.uuid4(), uuid.uuid4()

    send_telemetry("cognee.remember", FakeUser(user_id, tenant_id))

    properties = payloads[0]["properties"]
    assert properties["user_id"] == str(user_id)
    assert properties["tenant_id"] == str(tenant_id)
    assert payloads[0]["user_properties"]["tenant_id"] == str(tenant_id)


def test_payload_falls_back_when_deployment_has_no_tenancy(monkeypatch):
    payloads = _capture_http(monkeypatch)
    monkeypatch.setenv("TELEMETRY_SINK", "http")

    send_telemetry("cognee.recall", "sdk")

    assert payloads[0]["properties"]["tenant_id"] == "Single User Tenant"


def test_explicit_tenant_id_property_still_wins(monkeypatch):
    """Call sites that already pass their own tenant_id keep their behaviour."""
    payloads = _capture_http(monkeypatch)
    monkeypatch.setenv("TELEMETRY_SINK", "http")

    send_telemetry(
        "cognee.search EXECUTION STARTED",
        FakeUser(uuid.uuid4(), uuid.uuid4()),
        additional_properties={"tenant_id": "explicit-value"},
    )

    assert payloads[0]["properties"]["tenant_id"] == "explicit-value"


def test_legacy_user_id_keyword_is_still_accepted(monkeypatch):
    """Out-of-tree callers passing ``user_id=`` must not break."""
    payloads = _capture_http(monkeypatch)
    monkeypatch.setenv("TELEMETRY_SINK", "http")
    user_id = uuid.uuid4()

    send_telemetry("legacy_event", user_id=user_id)

    assert payloads[0]["properties"]["user_id"] == str(user_id)


def test_local_sink_receives_user_events_but_not_internal_task_events(monkeypatch):
    enqueued: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "cognee.modules.telemetry.postgres_sink.enqueue", lambda payload: enqueued.append(payload)
    )
    _capture_http(monkeypatch)
    monkeypatch.setenv("TELEMETRY_SINK", "postgres")
    user = FakeUser(uuid.uuid4(), uuid.uuid4())

    send_telemetry("cognee.improve", user)
    send_telemetry("Coroutine Task Started", user)

    assert [payload["event_name"] for payload in enqueued] == ["cognee.improve"]


def test_http_only_configuration_does_not_write_locally(monkeypatch):
    enqueued: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "cognee.modules.telemetry.postgres_sink.enqueue", lambda payload: enqueued.append(payload)
    )
    payloads = _capture_http(monkeypatch)
    monkeypatch.setenv("TELEMETRY_SINK", "http")

    send_telemetry("cognee.forget", FakeUser(uuid.uuid4(), uuid.uuid4()))

    assert enqueued == []
    assert len(payloads) == 1


def test_both_sinks_receive_the_same_event(monkeypatch):
    """The configuration the cross-tenant warehouse follow-up will switch on."""
    enqueued: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "cognee.modules.telemetry.postgres_sink.enqueue", lambda payload: enqueued.append(payload)
    )
    payloads = _capture_http(monkeypatch)
    monkeypatch.setenv("TELEMETRY_SINK", "postgres,http")

    send_telemetry("cognee.remember", FakeUser(uuid.uuid4(), uuid.uuid4()))

    assert len(enqueued) == 1
    assert len(payloads) == 1
    assert enqueued[0] == payloads[0]


def test_telemetry_disabled_beats_every_sink(monkeypatch):
    enqueued: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "cognee.modules.telemetry.postgres_sink.enqueue", lambda payload: enqueued.append(payload)
    )
    payloads = _capture_http(monkeypatch)
    monkeypatch.setenv("TELEMETRY_SINK", "postgres,http")
    monkeypatch.setenv("TELEMETRY_DISABLED", "1")

    send_telemetry("cognee.remember", FakeUser(uuid.uuid4(), uuid.uuid4()))

    assert enqueued == []
    assert payloads == []


def test_to_row_maps_payload_onto_the_model():
    from cognee.modules.telemetry.postgres_sink import _to_row

    user_id, tenant_id = uuid.uuid4(), uuid.uuid4()
    payload = {
        "anonymous_id": "anonymous-test-id",
        "event_name": "cognee.remember",
        "properties": {"user_id": str(user_id), "tenant_id": str(tenant_id), "mode": "test"},
    }

    class Row:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    row = _to_row(Row, payload)

    assert row.event_name == "cognee.remember"
    assert row.user_id == user_id
    assert row.tenant_id == tenant_id
    assert row.anonymous_id == "anonymous-test-id"
    assert row.properties["mode"] == "test"


def test_to_row_tolerates_a_non_uuid_tenant_sentinel():
    """``"Single User Tenant"`` must land as NULL, not raise."""
    from cognee.modules.telemetry.postgres_sink import _to_row

    class Row:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    row = _to_row(Row, {"event_name": "e", "properties": {"tenant_id": "Single User Tenant"}})

    assert row.tenant_id is None
