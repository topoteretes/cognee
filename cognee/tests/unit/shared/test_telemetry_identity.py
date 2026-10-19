"""Unit tests for telemetry identity resolution.

Guards the bug these tests exist for: ``send_telemetry`` used to ``str()`` its
second argument, while ``remember``/``improve``/``forget`` pass the ``User``
model — which defines no ``__str__`` — so those events recorded
``<...User object at 0x...>`` as the user id and carried no tenant at all.
"""

import uuid
from collections.abc import Coroutine
from typing import Any

from cognee.shared import utils
from cognee.shared.utils import _resolve_identity, send_telemetry


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


def _capture_telemetry(monkeypatch) -> list[dict[str, Any]]:
    """Capture emitted payloads without touching the network."""
    payloads: list[dict[str, Any]] = []

    def capture(payload: dict[str, Any]) -> Coroutine[Any, Any, None]:
        payloads.append(payload)

        async def noop() -> None:
            return None

        return noop()

    class StubTask:
        """send_telemetry tracks tasks in _TELEMETRY_TASKS via add_done_callback."""

        def add_done_callback(self, callback) -> None:
            callback(self)

    class CapturingLoop:
        def create_task(self, coroutine: Coroutine[Any, Any, None]) -> StubTask:
            coroutine.close()
            return StubTask()

    monkeypatch.setenv("ENV", "prod")
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setattr(utils, "get_anonymous_id", lambda: "anonymous-test-id")
    monkeypatch.setattr(utils, "get_persistent_id", lambda: "persistent-test-id")
    monkeypatch.setattr(utils, "_send_telemetry_request", capture)
    monkeypatch.setattr(utils.asyncio, "get_running_loop", lambda: CapturingLoop())
    return payloads


def test_payload_carries_tenant_id(monkeypatch):
    payloads = _capture_telemetry(monkeypatch)
    user_id, tenant_id = uuid.uuid4(), uuid.uuid4()

    send_telemetry("cognee.remember", FakeUser(user_id, tenant_id))

    properties = payloads[0]["properties"]
    assert properties["user_id"] == str(user_id)
    assert properties["tenant_id"] == str(tenant_id)
    assert payloads[0]["user_properties"]["tenant_id"] == str(tenant_id)


def test_payload_accepts_stored_ids_without_a_user_model(monkeypatch):
    payloads = _capture_telemetry(monkeypatch)
    user_id, tenant_id = uuid.uuid4(), uuid.uuid4()

    send_telemetry("Pipeline Run Errored", user_id, tenant_id=tenant_id)

    for properties in (payloads[0]["properties"], payloads[0]["user_properties"]):
        assert properties["user_id"] == str(user_id)
        assert properties["tenant_id"] == str(tenant_id)


def test_explicit_tenant_overrides_the_user_model_tenant(monkeypatch):
    payloads = _capture_telemetry(monkeypatch)
    user_id, tenant_id = uuid.uuid4(), uuid.uuid4()

    send_telemetry("event", FakeUser(user_id, uuid.uuid4()), tenant_id=tenant_id)

    for properties in (payloads[0]["properties"], payloads[0]["user_properties"]):
        assert properties["user_id"] == str(user_id)
        assert properties["tenant_id"] == str(tenant_id)


def test_payload_records_a_real_uuid_for_a_forwarded_user_model(monkeypatch):
    """The regression test proper: the emitted user_id must be the UUID, not a repr."""
    payloads = _capture_telemetry(monkeypatch)
    user_id = uuid.uuid4()

    send_telemetry("cognee.improve", FakeUser(user_id, uuid.uuid4()))

    assert payloads[0]["properties"]["user_id"] == str(user_id)
    assert "object at 0x" not in payloads[0]["properties"]["user_id"]


def test_payload_falls_back_when_deployment_has_no_tenancy(monkeypatch):
    payloads = _capture_telemetry(monkeypatch)

    send_telemetry("cognee.recall", "sdk")

    assert payloads[0]["properties"]["tenant_id"] == "Single User Tenant"


def test_explicit_tenant_id_property_still_wins(monkeypatch):
    """Call sites that already pass their own tenant_id keep their behaviour."""
    payloads = _capture_telemetry(monkeypatch)

    send_telemetry(
        "cognee.search EXECUTION STARTED",
        FakeUser(uuid.uuid4(), uuid.uuid4()),
        additional_properties={"tenant_id": "explicit-value"},
    )

    assert payloads[0]["properties"]["tenant_id"] == "explicit-value"


def test_legacy_user_id_keyword_is_still_accepted(monkeypatch):
    """Out-of-tree callers passing ``user_id=`` must not break."""
    payloads = _capture_telemetry(monkeypatch)
    user_id = uuid.uuid4()

    send_telemetry("legacy_event", user_id=user_id)

    assert payloads[0]["properties"]["user_id"] == str(user_id)


def test_telemetry_disabled_still_wins(monkeypatch):
    payloads = _capture_telemetry(monkeypatch)
    monkeypatch.setenv("TELEMETRY_DISABLED", "1")

    send_telemetry("cognee.remember", FakeUser(uuid.uuid4(), uuid.uuid4()))

    assert payloads == []


class DetachedUser:
    """Simulates an expired/detached ORM instance: attribute access raises
    something that is NOT AttributeError (DetachedInstanceError,
    MissingGreenlet), which getattr's default does not swallow."""

    def __init__(self, user_id=None, broken_attrs=("id", "tenant_id")):
        object.__setattr__(self, "_user_id", user_id)
        object.__setattr__(self, "_broken", set(broken_attrs))

    def __getattr__(self, name):
        if name in self._broken:
            raise RuntimeError(f"Instance is not bound to a Session ({name})")
        if name == "id" and self._user_id is not None:
            return self._user_id
        raise AttributeError(name)


def test_resolve_identity_survives_a_fully_detached_instance():
    """Telemetry identity must never break the operation that emitted it."""
    resolved_user, resolved_tenant = _resolve_identity(DetachedUser())

    assert resolved_user == "unknown-user"
    assert resolved_tenant is None


def test_resolve_identity_keeps_id_when_only_tenant_access_fails():
    user_id = uuid.uuid4()

    resolved_user, resolved_tenant = _resolve_identity(
        DetachedUser(user_id, broken_attrs=("tenant_id",))
    )

    assert resolved_user == str(user_id)
    assert resolved_tenant is None


def test_send_telemetry_emits_despite_detached_user(monkeypatch):
    payloads = _capture_telemetry(monkeypatch)

    send_telemetry("cognee.remember", DetachedUser())  # must not raise

    assert payloads[0]["properties"]["user_id"] == "unknown-user"
    assert payloads[0]["properties"]["tenant_id"] == "Single User Tenant"


class FakeAgentUser(FakeUser):
    """An agent: a child user whose email encodes its name and owner."""

    def __init__(self, name="claude-code", tenant_id=None):
        super().__init__(uuid.uuid4(), tenant_id)
        self.parent_user_id = uuid.uuid4()
        self.email = f"{name}+{self.parent_user_id}@cognee.agent"


def test_payload_attributes_an_agent_user_model(monkeypatch):
    payloads = _capture_telemetry(monkeypatch)
    agent = FakeAgentUser("codex")

    send_telemetry("cognee.remember", agent)

    properties = payloads[0]["properties"]
    assert properties["is_agent"] is True
    assert properties["agent_id"] == str(agent.id) == properties["user_id"]
    assert properties["agent_owner_id"] == str(agent.parent_user_id)
    assert properties["agent_plugin"] == "codex"
    assert payloads[0]["user_properties"]["is_agent"] is True


def test_payload_takes_the_request_agent_when_the_emitter_passes_a_sentinel(monkeypatch):
    """Emitters that pass ``"sdk"`` or a bare id still carry the request's agent."""
    from cognee.modules.operations.agent import agent_identity_from_user, operation_agent_scope

    payloads = _capture_telemetry(monkeypatch)
    agent = FakeAgentUser("claude-code")

    with operation_agent_scope(agent_identity_from_user(agent)):
        send_telemetry("cognee.search EXECUTION STARTED", "sdk")
        send_telemetry("cognee.search EXECUTION STARTED", agent.id)

    for payload in payloads:
        assert payload["properties"]["agent_id"] == str(agent.id)
        assert payload["properties"]["agent_plugin"] == "claude-code"


def test_payload_marks_ordinary_users_as_not_agents(monkeypatch):
    payloads = _capture_telemetry(monkeypatch)

    send_telemetry("cognee.remember", FakeUser(uuid.uuid4(), uuid.uuid4()))
    send_telemetry("cognee.recall", "sdk")

    for payload in payloads:
        properties = payload["properties"]
        assert properties["is_agent"] is False
        assert properties["agent_id"] is None
        assert properties["agent_owner_id"] is None
        assert properties["agent_plugin"] is None
        assert payload["user_properties"]["is_agent"] is False


def test_payload_labels_custom_agents_without_their_name(monkeypatch):
    payloads = _capture_telemetry(monkeypatch)

    send_telemetry("cognee.remember", FakeAgentUser("acme-internal-bot"))

    assert payloads[0]["properties"]["agent_plugin"] == "custom"
    assert "acme-internal-bot" not in str(payloads[0])


def test_payload_survives_a_detached_user_with_agent_fields(monkeypatch):
    payloads = _capture_telemetry(monkeypatch)

    send_telemetry(
        "cognee.improve", DetachedUser(broken_attrs=("id", "tenant_id", "parent_user_id"))
    )

    assert payloads[0]["properties"]["is_agent"] is False
