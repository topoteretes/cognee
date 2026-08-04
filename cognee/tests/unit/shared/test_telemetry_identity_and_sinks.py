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


def test_dataset_id_is_extracted_from_every_spelling_call_sites_use():
    """remember/forget use dataset_id, recall uses dataset_ids, improve uses dataset."""
    from cognee.modules.telemetry.postgres_sink import _dataset_id

    dataset = uuid.uuid4()

    assert _dataset_id({"dataset_id": str(dataset)}) == dataset
    assert _dataset_id({"dataset_ids": str(dataset)}) == dataset
    assert _dataset_id({"dataset": str(dataset)}) == dataset


def test_dataset_id_is_null_when_absent_ambiguous_or_a_name():
    from cognee.modules.telemetry.postgres_sink import _dataset_id

    one, two = uuid.uuid4(), uuid.uuid4()

    assert _dataset_id({}) is None
    assert _dataset_id({"dataset_id": ""}) is None
    # A multi-dataset recall has no single dataset; the list stays in properties.
    assert _dataset_id({"dataset_ids": f"{one},{two}"}) is None
    # ``dataset`` is usually a human name, not an id.
    assert _dataset_id({"dataset": "product-docs"}) is None


def test_to_row_populates_dataset_id():
    from cognee.modules.telemetry.postgres_sink import _to_row

    dataset = uuid.uuid4()

    class Row:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    row = _to_row(
        Row, {"event_name": "cognee.remember", "properties": {"dataset_id": str(dataset)}}
    )

    assert row.dataset_id == dataset


def test_origin_defaults_to_sdk(monkeypatch):
    payloads = _capture_http(monkeypatch)
    monkeypatch.setenv("TELEMETRY_SINK", "http")
    monkeypatch.delenv("TELEMETRY_ORIGIN", raising=False)

    send_telemetry("cognee.remember", FakeUser(uuid.uuid4(), uuid.uuid4()))

    assert payloads[0]["properties"]["origin"] == "sdk"


def test_origin_can_be_set_deployment_wide_by_env(monkeypatch):
    payloads = _capture_http(monkeypatch)
    monkeypatch.setenv("TELEMETRY_SINK", "http")
    monkeypatch.setenv("TELEMETRY_ORIGIN", "cloud")

    send_telemetry("cognee.remember", FakeUser(uuid.uuid4(), uuid.uuid4()))

    assert payloads[0]["properties"]["origin"] == "cloud"


def test_context_origin_beats_the_env_default(monkeypatch):
    """The API middleware sets this per request, so it must win."""
    from cognee.context_global_variables import telemetry_origin

    payloads = _capture_http(monkeypatch)
    monkeypatch.setenv("TELEMETRY_SINK", "http")
    monkeypatch.setenv("TELEMETRY_ORIGIN", "cloud")

    token = telemetry_origin.set("mcp")
    try:
        send_telemetry("cognee.recall", FakeUser(uuid.uuid4(), uuid.uuid4()))
    finally:
        telemetry_origin.reset(token)

    assert payloads[0]["properties"]["origin"] == "mcp"


def test_pipeline_run_id_is_picked_up_from_context(monkeypatch):
    from cognee.context_global_variables import current_pipeline_run_id

    payloads = _capture_http(monkeypatch)
    monkeypatch.setenv("TELEMETRY_SINK", "http")
    run_id = uuid.uuid4()

    token = current_pipeline_run_id.set(run_id)
    try:
        send_telemetry("Pipeline Run Completed", FakeUser(uuid.uuid4(), uuid.uuid4()))
    finally:
        current_pipeline_run_id.reset(token)

    assert payloads[0]["properties"]["pipeline_run_id"] == str(run_id)


def test_pipeline_run_id_absent_outside_a_pipeline(monkeypatch):
    payloads = _capture_http(monkeypatch)
    monkeypatch.setenv("TELEMETRY_SINK", "http")

    send_telemetry("cognee.forget", FakeUser(uuid.uuid4(), uuid.uuid4()))

    assert "pipeline_run_id" not in payloads[0]["properties"]


def test_to_row_populates_pipeline_run_id_and_origin():
    from cognee.modules.telemetry.postgres_sink import _to_row

    run_id = uuid.uuid4()

    class Row:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    row = _to_row(
        Row,
        {
            "event_name": "cognee.remember",
            "properties": {"pipeline_run_id": str(run_id), "origin": "mcp"},
        },
    )

    assert row.pipeline_run_id == run_id
    assert row.origin == "mcp"


def test_event_names_normalise_to_a_stable_operation_and_kind():
    """Five legacy naming styles must collapse to one contract for the UI."""
    from cognee.modules.telemetry.event_names import normalize_event

    cases = {
        # cognee.<op>
        "cognee.remember": ("remember", "operation"),
        "cognee.forget": ("forget", "operation"),
        "cognee.remember.code_graph": ("remember", "operation"),
        "cognee.session.add_qa": ("session", "operation"),
        # cognee.<op> EXECUTION <STATE>, with and without the prefix
        "cognee.search EXECUTION STARTED": ("search", "operation"),
        "cognee.add EXECUTION COMPLETED": ("add", "operation"),
        "code_description_to_code_part_search EXECUTION FAILED": (
            "code_description_to_code_part_search",
            "operation",
        ),
        # trailing prose must not leak into the operation
        "cognee.cognify DEFAULT TASKS CREATION ERRORED": ("cognify", "operation"),
        # <Thing> API Endpoint Invoked
        "Remember API Endpoint Invoked": ("remember", "endpoint"),
        "Remember Entry API Endpoint Invoked": ("remember", "endpoint"),
        "Cognify AIPTS API Endpoint Invoked": ("cognify", "endpoint"),
        "Add By Text API Endpoint Invoked": ("add", "endpoint"),
        "Api Key Management API Endpoint Invoked": ("api_key", "endpoint"),
        "List Principal Dataset Grants API Endpoint Invoked": ("permissions", "endpoint"),
        "Datasets API Endpoint Invoked": ("datasets", "endpoint"),
        # bookkeeping: owned by no single operation
        "Pipeline Run Started": (None, "pipeline"),
        "Pipeline Run Errored": (None, "pipeline"),
        "Coroutine Task Started": (None, "task"),
        "Async Generator Task Completed": (None, "task"),
        "Function Task Errored": (None, "task"),
    }
    for raw, expected in cases.items():
        assert normalize_event(raw) == expected, raw


def test_unknown_event_names_still_get_a_kind():
    """A new emitter must never produce a null event_kind."""
    from cognee.modules.telemetry.event_names import normalize_event

    assert normalize_event("Something Brand New") == ("something_brand_new", "operation")
    assert normalize_event("") == (None, "operation")


def test_to_row_populates_operation_and_event_kind():
    from cognee.modules.telemetry.postgres_sink import _to_row

    class Row:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    row = _to_row(Row, {"event_name": "Remember Entry API Endpoint Invoked", "properties": {}})

    assert row.event_name == "Remember Entry API Endpoint Invoked"  # raw preserved
    assert row.operation == "remember"
    assert row.event_kind == "endpoint"


def test_model_declaration_is_idempotent():
    """The table must survive its declaration running twice.

    A process that prunes metadata and re-runs migrations in-process (the
    performance benchmark does) re-executes the model module, which raises
    "Table 'telemetry_events' is already defined for this MetaData instance"
    unless the declaration opts into extend_existing.
    """
    import importlib

    module = importlib.import_module("cognee.modules.telemetry.models.TelemetryEvent")

    importlib.reload(module)  # must not raise

    from cognee.infrastructure.databases.relational import Base

    assert "telemetry_events" in Base.metadata.tables


def test_to_row_tolerates_a_non_uuid_tenant_sentinel():
    """``"Single User Tenant"`` must land as NULL, not raise."""
    from cognee.modules.telemetry.postgres_sink import _to_row

    class Row:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    row = _to_row(Row, {"event_name": "e", "properties": {"tenant_id": "Single User Tenant"}})

    assert row.tenant_id is None
