"""CloudClient must forward the params the agent integrations depend on.

Covers the params that were once silently dropped (improve's
``session_ids``, remember/add's ``node_set``), the tri-state
``search_type`` on recall (an explicit null opts the server into
auto-routing; an omitted key means GRAPH_COMPLETION), typed errors so
integration circuit breakers can tell transport failures from auth
failures from 4xx/5xx, and per-call timeouts for hook-path budgets.
"""

import asyncio
from uuid import UUID

import aiohttp
import pytest

from cognee.api.v1.serve import state
from cognee.api.v1.serve.cloud_client import CloudClient, UNSET
from cognee.api.v1.serve.exceptions import (
    CogneeAuthError,
    CogneeClientRequestError,
    CogneeServerError,
    CogneeTransportError,
)


class FakeResponse:
    def __init__(self, status=200, json_body=None, text_body="", raise_on_enter=None):
        self.status = status
        self._json = json_body if json_body is not None else {}
        self._text = text_body
        self._raise_on_enter = raise_on_enter

    async def json(self):
        return self._json

    async def text(self):
        return self._text

    async def __aenter__(self):
        if self._raise_on_enter is not None:
            raise self._raise_on_enter
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    closed = False

    def __init__(self, response=None):
        self.response = response or FakeResponse()
        self.calls = []

    def post(self, url, json=None, data=None, timeout=None):
        self.calls.append({"url": url, "json": json, "data": data, "timeout": timeout})
        return self.response


def make_client(response=None):
    client = CloudClient("http://cloud.example", "ck_test")
    session = FakeSession(response)
    client._session = session
    return client, session


def form_fields(form: aiohttp.FormData):
    """Flatten aiohttp FormData into (name, value) pairs."""
    return [(type_options["name"], value) for type_options, _headers, value in form._fields]


def field_values(form: aiohttp.FormData, name: str):
    return [value for field_name, value in form_fields(form) if field_name == name]


# ----- recall: tri-state search_type -----


def test_recall_omits_search_type_by_default():
    client, session = make_client(FakeResponse(json_body=[]))
    asyncio.run(client.recall("question"))
    assert "search_type" not in session.calls[0]["json"]


def test_recall_sends_explicit_null_for_auto_routing():
    client, session = make_client(FakeResponse(json_body=[]))
    asyncio.run(client.recall("question", None))
    payload = session.calls[0]["json"]
    assert "search_type" in payload
    assert payload["search_type"] is None


def test_recall_sends_concrete_search_type():
    client, session = make_client(FakeResponse(json_body=[]))
    asyncio.run(client.recall("question", "CHUNKS"))
    assert session.calls[0]["json"]["search_type"] == "CHUNKS"


def test_unset_sentinel_is_falsy_and_distinct_from_none():
    assert not UNSET
    assert UNSET is not None
    assert repr(UNSET) == "UNSET"


# ----- remember / add: node_set forwarding -----


def test_remember_forwards_node_set_as_repeated_fields():
    client, session = make_client()
    asyncio.run(client.remember("a memory", node_set=["qa", "trace"]))
    assert field_values(session.calls[0]["data"], "node_set") == ["qa", "trace"]


def test_remember_accepts_single_node_set_string():
    client, session = make_client()
    asyncio.run(client.remember("a memory", node_set="agent_actions"))
    assert field_values(session.calls[0]["data"], "node_set") == ["agent_actions"]


def test_remember_without_node_set_sends_no_field():
    client, session = make_client()
    asyncio.run(client.remember("a memory"))
    assert field_values(session.calls[0]["data"], "node_set") == []


def test_add_forwards_node_set_dataset_id_and_background():
    client, session = make_client()
    dataset_id = UUID("00000000-0000-0000-0000-000000000042")
    asyncio.run(
        client.add(
            "content",
            "my_dataset",
            node_set=["user_context"],
            dataset_id=dataset_id,
            run_in_background=True,
        )
    )
    form = session.calls[0]["data"]
    assert field_values(form, "node_set") == ["user_context"]
    assert field_values(form, "datasetId") == [str(dataset_id)]
    assert field_values(form, "run_in_background") == ["true"]
    assert field_values(form, "datasetName") == ["my_dataset"]


# ----- improve: session bridge params -----


def test_improve_forwards_session_ids_and_pipeline_params():
    client, session = make_client()
    asyncio.run(
        client.improve(
            "agent_sessions",
            session_ids=["s1", "s2"],
            node_name=["entity"],
            run_in_background=True,
            build_global_context_index=True,
            extraction_tasks=["extract"],
            enrichment_tasks=["enrich"],
            data="raw",
        )
    )
    payload = session.calls[0]["json"]
    assert payload["dataset_name"] == "agent_sessions"
    assert payload["session_ids"] == ["s1", "s2"]
    assert payload["node_name"] == ["entity"]
    assert payload["run_in_background"] is True
    assert payload["build_global_context_index"] is True
    assert payload["extraction_tasks"] == ["extract"]
    assert payload["enrichment_tasks"] == ["enrich"]
    assert payload["data"] == "raw"


def test_improve_uuid_dataset_becomes_dataset_id():
    client, session = make_client()
    dataset_id = UUID("00000000-0000-0000-0000-000000000007")
    asyncio.run(client.improve(dataset_id))
    payload = session.calls[0]["json"]
    assert payload["dataset_id"] == str(dataset_id)
    assert "dataset_name" not in payload


# ----- cognify dataset routing -----


def test_cognify_uuid_datasets_travel_as_dataset_ids():
    # The server resolves ``datasets`` entries by name and creates any it
    # cannot find — a stringified UUID there would create a junk dataset.
    client, session = make_client()
    dataset_id = UUID("00000000-0000-0000-0000-000000000009")
    asyncio.run(client.cognify([dataset_id]))
    payload = session.calls[0]["json"]
    assert payload["dataset_ids"] == [str(dataset_id)]
    assert "datasets" not in payload


def test_cognify_single_uuid_travels_as_dataset_id():
    client, session = make_client()
    dataset_id = UUID("00000000-0000-0000-0000-000000000010")
    asyncio.run(client.cognify(dataset_id))
    payload = session.calls[0]["json"]
    assert payload["dataset_ids"] == [str(dataset_id)]
    assert "datasets" not in payload


def test_cognify_names_still_travel_as_datasets():
    client, session = make_client()
    asyncio.run(client.cognify(["docs", "notes"]))
    payload = session.calls[0]["json"]
    assert payload["datasets"] == ["docs", "notes"]
    assert "dataset_ids" not in payload


def test_cognify_without_datasets_sends_neither_key():
    client, session = make_client()
    asyncio.run(client.cognify(None))
    payload = session.calls[0]["json"]
    assert "datasets" not in payload
    assert "dataset_ids" not in payload


# ----- typed errors -----


def run_expecting(client, coro_factory, error_type):
    with pytest.raises(error_type) as excinfo:
        asyncio.run(coro_factory())
    return excinfo.value


def test_auth_errors_raise_cognee_auth_error():
    for status in (401, 403):
        client, _ = make_client(FakeResponse(status=status, text_body="denied"))
        error = run_expecting(client, lambda c=client: c.forget(everything=True), CogneeAuthError)
        assert error.status == status
        assert error.operation == "forget"


def test_4xx_raises_client_request_error_with_parsed_body():
    client, _ = make_client(FakeResponse(status=404, text_body='{"detail": "no dataset"}'))
    error = run_expecting(client, lambda: client.recall("question"), CogneeClientRequestError)
    assert error.status == 404
    assert error.body == {"detail": "no dataset"}


def test_5xx_raises_server_error():
    client, _ = make_client(FakeResponse(status=503, text_body="unavailable"))
    error = run_expecting(client, lambda: client.cognify(["ds"]), CogneeServerError)
    assert error.status == 503


def test_transport_failure_raises_transport_error():
    client, _ = make_client(FakeResponse(raise_on_enter=aiohttp.ClientConnectionError("refused")))
    error = run_expecting(client, lambda: client.remember("data"), CogneeTransportError)
    assert error.operation == "remember"


def test_all_errors_remain_runtime_errors_for_legacy_callers():
    for error_type in (
        CogneeAuthError,
        CogneeClientRequestError,
        CogneeServerError,
        CogneeTransportError,
    ):
        assert issubclass(error_type, RuntimeError)


# ----- per-call timeouts -----


def test_default_timeout_used_when_not_overridden():
    client, session = make_client(FakeResponse(json_body=[]))
    asyncio.run(client.recall("question"))
    assert session.calls[0]["timeout"] is CloudClient.DEFAULT_TIMEOUT


def test_per_call_timeout_bounds_the_request():
    client, session = make_client(FakeResponse(json_body=[]))
    asyncio.run(client.recall("question", timeout=2.5))
    timeout = session.calls[0]["timeout"]
    assert timeout.total == 2.5
    assert timeout.sock_connect == 2.5


def test_long_per_call_timeout_caps_connect_at_thirty_seconds():
    client, session = make_client(FakeResponse(json_body=[]))
    asyncio.run(client.recall("question", timeout=120))
    timeout = session.calls[0]["timeout"]
    assert timeout.total == 120
    assert timeout.sock_connect == 30.0


def test_cogx_archive_upload_keeps_unbounded_total():
    client, session = make_client()
    asyncio.run(client.remember("archive-bytes", content_type="cogx-archive"))
    assert session.calls[0]["timeout"] is CloudClient.UPLOAD_TIMEOUT


# ----- no-silent-drop warning helper -----


def test_warn_reports_only_set_params(monkeypatch):
    warnings = []
    monkeypatch.setattr(state.logger, "warning", lambda *args, **kwargs: warnings.append(args))
    state.warn_unsupported_remote_params(
        "add", user=None, preferred_loaders=["pdf"], vector_db_config=None
    )
    assert len(warnings) == 1
    assert warnings[0][1] == "add"
    assert "preferred_loaders" in warnings[0][2]
    assert "vector_db_config" not in warnings[0][2]


def test_warn_is_silent_when_nothing_dropped(monkeypatch):
    warnings = []
    monkeypatch.setattr(state.logger, "warning", lambda *args, **kwargs: warnings.append(args))
    state.warn_unsupported_remote_params("add", user=None, preferred_loaders=None)
    assert warnings == []
