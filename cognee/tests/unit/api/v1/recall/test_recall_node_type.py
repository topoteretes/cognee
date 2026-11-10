"""node_type on recall(), its router and POST /api/v1/recall (SDK-1448).

recall() takes the class like search() does, threads it into routing and the
graph search, and sends its name over the wire; the router resolves the name back.
"""

import importlib
import types
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.recall.query_router import RoutingConstraints
from cognee.exceptions import CogneeValidationError
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
from cognee.modules.search.types import SearchType
from cognee.tasks.schema.models import SchemaColumn

recall_router_module = importlib.import_module("cognee.api.v1.recall.routers.get_recall_router")
RecallPayloadDTO = recall_router_module.RecallPayloadDTO


def _make_user():
    return types.SimpleNamespace(id=uuid4(), tenant_id=None)


@pytest.fixture
def api_recall_mod():
    return importlib.import_module("cognee.api.v1.recall.recall")


# --- routing constraints -----------------------------------------------------


def test_from_config_scopes_routing_by_node_type(monkeypatch):
    from cognee.modules import preflight

    monkeypatch.setattr(preflight, "llm_available", lambda _config: True)

    node_set = RoutingConstraints.from_config(node_name=["proj"])
    typed = RoutingConstraints.from_config(node_type=SchemaColumn, node_name=["genre:name"])

    assert SearchType.TEMPORAL in node_set.scoped_types
    assert SearchType.CODING_RULES in node_set.scoped_types
    assert SearchType.TEMPORAL not in typed.scoped_types
    assert SearchType.CODING_RULES not in typed.scoped_types
    assert SearchType.HYBRID_COMPLETION in typed.scoped_types
    assert SearchType.GRAPH_COMPLETION in typed.scoped_types


def test_from_config_without_a_filter_is_unscoped(monkeypatch):
    from cognee.modules import preflight

    monkeypatch.setattr(preflight, "llm_available", lambda _config: True)

    assert RoutingConstraints.from_config(node_type=SchemaColumn).scoped_types is None


# --- recall() local path -----------------------------------------------------


@pytest.fixture
def graph_lane(monkeypatch, api_recall_mod):
    """recall() running its graph lane against a recorded authorized_search."""
    from cognee.modules import preflight
    from cognee.modules.recall.config import RecallConfig

    calls = []

    async def dummy_set_session_user_context_variable(_user):
        return None

    async def dummy_authorized_search(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(
        api_recall_mod,
        "set_session_user_context_variable",
        dummy_set_session_user_context_variable,
    )
    monkeypatch.setattr(preflight, "llm_available", lambda _config: True)
    recall_config_mod = importlib.import_module("cognee.modules.recall.config")
    config = RecallConfig(_env_file=None, recall_warmup_shortcircuit=False)
    monkeypatch.setattr(recall_config_mod, "get_recall_config", lambda: config)

    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    search_methods = importlib.import_module("cognee.modules.search.methods.search")
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr(search_methods, "authorized_search", dummy_authorized_search)
    return calls


@pytest.mark.asyncio
async def test_recall_passes_node_type_to_the_graph_search(api_recall_mod, graph_lane):
    await api_recall_mod.recall(
        query_text="which genres sell best?",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4()],
        node_type=SchemaColumn,
        node_name=["genre:name"],
        user=_make_user(),
    )

    assert graph_lane[0]["node_type"] is SchemaColumn
    assert graph_lane[0]["node_name"] == ["genre:name"]


@pytest.mark.asyncio
async def test_recall_defaults_node_type_to_node_set(api_recall_mod, graph_lane):
    await api_recall_mod.recall(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4()],
        node_name=["proj"],
        user=_make_user(),
    )

    assert graph_lane[0]["node_type"] is NodeSet


@pytest.mark.asyncio
async def test_recall_refuses_names_with_an_explicit_none_type_before_routing(
    api_recall_mod, graph_lane
):
    with pytest.raises(QueryValidationError):
        await api_recall_mod.recall(
            query_text="q",
            dataset_ids=[uuid4()],
            node_type=None,
            node_name=["proj"],
            user=_make_user(),
        )

    assert graph_lane == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("node_type", "expected_type"),
    [
        # A node-set filter leaves the temporal route alone.
        (NodeSet, SearchType.TEMPORAL),
        # TEMPORAL applies node sets only, so a typed scope goes to the default.
        (SchemaColumn, SearchType.HYBRID_COMPLETION),
    ],
)
async def test_recall_routes_a_typed_scope_only_to_types_that_apply_it(
    api_recall_mod, graph_lane, node_type, expected_type
):
    await api_recall_mod.recall(
        query_text="What happened in 2019?",
        dataset_ids=[uuid4()],
        node_type=node_type,
        node_name=["genre:name"],
        user=_make_user(),
    )

    assert graph_lane[0]["query_type"] is expected_type
    assert graph_lane[0]["node_type"] is node_type


# --- recall() remote branch --------------------------------------------------


@pytest.fixture
def remote_recall(monkeypatch, api_recall_mod):
    captured = {}

    async def dummy_remote_recall(query_text, query_type, **kwargs):
        captured.update(kwargs)
        return []

    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(
        serve_state, "get_remote_client", lambda: SimpleNamespace(recall=dummy_remote_recall)
    )
    return captured


@pytest.mark.asyncio
async def test_recall_remote_client_forwards_node_type(api_recall_mod, remote_recall):
    await api_recall_mod.recall(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4()],
        node_type=SchemaColumn,
        node_name=["genre:name"],
        user=_make_user(),
    )

    assert remote_recall["node_type"] is SchemaColumn
    assert remote_recall["node_name"] == ["genre:name"]


@pytest.mark.asyncio
async def test_recall_remote_refuses_names_without_a_type_like_the_local_call(
    api_recall_mod, remote_recall
):
    with pytest.raises(QueryValidationError):
        await api_recall_mod.recall(
            query_text="q",
            dataset_ids=[uuid4()],
            node_type=None,
            node_name=["proj"],
            user=_make_user(),
        )

    assert remote_recall == {}


@pytest.mark.asyncio
async def test_cloud_client_recall_sends_the_node_type_name():
    from unittest.mock import AsyncMock

    from cognee.api.v1.serve.cloud_client import CloudClient

    class _Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def json(self):
            return []

    class _OpenApi(_Response):
        async def json(self):
            properties = {"nodeType": {}}
            return {
                "components": {
                    "schemas": {
                        "SearchPayloadDTO": {"properties": properties},
                        "RecallPayloadDTO": {"properties": properties},
                    }
                }
            }

    class _Session:
        payload = None

        def post(self, _url, *, json):
            self.payload = json
            return _Response()

        def get(self, _url):
            return _OpenApi()

    client = CloudClient("https://example.test", "key")
    session = _Session()
    client._get_session = AsyncMock(return_value=session)

    await client.recall("q", None, node_type=SchemaColumn, node_name=["genre:name"])
    assert session.payload["node_type"] == "SchemaColumn"

    await client.recall("q", None, node_type=NodeSet, node_name=["proj"])
    assert "node_type" not in session.payload


# --- router ------------------------------------------------------------------


def test_recall_dto_takes_node_type_in_both_casings():
    assert RecallPayloadDTO.model_validate(
        {"query": "q", "nodeType": "SchemaColumn"}
    ).node_type == ("SchemaColumn")
    assert RecallPayloadDTO.model_validate({"query": "q", "node_type": "Entity"}).node_type == (
        "Entity"
    )
    assert RecallPayloadDTO.model_validate({"query": "q"}).node_type is None


def test_node_type_is_in_the_openapi_schema_of_the_recall_endpoint():
    app = FastAPI()
    app.include_router(recall_router_module.get_recall_router(), prefix="/api/v1/recall")

    properties = app.openapi()["components"]["schemas"]["RecallPayloadDTO"]["properties"]

    assert "nodeType" in properties


def _router_client(monkeypatch, captured: dict):
    """The recall router behind the app's real CogneeApiError handler."""
    from cognee.api.client import exception_handler
    from cognee.exceptions import CogneeApiError

    app = FastAPI()
    app.include_router(recall_router_module.get_recall_router(), prefix="/api/v1/recall")
    app.add_exception_handler(CogneeApiError, exception_handler)
    app.dependency_overrides[recall_router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=uuid4(), tenant_id=None
    )
    monkeypatch.setattr(recall_router_module, "send_telemetry", lambda *a, **k: None)

    async def fake_recall(**kwargs):
        captured.update(kwargs)
        return []

    # The v1 package re-exports the function under the module's name, so import the
    # package by dotted path: the handler imports from that package.
    recall_pkg = importlib.import_module("cognee.api.v1.recall")
    monkeypatch.setattr(recall_pkg, "recall", fake_recall)
    return TestClient(app)


def test_recall_endpoint_resolves_node_type_to_the_class(monkeypatch):
    captured = {}
    client = _router_client(monkeypatch, captured)

    response = client.post(
        "/api/v1/recall",
        json={
            "searchType": "GRAPH_COMPLETION",
            "query": "which genres sell best?",
            "nodeType": "SchemaColumn",
            "nodeName": ["genre:name"],
        },
    )

    assert response.status_code == 200, response.text
    assert captured["node_type"] is SchemaColumn
    assert captured["node_name"] == ["genre:name"]


def test_recall_endpoint_without_node_type_reads_node_name_as_node_sets(monkeypatch):
    captured = {}
    client = _router_client(monkeypatch, captured)

    response = client.post("/api/v1/recall", json={"query": "q", "nodeName": ["proj"]})

    assert response.status_code == 200, response.text
    assert captured["node_type"] is NodeSet


@pytest.mark.parametrize("stream", [False, True])
def test_recall_endpoint_rejects_an_unknown_node_type_with_422_naming_it(monkeypatch, stream):
    """Resolution runs before the transport is chosen, so streaming answers 422 too."""
    captured = {}
    client = _router_client(monkeypatch, captured)

    response = client.post(
        "/api/v1/recall",
        json={"query": "q", "nodeType": "Nope", "nodeName": ["x"], "stream": stream},
    )

    assert response.status_code == 422, response.text
    assert "'Nope'" in response.json()["detail"]
    assert captured == {}
