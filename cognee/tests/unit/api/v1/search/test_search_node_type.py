"""node_type on POST /api/v1/search and on the remote branch of search() (SDK-1448).

The SDK takes a class; the wire carries its name. The router resolves the name
before calling search(), and the remote branch sends the name the router expects.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.serve.cloud_client import CloudClient
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
from cognee.modules.search.types import SearchType
from cognee.tasks.schema.models import SchemaColumn

# The routers package __init__ re-exports the same-named factory, so import by dotted path.
search_router_module = importlib.import_module("cognee.api.v1.search.routers.get_search_router")
SearchPayloadDTO = search_router_module.SearchPayloadDTO


def _client(monkeypatch, fake_search):
    app = FastAPI()
    app.include_router(search_router_module.get_search_router(), prefix="/api/v1/search")
    app.dependency_overrides[search_router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=uuid4(), tenant_id=None
    )
    monkeypatch.setattr(search_router_module, "send_telemetry", lambda *a, **k: None)

    import cognee.api.v1.search as search_pkg

    monkeypatch.setattr(search_pkg, "search", fake_search)
    return TestClient(app, raise_server_exceptions=False)


def _capturing_search(captured: dict):
    async def fake_search(**kwargs):
        captured.update(kwargs)
        return []

    return fake_search


def test_search_dto_takes_node_type_in_both_casings():
    assert SearchPayloadDTO.model_validate(
        {"query": "q", "nodeType": "SchemaColumn"}
    ).node_type == ("SchemaColumn")
    assert SearchPayloadDTO.model_validate({"query": "q", "node_type": "Entity"}).node_type == (
        "Entity"
    )
    assert SearchPayloadDTO.model_validate({"query": "q"}).node_type is None


def test_node_type_is_in_the_openapi_schema_of_the_search_endpoint():
    app = FastAPI()
    app.include_router(search_router_module.get_search_router(), prefix="/api/v1/search")

    properties = app.openapi()["components"]["schemas"]["SearchPayloadDTO"]["properties"]

    assert "nodeType" in properties
    assert "SchemaColumn" in properties["nodeType"]["description"]


def test_search_endpoint_resolves_node_type_to_the_class(monkeypatch):
    captured = {}
    client = _client(monkeypatch, _capturing_search(captured))

    response = client.post(
        "/api/v1/search",
        json={
            "searchType": "GRAPH_COMPLETION",
            "query": "which genres sell best?",
            "nodeType": "SchemaColumn",
            "nodeName": ["genre:name"],
            "onlyContext": True,
        },
    )

    assert response.status_code == 200, response.text
    assert captured["node_type"] is SchemaColumn
    assert captured["node_name"] == ["genre:name"]


def test_search_endpoint_without_node_type_reads_node_name_as_node_sets(monkeypatch):
    captured = {}
    client = _client(monkeypatch, _capturing_search(captured))

    response = client.post("/api/v1/search", json={"query": "q", "nodeName": ["proj"]})

    assert response.status_code == 200, response.text
    assert captured["node_type"] is NodeSet
    assert captured["node_name"] == ["proj"]


def test_search_endpoint_rejects_an_unknown_node_type_with_422_naming_it(monkeypatch):
    captured = {}
    client = _client(monkeypatch, _capturing_search(captured))

    # The global error handler lives on the real app; here the validation error
    # must leave the handler unhandled rather than be swallowed into a 500.
    from cognee.exceptions import CogneeValidationError

    with pytest.raises(CogneeValidationError) as error:
        TestClient(client.app).post(
            "/api/v1/search", json={"query": "q", "nodeType": "Nope", "nodeName": ["x"]}
        )

    assert error.value.status_code == 422
    assert "'Nope'" in error.value.message
    assert captured == {}


class _Response:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def json(self):
        return []


class _Session:
    def __init__(self):
        self.payload = None

    def post(self, _url, *, json):
        self.payload = json
        return _Response()


def _cloud_client():
    client = CloudClient("https://example.test", "key")
    session = _Session()
    client._get_session = AsyncMock(return_value=session)
    return client, session


@pytest.mark.asyncio
async def test_cloud_client_sends_the_node_type_name():
    client, session = _cloud_client()

    await client.search("q", node_type=SchemaColumn, node_name=["genre:name"])

    assert session.payload["nodeType"] == "SchemaColumn"
    assert session.payload["nodeName"] == ["genre:name"]


@pytest.mark.asyncio
@pytest.mark.parametrize("node_type", [NodeSet, None])
async def test_cloud_client_omits_node_type_for_the_node_set_default(node_type):
    client, session = _cloud_client()

    await client.search("q", node_type=node_type, node_name=["proj"])

    assert "nodeType" not in session.payload
    assert session.payload["nodeName"] == ["proj"]


@pytest.fixture
def remote_search(monkeypatch):
    """search() with a remote client that records what it was asked."""
    api_search_mod = importlib.import_module("cognee.api.v1.search.search")
    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    captured = {}

    async def dummy_remote_search(query_text, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        serve_state, "get_remote_client", lambda: SimpleNamespace(search=dummy_remote_search)
    )
    return api_search_mod.search, captured


@pytest.mark.asyncio
async def test_remote_search_forwards_node_type(remote_search):
    search, captured = remote_search

    await search(
        "q",
        query_type=SearchType.GRAPH_COMPLETION,
        node_type=SchemaColumn,
        node_name=["genre:name"],
    )

    assert captured["node_type"] is SchemaColumn
    assert captured["node_name"] == ["genre:name"]


@pytest.mark.asyncio
async def test_remote_search_forwards_the_node_set_default(remote_search):
    search, captured = remote_search

    await search("q", node_name=["proj"])

    assert captured["node_type"] is NodeSet


@pytest.mark.asyncio
async def test_remote_search_refuses_names_without_a_type_like_the_local_call(remote_search):
    search, captured = remote_search

    with pytest.raises(QueryValidationError):
        await search("q", node_type=None, node_name=["proj"])

    assert captured == {}
