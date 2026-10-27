"""``POST /datasets/{id}/sparql``, ``datasets.sparql`` and the pure SPARQL helpers."""

import asyncio
import importlib
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import XSD

from cognee.modules.graph.rdf import sparql as sparql_module
from cognee.modules.graph.rdf.export import DEFAULT_BASE_IRI
from cognee.modules.users.methods import get_authenticated_user

datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")
serve_state = importlib.import_module("cognee.api.v1.serve.state")

DATASET_ID = uuid.uuid4()
OWNER_ID = uuid.uuid4()

# A tiny property graph: two people, one knows the other, both typed Person.
NODES = [
    ("person-1", {"name": "Ada", "type": "Entity"}),
    ("person-2", {"name": "Grace", "type": "Entity"}),
    ("type-person", {"name": "Person", "type": "EntityType"}),
]
EDGES = [
    ("person-1", "person-2", "knows", {}),
    ("person-1", "type-person", "is_a", {}),
    ("person-2", "type-person", "is_a", {}),
]

SELECT_LABELS = "SELECT ?s ?name WHERE { ?s rdfs:label ?name } ORDER BY ?name"


class FakeGraphEngine:
    def __init__(self, nodes=NODES, edges=EDGES, *, num_nodes=None, data_delay=0.0):
        self._nodes = nodes
        self._edges = edges
        self._num_nodes = len(nodes) if num_nodes is None else num_nodes
        self._data_delay = data_delay
        self.get_graph_data_calls = 0

    async def get_graph_counts(self):
        return self._num_nodes, len(self._edges)

    async def get_graph_data(self):
        self.get_graph_data_calls += 1
        if self._data_delay:
            await asyncio.sleep(self._data_delay)
        return self._nodes, self._edges


@pytest.fixture
def graph(monkeypatch):
    """Scope the sparql module to a fake engine; records the dataset context it asked for."""
    engine = FakeGraphEngine()
    contexts = []

    @asynccontextmanager
    async def _context(dataset_id, owner_id, *args, **kwargs):
        contexts.append((dataset_id, owner_id))
        yield

    async def _get_graph_engine():
        return engine

    monkeypatch.setattr(sparql_module, "set_database_global_context_variables", _context)
    monkeypatch.setattr(sparql_module, "get_graph_engine", _get_graph_engine)
    monkeypatch.delenv("SPARQL_MAX_GRAPH_NODES", raising=False)
    monkeypatch.delenv("SPARQL_QUERY_TIMEOUT_SECONDS", raising=False)
    engine.contexts = contexts
    return engine


@pytest.fixture
def client(monkeypatch, graph):
    module = importlib.import_module("cognee.api.v1.datasets.routers.get_datasets_router")
    monkeypatch.setattr(module, "send_telemetry", lambda *a, **k: None)

    authorized = SimpleNamespace(
        datasets=[SimpleNamespace(id=DATASET_ID, owner_id=OWNER_ID)], calls=[]
    )

    async def _authorized(dataset_ids, permission, user):
        authorized.calls.append((list(dataset_ids), permission))
        return authorized.datasets

    monkeypatch.setattr(module, "get_authorized_existing_datasets", _authorized)

    app = FastAPI()
    app.include_router(module.get_datasets_router(), prefix="/api/v1/datasets")

    async def _user():
        return SimpleNamespace(
            id=str(uuid.uuid4()),
            email="default@example.com",
            is_active=True,
            tenant_id=str(uuid.uuid4()),
        )

    app.dependency_overrides[get_authenticated_user] = _user
    with TestClient(app) as c:
        c.authorized = authorized
        c.graph = graph
        yield c


def _sparql(client, query, **extra):
    return client.post(f"/api/v1/datasets/{DATASET_ID}/sparql", json={"query": query, **extra})


# --- HTTP: query forms --------------------------------------------------------


def test_select_returns_variables_and_rows(client):
    response = _sparql(client, SELECT_LABELS)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["variables"] == ["s", "name"]
    assert body["rows"] == [
        [f"{DEFAULT_BASE_IRI}node/person-1", "Ada"],
        [f"{DEFAULT_BASE_IRI}node/person-2", "Grace"],
        [f"{DEFAULT_BASE_IRI}node/type-person", "Person"],
    ]
    assert body["row_count"] == 3
    assert body["truncated"] is False
    # Only SELECT keys are present.
    assert set(body) == {"variables", "rows", "row_count", "truncated"}


def test_select_is_scoped_to_the_authorized_dataset(client):
    """Read permission is checked first, then the engine is opened under that dataset."""
    _sparql(client, SELECT_LABELS)

    assert client.authorized.calls == [([DATASET_ID], "read")]
    assert client.graph.contexts == [(DATASET_ID, OWNER_ID)]


def test_ask_returns_boolean(client):
    yes = _sparql(client, "ASK { ?s rdfs:label 'Ada' }")
    no = _sparql(client, "ASK { ?s rdfs:label 'Nobody' }")

    assert yes.status_code == 200 and yes.json() == {"boolean": True}
    assert no.status_code == 200 and no.json() == {"boolean": False}


def test_construct_returns_turtle(client):
    response = _sparql(
        client,
        "CONSTRUCT { ?a <urn:knows> ?b } WHERE { ?a <https://cognee.ai/graph/prop/knows> ?b }",
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"turtle"}
    # Parse the Turtle back rather than string-match: rdflib picks the prefixes.
    constructed = Graph().parse(data=body["turtle"], format="turtle")
    assert set(constructed) == {
        (
            URIRef(f"{DEFAULT_BASE_IRI}node/person-1"),
            URIRef("urn:knows"),
            URIRef(f"{DEFAULT_BASE_IRI}node/person-2"),
        )
    }


def test_is_a_edges_become_rdf_type(client):
    response = _sparql(client, "SELECT ?who WHERE { ?who a ?cls . ?cls rdfs:label 'Person' }")

    assert response.status_code == 200
    assert sorted(row[0] for row in response.json()["rows"]) == [
        f"{DEFAULT_BASE_IRI}node/person-1",
        f"{DEFAULT_BASE_IRI}node/person-2",
    ]


def test_base_iri_is_applied_to_minted_iris(client):
    response = _sparql(client, SELECT_LABELS, base_iri="https://example.org/mem/")

    assert response.status_code == 200
    assert response.json()["rows"][0][0] == "https://example.org/mem/node/person-1"


# --- HTTP: guards -------------------------------------------------------------


def test_limit_truncates_and_flags_it(client):
    response = _sparql(client, SELECT_LABELS, limit=2)

    assert response.status_code == 200
    body = response.json()
    assert body["row_count"] == 2
    assert len(body["rows"]) == 2
    assert body["truncated"] is True


@pytest.mark.parametrize("limit", [0, 10_001])
def test_limit_out_of_range_is_rejected(client, limit):
    response = _sparql(client, SELECT_LABELS, limit=limit)

    assert response.status_code == 422
    assert client.graph.get_graph_data_calls == 0


@pytest.mark.parametrize(
    "query",
    [
        "INSERT DATA { <urn:a> <urn:b> <urn:c> }",
        "DELETE WHERE { ?s ?p ?o }",
        "DELETE { ?s ?p ?o } INSERT { ?s ?p 'x' } WHERE { ?s ?p ?o }",
        "LOAD <http://example.org/data.ttl>",
        "CLEAR ALL",
        "DROP GRAPH <urn:g>",
        "CREATE GRAPH <urn:g>",
        "# a comment\ndelete where { ?s ?p ?o }",
        "SELECT * WHERE { ?s ?p ?o } # harmless\n; DROP ALL",
    ],
)
def test_update_queries_are_refused_with_400(client, query):
    response = _sparql(client, query)

    assert response.status_code == 400, response.text
    assert "SPARQL Update is not supported" in response.json()["message"]
    # Refused before the graph is read at all.
    assert client.graph.get_graph_data_calls == 0


def test_malformed_query_is_400(client):
    response = _sparql(client, "SELECT ?s WHERE { ?s rdfs:label")

    assert response.status_code == 400
    assert "Invalid SPARQL query" in response.json()["message"]


def test_over_size_graph_is_413_and_points_at_the_export(client, monkeypatch):
    monkeypatch.setenv("SPARQL_MAX_GRAPH_NODES", "2")

    response = _sparql(client, SELECT_LABELS)

    assert response.status_code == 413, response.text
    message = response.json()["message"]
    assert "3 nodes" in message and "limit of 2" in message
    assert "export" in message.lower()
    # The ceiling is checked on the count alone; the graph is never materialized.
    assert client.graph.get_graph_data_calls == 0


def test_slow_query_is_504(client, monkeypatch):
    """A query past SPARQL_QUERY_TIMEOUT_SECONDS answers 504 Gateway Timeout."""
    monkeypatch.setenv("SPARQL_QUERY_TIMEOUT_SECONDS", "0.05")
    client.graph._data_delay = 1.0

    response = _sparql(client, SELECT_LABELS)

    assert response.status_code == 504, response.text
    assert "0.05s" in response.json()["message"]


def test_unauthorized_dataset_is_404_like_the_other_dataset_routes(client):
    client.authorized.datasets = []

    response = _sparql(client, SELECT_LABELS)

    assert response.status_code == 404
    assert response.json() == {"message": f"Dataset ({DATASET_ID}) not found."}
    assert client.graph.contexts == []


# --- pure helpers -------------------------------------------------------------


def test_term_to_json_flattens_rdflib_terms():
    assert sparql_module.term_to_json(URIRef("https://example.org/x")) == "https://example.org/x"
    assert sparql_module.term_to_json(Literal("Ada")) == "Ada"
    assert sparql_module.term_to_json(Literal("Ada", lang="en")) == "Ada"
    assert sparql_module.term_to_json(Literal(42)) == "42"
    assert sparql_module.term_to_json(Literal("3.5", datatype=XSD.decimal)) == "3.5"
    assert sparql_module.term_to_json(BNode("b0")) == "_:b0"
    assert sparql_module.term_to_json(None) is None


@pytest.mark.parametrize(
    "query",
    [
        "SELECT ?s WHERE { ?s rdfs:label ?name } LIMIT 5",
        "ASK { ?s ?p ?o }",
        # Update keywords inside string literals, IRIs and comments are not operations.
        'SELECT ?s WHERE { ?s rdfs:label ?n FILTER(CONTAINS(?n, "delete")) }',
        "SELECT ?s WHERE { ?s <http://example.org/vocab#create> ?o }",
        "SELECT ?s WHERE { ?s ?p ?o } # do not INSERT anything",
        "SELECT ?s WHERE { ?s ?p ?o } # <urn:x#DROP>",
        "SELECT ?insert ?delete WHERE { ?insert ?p ?delete }",
        "PREFIX ex: <http://example.org/> SELECT ?s WHERE { ?s ex:load ?o }",
        "SELECT ?s WHERE { ?s ?p ?o } ORDER BY ?created",
    ],
)
def test_read_only_queries_pass_the_update_guard(query):
    assert sparql_module.is_update_query(query) is False
    sparql_module.assert_read_only(query)


def test_strip_sparql_comments_keeps_hashes_in_iris_and_strings():
    query = 'SELECT ?s WHERE { ?s <urn:x#frag> "issue #42" } # trailing\n# whole line\nLIMIT 1'
    stripped = sparql_module.strip_sparql_comments(query)

    assert "<urn:x#frag>" in stripped
    assert '"issue #42"' in stripped
    assert "trailing" not in stripped and "whole line" not in stripped
    assert stripped.endswith("LIMIT 1")


def test_format_sparql_result_select_truncation_walks_only_limit_plus_one():
    from cognee.modules.graph.rdf.export import graph_data_to_rdf

    graph = graph_data_to_rdf(NODES, EDGES)
    result = graph.query(SELECT_LABELS)

    body = sparql_module.format_sparql_result(result, limit=1)

    assert body == {
        "variables": ["s", "name"],
        "rows": [[f"{DEFAULT_BASE_IRI}node/person-1", "Ada"]],
        "row_count": 1,
        "truncated": True,
    }


# --- SDK: datasets.sparql -----------------------------------------------------


@pytest.mark.asyncio
async def test_sdk_sparql_runs_locally_under_the_authorized_dataset(monkeypatch, graph):
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    user = SimpleNamespace(id=uuid.uuid4(), tenant_id=None)
    dataset = SimpleNamespace(id=DATASET_ID, owner_id=OWNER_ID)
    authorize = AsyncMock(return_value=dataset)

    with patch.object(datasets_module, "get_authorized_dataset", authorize):
        body = await datasets_module.datasets.sparql(
            DATASET_ID, "ASK { ?s rdfs:label 'Ada' }", user=user
        )

    assert body == {"boolean": True}
    authorize.assert_awaited_once_with(user, DATASET_ID)
    assert graph.contexts == [(DATASET_ID, OWNER_ID)]


@pytest.mark.asyncio
async def test_sdk_sparql_refuses_an_inaccessible_dataset(monkeypatch, graph):
    from cognee.modules.data.exceptions.exceptions import UnauthorizedDataAccessError

    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    user = SimpleNamespace(id=uuid.uuid4(), tenant_id=None)

    with (
        patch.object(datasets_module, "get_authorized_dataset", AsyncMock(return_value=None)),
        pytest.raises(UnauthorizedDataAccessError),
    ):
        await datasets_module.datasets.sparql(DATASET_ID, SELECT_LABELS, user=user)

    assert graph.contexts == []


@pytest.mark.asyncio
async def test_sdk_sparql_routes_to_the_remote_instance_under_serve(monkeypatch):
    remote = {"variables": ["s"], "rows": [["urn:a"]], "row_count": 1, "truncated": False}
    client = SimpleNamespace(sparql=AsyncMock(return_value=remote))
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: client)
    authorize = AsyncMock()

    with patch.object(datasets_module, "get_authorized_dataset", authorize):
        body = await datasets_module.datasets.sparql(
            DATASET_ID, SELECT_LABELS, base_iri="https://x/", limit=7
        )

    assert body == remote
    client.sparql.assert_awaited_once_with(
        DATASET_ID, SELECT_LABELS, base_iri="https://x/", limit=7
    )
    authorize.assert_not_called()
