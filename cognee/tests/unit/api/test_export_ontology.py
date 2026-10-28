"""Tests for GET /api/v1/ontologies/{ontology_key}/export (SDK-1247).

Covers:
- OntologyService.load_ontology_graph / export_ontology: parse + serialize, user scoping
- Export endpoint: each of the four formats round-trips through rdflib with the
  same triple count, correct Content-Type / Content-Disposition
- Export endpoint: unsupported format -> 400 listing the options
- Export endpoint: unknown key -> 404, another user's key -> 404 (no existence leak)
- Export endpoint: stored file that is not RDF -> 422
- Export endpoint: JSON-LD referencing an external @context / @import -> 422, no request sent
- Export endpoint: an ontology the chosen writer cannot express -> 422 naming the format, not 404
"""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from rdflib import Graph

from cognee.api.client import app
from cognee.api.v1.ontologies.ontologies import (
    EXPORT_FORMATS,
    OntologyNotFoundError,
    OntologyParseError,
    OntologySerializationError,
    OntologyService,
)
from cognee.modules.users.methods import get_authenticated_user
from cognee.tests.utils.recording_http_server import recording_http_server

OWL_FIXTURE = b"""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:ex="http://example.org/medical#">
  <owl:Ontology rdf:about="http://example.org/medical"/>
  <owl:Class rdf:about="http://example.org/medical#Condition">
    <rdfs:label>Condition</rdfs:label>
  </owl:Class>
  <owl:Class rdf:about="http://example.org/medical#Diabetes">
    <rdfs:subClassOf rdf:resource="http://example.org/medical#Condition"/>
    <rdfs:label>Diabetes</rdfs:label>
  </owl:Class>
  <owl:ObjectProperty rdf:about="http://example.org/medical#treatedBy">
    <rdfs:domain rdf:resource="http://example.org/medical#Condition"/>
  </owl:ObjectProperty>
</rdf:RDF>
"""

EXPECTED_TRIPLES = len(Graph().parse(data=OWL_FIXTURE, format="xml"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_client():
    with TestClient(app) as c:
        yield c


def _make_user():
    return SimpleNamespace(
        id=str(uuid.uuid4()),
        email="test@example.com",
        is_active=True,
        tenant_id=str(uuid.uuid4()),
    )


@pytest.fixture
def mock_default_user():
    return _make_user()


@pytest.fixture
def client(test_client, mock_default_user):
    async def override_get_authenticated_user():
        return mock_default_user

    app.dependency_overrides[get_authenticated_user] = override_get_authenticated_user
    yield test_client
    app.dependency_overrides.pop(get_authenticated_user, None)


@pytest.fixture
def ontology_service(tmp_path):
    """OntologyService with base_dir pointed at tmp_path."""
    fake_config = SimpleNamespace(data_root_directory=tmp_path)
    with patch("cognee.api.v1.ontologies.ontologies.get_base_config", return_value=fake_config):
        yield OntologyService()


def _store(ontology_service, tmp_path, user_id: str, key: str, content: bytes) -> None:
    user_dir = tmp_path / user_id
    user_dir.mkdir(exist_ok=True)
    ontology_service._get_ontology_path(user_dir, key).write_bytes(content)
    metadata_path = user_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata[key] = {"filename": "f.owl", "size_bytes": len(content), "uploaded_at": "2024-01-01"}
    metadata_path.write_text(json.dumps(metadata))


def _upload(client, key: str, content: bytes = OWL_FIXTURE) -> None:
    response = client.post(
        "/api/v1/ontologies",
        files=[("ontology_file", ("test.owl", content, "application/xml"))],
        data={"ontology_key": key},
    )
    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Unit tests for OntologyService.load_ontology_graph / export_ontology
# ---------------------------------------------------------------------------


class TestOntologyServiceExport:
    def test_load_ontology_graph_parses_rdf_xml(self, ontology_service, tmp_path):
        user = SimpleNamespace(id="user-1")
        _store(ontology_service, tmp_path, "user-1", "med", OWL_FIXTURE)

        graph = ontology_service.load_ontology_graph("med", user)

        assert len(graph) == EXPECTED_TRIPLES

    def test_load_ontology_graph_guesses_turtle_in_owl_file(self, ontology_service, tmp_path):
        """A .owl upload that is really Turtle is still loadable."""
        user = SimpleNamespace(id="user-1")
        turtle = Graph().parse(data=OWL_FIXTURE, format="xml").serialize(format="turtle")
        _store(ontology_service, tmp_path, "user-1", "ttl", turtle.encode("utf-8"))

        graph = ontology_service.load_ontology_graph("ttl", user)

        assert len(graph) == EXPECTED_TRIPLES

    def test_load_ontology_graph_raises_for_unknown_key(self, ontology_service, tmp_path):
        user = SimpleNamespace(id="user-1")
        (tmp_path / "user-1").mkdir()
        (tmp_path / "user-1" / "metadata.json").write_text("{}")

        with pytest.raises(ValueError, match="not found"):
            ontology_service.load_ontology_graph("missing", user)

    def test_load_ontology_graph_is_scoped_to_user(self, ontology_service, tmp_path):
        _store(ontology_service, tmp_path, "owner", "shared_key", OWL_FIXTURE)

        with pytest.raises(ValueError, match="not found"):
            ontology_service.load_ontology_graph("shared_key", SimpleNamespace(id="intruder"))

    def test_load_ontology_graph_raises_parse_error_for_garbage(self, ontology_service, tmp_path):
        user = SimpleNamespace(id="user-1")
        _store(ontology_service, tmp_path, "user-1", "junk", b"\x00\x01 definitely not rdf <<<")

        with pytest.raises(OntologyParseError):
            ontology_service.load_ontology_graph("junk", user)

    @pytest.mark.parametrize("export_format", sorted(EXPORT_FORMATS))
    def test_export_ontology_round_trips(self, ontology_service, tmp_path, export_format):
        user = SimpleNamespace(id="user-1")
        _store(ontology_service, tmp_path, "user-1", "med", OWL_FIXTURE)

        payload = ontology_service.export_ontology("med", user, export_format=export_format)

        assert isinstance(payload, bytes)
        rdflib_format = EXPORT_FORMATS[export_format][0]
        assert len(Graph().parse(data=payload, format=rdflib_format)) == EXPECTED_TRIPLES

    def test_export_ontology_rejects_unknown_format(self, ontology_service, tmp_path):
        user = SimpleNamespace(id="user-1")
        _store(ontology_service, tmp_path, "user-1", "med", OWL_FIXTURE)

        with pytest.raises(KeyError):
            ontology_service.export_ontology("med", user, export_format="csv")


# ---------------------------------------------------------------------------
# API endpoint tests for GET /api/v1/ontologies/{key}/export
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("export_format", "media_type", "extension"),
    [
        pytest.param(name, media_type, ext, id=name)
        for name, (_, media_type, ext) in EXPORT_FORMATS.items()
    ],
)
def test_export_ontology_endpoint_each_format(client, export_format, media_type, extension):
    """Upload once, export in a given format, parse it back: same triple count."""
    key = f"test_export_{uuid.uuid4().hex[:8]}"
    _upload(client, key)
    try:
        response = client.get(f"/api/v1/ontologies/{key}/export", params={"format": export_format})

        assert response.status_code == 200, response.text
        assert response.headers["content-type"].split(";")[0] == media_type
        assert response.headers["content-disposition"] == (
            f'attachment; filename="{key}.{extension}"'
        )
        rdflib_format = EXPORT_FORMATS[export_format][0]
        assert len(Graph().parse(data=response.content, format=rdflib_format)) == EXPECTED_TRIPLES
    finally:
        client.delete(f"/api/v1/ontologies/{key}")


def test_export_ontology_endpoint_defaults_to_turtle(client):
    key = f"test_export_default_{uuid.uuid4().hex[:8]}"
    _upload(client, key)
    try:
        response = client.get(f"/api/v1/ontologies/{key}/export")

        assert response.status_code == 200, response.text
        assert response.headers["content-type"].split(";")[0] == "text/turtle"
        assert response.headers["content-disposition"] == f'attachment; filename="{key}.ttl"'
        assert len(Graph().parse(data=response.content, format="turtle")) == EXPECTED_TRIPLES
    finally:
        client.delete(f"/api/v1/ontologies/{key}")


def test_export_ontology_endpoint_rejects_unknown_format(client):
    key = f"test_export_badfmt_{uuid.uuid4().hex[:8]}"
    _upload(client, key)
    try:
        response = client.get(f"/api/v1/ontologies/{key}/export", params={"format": "csv"})

        assert response.status_code == 400
        error = response.json()["error"]
        for option in EXPORT_FORMATS:
            assert option in error
    finally:
        client.delete(f"/api/v1/ontologies/{key}")


def test_export_ontology_endpoint_unknown_key_returns_404(client):
    response = client.get(f"/api/v1/ontologies/nonexistent_{uuid.uuid4().hex[:8]}/export")

    assert response.status_code == 404
    assert response.json() == {"error": "Ontology key not found."}


def test_export_ontology_endpoint_invalid_key_returns_404(client):
    """A syntactically invalid key is reported the same way as a missing one."""
    response = client.get("/api/v1/ontologies/bad%20key%21/export")

    assert response.status_code == 404
    assert response.json() == {"error": "Ontology key not found."}


def test_export_ontology_endpoint_other_users_key_returns_404(test_client):
    """Another user's ontology is indistinguishable from a missing one (404, not 403)."""
    owner, intruder = _make_user(), _make_user()
    key = f"test_export_scope_{uuid.uuid4().hex[:8]}"

    async def as_owner():
        return owner

    async def as_intruder():
        return intruder

    app.dependency_overrides[get_authenticated_user] = as_owner
    try:
        _upload(test_client, key)
        assert test_client.get(f"/api/v1/ontologies/{key}/export").status_code == 200

        app.dependency_overrides[get_authenticated_user] = as_intruder
        response = test_client.get(f"/api/v1/ontologies/{key}/export")
        assert response.status_code == 404
        assert response.json() == {"error": "Ontology key not found."}
    finally:
        app.dependency_overrides[get_authenticated_user] = as_owner
        test_client.delete(f"/api/v1/ontologies/{key}")
        app.dependency_overrides.pop(get_authenticated_user, None)


def test_export_ontology_endpoint_unparseable_file_returns_422(client):
    key = f"test_export_junk_{uuid.uuid4().hex[:8]}"
    _upload(client, key, content=b"\x00\x01 definitely not rdf <<<")
    try:
        response = client.get(f"/api/v1/ontologies/{key}/export")

        assert response.status_code == 422
        assert response.json() == {"error": "Stored ontology file could not be parsed as RDF."}
    finally:
        client.delete(f"/api/v1/ontologies/{key}")


# ---------------------------------------------------------------------------
# JSON-LD contexts: never fetched (SSRF / local file read)
# ---------------------------------------------------------------------------

REMOTE_CONTEXT = b'{"@context": {"label": "http://www.w3.org/2000/01/rdf-schema#label"}}'


def _jsonld(document: dict) -> bytes:
    return json.dumps(document).encode()


def _remote_context_documents(url: str) -> list[bytes]:
    node = {"@id": "http://example.org/a", "label": "A"}
    return [
        _jsonld({"@context": f"{url}/context.jsonld", **node}),
        _jsonld({"@context": [f"{url}/context.jsonld", {"ex": "http://example.org/"}], **node}),
        _jsonld({"@graph": [{"@context": f"{url}/context.jsonld", **node}]}),
        # A scoped context inside a term definition.
        _jsonld(
            {
                "@context": {"knows": {"@id": "http://x/knows", "@context": f"{url}/scoped"}},
                **node,
            }
        ),
        _jsonld({"@context": {"@version": 1.1, "@import": f"{url}/imported"}, **node}),
    ]


def test_export_refuses_jsonld_that_references_an_external_context(client):
    """Each document would make the server fetch the URL it names while parsing."""
    with recording_http_server(REMOTE_CONTEXT, "application/ld+json") as (url, requests):
        for index, document in enumerate(_remote_context_documents(url)):
            key = f"test_export_remote_ctx_{index}_{uuid.uuid4().hex[:8]}"
            _upload(client, key, content=document)
            try:
                response = client.get(f"/api/v1/ontologies/{key}/export")

                assert response.status_code == 422, (index, response.text)
            finally:
                client.delete(f"/api/v1/ontologies/{key}")

    assert requests == []


def test_export_refuses_jsonld_that_references_a_local_file(client, tmp_path):
    """A path or file: URI as @context would read a file from the server's disk."""
    context_file = tmp_path / "context.jsonld"
    context_file.write_bytes(REMOTE_CONTEXT)
    node = {"@id": "http://example.org/a", "label": "A"}

    for index, reference in enumerate([context_file.as_uri(), str(context_file)]):
        key = f"test_export_file_ctx_{index}_{uuid.uuid4().hex[:8]}"
        _upload(client, key, content=_jsonld({"@context": reference, **node}))
        try:
            response = client.get(f"/api/v1/ontologies/{key}/export")

            assert response.status_code == 422, (reference, response.text)
        finally:
            client.delete(f"/api/v1/ontologies/{key}")


def test_export_still_reads_jsonld_with_an_inline_context(client):
    key = f"test_export_inline_ctx_{uuid.uuid4().hex[:8]}"
    document = _jsonld(
        {
            "@context": {"label": "http://www.w3.org/2000/01/rdf-schema#label"},
            "@id": "http://example.org/a",
            "label": "A",
        }
    )
    _upload(client, key, content=document)
    try:
        response = client.get(f"/api/v1/ontologies/{key}/export")

        assert response.status_code == 200, response.text
        assert len(Graph().parse(data=response.content, format="turtle")) == 1
    finally:
        client.delete(f"/api/v1/ontologies/{key}")


def test_export_does_not_resolve_xml_external_entities(client):
    """RDF/XML is parsed with external entities off; pins that against a parser change."""
    with recording_http_server(b"<x/>", "application/xml") as (url, requests):
        document = f"""<?xml version="1.0"?>
<!DOCTYPE rdf:RDF [<!ENTITY ext SYSTEM "{url}/entity">]>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
  <rdf:Description rdf:about="http://example.org/a"><rdfs:label>&ext;</rdfs:label></rdf:Description>
</rdf:RDF>""".encode()
        key = f"test_export_xxe_{uuid.uuid4().hex[:8]}"
        _upload(client, key, content=document)
        try:
            client.get(f"/api/v1/ontologies/{key}/export")
        finally:
            client.delete(f"/api/v1/ontologies/{key}")

    assert requests == []


# ---------------------------------------------------------------------------
# Serialization failures are not "not found"
# ---------------------------------------------------------------------------

# rdflib's RDF/XML writer cannot shorten a predicate IRI ending in digits to a QName.
DIGIT_PREDICATE_FIXTURE = (
    b"<http://example.org/a> <http://example.org/123> <http://example.org/b> ."
)


def test_export_ontology_raises_serialization_error_for_inexpressible_format(
    ontology_service, tmp_path
):
    user = SimpleNamespace(id="user-1")
    _store(ontology_service, tmp_path, "user-1", "digits", DIGIT_PREDICATE_FIXTURE)

    with pytest.raises(OntologySerializationError, match="cannot be serialized as xml"):
        ontology_service.export_ontology("digits", user, export_format="xml")
    assert ontology_service.export_ontology("digits", user, export_format="turtle")


def test_load_ontology_graph_raises_not_found_for_unknown_and_invalid_keys(
    ontology_service, tmp_path
):
    user = SimpleNamespace(id="user-1")

    with pytest.raises(OntologyNotFoundError):
        ontology_service.load_ontology_graph("missing", user)
    with pytest.raises(OntologyNotFoundError):
        ontology_service.load_ontology_graph("../escape", user)


def test_export_ontology_endpoint_inexpressible_format_is_422_not_404(client):
    key = f"test_export_digits_{uuid.uuid4().hex[:8]}"
    _upload(client, key, content=DIGIT_PREDICATE_FIXTURE)
    try:
        as_xml = client.get(f"/api/v1/ontologies/{key}/export", params={"format": "xml"})
        as_turtle = client.get(f"/api/v1/ontologies/{key}/export", params={"format": "turtle"})

        assert as_xml.status_code == 422, as_xml.text
        assert as_xml.json() == {
            "error": "This ontology cannot be exported as xml. "
            "Try format=turtle, which can represent any RDF graph."
        }
        assert as_turtle.status_code == 200
    finally:
        client.delete(f"/api/v1/ontologies/{key}")
