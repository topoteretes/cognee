"""POST /api/v1/ontologies accepts every RDF serialization rdflib parses (SDK-1246).

The format is taken from the filename extension, or from the multipart part's
Content-Type when the extension is unknown. Uploads matching neither are
rejected with a 400 that lists the accepted formats. The stored upload keeps
its filename and content type so ``RDFLibOntologyResolver`` parses it with the
right parser rather than trial-and-error.
"""

import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from rdflib import Graph

from cognee.api.v1.ontologies.ontologies import OntologyFile, OntologyService
from cognee.api.v1.ontologies.routers.get_ontology_router import get_ontology_router
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.modules.users.methods import get_authenticated_user

_ROUTER_MODULE = "cognee.api.v1.ontologies.routers.get_ontology_router"

OWL_FIXTURE = b"""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:ex="http://example.org/vehicles#">
  <owl:Class rdf:about="http://example.org/vehicles#Vehicle"/>
  <owl:Class rdf:about="http://example.org/vehicles#Car">
    <rdfs:subClassOf rdf:resource="http://example.org/vehicles#Vehicle"/>
  </owl:Class>
  <owl:NamedIndividual rdf:about="http://example.org/vehicles#Tesla_Model_3">
    <rdf:type rdf:resource="http://example.org/vehicles#Car"/>
  </owl:NamedIndividual>
</rdf:RDF>
"""


def _serialize(rdf_format: str) -> bytes:
    """Convert the OWL fixture into another rdflib serialization."""
    graph = Graph()
    graph.parse(data=OWL_FIXTURE, format="xml")
    serialized = graph.serialize(format=rdf_format)
    return serialized.encode("utf-8") if isinstance(serialized, str) else serialized


@pytest.fixture
def user():
    return SimpleNamespace(id=str(uuid.uuid4()))


@pytest.fixture
def client(monkeypatch, tmp_path, user):
    # Keep uploads out of the real data root and skip telemetry.
    monkeypatch.setattr(OntologyService, "base_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(f"{_ROUTER_MODULE}.send_telemetry", lambda *args, **kwargs: None)

    app = FastAPI()
    app.include_router(get_ontology_router(), prefix="/api/v1/ontologies")

    async def override_user():
        return user

    app.dependency_overrides[get_authenticated_user] = override_user
    return TestClient(app)


def _upload(client, filename, content, content_type=None, key=None):
    key = key or f"onto_{uuid.uuid4().hex[:8]}"
    file_part = (filename, content, content_type) if content_type else (filename, content)
    response = client.post(
        "/api/v1/ontologies",
        files=[("ontology_file", file_part)],
        data={"ontology_key": key},
    )
    return key, response


def test_turtle_upload_is_accepted(client, user):
    key, response = _upload(client, "vehicles.ttl", _serialize("turtle"), "text/turtle")

    assert response.status_code == 200, response.text
    assert response.json()["uploaded_ontologies"][0]["filename"] == "vehicles.ttl"
    assert OntologyService().list_ontologies(user)[key]["format"] == "turtle"


def test_jsonld_upload_is_accepted(client, user):
    key, response = _upload(client, "vehicles.jsonld", _serialize("json-ld"), "application/ld+json")

    assert response.status_code == 200, response.text
    assert OntologyService().list_ontologies(user)[key]["format"] == "json-ld"


@pytest.mark.parametrize(
    ("filename", "expected_format"),
    [("vehicles.owl", "xml"), ("vehicles.rdf", "xml"), ("vehicles.xml", "xml")],
)
def test_rdf_xml_extensions_are_accepted_without_content_type(
    client, user, filename, expected_format
):
    key, response = _upload(client, filename, OWL_FIXTURE)

    assert response.status_code == 200, response.text
    assert OntologyService().list_ontologies(user)[key]["format"] == expected_format


@pytest.mark.parametrize(
    ("rdf_format", "content_type"),
    [("n3", "text/n3"), ("nt", "application/n-triples")],
)
def test_n3_and_ntriples_uploads_are_accepted(client, rdf_format, content_type):
    _, response = _upload(client, f"vehicles.{rdf_format}", _serialize(rdf_format), content_type)

    assert response.status_code == 200, response.text


def test_content_type_only_upload_is_accepted(client, user):
    """A filename without an extension is admitted on the strength of its Content-Type."""
    key, response = _upload(client, "vehicles", _serialize("turtle"), "text/turtle")

    assert response.status_code == 200, response.text
    stored = OntologyService().list_ontologies(user)[key]
    assert stored["content_type"] == "text/turtle"
    assert stored["format"] == "turtle"


def test_content_type_parameters_are_ignored(client, user):
    key, response = _upload(client, "vehicles", _serialize("turtle"), "text/turtle; charset=utf-8")

    assert response.status_code == 200, response.text
    assert OntologyService().list_ontologies(user)[key]["content_type"] == "text/turtle"


def test_unknown_extension_and_content_type_are_rejected_with_format_list(client, user):
    key, response = _upload(client, "vehicles.txt", b"Vehicle, Car", "text/plain")

    assert response.status_code == 400
    error = response.json()["error"]
    assert error.startswith("Unsupported ontology format.")
    for accepted in (".owl", ".rdf", ".xml", ".ttl", ".n3", ".jsonld", ".nt"):
        assert accepted in error
    for accepted in (
        "application/rdf+xml",
        "text/turtle",
        "text/n3",
        "application/ld+json",
        "application/n-triples",
    ):
        assert accepted in error
    # Nothing from the request is echoed back.
    assert "vehicles.txt" not in error
    assert "text/plain" not in error
    assert key not in OntologyService().list_ontologies(user)


def test_unknown_extension_without_content_type_is_rejected(client):
    _, response = _upload(client, "vehicles.txt", b"Vehicle, Car")

    assert response.status_code == 400
    assert "Unsupported ontology format" in response.json()["error"]


def test_stored_file_carries_filename_and_content_type_to_resolver(client, user):
    """The content type must reach ``_get_content_type_format`` for extension-less uploads."""
    key, response = _upload(client, "vehicles", _serialize("turtle"), "text/turtle")
    assert response.status_code == 200, response.text

    (ontology_file,) = OntologyService().get_ontology_files([key], user)
    resolver = RDFLibOntologyResolver(ontology_file=None)

    assert isinstance(ontology_file, OntologyFile)
    assert ontology_file.filename == "vehicles"
    assert resolver._get_content_type_format(ontology_file) == "turtle"
    # No extension to guess from, so the content type is the first parser tried.
    assert resolver._get_candidate_formats(ontology_file)[0] == "turtle"


def test_turtle_upload_resolves_same_classes_as_owl(client, user):
    """A Turtle upload yields the same lookup as the equivalent RDF/XML upload."""
    owl_key, owl_response = _upload(client, "vehicles.owl", OWL_FIXTURE, "application/rdf+xml")
    ttl_key, ttl_response = _upload(client, "vehicles.ttl", _serialize("turtle"), "text/turtle")
    assert owl_response.status_code == 200, owl_response.text
    assert ttl_response.status_code == 200, ttl_response.text

    service = OntologyService()
    owl_resolver = RDFLibOntologyResolver(ontology_file=service.get_ontology_files([owl_key], user))
    ttl_resolver = RDFLibOntologyResolver(ontology_file=service.get_ontology_files([ttl_key], user))

    assert set(owl_resolver.lookup["classes"]) == {"vehicle", "car"}
    assert ttl_resolver.lookup["classes"] == owl_resolver.lookup["classes"]
    assert ttl_resolver.lookup["individuals"] == owl_resolver.lookup["individuals"]

    _, owl_edges, _ = owl_resolver.get_subgraph("Tesla Model 3")
    _, ttl_edges, _ = ttl_resolver.get_subgraph("Tesla Model 3")
    assert sorted(ttl_edges) == sorted(owl_edges)
    assert ("tesla_model_3", "is_a", "car") in ttl_edges
