"""parse_rdf_content: format fallback, and JSON-LD never fetching an external context.

The resolver used by cognify / remember parses uploaded ontologies through it, so the
same documents that the export endpoint refuses must not reach the network here either.
"""

import json
from io import StringIO

import pytest
from rdflib import Graph

from cognee.modules.ontology.exceptions import OntologyInitializationError
from cognee.modules.ontology.rdf_xml.parse_rdf_content import (
    FALLBACK_FORMATS,
    parse_rdf_content,
    parse_rdf_file,
    refuse_remote_jsonld_contexts,
)
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.tasks.graph.gliner_demo.schema import schema_from_ontology
from cognee.tests.utils.recording_http_server import recording_http_server

LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
CONTEXT_BODY = json.dumps({"@context": {"label": LABEL}}).encode()
NODE = {"@id": "http://example.org/a", "label": "A"}


def _remote(url: str) -> str:
    return json.dumps({"@context": f"{url}/context.jsonld", **NODE})


def test_parse_rdf_content_returns_the_format_that_parsed():
    graph, rdf_format = parse_rdf_content(
        "<http://example.org/a> <http://example.org/p> <http://example.org/b> .",
        FALLBACK_FORMATS,
    )

    assert rdf_format == "turtle"
    assert len(graph) == 1


def test_parse_rdf_content_lists_every_format_tried_when_none_parses():
    with pytest.raises(ValueError, match="Tried formats: xml: .*; json-ld: "):
        parse_rdf_content("not rdf <<<", ("xml", "json-ld"))


def test_inline_jsonld_context_parses():
    graph, rdf_format = parse_rdf_content(
        json.dumps({"@context": {"label": LABEL}, **NODE}), ("json-ld",)
    )

    assert rdf_format == "json-ld"
    assert len(graph) == 1


@pytest.mark.parametrize("rdf_format", ["json-ld", "application/ld+json"])
def test_remote_jsonld_context_is_refused_without_a_request(rdf_format):
    """Every name rdflib maps to its JSON-LD parser is guarded, not just ``json-ld``."""
    with (
        recording_http_server(CONTEXT_BODY, "application/ld+json") as (url, requests),
        pytest.raises(ValueError, match="references an external document"),
    ):
        parse_rdf_content(_remote(url), (rdf_format,))

    assert requests == []


@pytest.mark.parametrize(
    "document",
    [
        {"@context": "http://x/ctx", **NODE},
        {"@context": ["http://x/ctx", {"ex": "http://example.org/"}], **NODE},
        {"@context": "other.jsonld", **NODE},
        {"@context": "file:///etc/ctx.jsonld", **NODE},
        {"@graph": [{"@context": "http://x/ctx", **NODE}]},
        {"@context": {"knows": {"@id": "http://x/knows", "@context": "http://x/scoped"}}},
        {"@context": {"@import": "http://x/imported"}, **NODE},
    ],
)
def test_refuse_remote_jsonld_contexts_finds_every_reference(document):
    with pytest.raises(ValueError, match="references an external document"):
        refuse_remote_jsonld_contexts(json.dumps(document))


@pytest.mark.parametrize(
    "document",
    [
        {"@context": {"label": LABEL}, **NODE},
        {"@context": [{"label": LABEL}, None], **NODE},
        # A string value under an ordinary key is data, not a context reference.
        {"@context": {"label": LABEL}, "@id": "http://example.org/a", "label": "http://x/ctx"},
    ],
)
def test_refuse_remote_jsonld_contexts_allows_inline_contexts(document):
    refuse_remote_jsonld_contexts(json.dumps(document))


def test_refuse_remote_jsonld_contexts_refuses_content_it_cannot_inspect():
    """Not JSON means not inspected: the JSON-LD attempt is refused, never let through."""
    with pytest.raises(ValueError):
        refuse_remote_jsonld_contexts("@prefix ex: <http://example.org/> .")


def test_resolver_skips_an_uploaded_ontology_with_a_remote_context():
    """The cognify / remember path: uploaded ontologies arrive as file objects."""
    with (
        recording_http_server(CONTEXT_BODY, "application/ld+json") as (url, requests),
        pytest.raises(OntologyInitializationError, match="No valid ontology file objects"),
    ):
        RDFLibOntologyResolver(ontology_file=[StringIO(_remote(url))])

    assert requests == []


def test_resolver_keeps_the_safe_file_when_one_upload_is_refused():
    with recording_http_server(CONTEXT_BODY, "application/ld+json") as (url, requests):
        resolver = RDFLibOntologyResolver(
            ontology_file=[
                StringIO(_remote(url)),
                StringIO(json.dumps({"@context": {"label": LABEL}, **NODE})),
            ]
        )

    assert requests == []
    assert len(resolver.graph) == 1


@pytest.mark.parametrize("suffix", [".jsonld", ".json"])
def test_resolver_refuses_an_ontology_file_path_with_a_remote_context(tmp_path, suffix):
    """Paths from ONTOLOGY_FILE_PATH / SDK arguments get the same check as uploads."""
    with recording_http_server(CONTEXT_BODY, "application/ld+json") as (url, requests):
        ontology_file = tmp_path / f"ontology{suffix}"
        ontology_file.write_text(_remote(url))

        with pytest.raises(OntologyInitializationError, match="references an external document"):
            RDFLibOntologyResolver(ontology_file=str(ontology_file))

    assert requests == []


def test_resolver_loads_an_ontology_file_path_with_an_inline_context(tmp_path):
    ontology_file = tmp_path / "ontology.jsonld"
    ontology_file.write_text(json.dumps({"@context": {"label": LABEL}, **NODE}))

    resolver = RDFLibOntologyResolver(ontology_file=str(ontology_file))

    assert len(resolver.graph) == 1


def test_parse_rdf_file_keeps_the_file_location_as_base_iri(tmp_path):
    """Relative IRIs resolve against the file, exactly as ``Graph.parse(path)`` did."""
    ontology_file = tmp_path / "ontology.ttl"
    ontology_file.write_text("<#a> <http://example.org/p> <#b> .")

    graph = Graph()
    parse_rdf_file(graph, str(ontology_file))

    expected = Graph()
    expected.parse(str(ontology_file))
    assert set(graph) == set(expected)
    assert str(next(iter(graph))[0]).startswith("file://")


def test_gliner_schema_refuses_an_ontology_file_with_a_remote_context(tmp_path):
    """The keyless extractor reads ONTOLOGY_FILE_PATH itself; it gets the same check."""
    with recording_http_server(CONTEXT_BODY, "application/ld+json") as (url, requests):
        ontology_file = tmp_path / "ontology.jsonld"
        ontology_file.write_text(_remote(url))

        with pytest.raises(ValueError, match="references an external document"):
            schema_from_ontology(str(ontology_file))

    assert requests == []
