"""Ontology requests must not block the event loop.

The store reads and writes through cognee's async file storage, so the routes
await it directly. The steps that still block a thread are a file read (a
network request on S3) and rdflib's parse and serialize; the store runs those
in worker threads.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from cognee.api.v1.ontologies import ontologies as ontologies_module
from cognee.api.v1.ontologies.ontologies import OntologyService
from cognee.api.v1.ontologies.routers.get_ontology_router import get_ontology_router

_MODULE = "cognee.api.v1.ontologies.routers.get_ontology_router"

TURTLE = b"<http://example.org/a> <http://example.org/knows> <http://example.org/b> ."


def _endpoint_for(method: str):
    router = get_ontology_router()
    for route in router.routes:
        if method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"no {method} route found on the ontology router")


@pytest.mark.asyncio
async def test_delete_ontology_route_awaits_the_store():
    delete_route = _endpoint_for("DELETE")
    user = Mock()
    user.id = "u1"

    with (
        patch.object(OntologyService, "delete_ontology", new_callable=AsyncMock) as mock_delete,
        patch(f"{_MODULE}.send_telemetry"),
    ):
        result = await delete_route(ontology_key="k", user=user)

    mock_delete.assert_awaited_once_with(ontology_key="k", user=user)
    assert result == {"status": "success", "ontology_key": "k"}


@pytest.mark.asyncio
async def test_list_ontologies_route_awaits_the_store():
    list_route = _endpoint_for("GET")
    user = Mock()
    user.id = "u1"
    sentinel = {"k": {"filename": "k.owl"}}

    with (
        patch.object(
            OntologyService, "list_ontologies", new_callable=AsyncMock, return_value=sentinel
        ) as mock_list,
        patch(f"{_MODULE}.send_telemetry"),
    ):
        result = await list_route(user=user)

    mock_list.assert_awaited_once_with(user)
    assert result == sentinel


@pytest.mark.asyncio
async def test_export_ontology_route_awaits_the_store():
    router = get_ontology_router()
    export_route = next(r.endpoint for r in router.routes if r.path.endswith("/export"))
    user = Mock()
    user.id = "u1"

    with (
        patch.object(
            OntologyService,
            "export_ontology",
            new_callable=AsyncMock,
            return_value=b"@prefix : <#> .",
        ) as mock_export,
        patch(f"{_MODULE}.send_telemetry"),
    ):
        result = await export_route(ontology_key="k", format="turtle", user=user)

    mock_export.assert_awaited_once_with(ontology_key="k", user=user, export_format="turtle")
    assert result.media_type == "text/turtle"
    assert result.headers["content-disposition"] == 'attachment; filename="k.ttl"'


@pytest.mark.asyncio
async def test_store_reads_parses_and_serializes_in_worker_threads(tmp_path):
    """The blocking steps of an export run off the event loop's thread."""
    user = SimpleNamespace(id="user-1")
    event_loop_thread = threading.current_thread()
    seen_threads: dict[str, threading.Thread] = {}

    real_parse = ontologies_module.parse_rdf_content

    def recording_parse(content, formats):
        seen_threads["parse"] = threading.current_thread()
        graph, parsed_format = real_parse(content, formats)
        real_serialize = graph.serialize

        def recording_serialize(*args, **kwargs):
            seen_threads["serialize"] = threading.current_thread()
            return real_serialize(*args, **kwargs)

        graph.serialize = recording_serialize
        return graph, parsed_format

    real_to_thread = asyncio.to_thread
    offloaded = []

    async def recording_to_thread(function, *args, **kwargs):
        offloaded.append(getattr(function, "__name__", repr(function)))
        return await real_to_thread(function, *args, **kwargs)

    fake_config = SimpleNamespace(data_root_directory=tmp_path)
    with (
        patch.object(ontologies_module, "get_base_config", return_value=fake_config),
        patch.object(ontologies_module, "parse_rdf_content", recording_parse),
        patch.object(ontologies_module.asyncio, "to_thread", recording_to_thread),
    ):
        service = OntologyService()
        upload = SimpleNamespace(
            filename="graph.nt", content_type="application/n-triples", read=AsyncMock()
        )
        upload.read.return_value = TURTLE
        await service.upload_ontology("graph", upload, user)
        offloaded.clear()

        payload = await service.export_ontology("graph", user, export_format="turtle")

    assert b"knows" in payload
    assert seen_threads["parse"] is not event_loop_thread
    assert seen_threads["serialize"] is not event_loop_thread
    # The index and the ontology file are each read through a worker thread too.
    assert offloaded.count("read") == 2
