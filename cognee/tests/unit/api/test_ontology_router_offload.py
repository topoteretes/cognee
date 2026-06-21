"""Ontology routes must not block the event loop on synchronous file IO.

`OntologyService.delete_ontology` / `list_ontologies` do blocking filesystem IO
(stat / unlink / metadata read). The async routes must dispatch them via
`asyncio.to_thread` rather than calling them inline on the event loop.
"""

import asyncio
from unittest.mock import Mock, patch

import pytest

from cognee.api.v1.ontologies.ontologies import OntologyService
from cognee.api.v1.ontologies.routers.get_ontology_router import get_ontology_router

_MODULE = "cognee.api.v1.ontologies.routers.get_ontology_router"


def _endpoint_for(method: str):
    router = get_ontology_router()
    for route in router.routes:
        if method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"no {method} route found on the ontology router")


@pytest.mark.asyncio
async def test_delete_ontology_route_offloads_blocking_io():
    delete_route = _endpoint_for("DELETE")
    user = Mock()
    user.id = "u1"

    with (
        patch.object(OntologyService, "delete_ontology", return_value=None) as mock_delete,
        patch(f"{_MODULE}.send_telemetry"),
        patch(f"{_MODULE}.asyncio.to_thread", wraps=asyncio.to_thread) as to_thread_spy,
    ):
        result = await delete_route(ontology_key="k", user=user)

    assert to_thread_spy.called, "delete route did not offload via asyncio.to_thread"
    mock_delete.assert_called_once()
    assert result == {"status": "success", "ontology_key": "k"}


@pytest.mark.asyncio
async def test_list_ontologies_route_offloads_blocking_io():
    list_route = _endpoint_for("GET")
    user = Mock()
    user.id = "u1"
    sentinel = {"k": {"filename": "k.owl"}}

    with (
        patch.object(OntologyService, "list_ontologies", return_value=sentinel) as mock_list,
        patch(f"{_MODULE}.send_telemetry"),
        patch(f"{_MODULE}.asyncio.to_thread", wraps=asyncio.to_thread) as to_thread_spy,
    ):
        result = await list_route(user=user)

    assert to_thread_spy.called, "list route did not offload via asyncio.to_thread"
    mock_list.assert_called_once()
    assert result == sentinel
