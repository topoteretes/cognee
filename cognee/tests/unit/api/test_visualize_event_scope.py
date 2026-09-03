"""fetch_visualization_data: which dataset the Memory-tab events are scoped to.

The predicate itself is covered in
cognee/tests/unit/modules/visualization/test_session_event_scoping.py. What
this pins is the wiring on the /visualize/json and HTML-render side (COG-6121):
that the authorized dataset reaches the collector, and that a *failed* scope
check collects nothing rather than everything.
"""

import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest

from cognee.api.v1.visualize.visualize import fetch_visualization_data  # noqa: F401

visualize_module = sys.modules["cognee.api.v1.visualize.visualize"]

DATASET_ID = UUID("aaaaaaaa-1111-4111-8111-1111111111ab")
USER = SimpleNamespace(id=UUID("cccccccc-3333-4333-8333-3333333333ef"))


@asynccontextmanager
async def _noop_db_context(*_args, **_kwargs):
    yield


def _patches(collect, *, authorized):
    return (
        patch.object(
            visualize_module,
            "get_authorized_existing_datasets",
            AsyncMock(return_value=[SimpleNamespace(id=DATASET_ID)] if authorized else []),
        ),
        patch.object(
            visualize_module, "fetch_dataset_graph_data", AsyncMock(return_value=([], []))
        ),
        patch.object(visualize_module, "set_database_global_context_variables", _noop_db_context),
        patch.object(visualize_module, "collect_session_events", collect),
    )


async def _fetch(collect, *, dataset, authorized=True):
    a, b, c, d = _patches(collect, authorized=authorized)
    with a, b, c, d:
        return await visualize_module.fetch_visualization_data(user=USER, dataset=dataset)


@pytest.mark.asyncio
async def test_the_authorized_dataset_becomes_the_collection_scope():
    collect = AsyncMock(return_value=[])

    await _fetch(collect, dataset="some-dataset")

    collect.assert_awaited_once_with(user=USER, session_ids=None, dataset_id=DATASET_ID)


@pytest.mark.asyncio
async def test_a_named_but_unauthorized_dataset_collects_nothing():
    """A failed scope check must scope to nothing, not fall back to unscoped.

    get_authorized_existing_datasets reports "missing or unreadable" by
    returning [], so the collector must not be reached at all here.
    """
    collect = AsyncMock(return_value=[])

    _graph_data, search_events = await _fetch(collect, dataset="denied", authorized=False)

    collect.assert_not_awaited()
    assert search_events == []


@pytest.mark.asyncio
async def test_no_dataset_asked_for_stays_unscoped():
    """Nothing to scope to: a datasetless render can only show everything."""
    collect = AsyncMock(return_value=[])

    await _fetch(collect, dataset=None)

    collect.assert_awaited_once_with(user=USER, session_ids=None, dataset_id=None)
