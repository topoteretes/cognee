"""fetch_visualization_data: which dataset the Memory-tab events are scoped to.

The predicate itself is covered in
cognee/tests/unit/modules/visualization/test_session_event_scoping.py. What
this pins is the wiring on the /visualize/json and HTML-render side (COG-6121):
that the authorized dataset reaches the collector, and that a *failed* scope
check collects nothing rather than everything.

Also covers fetch_visualization_data_for_dataset, the authorization-free half
fetch_visualization_data delegates to once it has resolved a dataset
(SDK-972), and confirms visualize_graph_json still authorizes on its own now
that the router calls fetch_visualization_data_for_dataset directly instead.
"""

import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest

from cognee.api.v1.visualize.visualize import fetch_visualization_data

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


# fetch_visualization_data_for_dataset: the authorization-free half of
# fetch_visualization_data (SDK-972). The /visualize/json router calls this
# directly with a Dataset it already authorized itself, so this function must
# never re-authorize: it is given the resolved dataset, not a name or id.


async def _fetch_for_dataset(collect, *, dataset, include_session_events=True):
    with (
        patch.object(
            visualize_module, "fetch_dataset_graph_data", AsyncMock(return_value=([], []))
        ),
        patch.object(visualize_module, "set_database_global_context_variables", _noop_db_context),
        patch.object(visualize_module, "collect_session_events", collect),
    ):
        return await visualize_module.fetch_visualization_data_for_dataset(
            dataset, USER, include_session_events=include_session_events
        )


@pytest.mark.asyncio
async def test_fetch_for_dataset_scopes_events_to_the_given_dataset():
    collect = AsyncMock(return_value=[])
    dataset = SimpleNamespace(id=DATASET_ID)

    await _fetch_for_dataset(collect, dataset=dataset)

    collect.assert_awaited_once_with(user=USER, session_ids=None, dataset_id=DATASET_ID)


@pytest.mark.asyncio
async def test_fetch_for_dataset_include_session_events_false_skips_collection():
    collect = AsyncMock(return_value=[])
    dataset = SimpleNamespace(id=DATASET_ID)

    _graph_data, search_events = await _fetch_for_dataset(
        collect, dataset=dataset, include_session_events=False
    )

    collect.assert_not_awaited()
    assert search_events is None


@pytest.mark.asyncio
async def test_fetch_for_dataset_with_none_collects_unscoped():
    """dataset=None here means "render the current context", not "denied".
    fetch_visualization_data handles the denied case itself and never asks
    this function to collect events for None."""
    collect = AsyncMock(return_value=[])

    await _fetch_for_dataset(collect, dataset=None)

    collect.assert_awaited_once_with(user=USER, session_ids=None, dataset_id=None)


# visualize_graph_json: the public SDK function the task requires keeps
# authorizing exactly as before, now that the router calls
# fetch_visualization_data_for_dataset directly instead of this.


@pytest.mark.asyncio
async def test_visualize_graph_json_still_authorizes_the_dataset():
    authorize = AsyncMock(return_value=[SimpleNamespace(id=DATASET_ID)])
    with (
        patch.object(visualize_module, "get_authorized_existing_datasets", authorize),
        patch.object(
            visualize_module, "fetch_dataset_graph_data", AsyncMock(return_value=([], []))
        ),
        patch.object(visualize_module, "set_database_global_context_variables", _noop_db_context),
        patch.object(visualize_module, "collect_session_events", AsyncMock(return_value=[])),
    ):
        payload = await visualize_module.visualize_graph_json(user=USER, dataset="some-dataset")

    authorize.assert_awaited_once_with(["some-dataset"], "read", USER)
    assert payload["search_events"] == []


@pytest.mark.asyncio
async def test_visualize_graph_json_denied_dataset_still_scopes_to_nothing():
    """A caller hitting the public function directly with a denied dataset
    must get the same empty scope fetch_visualization_data already gives,
    not an authorization bypass through the new helper."""
    with (
        patch.object(
            visualize_module, "get_authorized_existing_datasets", AsyncMock(return_value=[])
        ),
        patch.object(
            visualize_module, "fetch_dataset_graph_data", AsyncMock(return_value=([], []))
        ),
        patch.object(visualize_module, "set_database_global_context_variables", _noop_db_context),
        patch.object(
            visualize_module,
            "collect_session_events",
            AsyncMock(side_effect=AssertionError("must not be reached for a denied dataset")),
        ),
    ):
        payload = await visualize_module.visualize_graph_json(user=USER, dataset="denied")

    assert payload["search_events"] == []
