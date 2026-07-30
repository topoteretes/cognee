from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4
import importlib

import pytest

visualize_module = importlib.import_module("cognee.api.v1.visualize.visualize")


class _DummyContext:
    """Stand-in for the DatabaseContextManager returned by
    set_database_global_context_variables (used via `async with`)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _capturing_set_context(calls):
    def fake(*args, **kwargs):
        calls.append(args)
        return _DummyContext()

    return fake


@pytest.mark.asyncio
@pytest.mark.parametrize("single_dataset", ["NLP", uuid4()])
async def test_visualize_graph_wraps_single_dataset_into_list(monkeypatch, single_dataset):
    """A single dataset (name or UUID) must be passed to
    get_authorized_existing_datasets as a one-element list."""
    dataset = SimpleNamespace(id="dataset-id", owner_id="owner-id")
    default_user = object()
    authorize = AsyncMock(return_value=[dataset])
    graph_engine = SimpleNamespace(get_graph_data=AsyncMock(return_value=([], [])))
    set_context_calls = []

    monkeypatch.setattr(visualize_module, "get_default_user", AsyncMock(return_value=default_user))
    monkeypatch.setattr(visualize_module, "get_authorized_existing_datasets", authorize)
    monkeypatch.setattr(
        visualize_module,
        "set_database_global_context_variables",
        _capturing_set_context(set_context_calls),
    )
    monkeypatch.setattr(visualize_module, "get_graph_engine", AsyncMock(return_value=graph_engine))
    monkeypatch.setattr(
        visualize_module, "cognee_network_visualization", AsyncMock(return_value="<html></html>")
    )

    result = await visualize_module.visualize_graph(
        dataset=single_dataset, include_session_events=False
    )

    assert result == "<html></html>"
    # The crux of the feature: the single input is wrapped into [dataset].
    authorize.assert_awaited_once_with([single_dataset], "read", default_user)
    # And the resolved dataset drives the DB context.
    assert set_context_calls == [(dataset.id, dataset.owner_id)]


@pytest.mark.asyncio
async def test_visualize_graph_with_user_and_dataset_provided(monkeypatch):
    """The router path: when both user and dataset are passed, the provided
    user is used directly (get_default_user is NOT called) and the dataset is
    still wrapped into a one-element list."""
    provided_user = object()
    dataset = SimpleNamespace(id="dataset-id", owner_id="owner-id")
    dataset_id = uuid4()
    authorize = AsyncMock(return_value=[dataset])
    graph_engine = SimpleNamespace(get_graph_data=AsyncMock(return_value=([], [])))
    get_default = AsyncMock()
    network_viz = AsyncMock(return_value="<html></html>")
    set_context_calls = []

    monkeypatch.setattr(visualize_module, "get_default_user", get_default)
    monkeypatch.setattr(visualize_module, "get_authorized_existing_datasets", authorize)
    monkeypatch.setattr(
        visualize_module,
        "set_database_global_context_variables",
        _capturing_set_context(set_context_calls),
    )
    monkeypatch.setattr(visualize_module, "get_graph_engine", AsyncMock(return_value=graph_engine))
    monkeypatch.setattr(visualize_module, "cognee_network_visualization", network_viz)

    result = await visualize_module.visualize_graph(
        dataset=dataset_id,
        destination_file_path="/tmp/graph.html",
        user=provided_user,
        include_session_events=False,
    )

    assert result == "<html></html>"
    get_default.assert_not_awaited()
    authorize.assert_awaited_once_with([dataset_id], "read", provided_user)
    assert set_context_calls == [(dataset.id, dataset.owner_id)]
    network_viz.assert_awaited_once_with(([], []), "/tmp/graph.html", search_events=None)


@pytest.mark.asyncio
async def test_visualize_graph_defaults_to_main_dataset(monkeypatch):
    """With no dataset argument, it defaults to "main_dataset" (the same
    default as add/cognify/remember) and authorizes that."""
    dataset = SimpleNamespace(id="dataset-id", owner_id="owner-id")
    default_user = object()
    authorize = AsyncMock(return_value=[dataset])
    graph_engine = SimpleNamespace(get_graph_data=AsyncMock(return_value=([], [])))
    set_context_calls = []

    monkeypatch.setattr(visualize_module, "get_default_user", AsyncMock(return_value=default_user))
    monkeypatch.setattr(visualize_module, "get_authorized_existing_datasets", authorize)
    monkeypatch.setattr(
        visualize_module,
        "set_database_global_context_variables",
        _capturing_set_context(set_context_calls),
    )
    monkeypatch.setattr(visualize_module, "get_graph_engine", AsyncMock(return_value=graph_engine))
    monkeypatch.setattr(
        visualize_module, "cognee_network_visualization", AsyncMock(return_value="<html></html>")
    )

    result = await visualize_module.visualize_graph(include_session_events=False)

    assert result == "<html></html>"
    authorize.assert_awaited_once_with(["main_dataset"], "read", default_user)
    assert set_context_calls == [(dataset.id, dataset.owner_id)]


@pytest.mark.asyncio
async def test_visualize_graph_explicit_none_skips_authorization(monkeypatch):
    """Passing dataset=None explicitly skips authorization; the DB context is
    set with None, which is a no-op when access control is off."""
    authorize = AsyncMock()
    graph_engine = SimpleNamespace(get_graph_data=AsyncMock(return_value=([], [])))
    set_context_calls = []

    monkeypatch.setattr(visualize_module, "get_default_user", AsyncMock(return_value=object()))
    monkeypatch.setattr(visualize_module, "get_authorized_existing_datasets", authorize)
    monkeypatch.setattr(
        visualize_module,
        "set_database_global_context_variables",
        _capturing_set_context(set_context_calls),
    )
    monkeypatch.setattr(visualize_module, "get_graph_engine", AsyncMock(return_value=graph_engine))
    monkeypatch.setattr(
        visualize_module, "cognee_network_visualization", AsyncMock(return_value="<html></html>")
    )

    result = await visualize_module.visualize_graph(dataset=None, include_session_events=False)

    assert result == "<html></html>"
    authorize.assert_not_awaited()
    assert set_context_calls == [(None, None)]
