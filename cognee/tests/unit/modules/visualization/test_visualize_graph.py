from types import SimpleNamespace
from unittest.mock import AsyncMock
import importlib

import pytest

visualize_module = importlib.import_module("cognee.api.v1.visualize.visualize")


@pytest.mark.asyncio
async def test_visualize_graph_uses_dataset_context(monkeypatch):
    dataset = SimpleNamespace(id="dataset-id", owner_id="owner-id")
    graph_engine = SimpleNamespace(get_graph_data=AsyncMock(return_value=([], [])))
    set_context = AsyncMock()
    monkeypatch.setattr(visualize_module, "get_default_user", AsyncMock(return_value=object()))
    monkeypatch.setattr(visualize_module, "get_authorized_existing_datasets", AsyncMock(return_value=[dataset]))
    monkeypatch.setattr(visualize_module, "set_database_global_context_variables", set_context)
    monkeypatch.setattr(visualize_module, "get_graph_engine", AsyncMock(return_value=graph_engine))
    monkeypatch.setattr(visualize_module, "cognee_network_visualization", AsyncMock(return_value="<html></html>"))

    assert await visualize_module.visualize_graph(datasets="NLP") == "<html></html>"
    set_context.assert_awaited_once_with(dataset.id, dataset.owner_id)
