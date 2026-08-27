"""Tests for delete_isolated_dataset_storage (COG-6335).

This is the same graph+vector handler-drop primitive
``cognee.modules.data.methods.delete_dataset`` uses before it removes the
``Dataset`` row, factored out so a memory-only reset can reuse it without any
relational cleanup. These tests only prove the two handler calls happen with
the right arguments — the handlers' own delete_dataset() implementations
(file removal, DROP DATABASE, ...) are covered by their own tests and by
``cognee/tests/test_dataset_delete.py``.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

module = importlib.import_module(
    "cognee.infrastructure.databases.utils.delete_isolated_dataset_storage"
)

pytestmark = pytest.mark.asyncio


async def test_delete_isolated_dataset_storage_drops_graph_and_vector(monkeypatch):
    dataset_database = SimpleNamespace(
        graph_dataset_database_handler="ladybug",
        vector_dataset_database_handler="lancedb",
    )

    graph_delete = AsyncMock()
    vector_delete = AsyncMock()
    monkeypatch.setattr(
        module,
        "get_graph_dataset_database_handler",
        lambda _dataset_database: {
            "handler_instance": SimpleNamespace(delete_dataset=graph_delete)
        },
    )
    monkeypatch.setattr(
        module,
        "get_vector_dataset_database_handler",
        lambda _dataset_database: {
            "handler_instance": SimpleNamespace(delete_dataset=vector_delete)
        },
    )

    await module.delete_isolated_dataset_storage(dataset_database)

    graph_delete.assert_awaited_once_with(dataset_database)
    vector_delete.assert_awaited_once_with(dataset_database)
