import importlib
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest

from cognee.infrastructure.databases.vector.exceptions import SharedDatabasePruneError

prune_system_module = importlib.import_module("cognee.modules.data.deletion.prune_system")


@asynccontextmanager
async def _record_operation(_operation_name):
    yield


@pytest.mark.asyncio
async def test_full_prune_uses_relational_delete_for_shared_pgvector(monkeypatch):
    vector_engine = Mock()
    vector_engine.prune = AsyncMock(side_effect=SharedDatabasePruneError)
    relational_engine = Mock()
    relational_engine.delete_database = AsyncMock()

    monkeypatch.setattr(prune_system_module, "backend_access_control_enabled", lambda: False)
    monkeypatch.setattr(
        prune_system_module,
        "get_vector_engine_async",
        AsyncMock(return_value=vector_engine),
    )
    monkeypatch.setattr(
        prune_system_module,
        "get_relational_engine",
        Mock(return_value=relational_engine),
    )
    monkeypatch.setattr(prune_system_module._create_vector_engine, "cache_clear", Mock())

    await prune_system_module.prune_system(
        graph=False,
        vector=True,
        metadata=True,
        cache=False,
    )

    vector_engine.prune.assert_awaited_once_with()
    relational_engine.delete_database.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_vector_only_prune_propagates_shared_database_refusal(monkeypatch):
    vector_engine = Mock()
    vector_engine.prune = AsyncMock(side_effect=SharedDatabasePruneError)

    monkeypatch.setattr(prune_system_module, "backend_access_control_enabled", lambda: False)
    monkeypatch.setattr(
        prune_system_module,
        "get_vector_engine_async",
        AsyncMock(return_value=vector_engine),
    )
    monkeypatch.setattr(prune_system_module, "record_operation", _record_operation)

    with pytest.raises(SharedDatabasePruneError):
        await prune_system_module.prune_system(
            graph=False,
            vector=True,
            metadata=False,
            cache=False,
        )


@pytest.mark.asyncio
async def test_shared_database_refusal_leaves_graph_untouched(monkeypatch):
    vector_engine = Mock()
    vector_engine.prune = AsyncMock(side_effect=SharedDatabasePruneError)
    graph_engine = Mock()
    graph_engine.delete_graph = AsyncMock()

    monkeypatch.setattr(prune_system_module, "backend_access_control_enabled", lambda: False)
    monkeypatch.setattr(
        prune_system_module,
        "get_vector_engine_async",
        AsyncMock(return_value=vector_engine),
    )
    monkeypatch.setattr(
        prune_system_module,
        "get_graph_engine",
        AsyncMock(return_value=graph_engine),
    )
    monkeypatch.setattr(prune_system_module, "record_operation", _record_operation)

    with pytest.raises(SharedDatabasePruneError):
        await prune_system_module.prune_system(
            graph=True,
            vector=True,
            metadata=False,
            cache=False,
        )

    graph_engine.delete_graph.assert_not_awaited()


@pytest.mark.asyncio
async def test_dry_run_instantiates_no_engine_and_reports_scope(monkeypatch):
    """dry_run=True reports the prune scope without touching any engine.

    Even instantiating an engine can create empty database files, so the
    preview must not construct one — the report comes from the flags alone.
    """
    vector_getter = AsyncMock()
    graph_getter = AsyncMock()
    relational_getter = Mock()
    cache_deleter = AsyncMock()
    record = Mock()

    monkeypatch.setattr(prune_system_module, "get_vector_engine_async", vector_getter)
    monkeypatch.setattr(prune_system_module, "get_graph_engine", graph_getter)
    monkeypatch.setattr(prune_system_module, "get_relational_engine", relational_getter)
    monkeypatch.setattr(prune_system_module, "delete_cache", cache_deleter)
    monkeypatch.setattr(prune_system_module, "record_operation", record)

    report = await prune_system_module.prune_system(
        graph=True,
        vector=True,
        metadata=True,
        cache=True,
        dry_run=True,
    )

    vector_getter.assert_not_called()
    graph_getter.assert_not_called()
    relational_getter.assert_not_called()
    cache_deleter.assert_not_called()
    record.assert_not_called()

    assert report == {
        "status": "dry_run",
        "dry_run": True,
        "operation": "prune_system",
        "would_prune": {"graph": True, "vector": True, "metadata": True, "cache": True},
    }


@pytest.mark.asyncio
async def test_dry_run_reports_partial_scope(monkeypatch):
    monkeypatch.setattr(prune_system_module, "get_vector_engine_async", AsyncMock())
    monkeypatch.setattr(prune_system_module, "get_graph_engine", AsyncMock())
    monkeypatch.setattr(prune_system_module, "get_relational_engine", Mock())

    report = await prune_system_module.prune_system(
        graph=False,
        vector=True,
        metadata=False,
        cache=False,
        dry_run=True,
    )

    assert report["would_prune"] == {
        "graph": False,
        "vector": True,
        "metadata": False,
        "cache": False,
    }
