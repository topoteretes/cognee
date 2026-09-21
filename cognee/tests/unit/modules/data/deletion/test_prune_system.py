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
