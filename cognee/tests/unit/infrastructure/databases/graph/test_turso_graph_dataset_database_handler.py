"""Tests for TursoGraphDatasetDatabaseHandler.delete_dataset (Finding 5,
COG-6335 review).

Covers:
- eviction is by database name (aevict_for_database), matching the generic
  key ensure_graph_memory_cleared's get_graph_engine() resolves, instead of
  the old narrower exact-key evict() that could miss the live cache entry
- the dataset's libSQL file and its WAL-mode companions (-wal/-shm) are
  removed
- a dataset with no graph_database_name never calls the cache at all
  (nothing to evict by)
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

handler_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.turso.TursoGraphDatasetDatabaseHandler"
)
TursoGraphDatasetDatabaseHandler = handler_module.TursoGraphDatasetDatabaseHandler

pytestmark = pytest.mark.asyncio


def _dataset_database(url, graph_db_name="dataset-id"):
    return SimpleNamespace(graph_database_url=url, graph_database_name=graph_db_name)


async def test_delete_dataset_evicts_by_database_name(monkeypatch, tmp_path):
    aevict = AsyncMock(return_value=1)
    monkeypatch.setattr(
        handler_module, "graph_engine_cache", SimpleNamespace(aevict_for_database=aevict)
    )

    await TursoGraphDatasetDatabaseHandler.delete_dataset(
        _dataset_database(str(tmp_path / "graph_dataset-id.db"), graph_db_name="dataset-id")
    )

    aevict.assert_awaited_once_with("dataset-id")


async def test_delete_dataset_removes_db_file_and_wal_companions(monkeypatch, tmp_path):
    db_path = tmp_path / "graph_dataset-id.db"
    wal_path = tmp_path / "graph_dataset-id.db-wal"
    shm_path = tmp_path / "graph_dataset-id.db-shm"
    for path in (db_path, wal_path, shm_path):
        path.write_text("data")

    monkeypatch.setattr(
        handler_module, "graph_engine_cache", SimpleNamespace(aevict_for_database=AsyncMock())
    )

    await TursoGraphDatasetDatabaseHandler.delete_dataset(
        _dataset_database(str(db_path), graph_db_name="dataset-id")
    )

    assert not db_path.exists()
    assert not wal_path.exists()
    assert not shm_path.exists()


async def test_delete_dataset_tolerates_missing_wal_companions(monkeypatch, tmp_path):
    """No -wal/-shm files (clean checkpointed close) must not raise."""
    db_path = tmp_path / "graph_dataset-id.db"
    db_path.write_text("data")

    monkeypatch.setattr(
        handler_module, "graph_engine_cache", SimpleNamespace(aevict_for_database=AsyncMock())
    )

    await TursoGraphDatasetDatabaseHandler.delete_dataset(
        _dataset_database(str(db_path), graph_db_name="dataset-id")
    )

    assert not db_path.exists()


async def test_delete_dataset_skips_eviction_when_no_database_name(monkeypatch, tmp_path):
    aevict = AsyncMock()
    monkeypatch.setattr(
        handler_module, "graph_engine_cache", SimpleNamespace(aevict_for_database=aevict)
    )

    await TursoGraphDatasetDatabaseHandler.delete_dataset(
        _dataset_database(str(tmp_path / "graph_x.db"), graph_db_name="")
    )

    aevict.assert_not_called()
