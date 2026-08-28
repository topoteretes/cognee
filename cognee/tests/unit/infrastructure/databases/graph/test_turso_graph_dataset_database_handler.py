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
- delete_dataset actually removes the file at the URL create_dataset itself
  produces, on the OS the test runs on -- not a hand-built clean path. On
  POSIX, create_dataset's URL equals os.path.join's result verbatim; on
  Windows, os.path.join returns a backslash drive-letter path that does not
  start with "/", so create_dataset prepends one ("sqlite+aiosqlite:///
  needs three slashes for absolute path"), and delete_dataset must still
  find and remove the resulting file. Round-tripping through the real
  create_dataset (mocking only get_base_config/get_graph_config/
  create_graph_engine, not the path construction itself) is what lets
  Windows CI -- not a POSIX dev machine -- be the actual judge of whether
  the absolute-path check in delete_dataset works for what create_dataset
  really produces on that platform.
"""

import importlib
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

handler_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.turso.TursoGraphDatasetDatabaseHandler"
)
TursoGraphDatasetDatabaseHandler = handler_module.TursoGraphDatasetDatabaseHandler

pytestmark = pytest.mark.asyncio


def _dataset_database(url, graph_db_name="dataset-id"):
    return SimpleNamespace(graph_database_url=url, graph_database_name=graph_db_name)


async def _create_real_dataset_url(tmp_path, dataset_id="dataset-id"):
    """Round-trip through the real create_dataset to get the exact
    graph_database_url it produces on this OS, without a live DB connection."""
    fake_engine = MagicMock()
    fake_engine.initialize = AsyncMock()

    with (
        patch.object(
            handler_module,
            "get_graph_config",
            return_value=SimpleNamespace(graph_database_provider="turso"),
        ),
        patch.object(
            handler_module,
            "get_base_config",
            return_value=SimpleNamespace(system_root_directory=str(tmp_path)),
        ),
        patch.object(handler_module, "create_graph_engine", return_value=fake_engine),
    ):
        info = await TursoGraphDatasetDatabaseHandler.create_dataset(dataset_id, None)

    return info["graph_database_url"]


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


async def test_delete_dataset_removes_file_at_real_create_dataset_url(monkeypatch, tmp_path):
    """Same assertion as test_delete_dataset_removes_db_file_and_wal_companions,
    but against the URL create_dataset itself actually builds on this OS,
    not a hand-constructed clean path -- see module docstring."""
    dataset_url = await _create_real_dataset_url(tmp_path)

    # Write the fixture file the same way delete_dataset checks/removes it:
    # raw os-level calls on the exact URL string, not pathlib (pathlib parses
    # a path string into components and could disagree with os.path's/the
    # OS's own resolution of an unusual string like this one).
    fd = os.open(dataset_url, os.O_CREAT | os.O_WRONLY)
    os.close(fd)
    assert os.path.exists(dataset_url)

    monkeypatch.setattr(
        handler_module, "graph_engine_cache", SimpleNamespace(aevict_for_database=AsyncMock())
    )

    await TursoGraphDatasetDatabaseHandler.delete_dataset(
        _dataset_database(dataset_url, graph_db_name="dataset-id")
    )

    assert not os.path.exists(dataset_url)


async def test_delete_dataset_evicts_before_removing_file(monkeypatch, tmp_path):
    """Eviction must complete (and wait for any in-flight close) BEFORE the
    file is removed -- the whole reason for aevict_for_database over a bare
    evict() is to not delete a file a just-evicted engine might still be
    closing. A future refactor that reordered these two steps would still
    pass every other test in this file, since they assert eviction and file
    removal independently; this one asserts the sequence."""
    db_path = tmp_path / "graph_dataset-id.db"
    db_path.write_text("data")

    call_order = []

    async def fake_aevict(name):
        call_order.append("evict")

    real_remove = os.remove

    def tracking_remove(path):
        call_order.append("remove")
        real_remove(path)

    monkeypatch.setattr(
        handler_module, "graph_engine_cache", SimpleNamespace(aevict_for_database=fake_aevict)
    )
    monkeypatch.setattr(handler_module.os, "remove", tracking_remove)

    await TursoGraphDatasetDatabaseHandler.delete_dataset(
        _dataset_database(str(db_path), graph_db_name="dataset-id")
    )

    assert call_order == ["evict", "remove"]
