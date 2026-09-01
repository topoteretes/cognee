"""Tests for TursoGraphDatasetDatabaseHandler (delete_dataset from Finding 5,
COG-6335 review; create_dataset from COG-6491).

delete_dataset covers:
- eviction is by database name (aevict_for_database), matching the generic
  key ensure_graph_memory_cleared's get_graph_engine() resolves, instead of
  the old narrower exact-key evict() that could miss the live cache entry
- the dataset's libSQL file and its WAL-mode companions (-wal/-shm) are
  removed
- a dataset with no graph_database_name never calls the cache at all
  (nothing to evict by)
- -wal/-shm companions absent (clean checkpointed close) does not raise
- eviction completes before the file is removed, not just both happening
- delete_dataset actually removes the file at the URL create_dataset itself
  produces, on the OS the test runs on -- not a hand-built clean path.
  Round-tripping through the real create_dataset (mocking only
  get_base_config/get_graph_config/create_graph_engine, not the path
  construction itself) is what lets Windows CI -- not a POSIX dev machine --
  judge whether delete_dataset's absolute-path check works for what
  create_dataset really produces on that platform.

create_dataset covers:
- the URL is os.path.join's result verbatim, with no added prefix
- the same holds for a Windows-shaped path, asserted on any OS
- a non-absolute system root is rejected, and rejected before makedirs
- the dataset opens for real: a live engine round trip leaves a file at the
  URL, which on windows-latest is a real drive-letter path

COG-6491: an earlier create_dataset appended a leading "/" whenever the
joined path did not already start with one. On Windows that produced
"/C:\\...", which ntpath.isabs() rejects on Python 3.13 (on 3.12 isabs
passed and os.path.exists failed instead -- same silent no-op, different
route), so delete_dataset cleaned up nothing. It also fired on POSIX for any
system root not starting with "/" -- an s3:// one, or a relative one --
where the extra "/" was in fact what made sqlite refuse the path outright.
Dropping it alone would have turned that loud failure into an accepted,
CWD-relative file that the isabs() guard makes undeletable, so create_dataset
now rejects those roots instead.
"""

import importlib
import ntpath
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


async def _create_dataset_url_for_root(system_root, fake_os=None):
    """create_dataset against an arbitrary system root, with the engine
    stubbed so only path construction runs. fake_os replaces the handler's
    own os global -- patching handler_module.os's attributes would mutate the
    stdlib process-wide, since handler_module.os *is* os."""
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
            return_value=SimpleNamespace(system_root_directory=system_root),
        ),
        patch.object(handler_module, "create_graph_engine", return_value=fake_engine),
        patch.object(
            handler_module, "os", fake_os or SimpleNamespace(path=os.path, makedirs=MagicMock())
        ),
    ):
        info = await TursoGraphDatasetDatabaseHandler.create_dataset("dataset-id", None)

    return info["graph_database_url"]


async def test_create_dataset_url_is_the_plain_join(tmp_path):
    """The URL is os.path.join's result verbatim -- no added prefix."""
    dataset_url = await _create_dataset_url_for_root(str(tmp_path))

    assert dataset_url == os.path.join(str(tmp_path), "databases", "graph_dataset-id.db")
    assert os.path.isabs(dataset_url)


async def test_create_dataset_does_not_prefix_windows_style_path():
    """Platform-independent guard for the COG-6491 mangling.

    On POSIX the old and new expressions are byte-identical, since
    os.path.join already yields a leading "/" -- so a POSIX run cannot catch
    a reintroduction on its own and only windows-latest would. Handing the
    handler an os whose path is ntpath reproduces the Windows string
    arithmetic here: the old code turned this into "/C:\\fake\\databases\\...",
    which ntpath.isabs rejects and delete_dataset therefore never cleans up."""
    dataset_url = await _create_dataset_url_for_root(
        r"C:\fake", fake_os=SimpleNamespace(path=ntpath, makedirs=MagicMock())
    )

    assert dataset_url == r"C:\fake\databases\graph_dataset-id.db"
    assert ntpath.isabs(dataset_url)
    assert not dataset_url.startswith("/")


@pytest.mark.parametrize(
    "system_root",
    ["s3://bucket/cognee/system", "relative/system/root"],
    ids=["s3_root", "relative_root"],
)
async def test_create_dataset_rejects_non_absolute_system_root(system_root):
    """A libSQL dataset is a local file, and delete_dataset's os.path.isabs
    guard silently skips a non-absolute one, so such a database could never
    be removed. Both roots are reachable: ensure_absolute_path passes s3://
    through untouched, and cognee.config.system_root_directory assigns a
    relative path without revalidating. sqlite accepts both (creating a file
    relative to the CWD), so without this check they would be orphans."""
    with pytest.raises(EnvironmentError, match="absolute local path"):
        await _create_dataset_url_for_root(system_root)


async def test_create_dataset_makes_no_directory_for_non_absolute_root():
    """The check runs before makedirs, so a non-local root leaves no local
    directory tree behind -- an s3:// root would otherwise have os.makedirs
    build a literal "./s3:/bucket/cognee/system/databases"."""
    fake_os = SimpleNamespace(path=os.path, makedirs=MagicMock())

    with pytest.raises(EnvironmentError):
        await _create_dataset_url_for_root("s3://bucket/cognee/system", fake_os=fake_os)

    fake_os.makedirs.assert_not_called()


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
    # Replace the module's os global, not os.remove itself -- handler_module.os
    # *is* os, so setting an attribute on it would swap the stdlib's remove
    # process-wide for the duration of the test.
    monkeypatch.setattr(handler_module, "os", SimpleNamespace(path=os.path, remove=tracking_remove))

    await TursoGraphDatasetDatabaseHandler.delete_dataset(
        _dataset_database(str(db_path), graph_db_name="dataset-id")
    )

    assert call_order == ["evict", "remove"]


async def test_create_dataset_opens_a_real_graph(tmp_path):
    """The other half of COG-6491: cleanup was the visible failure, but the
    same value also builds the connection string, interpolated into
    "sqlite+aiosqlite:///". Nothing else here lets the real engine run, so
    nothing else would notice a URL sqlite cannot open.

    create_graph_engine is deliberately NOT patched: this drives create_dataset
    end to end and asserts a file exists at the URL it returned. On
    windows-latest tmp_path is a real drive-letter path, which makes this the
    acceptance test for "a per-dataset Turso graph opens on Windows"; on POSIX
    it is an ordinary sqlite round trip."""
    kwargs = dict(
        graph_database_provider="turso",
        graph_file_path="",
        graph_database_key="",
    )

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
    ):
        info = await TursoGraphDatasetDatabaseHandler.create_dataset("dataset-id", None)

    dataset_url = info["graph_database_url"]
    try:
        assert os.path.exists(dataset_url)
    finally:
        # create_graph_engine is closing_lru_cache-wrapped; without this the
        # adapter stays cached under a tmp_path key that is gone next test.
        handler_module.graph_engine_cache.evict(graph_database_url=dataset_url, **kwargs)
