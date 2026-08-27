"""Tests for LadybugDatasetDatabaseHandler.delete_dataset's shared-lock use
(Findings 3/4, COG-6335 review).

LadybugAdapter.query() takes a cross-process Redis lock before opening the
native ladybug.Database when SHARED_LADYBUG_LOCK is enabled, and
LadybugAdapter.delete_graph() already takes that same lock before removing
files, for the same reason. delete_dataset must take the identical lock
before evicting the cache and removing files, or a concurrent query()/search()
in another process could be mid-query (holding the on-disk file lock) when
this method deletes its files, or could open the file right between the drop
and a later recreate.

Covers:
- SHARED_LADYBUG_LOCK enabled: the lock is acquired before eviction/file
  removal and released afterward, even when eviction/removal raises
- the lock key matches LadybugAdapter's own key derivation for this exact
  dataset's db path
- SHARED_LADYBUG_LOCK disabled (default): no lock is touched at all --
  behavior is unchanged for deployments that have not opted in
"""

import importlib
import os
from types import SimpleNamespace
from uuid import NAMESPACE_OID, uuid4, uuid5
from unittest.mock import AsyncMock, MagicMock

import pytest

# Import order matters here: cognee.infrastructure.databases.dataset_database_handler
# (the registry) and LadybugDatasetDatabaseHandler import each other (the
# registry lists every handler; each handler implements the registry's
# interface). Importing the registry package first lets that pre-existing
# cycle resolve normally -- importing LadybugDatasetDatabaseHandler as the
# very first thing raises "cannot import name ... from partially initialized
# module", unrelated to this fix and reproducible on main too.
import cognee.infrastructure.databases.dataset_database_handler  # noqa: F401

handler_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.ladybug.LadybugDatasetDatabaseHandler"
)
get_cache_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.cache.get_cache_engine"
)
LadybugDatasetDatabaseHandler = handler_module.LadybugDatasetDatabaseHandler

pytestmark = pytest.mark.asyncio

OWNER_ID = uuid4()


def _dataset_database(graph_db_name="dataset.lbug"):
    return SimpleNamespace(owner_id=OWNER_ID, graph_database_name=graph_db_name)


def _fake_file_storage():
    storage = SimpleNamespace()
    storage.is_file = AsyncMock(return_value=False)
    storage.remove = AsyncMock()
    storage.remove_all = AsyncMock()
    return storage


async def test_delete_dataset_holds_shared_lock_around_evict_and_file_removal(monkeypatch):
    monkeypatch.setattr(
        handler_module, "get_cache_config", lambda: SimpleNamespace(shared_ladybug_lock=True)
    )
    calls = []

    class _FakeLock:
        def acquire_lock(self):
            calls.append("acquire")
            return "lock-handle"

        def release_lock(self, handle):
            calls.append(("release", handle))

    fake_lock = _FakeLock()

    def _fake_get_cache_engine(lock_key):
        calls.append(("get_cache_engine", lock_key))
        return fake_lock

    monkeypatch.setattr(get_cache_engine_module, "get_cache_engine", _fake_get_cache_engine)

    async def _fake_aevict(_name):
        calls.append("evict")
        return 1

    monkeypatch.setattr(
        handler_module, "graph_engine_cache", SimpleNamespace(aevict_for_database=_fake_aevict)
    )
    file_storage = _fake_file_storage()
    monkeypatch.setattr(handler_module, "get_file_storage", lambda _path: file_storage)

    dataset_database = _dataset_database()
    await LadybugDatasetDatabaseHandler.delete_dataset(dataset_database)

    assert calls[0][0] == "get_cache_engine"
    assert calls[1] == "acquire"
    assert calls[2] == "evict"
    assert calls[-1] == ("release", "lock-handle")


async def test_delete_dataset_lock_key_matches_ladybug_adapter_db_path(monkeypatch):
    """The lock key must be derived from the exact same db_path LadybugAdapter
    uses -- get_graph_engine() resolves it as
    os.path.join(system_root_directory/databases/<owner_id>, graph_database_name)."""
    monkeypatch.setattr(
        handler_module,
        "get_base_config",
        lambda: SimpleNamespace(system_root_directory="/system/root"),
    )
    monkeypatch.setattr(
        handler_module, "get_cache_config", lambda: SimpleNamespace(shared_ladybug_lock=True)
    )
    seen_lock_keys = []

    def _fake_get_cache_engine(lock_key):
        seen_lock_keys.append(lock_key)
        return SimpleNamespace(acquire_lock=lambda: None, release_lock=lambda _h: None)

    monkeypatch.setattr(get_cache_engine_module, "get_cache_engine", _fake_get_cache_engine)
    monkeypatch.setattr(
        handler_module,
        "graph_engine_cache",
        SimpleNamespace(aevict_for_database=AsyncMock(return_value=0)),
    )
    monkeypatch.setattr(handler_module, "get_file_storage", lambda _path: _fake_file_storage())

    dataset_database = _dataset_database(graph_db_name="my-dataset.lbug")
    await LadybugDatasetDatabaseHandler.delete_dataset(dataset_database)

    expected_db_path = os.path.join("/system/root", "databases", str(OWNER_ID), "my-dataset.lbug")
    expected_key = "ladybug-lock-" + str(uuid5(NAMESPACE_OID, expected_db_path))
    assert seen_lock_keys == [expected_key]


async def test_delete_dataset_releases_lock_even_when_eviction_raises(monkeypatch):
    monkeypatch.setattr(
        handler_module, "get_cache_config", lambda: SimpleNamespace(shared_ladybug_lock=True)
    )
    released = []
    monkeypatch.setattr(
        get_cache_engine_module,
        "get_cache_engine",
        lambda lock_key: SimpleNamespace(
            acquire_lock=lambda: "handle",
            release_lock=lambda handle: released.append(handle),
        ),
    )
    monkeypatch.setattr(
        handler_module,
        "graph_engine_cache",
        SimpleNamespace(
            aevict_for_database=AsyncMock(side_effect=RuntimeError("cache eviction failed"))
        ),
    )

    with pytest.raises(RuntimeError, match="cache eviction failed"):
        await LadybugDatasetDatabaseHandler.delete_dataset(_dataset_database())

    assert released == ["handle"]


async def test_delete_dataset_does_not_touch_lock_when_shared_lock_disabled(monkeypatch):
    """Default (SHARED_LADYBUG_LOCK unset/false): behavior is unchanged."""
    monkeypatch.setattr(
        handler_module, "get_cache_config", lambda: SimpleNamespace(shared_ladybug_lock=False)
    )
    get_cache_engine_mock = MagicMock()
    monkeypatch.setattr(get_cache_engine_module, "get_cache_engine", get_cache_engine_mock)
    monkeypatch.setattr(
        handler_module,
        "graph_engine_cache",
        SimpleNamespace(aevict_for_database=AsyncMock(return_value=0)),
    )
    monkeypatch.setattr(handler_module, "get_file_storage", lambda _path: _fake_file_storage())

    await LadybugDatasetDatabaseHandler.delete_dataset(_dataset_database())

    get_cache_engine_mock.assert_not_called()
