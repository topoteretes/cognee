"""The filesystem cache stores text only and refuses pickled rows (CVE-2025-69872)."""

import pickle
import tempfile
from unittest.mock import patch

import diskcache as dc
import pytest

from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import FSCacheAdapter
from cognee.infrastructure.databases.exceptions.exceptions import CacheConnectionError


@pytest.fixture
def adapter():
    """An adapter on a temporary data root, closed before the directory is removed
    (Windows cannot delete the SQLite file while diskcache holds it open)."""
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch(
            "cognee.infrastructure.databases.cache.fscache.FsCacheAdapter.get_storage_config",
            return_value={"data_root_directory": tmpdir},
        ),
    ):
        inst = FSCacheAdapter()
        yield inst
        inst.cache.close()


@pytest.mark.asyncio
async def test_string_values_round_trip_without_pickle(adapter):
    await adapter.set_value("k", '{"a": 1}')
    assert await adapter.get_value("k") == '{"a": 1}'

    # The row diskcache wrote is text (inline when small, a file when large), never pickle.
    (mode,) = adapter.cache._sql("SELECT mode FROM Cache WHERE key = 'k'").fetchone()
    assert mode in (dc.core.MODE_RAW, dc.core.MODE_TEXT)


@pytest.mark.asyncio
async def test_pickled_value_planted_in_cache_directory_is_refused(adapter):
    # Something else writes a pickled row into the same directory with diskcache's
    # default serializer, as an attacker with write access to the directory could.
    foreign = dc.Cache(directory=adapter.cache_directory)
    foreign.set("planted", {"not": "text"})
    foreign.close()

    with pytest.raises(CacheConnectionError, match="pickled value"):
        await adapter.get_value("planted")


@pytest.mark.asyncio
async def test_pickled_key_planted_in_cache_directory_is_refused(adapter):
    foreign = dc.Cache(directory=adapter.cache_directory)
    foreign.set(("tuple", "key"), "text")
    foreign.close()

    # Iterating keys goes through Disk.get for every row, pickled key included.
    with pytest.raises(ValueError, match="pickled key"):
        list(adapter.cache)


@pytest.mark.asyncio
async def test_non_string_value_is_refused_at_write_time(adapter):
    with pytest.raises(CacheConnectionError, match="must be str"):
        await adapter.set_value("k", pickle.dumps({"a": 1}))  # bytes, not str
