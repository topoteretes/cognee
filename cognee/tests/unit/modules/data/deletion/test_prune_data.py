import pathlib

import pytest

import cognee
from cognee.base_config import get_base_config
from cognee.infrastructure.databases.cache.config import get_cache_config
from cognee.infrastructure.databases.cache.get_cache_engine import (
    close_cache_engine,
    create_cache_engine,
    get_cache_engine,
)
from cognee.modules.data.deletion.prune_data import prune_data


@pytest.mark.asyncio
async def test_prune_data_closes_cached_fs_cache(tmp_path: pathlib.Path):
    base_config = get_base_config()
    cache_config = get_cache_config()
    previous_data_root_directory = base_config.data_root_directory
    previous_cache_backend = cache_config.cache_backend
    previous_caching = cache_config.caching

    try:
        await close_cache_engine()
        cognee.config.data_root_directory(str(tmp_path))
        cache_config.cache_backend = "fs"
        cache_config.caching = True

        cache_engine = get_cache_engine()
        assert cache_engine is not None
        assert (tmp_path / ".cognee_fs_cache" / "sessions_db").exists()

        await prune_data()

        assert not tmp_path.exists()
        assert create_cache_engine.cache_info().currsize == 0
    finally:
        await close_cache_engine()
        cognee.config.data_root_directory(previous_data_root_directory)
        cache_config.cache_backend = previous_cache_backend
        cache_config.caching = previous_caching
