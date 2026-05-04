from cognee.infrastructure.databases.cache.get_cache_engine import (
    close_and_clear_cache_engine,
)
from cognee.infrastructure.files.storage import get_file_storage, get_storage_config


async def prune_data():
    # Release any open cache adapter handle for the current data_root_directory
    # before removing the directory; on Windows an open diskcache sqlite handle
    # would block rmtree of cache.db.
    await close_and_clear_cache_engine()

    storage_config = get_storage_config()
    data_root_directory = storage_config["data_root_directory"]
    await get_file_storage(data_root_directory).remove_all()
