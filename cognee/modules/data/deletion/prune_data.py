from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.databases.cache.get_cache_engine import close_cache_engine
from cognee.modules.operations import record_operation


async def prune_data():
    async with record_operation("prune_data"):
        storage_config = get_storage_config()
        data_root_directory = storage_config["data_root_directory"]
        await close_cache_engine()
        await get_file_storage(data_root_directory).remove_all()
