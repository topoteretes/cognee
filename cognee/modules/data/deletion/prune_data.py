from cognee.infrastructure.files.storage import get_file_storage, get_storage_config


async def prune_data():
    storage_config = get_storage_config()
    data_root_directory = storage_config["data_root_directory"]
    await get_file_storage(data_root_directory).remove_all()
