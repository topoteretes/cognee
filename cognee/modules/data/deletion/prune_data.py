from cognee.base_config import get_base_config
from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage


async def prune_data():
    base_config = get_base_config()
    data_root_directory = base_config.data_root_directory
    LocalFileStorage(data_root_directory).remove_all()
