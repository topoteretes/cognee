from cognee.base_config import get_base_config
from cognee.infrastructure.files.storage import LocalStorage


async def prune_data():
    base_config = get_base_config()
    data_root_directory = base_config.data_root_directory
    LocalStorage.remove_all(data_root_directory)
