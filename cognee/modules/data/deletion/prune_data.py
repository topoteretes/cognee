from os import path
from cognee.base_config import get_base_config
from cognee.infrastructure.files.storage import LocalStorage


async def prune_data(user = None):
    base_config = get_base_config()
    data_root_directory = base_config.data_root_directory

    if user:
        LocalStorage.remove_all(path.join(data_root_directory, str(user.id)))
    else:
        LocalStorage.remove_all(data_root_directory)
