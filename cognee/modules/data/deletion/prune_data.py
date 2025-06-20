from cognee.base_config import get_base_config
from cognee.infrastructure.files.storage import get_file_storage


async def prune_data():
    base_config = get_base_config()
    data_root_directory = base_config.data_root_directory
    get_file_storage(data_root_directory).remove_all()
