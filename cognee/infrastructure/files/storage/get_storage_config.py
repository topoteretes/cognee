from cognee.base_config import get_base_config

from .config import file_storage_config


def get_global_storage_config() -> dict[str, str]:
    base_config = get_base_config()

    return {
        "data_root_directory": base_config.data_root_directory,
    }


def get_storage_config() -> dict[str, str]:
    context_config = file_storage_config.get()
    if context_config:
        return context_config

    return get_global_storage_config()
