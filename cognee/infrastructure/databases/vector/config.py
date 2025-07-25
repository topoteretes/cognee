import os
from typing import Optional
import pydantic
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.base_config import get_base_config


class VectorConfig(BaseSettings):
    """
    Manage the configuration settings for the vector database.

    Instance variables:

    - vector_db_url: The URL of the vector database.
    - vector_db_port: The port for the vector database.
    - vector_db_key: The key for accessing the vector database.
    - vector_db_provider: The provider for the vector database.
    """

    vector_db_url: str = ""
    vector_db_port: int = 1234
    vector_db_key: str = ""
    vector_db_provider: str = "lancedb"

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    @pydantic.model_validator(mode="after")
    def fill_derived(cls, values):
        # Set file path based on graph database provider if no file path is provided
        if not values.vector_db_url:
            base_config = get_base_config()
            databases_directory_path = os.path.join(base_config.system_root_directory, "databases")
            values.vector_db_url = os.path.join(databases_directory_path, "cognee.lancedb")

        return values


@lru_cache
def get_vectordb_config():
    """
    Retrieve the cached vector database configuration.

    This function uses the LRU cache to store the instance of `VectorConfig`, allowing for
    efficient reuse without needing to recreate the object multiple times. If a
    configuration is already cached, it returns that instead of creating a new one.

    Returns:
    --------

        - VectorConfig: An instance of `VectorConfig` containing the vector database
          configuration.
    """
    context_config = get_vectordb_context_config()

    if context_config:
        return context_config

    return VectorConfig()


def get_vectordb_context_config() -> Optional[VectorConfig]:
    """This function will get the appropriate vector db config based on async context."""
    from cognee.context_global_variables import vector_db_config

    return vector_db_config.get()
