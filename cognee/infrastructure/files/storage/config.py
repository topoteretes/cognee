from typing import Optional
from contextvars import ContextVar
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageConfig(BaseSettings):
    """
    Manage configuration settings for file storage.
    """

    data_root_directory: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


file_storage_config = ContextVar[Optional[StorageConfig]]("file_storage_config", default=None)
