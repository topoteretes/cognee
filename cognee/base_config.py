import os
from typing import Optional
from functools import lru_cache
from cognee.root_dir import get_absolute_path
from cognee.modules.observability.observers import Observer
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    data_root_directory: str = get_absolute_path(".data_storage")
    system_root_directory: str = get_absolute_path(".cognee_system")
    monitoring_tool: object = Observer.LANGFUSE
    default_user_email: Optional[str] = os.getenv("DEFAULT_USER_EMAIL")
    default_user_password: Optional[str] = os.getenv("DEFAULT_USER_PASSWORD")

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_base_config():
    return BaseConfig()
