import os
from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from cognee.root_dir import get_absolute_path
from cognee.shared.data_models import MonitoringTool


class BaseConfig(BaseSettings):
    data_root_directory: str = get_absolute_path(".data_storage")
    monitoring_tool: object = MonitoringTool.LANGFUSE
    graphistry_username: Optional[str] = os.getenv("GRAPHISTRY_USERNAME")
    graphistry_password: Optional[str] = os.getenv("GRAPHISTRY_PASSWORD")
    langfuse_public_key: Optional[str] = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[str] = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_host: Optional[str] = os.getenv("LANGFUSE_HOST")
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "data_root_directory": self.data_root_directory,
            "monitoring_tool": self.monitoring_tool,
        }


@lru_cache
def get_base_config():
    return BaseConfig()
