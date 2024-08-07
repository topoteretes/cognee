from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from cognee.root_dir import get_absolute_path
from cognee.shared.data_models import MonitoringTool

class BaseConfig(BaseSettings):
    data_root_directory: str = get_absolute_path(".data_storage")
    monitoring_tool: object = MonitoringTool.LANGFUSE
    graphistry_username: Optional[str] = None
    graphistry_password: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    model_config = SettingsConfigDict(env_file = ".env", extra = "allow")

    def to_dict(self) -> dict:
        return {
            "data_root_directory": self.data_root_directory,
            "monitoring_tool": self.monitoring_tool,
        }

@lru_cache
def get_base_config():
    return BaseConfig()
