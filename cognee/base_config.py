from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.root_dir import get_absolute_path
from cognee.shared.data_models import MonitoringTool

class BaseConfig(BaseSettings):
    system_root_directory: str = get_absolute_path(".cognee_system")
    data_root_directory: str = get_absolute_path(".data")
    monitoring_tool: object = MonitoringTool.LANGFUSE

    model_config = SettingsConfigDict(env_file = ".env", extra = "allow")

    def to_dict(self) -> dict:
        return {
            "system_root_directory": self.system_root_directory,
            "data_root_directory": self.data_root_directory,
            "monitoring_tool": self.monitoring_tool,
        }

@lru_cache
def get_base_config():
    return BaseConfig()
