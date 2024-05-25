from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.root_dir import get_absolute_path


class BaseConfig(BaseSettings):
    system_root_directory: str = get_absolute_path(".cognee_system")
    data_root_directory: str = get_absolute_path(".data")



    model_config = SettingsConfigDict(env_file = ".env", extra = "allow")

    def to_dict(self) -> dict:
        return {
            "system_root_directory": self.system_root_directory,
            "data_root_directory": self.data_root_directory
        }

@lru_cache
def get_llm_config():
    return BaseConfig()