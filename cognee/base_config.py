import asyncio
import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

import cognee.shared.utils
from cognee.modules.users.methods import get_default_user
from cognee.root_dir import get_absolute_path
from cognee.shared.data_models import MonitoringTool


class BaseConfig(BaseSettings):
    data_root_directory: str = get_absolute_path(".data_storage")
    monitoring_tool: object = MonitoringTool.LANGFUSE
    graphistry_username: Optional[str] = os.getenv("GRAPHISTRY_USERNAME")
    graphistry_password: Optional[str] = os.getenv("GRAPHISTRY_PASSWORD")

    model_config = SettingsConfigDict(env_file = ".env", extra = "allow")

    def to_dict(self) -> dict:
        return {
            "data_root_directory": self.data_root_directory,
            "monitoring_tool": self.monitoring_tool,
        }

@lru_cache
def get_base_config():
    config = BaseConfig()
    keys_to_show = [
        'data_root_directory', 
        'monitoring_tool', 
        'graphistry_username', 
        'env', 
        'tokenizers_parallelism', 
        'graph_database_provider', 
        'graph_database_url', 
        'vector_db_provider', 
        'db_provider', 
        'db_name', 
        'db_host', 
        'db_port'
    ]
    conf_dict = dict(config)
    user = asyncio.run(get_default_user())
    filtered_config = {key: conf_dict[key] for key in keys_to_show if key in conf_dict}
    cognee.shared.utils.send_telemetry("cognee base config", user.id, filtered_config)
    return config