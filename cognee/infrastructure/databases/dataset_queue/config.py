from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetQueueConfig(BaseSettings):
    database_max_lru_cache_size: int = 6
    dataset_queue_enabled: bool = True
    dataset_queue_max_concurrent: Optional[int] = None

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "database_max_lru_cache_size": self.database_max_lru_cache_size,
            "dataset_queue_enabled": self.dataset_queue_enabled,
            "dataset_queue_max_concurrent": self.dataset_queue_max_concurrent,
        }


@lru_cache
def get_dataset_queue_config():
    return DatasetQueueConfig()
