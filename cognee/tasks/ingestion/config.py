from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionConfig(BaseSettings):
    dlt_max_rows_per_table: int = 50

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "dlt_max_rows_per_table": self.dlt_max_rows_per_table,
        }


@lru_cache
def get_ingestion_config():
    return IngestionConfig()
