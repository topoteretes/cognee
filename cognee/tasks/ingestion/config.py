from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionConfig(BaseSettings):
    # Cap on rows read per table from a DLT source; 0 (default) means
    # unlimited — everything is ingested. Set a positive value (env:
    # DLT_MAX_ROWS_PER_TABLE, or add(..., max_rows_per_table=N)) to bound
    # ingestion of large sources.
    dlt_max_rows_per_table: int = 0

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "dlt_max_rows_per_table": self.dlt_max_rows_per_table,
        }


@lru_cache
def get_ingestion_config():
    return IngestionConfig()
