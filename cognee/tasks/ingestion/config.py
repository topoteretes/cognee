from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionConfig(BaseSettings):
    # Cap on rows read per table from a DLT source; 0 (default) means
    # unlimited — everything is ingested. Set a positive value (env:
    # DLT_MAX_ROWS_PER_TABLE, or add(..., max_rows_per_table=N)) to bound
    # ingestion of large sources.
    dlt_max_rows_per_table: int = 0

    # Optional cell-level graph nodes for DLT rows: maps table name to the
    # columns whose values become shared ColumnValue nodes. "*" is a wildcard
    # on either side — {"*": ["*"]} emits value nodes for every column of
    # every table, {"orders": ["*"]} for all columns of one table. Empty
    # (default) disables value-node emission. Beware high-cardinality columns:
    # every unique value becomes one node and one embedding. Env:
    # DLT_COLUMN_VALUE_COLUMNS as JSON, or add(..., column_value_columns={...}).
    dlt_column_value_columns: dict[str, list[str]] = {}

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "dlt_max_rows_per_table": self.dlt_max_rows_per_table,
            "dlt_column_value_columns": self.dlt_column_value_columns,
        }


@lru_cache
def get_ingestion_config():
    return IngestionConfig()
