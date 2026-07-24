from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionConfig(BaseSettings):
    # Cap on rows read per table from a DLT source; 0 (default) means
    # unlimited — everything is ingested. Set a positive value (env:
    # DLT_MAX_ROWS_PER_TABLE, or add(..., max_rows_per_table=N)) to bound
    # ingestion of large sources.
    dlt_max_rows_per_table: int = 0

    # Cell-level graph nodes for DLT rows: maps table name to the columns
    # whose values become shared ColumnValue nodes. "*" is a wildcard on
    # either side. Column specs:
    #   ["auto"] (default, {"*": ["auto"]}) — only columns whose values repeat
    #       across rows qualify; ids, timestamps, and free text are skipped so
    #       high-cardinality sources (e.g. Slack messages) stay clean.
    #   ["*"] — every column, no cardinality gate.
    #   ["col", ...] — exactly these columns.
    #   {} — disable value-node emission entirely.
    # Env: DLT_COLUMN_VALUE_COLUMNS as JSON, or add(..., column_value_columns=...).
    dlt_column_value_columns: dict[str, list[str]] = {"*": ["auto"]}

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "dlt_max_rows_per_table": self.dlt_max_rows_per_table,
            "dlt_column_value_columns": self.dlt_column_value_columns,
        }


@lru_cache
def get_ingestion_config():
    return IngestionConfig()
