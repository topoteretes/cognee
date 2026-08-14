from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionConfig(BaseSettings):
    # Cap on rows read per table from a DLT source; 0 (default) means
    # unlimited — everything is ingested. Set a positive value (env:
    # DLT_MAX_ROWS_PER_TABLE, or add(..., max_rows_per_table=N)) to bound
    # ingestion of large sources.
    dlt_max_rows_per_table: int = 0

    # Cell-level graph nodes for DLT rows: maps table name to the columns
    # whose values become shared ColumnValue nodes, e.g. {"orders": ["status"]};
    # "*" is a wildcard on either side ({"*": ["*"]} selects every column of
    # every table). Default {} (off): value nodes serve deterministic traversal
    # (CYPHER, degree-based aggregates, visualization joins) and cost one
    # embedding per unique value, so they are opt-in — prefer selecting a few
    # low-cardinality columns over "*" to avoid one-off nodes for ids,
    # timestamps, and free text.
    # Env: DLT_COLUMN_VALUE_COLUMNS as JSON, or add(..., column_value_columns=...).
    dlt_column_value_columns: dict[str, list[str]] = {}

    # Length bound on cell values selected for ColumnValue nodes; 0 (default)
    # means unlimited — every selected value becomes a node regardless of
    # length. Set a positive value (env: DLT_MAX_COLUMN_VALUE_LENGTH) to skip
    # long free-text cells, which make poor shared categorical nodes and cost
    # one embedding per unique value.
    dlt_max_column_value_length: int = 0

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "dlt_max_rows_per_table": self.dlt_max_rows_per_table,
            "dlt_column_value_columns": self.dlt_column_value_columns,
            "dlt_max_column_value_length": self.dlt_max_column_value_length,
        }


@lru_cache
def get_ingestion_config():
    return IngestionConfig()
