from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionConfig(BaseSettings):
    # Cap on rows read per table from a DLT source; 0 (default) means
    # unlimited — everything is ingested. Set a positive value (env:
    # DLT_MAX_ROWS_PER_TABLE, or add(..., max_rows_per_table=N)) to bound
    # ingestion of large sources.
    dlt_max_rows_per_table: int = 0

    # Cell-level graph nodes for DLT rows: maps table name to the columns
    # whose values become shared ColumnValue nodes; "*" is a wildcard on either
    # side. Default {"*": ["*"]}: every column of every table, so rows that share
    # a value (a country, a genre, a status) meet in one node and deterministic
    # traversal, degree-based aggregates and the visualization see every cell.
    # Key columns never become value nodes: a primary key is the row's identity
    # and a foreign key is already an edge to the row it references. A date cell
    # that anchors to a Timestamp (dlt_temporal_columns) is that node. Each
    # unique value costs one embedding; narrow the selection, e.g.
    # {"orders": ["status"]}, to keep one-off values such as free text out.
    # {} turns value nodes off.
    # Env: DLT_COLUMN_VALUE_COLUMNS as JSON, or add(..., column_value_columns=...).
    dlt_column_value_columns: dict[str, list[str]] = {"*": ["*"]}

    # Length bound on cell values selected for ColumnValue nodes; 0 (default)
    # means unlimited — every selected value becomes a node regardless of
    # length. Set a positive value (env: DLT_MAX_COLUMN_VALUE_LENGTH) to skip
    # long free-text cells, which make poor shared categorical nodes and cost
    # one embedding per unique value.
    dlt_max_column_value_length: int = 0

    # Time edges for DLT rows: maps table name to the columns whose ISO-shaped
    # cells (YYYY-MM, YYYY-MM-DD, date-times) become edges into shared Timestamp
    # nodes, which temporal search anchors on. Same shape as
    # dlt_column_value_columns; "*" is a wildcard on either side. Default
    # {"*": ["*"]}: every date cell of every table. Narrow it when a table
    # carries bookkeeping dates — {"orders": ["order_date"]} keeps an
    # ``updated_at`` column from anchoring rows to the time they were revised.
    # {} turns time edges off. Env: DLT_TEMPORAL_COLUMNS as JSON, or
    # add(..., temporal_columns=...).
    dlt_temporal_columns: dict[str, list[str]] = {"*": ["*"]}

    model_config = SettingsConfigDict(extra="allow")

    def to_dict(self) -> dict:
        return {
            "dlt_max_rows_per_table": self.dlt_max_rows_per_table,
            "dlt_column_value_columns": self.dlt_column_value_columns,
            "dlt_max_column_value_length": self.dlt_max_column_value_length,
            "dlt_temporal_columns": self.dlt_temporal_columns,
        }


@lru_cache
def get_ingestion_config():
    return IngestionConfig()
