from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class DltRowData:
    """Carries per-row information from ingest_dlt_source back to ingest_data."""

    table_name: str
    primary_key_column: str
    primary_key_value: str
    row_data: dict
    content_hash: str
    schema_info: object  # list[dict] for SQLite, dict for Postgres
    schema_hash: str  # Hash of schema structure for evolution detection
    foreign_keys: list
    dlt_db_name: str
    dataset_name: str


class DltRows(list[DltRowData]):
    """Rows plus the tables successfully loaded and read back, including empty ones.

    An empty incremental run with no load jobs is not an empty source. Keeping
    this evidence separate lets document cleanup distinguish it from a table
    emptied by a successful hard-delete merge.
    """

    def __init__(self, rows: Iterable[DltRowData], *, loaded_tables: Iterable[str]):
        super().__init__(rows)
        self.loaded_tables = frozenset(loaded_tables)
