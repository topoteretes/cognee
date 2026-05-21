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
