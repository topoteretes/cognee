from collections.abc import Iterable
from dataclasses import dataclass

# Joins the column values of a composite key into the single string that
# identifies a row (``primary_key_value``, FK lookup keys). A one-column key
# joins to exactly ``str(value)``, so single-key behaviour is unchanged.
KEY_VALUE_SEPARATOR = "|"


@dataclass
class DltRowData:
    """Carries per-row information from ingest_dlt_source back to ingest_data."""

    table_name: str
    primary_key_column: str  # comma-joined names for a composite key
    primary_key_value: str  # KEY_VALUE_SEPARATOR-joined values for a composite key
    row_data: dict
    content_hash: str
    schema_info: object  # list[dict] for SQLite, dict for Postgres
    schema_hash: str  # Hash of schema structure for evolution detection
    foreign_keys: list
    dlt_db_name: str
    dataset_name: str
    primary_key_columns: list[str] | None = None  # the key's columns, in key order


def join_key_values(values: Iterable) -> str:
    return KEY_VALUE_SEPARATOR.join(str(value) for value in values)


def pk_columns(row: DltRowData) -> list[str]:
    """Key columns of a row, in key order (falls back to the single-column field)."""
    if row.primary_key_columns:
        return list(row.primary_key_columns)
    return [row.primary_key_column] if row.primary_key_column else []


def fk_columns(fk: dict) -> tuple[list[str], list[str]]:
    """(local columns, referenced columns) of a foreign key entry.

    Entries carry ``columns``/``ref_columns`` lists for composite keys and the
    legacy single-column ``column``/``ref_column`` fields (comma-joined names
    when composite) so older readers keep working.
    """
    columns = fk.get("columns") or ([fk["column"]] if fk.get("column") else [])
    ref_columns = fk.get("ref_columns") or ([fk["ref_column"]] if fk.get("ref_column") else [])
    return list(columns), list(ref_columns)
