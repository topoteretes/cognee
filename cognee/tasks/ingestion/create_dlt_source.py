import os
import re
from typing import Optional

DB_CONNECTION_PATTERNS = [
    "postgresql://",
    "postgres://",
    "mysql://",
    "mysql+pymysql://",
    "sqlite:///",
    "mssql://",
    "oracle://",
]


def is_connection_string(data: str) -> bool:
    return any(data.startswith(p) for p in DB_CONNECTION_PATTERNS)


def is_csv_path(data: str) -> bool:
    return data.lower().endswith(".csv") and not data.startswith(("http://", "https://"))


def is_csv_upload(item) -> bool:
    """A file-like CSV input: an API upload (``.file`` + ``.filename``) or a
    binary handle (``.read`` + ``.name``) whose filename ends in .csv."""
    if isinstance(item, (str, bytes)):
        return False
    filename = getattr(item, "filename", None) or getattr(item, "name", None)
    if not isinstance(filename, str) or not filename.lower().endswith(".csv"):
        return False
    return hasattr(item, "file") or hasattr(item, "read")


def csv_source_name(filename: str) -> str:
    """Deterministic dlt resource name for a CSV file, derived from its
    basename stem. The manifest identity is seeded from this name, so it must
    be stable across runs and distinct across files."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_").lower()
    return safe or "csv_source"


def create_dlt_source_from_connection_string(
    connection_string: str,
    query: Optional[str] = None,
):
    """Auto-generate a dlt source from a database connection string with optional SQL query filtering."""
    from dlt.sources.sql_database import sql_database
    import sqlalchemy

    # SQLite paths must be absolute for SQLAlchemy to find the file.
    # sqlite:/// = relative, sqlite://// = absolute
    if connection_string.startswith("sqlite:///") and not connection_string.startswith(
        "sqlite:////"
    ):
        relative_path = connection_string[len("sqlite:///") :]
        connection_string = "sqlite:///" + os.path.abspath(relative_path)

    if query:
        table_name, where_clause = _parse_sql_query(query)

        def query_adapter_callback(q, table):
            if table.name == table_name:
                return q.where(sqlalchemy.text(where_clause))
            return q

        source = sql_database(
            credentials=connection_string,
            table_names=[table_name],
            query_adapter_callback=query_adapter_callback,
        )
    else:
        source = sql_database(credentials=connection_string)

    return source


def create_dlt_source_from_csv(csv_path: str):
    """Auto-generate a dlt resource from a CSV file path."""
    from dlt.sources.filesystem import filesystem, read_csv

    parent_dir = os.path.dirname(os.path.abspath(csv_path))
    filename = os.path.basename(csv_path)

    source = (
        filesystem(
            bucket_url=f"file://{parent_dir}",
            file_glob=filename,
        )
        | read_csv()
    )
    # A piped read_csv resource is otherwise always named "_read_csv", and the
    # manifest identity is seeded from (dataset, source name) — every CSV in a
    # dataset would collapse into one identity. Name per file instead.
    return source.with_name(csv_source_name(filename))


def _parse_sql_query(query: str) -> tuple:
    """Extract table name and WHERE clause from a SELECT query.
    Returns (table_name, where_clause) or raises ValueError."""
    match = re.match(
        r"SELECT\s+.+?\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+))?",
        query.strip(),
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"Cannot parse SQL query: {query}")
    table_name = match.group(1)
    where_clause = match.group(2) or "1=1"
    return table_name, where_clause
