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


# fsspec-style URL scheme (s3://, gs://, az://, file://, memory://, ...).
# http(s) is already excluded by is_csv_path. A URL must NOT go through
# os.path.abspath below — abspath treats "s3://bucket/x.csv" as a relative
# local path and mangles it into "<cwd>/s3:/bucket/x.csv".
_URL_SCHEME = re.compile(r"^[a-z0-9]+://", re.IGNORECASE)


def create_dlt_source_from_csv(csv_path: str):
    """Auto-generate a dlt resource from a CSV file path or fsspec URL.

    Local paths are resolved to absolute file:// URLs. URL inputs (s3://,
    gs://, az://, file://, ...) are split into parent-URL + filename and
    passed to dlt's filesystem source as-is; the protocol's fsspec backend
    must be installed (s3 ships with the ``aws`` extra) and credentials come
    from the standard environment (e.g. AWS_ACCESS_KEY_ID / _SECRET / region
    for s3).
    """
    from dlt.sources.filesystem import filesystem, read_csv

    if _URL_SCHEME.match(csv_path):
        bucket_url, _, filename = csv_path.rpartition("/")
    else:
        bucket_url = f"file://{os.path.dirname(os.path.abspath(csv_path))}"
        filename = os.path.basename(csv_path)

    return (
        filesystem(
            bucket_url=bucket_url,
            file_glob=filename,
        )
        | read_csv()
    )


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
