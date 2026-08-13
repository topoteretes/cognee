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


def is_remote_csv_path(data: str) -> bool:
    """A CSV living behind cognee's remote storage layer (currently S3)."""
    return is_csv_path(data) and data.startswith("s3://")


async def download_csv_for_staging(csv_url: str, temp_dir: str) -> str:
    """Localize a remote CSV through cognee's file layer for dlt staging.

    The storage layer (``open_data_file`` -> ``S3FileStorage``) owns the
    local-vs-S3 distinction and the credentials: cognee's ``S3Config``
    (access key / endpoint / profile / session token) or the IAM chain —
    never a parallel fsspec configuration. dlt's staging reader then treats
    the result like any local CSV, keeping one read path and identical
    typing behavior for both origins.
    """
    from uuid import uuid4

    from cognee.infrastructure.files.utils.open_data_file import open_data_file

    filename = csv_url.rpartition("/")[2]
    # Per-download subdirectory: two sources may share a filename.
    local_dir = os.path.join(temp_dir, uuid4().hex)
    os.makedirs(local_dir)
    local_path = os.path.join(local_dir, filename)

    async with open_data_file(csv_url, mode="rb") as remote_file:
        content = remote_file.read()
    with open(local_path, "wb") as local_file:
        local_file.write(content)

    return local_path


def create_dlt_source_from_csv(csv_path: str):
    """Auto-generate a dlt resource from a local CSV file path.

    Accepts plain paths and ``file://`` URLs. Remote CSVs (s3://) must be
    localized first via ``download_csv_for_staging`` — resolve_dlt_sources
    does this — so this function never needs to know about remote backends.
    """
    from dlt.sources.filesystem import filesystem, read_csv

    if csv_path.startswith("file://"):
        csv_path = csv_path[len("file://") :]

    parent_dir = os.path.dirname(os.path.abspath(csv_path))
    filename = os.path.basename(csv_path)

    return (
        filesystem(
            bucket_url=f"file://{parent_dir}",
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
