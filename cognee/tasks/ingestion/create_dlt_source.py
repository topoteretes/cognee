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
        schema_name, bare_table_name = _split_schema_and_table(table_name)

        def query_adapter_callback(q, table):
            # sqlalchemy Table.name is never schema-qualified
            if table.name == bare_table_name:
                return q.where(sqlalchemy.text(where_clause))
            return q

        source_kwargs = {
            "credentials": connection_string,
            "table_names": [bare_table_name],
            "query_adapter_callback": query_adapter_callback,
        }
        if schema_name:
            source_kwargs["schema"] = schema_name

        source = sql_database(**source_kwargs)
    else:
        source = sql_database(credentials=connection_string)

    return source


def create_dlt_source_from_csv(csv_path: str, source_name: Optional[str] = None):
    """Auto-generate a dlt resource from a CSV file path.

    ``source_name`` overrides the filename-derived resource name — callers
    reading from a localized copy (temp download, stored upload) pass the
    name derived from the ORIGINAL file so the manifest identity is stable
    across runs regardless of where the bytes were staged.
    """
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
    return source.with_name(source_name or csv_source_name(filename))


def _split_schema_and_table(qualified_name: str) -> tuple:
    """Split a possibly schema-qualified table ref into (schema, table).

    ``public.users`` -> (``public``, ``users``). A bare name has no schema.
    """
    if "." not in qualified_name:
        return None, qualified_name
    schema, table = qualified_name.rsplit(".", 1)
    return schema, table


def _parse_sql_query(query: str) -> tuple:
    """Extract table name and WHERE clause from a SELECT query.
    Returns (table_name, where_clause) or raises ValueError.

    Table names may be schema-qualified (``public.users``). ``\\w+`` alone
    stops at the first dot, which used to drop both the schema and the WHERE
    clause.

    A table alias (``users u`` / ``users AS u``) is allowed as long as the
    WHERE clause qualifies columns with the table name, not the alias: dlt's
    select is built on the bare table, so an alias reference (``u.age``)
    cannot be replayed and raises instead of silently ingesting the whole
    table. JOIN queries raise for the same reason — their WHERE spans tables
    the single-table source cannot express.
    """
    # Words that can follow a table name without being an alias.
    _RESERVED_AFTER_TABLE = (
        "WHERE",
        "JOIN",
        "INNER",
        "LEFT",
        "RIGHT",
        "FULL",
        "CROSS",
        "ORDER",
        "GROUP",
        "LIMIT",
        "UNION",
        "OFFSET",
        "HAVING",
    )
    alias_pattern = "|".join(_RESERVED_AFTER_TABLE)
    match = re.match(
        r"SELECT\s+.+?\s+FROM\s+"
        r"(?P<table>\w+(?:\.\w+)*)"
        rf"(?:\s+(?:AS\s+)?(?!(?:{alias_pattern})\b)(?P<alias>\w+))?"
        rf"(?P<join>\s+(?:(?:INNER|LEFT|RIGHT|FULL|CROSS)\s+)?JOIN\b)?"
        r"(?:\s+WHERE\s+(?P<where>.+))?",
        query.strip(),
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"Cannot parse SQL query: {query}")
    if match.group("join"):
        raise ValueError(
            "JOIN queries are not supported for filtered ingestion (the WHERE clause spans "
            f"tables this source cannot express): {query}. Ingest the table without a filter, "
            "or create a view and ingest that."
        )
    table_name = match.group("table")
    where_clause = match.group("where") or "1=1"
    alias = match.group("alias")
    if alias and re.search(rf"\b{re.escape(alias)}\.\w+", where_clause, re.IGNORECASE):
        raise ValueError(
            f"WHERE clause references the table alias '{alias}', but the filter is replayed against "
            f"the bare table '{table_name}'. Qualify columns with the table name instead: {query}"
        )
    return table_name, where_clause
