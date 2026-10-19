import csv
import os
import re

from .dlt_utils import SOURCE_LABEL_ATTR

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
    query: str | None = None,
):
    """Auto-generate a dlt source from a database connection string with optional SQL query filtering.

    ``resolve_foreign_keys`` makes dlt record the source's foreign keys in its schema;
    the staging tables dlt creates carry no key constraints to reflect them from.
    """
    import sqlalchemy
    from dlt.sources.sql_database import sql_database

    if connection_string.startswith("sqlite:///"):
        connection_string = _allowed_sqlite_connection_string(connection_string)

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
            "resolve_foreign_keys": True,
        }
        if schema_name:
            source_kwargs["schema"] = schema_name

        source = sql_database(**source_kwargs)
    else:
        source = sql_database(credentials=connection_string, resolve_foreign_keys=True)

    setattr(source, SOURCE_LABEL_ATTR, _database_label(connection_string))
    return source


def _database_label(connection_string: str) -> str:
    """The database a connection string names, without host or credentials: a
    SQLite file's name, or the server database's name."""
    from sqlalchemy.engine import make_url

    url = make_url(connection_string)
    database = url.database or ""
    return os.path.basename(database) if url.get_backend_name() == "sqlite" else database


def _allowed_sqlite_connection_string(connection_string: str) -> str:
    """Return the SQLite connection string with its file path made absolute, or
    raise when the file is one cognee must not read as a source.

    A SQLite file has no credentials, so whoever names it reads it. Refused:
    files inside cognee's own storage (every tenant's users, ACLs and data), and
    files outside COGNEE_ALLOWED_LOCAL_FILE_ROOTS when that allowlist is set.
    """
    from sqlalchemy.engine import make_url

    from cognee.exceptions import CogneeValidationError
    from cognee.infrastructure.files.utils.local_path_safety import (
        get_cognee_internal_roots,
        is_within_roots,
        resolve_local_path,
    )

    # Parsed with the same parser the connection uses, so the path checked here
    # is the path that gets opened.
    url = make_url(connection_string)
    # `uri=true` makes SQLite read the path as a `file:` URI, a second path
    # syntax; refuse it rather than parse it.
    if "uri" in url.query:
        raise CogneeValidationError(
            message="SQLite connection strings with the 'uri' option are not supported.",
            name="SqliteSourceRefused",
        )
    if not url.database:
        # In-memory database: there is no file to read.
        return connection_string

    # realpath, not abspath: a symlink is resolved to the file it points at.
    database_path = os.path.realpath(os.path.expanduser(url.database))
    if is_within_roots(database_path, get_cognee_internal_roots()):
        raise CogneeValidationError(
            message="This SQLite file is part of cognee's own storage and cannot be "
            "ingested as a source.",
            name="SqliteSourceRefused",
        )
    try:
        resolve_local_path(database_path)
    except ValueError as error:
        raise CogneeValidationError(
            message="This SQLite file is outside COGNEE_ALLOWED_LOCAL_FILE_ROOTS.",
            name="SqliteSourceRefused",
        ) from error

    return url.set(database=database_path).render_as_string(hide_password=False)


CSV_DELIMITER_CANDIDATES = (",", ";", "\t", "|")


def detect_csv_delimiter(csv_path: str, sample_lines: int = 20) -> str:
    """Pick the delimiter that splits the first lines into a consistent number
    of columns. Candidates are tried in order, so a comma wins ties.

    ``csv.Sniffer`` is not used: it counts characters, so a comma-heavy text
    column (JSON lists, prose) makes it pick "," for a semicolon-delimited
    file. Parsing with ``csv.reader`` respects quoting, and a wrong delimiter
    shows up as a ragged column count. Falls back to "," when nothing fits.
    """
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as handle:
        lines = [line for _, line in zip(range(sample_lines), handle) if line.strip()]
    if not lines:
        return ","

    for delimiter in CSV_DELIMITER_CANDIDATES:
        try:
            counts = {len(row) for row in csv.reader(lines, delimiter=delimiter)}
        except csv.Error:
            continue
        if len(counts) == 1 and counts.pop() > 1:
            return delimiter
    return ","


def create_dlt_source_from_csv(csv_path: str, source_name: str | None = None):
    """Auto-generate a dlt resource from a CSV file path.

    ``source_name`` overrides the filename-derived resource name — callers
    reading from a localized copy (temp download, stored upload) pass the
    name derived from the ORIGINAL file so the manifest identity is stable
    across runs regardless of where the bytes were staged.

    The delimiter is detected from the file (see :func:`detect_csv_delimiter`)
    and forwarded to pandas through dlt's ``read_csv`` — semicolon, tab, and
    pipe files load as tables instead of failing on a ragged comma split.
    """
    from dlt.sources.filesystem import filesystem, read_csv

    parent_dir = os.path.dirname(os.path.abspath(csv_path))
    filename = os.path.basename(csv_path)

    source = filesystem(
        bucket_url=f"file://{parent_dir}",
        file_glob=filename,
    ) | read_csv(sep=detect_csv_delimiter(csv_path))
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
