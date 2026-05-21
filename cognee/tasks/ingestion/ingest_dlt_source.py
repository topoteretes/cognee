import asyncio
import hashlib
import json
import os
import re
from collections import Counter
from typing import List, Optional

from sqlalchemy import URL, text
from sqlalchemy.ext.asyncio import create_async_engine

from cognee.modules.data.models import Data
from cognee.infrastructure.databases.relational.config import get_relational_config
from cognee.tasks.ingestion.dlt_row_data import DltRowData
from cognee.tasks.ingestion.exceptions.exceptions import (
    UnsupportedDBProviderError,
    DLTIngestionError,
    InvalidDLTArgumentError,
)
from cognee.tasks.ingestion.get_dlt_destination import get_dlt_destination
from cognee.shared.logging_utils import get_logger

try:
    import dlt
except ImportError:
    dlt = None

logger = get_logger("ingest_dlt_source")

# Strict identifier pattern — only allow alphanumerics, underscores, dots, and hyphens
_SAFE_IDENT_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


async def ingest_dlt_source(
    dlt_source,
    dataset_name: str,
    primary_key: Optional[str] = None,
    write_disposition: str = "replace",
    max_rows_per_table: Optional[int] = None,
) -> List[DltRowData]:
    """
    Ingests a dlt (re)source by running the dlt pipeline on it.
    Returns a list of DltRowData, one per row in the ingested tables.

    Supports both SQLite and PostgreSQL as the dlt destination, based on
    the cognee relational config.

    Parameters:
        dlt_source: The dlt resource or source to ingest.
        dataset_name: The name of the dataset.
        primary_key: Optional primary key column name. If not provided, auto-detected
                     from schema or defaults to 'id' or first column.
        write_disposition: DLT write disposition. One of:
            - "merge" (default): Upsert rows by primary key. Existing rows with the same
              PK are updated; new rows are inserted.
            - "append": Always insert new rows without deduplication.
            - "replace": Drop and recreate the table on each run.
    Returns:
        List[DltRowData]: Per-row data objects for downstream processing.
    """

    valid_dispositions = ("merge", "append", "replace")
    if write_disposition not in valid_dispositions:
        raise InvalidDLTArgumentError(
            message=f"Invalid write_disposition '{write_disposition}'. Must be one of: {valid_dispositions}"
        )

    original_dataset_name = dataset_name
    dataset_name = _to_safe_ident(dataset_name)

    relational_config = get_relational_config()
    dlt_db_name = f"dlt_database_{dataset_name}"

    if relational_config.db_provider == "postgres":
        await _create_pg_database(dlt_db_name)

    destination = get_dlt_destination(dlt_db_name=dlt_db_name)
    if destination is None:
        raise UnsupportedDBProviderError(
            message=f"Unsupported db_provider for DLT ingestion: {relational_config.db_provider}. "
            "Only 'sqlite' and 'postgres' are supported."
        )

    # Execute dlt pipeline with error handling
    pipeline = dlt.pipeline(
        pipeline_name="ingest_dlt_source",
        destination=destination,
        dataset_name=dataset_name,
    )

    # Build run kwargs based on disposition
    run_kwargs = {
        "write_disposition": write_disposition,
    }
    # Only pass primary_key for merge disposition (required for upsert).
    # When the user didn't provide an explicit primary_key, omit it so dlt
    # can auto-detect PKs from the source schema per table.
    if write_disposition == "merge" and primary_key:
        run_kwargs["primary_key"] = primary_key

    try:
        # dlt's pipeline.run() is synchronous and potentially long-running;
        # run it in a thread to avoid blocking the async event loop.
        load_info = await asyncio.to_thread(pipeline.run, dlt_source, **run_kwargs)
    except Exception as e:
        raise DLTIngestionError(
            message=f"DLT pipeline execution failed for dataset '{original_dataset_name}': {e}"
        ) from e

    # Validate load_info for failed jobs
    if load_info is not None:
        for package in load_info.load_packages:
            failed_jobs = [job for job in package.jobs.get("failed_jobs", [])]
            if failed_jobs:
                failure_messages = [
                    f"Table '{job.job_file_info.table_name}': {job.failed_message}"
                    for job in failed_jobs
                ]
                raise DLTIngestionError(
                    message=f"DLT load had {len(failed_jobs)} failed job(s) for dataset "
                    f"'{original_dataset_name}':\n" + "\n".join(failure_messages)
                )

    # Extract schema from the dlt database
    try:
        _, filtered_schema = await _extract_dlt_schema(relational_config, dlt_db_name, dataset_name)
    except Exception as e:
        raise DLTIngestionError(
            message=f"Failed to extract schema from DLT database '{dlt_db_name}': {e}"
        ) from e

    # Read rows from each table and produce DltRowData objects
    from cognee.tasks.ingestion.config import get_ingestion_config

    default_max_rows = get_ingestion_config().dlt_max_rows_per_table
    effective_max_rows = max_rows_per_table if max_rows_per_table is not None else default_max_rows
    try:
        row_data_list = await _read_rows_from_tables(
            dlt_db_name=dlt_db_name,
            dataset_name=dataset_name,
            schema=filtered_schema,
            primary_key=primary_key,
            relational_config=relational_config,
            max_rows_per_table=effective_max_rows,
        )
    except Exception as e:
        raise DLTIngestionError(
            message=f"Failed to read rows from DLT database '{dlt_db_name}': {e}"
        ) from e

    return row_data_list


async def _extract_dlt_schema(relational_config, dlt_db_name: str, dataset_name: str):
    """Extract and filter schema from the dlt-populated database."""
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    # Build engine directly instead of mutating the cached migration config
    # singleton — concurrent calls would clobber each other's settings.
    if relational_config.db_provider == "sqlite":
        dlt_sqlite_db_name = f"{dlt_db_name}__{dataset_name}"
        db_path = relational_config.db_path
        db_name = dlt_sqlite_db_name
    else:
        db_path = None
        db_name = dlt_db_name

    engine = create_relational_engine(
        db_path=db_path,
        db_name=db_name,
        db_host=relational_config.db_host,
        db_port=relational_config.db_port,
        db_username=relational_config.db_username,
        db_password=relational_config.db_password,
        db_provider=relational_config.db_provider,
    )
    schema = await engine.extract_schema()

    # Filter out dlt internal tables (those starting with _dlt_ or containing staging)
    filtered_schema = {k: v for k, v in schema.items() if "_dlt_" not in k and "staging" not in k}

    return schema, filtered_schema


def _quote_identifier(name: str) -> str:
    """Safely quote a SQL identifier to prevent injection.

    Validates the name against a strict pattern and double-quotes it.
    Raises ValueError if the name contains unexpected characters.
    """
    if not _SAFE_IDENT_RE.match(name):
        raise ValueError(
            f"Unsafe SQL identifier rejected: {name!r}. "
            "Only alphanumerics, underscores, dots, and hyphens are allowed."
        )
    # Escape any embedded double-quotes (defensive — regex above disallows them)
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _compute_schema_hash(schema_info) -> str:
    """Compute a stable hash of the schema structure for evolution detection."""
    if isinstance(schema_info, list):
        # SQLite format: [{"name": "col", "type": "TEXT"}, ...]
        normalized = [(c.get("name"), c.get("type")) for c in schema_info]
    elif isinstance(schema_info, dict):
        # Postgres format: {"col_name": {"type": "TEXT", ...}, ...}
        normalized = sorted((k, str(v)) for k, v in schema_info.items())
    else:
        normalized = str(schema_info)
    return hashlib.md5(json.dumps(normalized, sort_keys=True).encode()).hexdigest()


async def _read_rows_from_tables(
    dlt_db_name: str,
    dataset_name: str,
    schema: dict,
    primary_key: Optional[str],
    relational_config,
    max_rows_per_table: int = 50,
) -> List[DltRowData]:
    """Read rows from the dlt database tables and return DltRowData objects."""
    if relational_config.db_provider == "sqlite":
        # DLT creates a separate SQLite file: {db_name}__{dataset_name}
        dlt_sqlite_db_name = f"{dlt_db_name}__{dataset_name}"
        db_path = os.path.join(relational_config.db_path, dlt_sqlite_db_name)
        async_url = f"sqlite+aiosqlite:///{db_path}"
    else:
        async_url = URL.create(
            "postgresql+asyncpg",
            username=relational_config.db_username,
            password=relational_config.db_password,
            host=relational_config.db_host,
            port=int(relational_config.db_port),
            database=dlt_db_name,
        )

    async_engine = create_async_engine(async_url)
    row_data_list = []

    try:
        async with async_engine.connect() as conn:
            for table_name, table_info in schema.items():
                try:
                    table_rows = await _read_single_table(
                        conn=conn,
                        table_name=table_name,
                        table_info=table_info,
                        primary_key=primary_key,
                        dataset_name=dataset_name,
                        dlt_db_name=dlt_db_name,
                        relational_config=relational_config,
                        max_rows=max_rows_per_table,
                    )
                    row_data_list.extend(table_rows)
                except Exception as e:
                    logger.error(
                        "Failed to read table '%s' from DLT database '%s': %s",
                        table_name,
                        dlt_db_name,
                        e,
                    )
                    raise
    finally:
        await async_engine.dispose()

    return row_data_list


async def _read_single_table(
    conn,
    table_name: str,
    table_info: dict,
    primary_key: Optional[str],
    dataset_name: str,
    dlt_db_name: str,
    relational_config,
    max_rows: int = 50,
) -> List[DltRowData]:
    """Read rows from a single table and return DltRowData objects.

    At most ``max_rows`` rows are read.  Pass 0 or a negative value to
    read all rows (no limit).
    """
    raw_columns = table_info.get("columns", [])
    foreign_keys = table_info.get("foreign_keys", [])

    # extract_schema() returns columns as a list of dicts for SQLite
    # (e.g. [{"name": "col", "type": "TEXT"}, ...]) or a dict for Postgres
    if isinstance(raw_columns, list):
        column_names = [c["name"] for c in raw_columns]
    else:
        column_names = list(raw_columns.keys())

    # Auto-detect primary key with validation
    pk_col = _resolve_primary_key(primary_key, table_info, column_names, table_name)

    # Compute schema hash for evolution detection
    schema_hash = _compute_schema_hash(raw_columns)

    # For PostgreSQL, dlt uses dataset_name as schema prefix;
    # for SQLite, dlt uses a separate database file and tables have no prefix.
    if relational_config.db_provider == "sqlite":
        quoted_table = _quote_identifier(table_name)
        query = f"SELECT * FROM {quoted_table}"
    else:
        schema_name, table_name_only = table_name.split(".", 1)
        quoted_schema = _quote_identifier(schema_name)
        quoted_tbl = _quote_identifier(table_name_only)
        query = f"SELECT * FROM {quoted_schema}.{quoted_tbl}"

    if max_rows > 0:
        query += f" LIMIT {int(max_rows)}"

    result = await conn.execute(text(query))
    rows = result.mappings().all()

    # Validate PK uniqueness
    if rows:
        pk_values = [str(row.get(pk_col, "")) for row in rows]
        pk_counts = Counter(pk_values)
        duplicates = {v: c for v, c in pk_counts.items() if c > 1}
        if duplicates:
            dup_sample = dict(list(duplicates.items())[:5])
            logger.warning(
                "Table '%s': primary key column '%s' has %d duplicate values "
                "(sample: %s). Rows with duplicate PKs will overwrite each other "
                "during upsert. Consider specifying a unique primary_key.",
                table_name,
                pk_col,
                len(duplicates),
                dup_sample,
            )

    row_data_list = []
    for row in rows:
        row_dict = {k: v for k, v in row.items()}
        pk_value = str(row_dict.get(pk_col, ""))

        row_keys = list(row_dict.keys())
        for k in row_keys:
            if "_dlt_" in k:
                row_dict.pop(k)

        content_hash = hashlib.md5(
            json.dumps(row_dict, sort_keys=True, default=str).encode()
        ).hexdigest()

        row_data_list.append(
            DltRowData(
                table_name=table_name,
                primary_key_column=pk_col,
                primary_key_value=pk_value,
                row_data=row_dict,
                content_hash=content_hash,
                schema_info=raw_columns,
                schema_hash=schema_hash,
                foreign_keys=foreign_keys,
                dlt_db_name=dlt_db_name,
                dataset_name=dataset_name,
            )
        )

    return row_data_list


def _resolve_primary_key(
    provided_pk: Optional[str],
    table_info: dict,
    column_names: list,
    table_name: str = "",
) -> str:
    """Resolve the primary key column for a table with validation and logging."""
    if provided_pk and provided_pk in column_names:
        return provided_pk

    if provided_pk and provided_pk not in column_names:
        logger.warning(
            "Table '%s': provided primary_key '%s' not found in columns %s. "
            "Falling back to auto-detection.",
            table_name,
            provided_pk,
            column_names,
        )

    # Check schema-level primary_key
    schema_pk = table_info.get("primary_key")
    if schema_pk:
        if isinstance(schema_pk, list) and len(schema_pk) > 0:
            return schema_pk[0]
        if isinstance(schema_pk, str):
            return schema_pk

    # Fallback to 'id' column
    if "id" in column_names:
        logger.info("Table '%s': no explicit primary key found, using 'id' column.", table_name)
        return "id"

    # Last resort: first column (with warning)
    if column_names:
        logger.warning(
            "Table '%s': no primary key detected, falling back to first column '%s'. "
            "This may cause incorrect upsert behavior if values are not unique. "
            "Specify primary_key explicitly for reliable deduplication.",
            table_name,
            column_names[0],
        )
        return column_names[0]

    return "id"


async def _create_pg_database(db_name):
    relational_config = get_relational_config()
    maintenance_db_name = "postgres"
    maintenance_db_url = URL.create(
        "postgresql+asyncpg",
        username=relational_config.db_username,
        password=relational_config.db_password,
        host=relational_config.db_host,
        port=int(relational_config.db_port),
        database=maintenance_db_name,
    )
    maintenance_engine = create_async_engine(maintenance_db_url)

    try:
        connection = await maintenance_engine.connect()
        connection = await connection.execution_options(isolation_level="AUTOCOMMIT")
        exists_result = await connection.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
            {"db_name": db_name},
        )
        if exists_result.scalar() is None:
            await connection.execute(text(f'CREATE DATABASE "{db_name}";'))
        await connection.close()
    finally:
        await maintenance_engine.dispose()


def _to_safe_ident(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_").lower()
    if not s:
        raise InvalidDLTArgumentError(message="Invalid dataset name given for dlt ingestion.")
    if s[0].isdigit():
        s = f"d_{s}"
    return s[:63]


async def migrate_dlt_database(data: List[Data]):
    """Legacy function for migrating dlt database schema to graph database."""
    from cognee.tasks.ingestion.migrate_relational_database import migrate_relational_database
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
    from cognee.infrastructure.files.utils.open_data_file import open_data_file

    graph_engine = await get_graph_engine()

    data_object = data[0]

    async def read_json_file(path: str):
        async with open_data_file(path, "r", encoding="utf-8") as f:
            return json.load(f)

    schema = await read_json_file(data_object.raw_data_location)

    await migrate_relational_database(graph_db=graph_engine, schema=schema)
