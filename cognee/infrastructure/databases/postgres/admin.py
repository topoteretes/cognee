"""Postgres admin operations against the cluster's 'postgres' maintenance database.

These helpers handle the boilerplate of opening an AUTOCOMMIT connection to the
admin DB, running a CREATE/DROP DATABASE statement, and disposing of the
maintenance engine. Callers supply connection credentials directly; this module
is intentionally agnostic about which subsystem (graph, vector, dlt) the
credentials came from.
"""

import json
import os
from typing import Union
from uuid import UUID

from sqlalchemy import URL, text
from sqlalchemy.ext.asyncio import create_async_engine


_MAINTENANCE_DB_NAME = "postgres"


def dataset_schema_name(dataset_id: Union[UUID, str]) -> str:
    """Postgres schema name used to isolate a dataset in shared-database mode.

    Returns ``ds_<dataset_id_hex>`` — a valid, lower-case Postgres identifier
    (``ds_`` + 32 hex chars = 35 chars, well under the 63-char limit). The
    leading ``ds_`` keeps the name a valid identifier even though a raw UUID
    starts with a digit, and namespaces these schemas so they never collide
    with ``public`` or cognee's relational tables.
    """
    raw = dataset_id.hex if isinstance(dataset_id, UUID) else str(dataset_id).replace("-", "")
    return f"ds_{raw}"


def _admin_connect_args() -> dict:
    """SSL/connect args for the maintenance engine, from DATABASE_CONNECT_ARGS.

    Managed Postgres (e.g. Neon, RDS) requires SSL,
    but asyncpg uses ``ssl`` not ``sslmode``. Returns {} for in-cluster Postgres
    (env unset) so this stays a no-op there.
    """
    raw = os.environ.get("DATABASE_CONNECT_ARGS")
    if not raw:
        return {}
    try:
        args = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    out = {}
    if args.get("ssl") is not None:
        out["ssl"] = args["ssl"]
    elif args.get("sslmode") is not None:  # map libpq sslmode -> asyncpg ssl
        out["ssl"] = args["sslmode"]
    return out


def _direct_host(host: str) -> str:
    """CREATE/DROP DATABASE can't run through Neon's PgBouncer pooler — use the
    direct endpoint. No-op for non-Neon hosts."""
    return host.replace("-pooler.", ".") if host and "-pooler." in host else host


def _build_maintenance_url(host: str, port: Union[int, str], username: str, password: str) -> URL:
    return URL.create(
        "postgresql+asyncpg",
        username=username,
        password=password,
        host=_direct_host(host),
        port=int(port),
        database=_MAINTENANCE_DB_NAME,
    )


def _build_db_url(
    db_name: str, host: str, port: Union[int, str], username: str, password: str
) -> URL:
    """Connection URL to a specific (already existing) database.

    Unlike ``_build_maintenance_url`` this targets ``db_name`` directly, since
    schema DDL (CREATE/DROP SCHEMA) runs inside the target database rather than
    against the cluster's maintenance database.
    """
    return URL.create(
        "postgresql+asyncpg",
        username=username,
        password=password,
        host=_direct_host(host),
        port=int(port),
        database=db_name,
    )


async def create_pg_schema_if_not_exists(
    db_name: str,
    schema: str,
    host: str,
    port: Union[int, str],
    username: str,
    password: str,
    with_vector_extension: bool = False,
) -> None:
    """Create a Postgres schema inside ``db_name`` if it does not already exist.

    Used by the shared-database dataset handlers, which isolate each dataset in
    its own schema (``ds_<dataset_id>``) inside one shared database instead of
    provisioning a whole database per dataset. When ``with_vector_extension`` is
    set the pgvector extension is ensured in the database (it installs into the
    database's ``public`` schema and is reachable from any schema via the
    search path).
    """
    engine = create_async_engine(
        _build_db_url(db_name, host, port, username, password),
        connect_args=_admin_connect_args(),
    )
    try:
        async with engine.begin() as connection:
            if with_vector_extension:
                await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}";'))
    finally:
        await engine.dispose()


async def drop_pg_schema_if_exists(
    db_name: str,
    schema: str,
    host: str,
    port: Union[int, str],
    username: str,
    password: str,
) -> None:
    """Drop a dataset schema (and everything in it) from ``db_name``.

    ``DROP SCHEMA ... CASCADE`` removes every table/index the dataset created,
    giving the shared-database handlers an atomic, single-statement cleanup that
    mirrors ``drop_pg_database_if_exists`` for the database-per-dataset mode.
    """
    engine = create_async_engine(
        _build_db_url(db_name, host, port, username, password),
        connect_args=_admin_connect_args(),
    )
    try:
        async with engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE;'))
    finally:
        await engine.dispose()


async def create_pg_database_if_not_exists(
    db_name: str,
    host: str,
    port: Union[int, str],
    username: str,
    password: str,
) -> bool:
    """Create a Postgres database if it does not already exist.

    Connects to the cluster's 'postgres' maintenance database in AUTOCOMMIT
    mode and runs ``CREATE DATABASE`` guarded by an existence check.

    Returns:
        True if the database was created, False if it already existed.
    """
    engine = create_async_engine(
        _build_maintenance_url(host, port, username, password),
        connect_args=_admin_connect_args(),
    )
    try:
        connection = await engine.connect()
        try:
            connection = await connection.execution_options(isolation_level="AUTOCOMMIT")
            result = await connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :db"),
                {"db": db_name},
            )
            if result.scalar():
                return False
            await connection.execute(text(f'CREATE DATABASE "{db_name}";'))
            return True
        finally:
            await connection.close()
    finally:
        await engine.dispose()


async def drop_pg_database_if_exists(
    db_name: str,
    host: str,
    port: Union[int, str],
    username: str,
    password: str,
) -> None:
    """Drop a Postgres database after terminating any active backends.

    Connects to the cluster's 'postgres' maintenance database in AUTOCOMMIT
    mode, terminates every backend currently connected to ``db_name`` other
    than our own, then runs ``DROP DATABASE IF EXISTS``.
    """
    engine = create_async_engine(
        _build_maintenance_url(host, port, username, password),
        connect_args=_admin_connect_args(),
    )
    try:
        connection = await engine.connect()
        try:
            connection = await connection.execution_options(isolation_level="AUTOCOMMIT")
            await connection.execute(
                text(
                    "SELECT pg_terminate_backend(pid) "
                    "FROM pg_stat_activity "
                    "WHERE datname = :db AND pid <> pg_backend_pid()"
                ),
                {"db": db_name},
            )
            await connection.execute(text(f'DROP DATABASE IF EXISTS "{db_name}";'))
        finally:
            await connection.close()
    finally:
        await engine.dispose()
