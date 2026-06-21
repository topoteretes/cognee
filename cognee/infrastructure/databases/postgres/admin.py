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

from sqlalchemy import URL, text
from sqlalchemy.ext.asyncio import create_async_engine


_MAINTENANCE_DB_NAME = "postgres"


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
