"""The Postgres create helpers must be idempotent under concurrent callers (SDK-603).

``CREATE DATABASE`` has no ``IF NOT EXISTS`` in Postgres, and even
``CREATE SCHEMA/EXTENSION IF NOT EXISTS`` can fail with a unique violation when
two transactions race. Both helpers back the first-use provisioning of a
dataset's databases, which a query can trigger at the same moment as the first
ingestion, so concurrent calls must all succeed and create the resource once.

Runs against a live Postgres (default: cognee:cognee@localhost:5432/cognee_db)
and skips if it is unreachable.
"""

import asyncio
import logging
import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from cognee.infrastructure.databases.postgres import (
    create_pg_database_if_not_exists,
    create_pg_schema_if_not_exists,
    drop_pg_database_if_exists,
    drop_pg_schema_if_exists,
)

logger = logging.getLogger(__name__)

CONCURRENCY = 6


def _db() -> dict:
    return {
        "host": os.environ.get("DB_HOST", "localhost"),
        "port": os.environ.get("DB_PORT", "5432"),
        "username": os.environ.get("DB_USERNAME", "cognee"),
        "password": os.environ.get("DB_PASSWORD", "cognee"),
        "name": os.environ.get("DB_NAME", "cognee_db"),
    }


def _url(database: str) -> str:
    d = _db()
    return (
        f"postgresql+asyncpg://{d['username']}:{d['password']}@{d['host']}:{d['port']}/{database}"
    )


async def _postgres_reachable() -> bool:
    engine = create_async_engine(_url(_db()["name"]))
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except Exception:
        # Any failure to connect means the suite skips, and the cause goes
        # to the log with its traceback rather than being swallowed.
        logger.debug("Postgres not reachable; skipping", exc_info=True)
        return False
    finally:
        await engine.dispose()


async def _count(database: str, query: str, **params) -> int:
    engine = create_async_engine(_url(database))
    try:
        async with engine.connect() as connection:
            return (await connection.execute(text(query), params)).scalar()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_create_database_creates_it_once():
    if not await _postgres_reachable():
        pytest.skip("Postgres not reachable")
    d = _db()
    creds = {k: d[k] for k in ("host", "port", "username", "password")}
    db_name = f"sdk603_{uuid.uuid4().hex[:12]}"

    try:
        created = await asyncio.gather(
            *(create_pg_database_if_not_exists(db_name, **creds) for _ in range(CONCURRENCY))
        )

        assert sorted(created) == [False] * (CONCURRENCY - 1) + [True]
        assert (
            await _count(
                d["name"], "SELECT count(*) FROM pg_database WHERE datname = :n", n=db_name
            )
            == 1
        )
    finally:
        await drop_pg_database_if_exists(db_name, **creds)


@pytest.mark.asyncio
async def test_concurrent_create_schema_creates_it_once():
    if not await _postgres_reachable():
        pytest.skip("Postgres not reachable")
    d = _db()
    creds = {k: d[k] for k in ("host", "port", "username", "password")}
    schema = f"ds_sdk603_{uuid.uuid4().hex[:12]}"

    try:
        await asyncio.gather(
            *(
                create_pg_schema_if_not_exists(
                    d["name"], schema, with_vector_extension=True, **creds
                )
                for _ in range(CONCURRENCY)
            )
        )

        assert (
            await _count(
                d["name"], "SELECT count(*) FROM pg_namespace WHERE nspname = :n", n=schema
            )
            == 1
        )
    finally:
        await drop_pg_schema_if_exists(d["name"], schema, **creds)
