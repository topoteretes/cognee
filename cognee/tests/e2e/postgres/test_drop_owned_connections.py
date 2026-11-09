"""Run against an explicitly opted-in disposable PostgreSQL, never a tenant DB."""

import os
from uuid import uuid4

import asyncpg
import pytest
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError

from cognee.infrastructure.databases.postgres import drop_pg_database_if_exists


@pytest.mark.asyncio
async def test_non_superuser_drop_leaves_foreign_role_connected():
    url = os.getenv("COGNEE_PG_TEST_URL")
    if not url:
        pytest.skip("set COGNEE_PG_TEST_URL to a disposable superuser Postgres")
    admin = await asyncpg.connect(url)
    suffix = uuid4().hex[:12]
    owner, foreign, database = (
        f"clo1521_{prefix}_{suffix}" for prefix in ("owner", "foreign", "db")
    )
    own_connection = foreign_connection = None
    try:
        await admin.execute(
            f"CREATE ROLE \"{owner}\" LOGIN NOSUPERUSER CREATEDB PASSWORD 'test-only'"
        )
        await admin.execute(f"CREATE ROLE \"{foreign}\" LOGIN NOSUPERUSER PASSWORD 'test-only'")
        await admin.execute(f'CREATE DATABASE "{database}" OWNER "{owner}"')
        parsed = make_url(url)
        params = {"host": parsed.host, "port": parsed.port or 5432, "password": "test-only"}
        own_connection = await asyncpg.connect(**params, user=owner, database=database)
        foreign_connection = await asyncpg.connect(**params, user=foreign, database=database)
        # The old helper raises 42501 (insufficient privilege) while trying to
        # terminate the other role. The fixed one reaches DROP, which honestly
        # reports 55006 (database in use) without touching that session.
        with pytest.raises(DBAPIError) as failure:
            await drop_pg_database_if_exists(database, username=owner, **params)
        assert failure.value.orig.sqlstate == "55006"
        assert await foreign_connection.fetchval("SELECT 1") == 1
        assert own_connection.is_closed()
        await foreign_connection.close()
        await drop_pg_database_if_exists(database, username=owner, **params)
        assert not await admin.fetchval("SELECT 1 FROM pg_database WHERE datname=$1", database)
        # Missing databases remain an idempotent successful delete.
        await drop_pg_database_if_exists(database, username=owner, **params)
    finally:
        for connection in (own_connection, foreign_connection):
            if connection is not None:
                await connection.close()
        await admin.execute(f'DROP DATABASE IF EXISTS "{database}"')
        await admin.execute(f'DROP ROLE IF EXISTS "{owner}"')
        await admin.execute(f'DROP ROLE IF EXISTS "{foreign}"')
        await admin.close()
