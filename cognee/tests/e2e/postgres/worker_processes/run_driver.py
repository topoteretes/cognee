"""Run a driver script on a PostgreSQL database of its own and read what it reports.

Shared by the tests that run pipelines through worker processes: the driver gets
the full PostgreSQL setup in its environment (the relational store, pgvector and
the Postgres graph adapter, with databases of its own for every dataset), mocked
embeddings and no model key. Every database the run created is dropped after it.
"""

import asyncio
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from sqlalchemy.engine import make_url


async def _databases(connect: dict) -> set[str]:
    import asyncpg

    admin = await asyncpg.connect(**connect)
    try:
        return {row["datname"] for row in await admin.fetch("select datname from pg_database")}
    finally:
        await admin.close()


async def _execute(connect: dict, statements: list[str]) -> None:
    import asyncpg

    admin = await asyncpg.connect(**connect)
    try:
        for statement in statements:
            await admin.execute(statement)
    finally:
        await admin.close()


@contextmanager
def driver_results(driver: Path, pg_url: str, root: Path) -> Iterator[dict]:
    """What ``driver`` reports (its last ``RESULT`` line), from runs on a database
    of their own on the server ``pg_url`` names, with their files under ``root``."""
    url = make_url(pg_url)
    connect = {
        "host": url.host,
        "port": url.port or 5432,
        "user": url.username,
        "password": url.password,
        "database": url.database or "postgres",
    }
    database = f"cognee_workers_{uuid4().hex[:10]}"
    existing = asyncio.run(_databases(connect))
    asyncio.run(_execute(connect, [f'create database "{database}"']))
    stores = {
        f"{prefix}_{name}": value
        for prefix in ("DB", "VECTOR_DB", "GRAPH_DATABASE")
        for name, value in (
            ("HOST", url.host),
            ("PORT", str(url.port or 5432)),
            ("USERNAME", url.username),
            ("PASSWORD", url.password),
            ("NAME", database),
        )
    }
    environment = {
        **os.environ,
        **stores,
        "DB_PROVIDER": "postgres",
        "VECTOR_DB_PROVIDER": "pgvector",
        "VECTOR_DATASET_DATABASE_HANDLER": "pgvector",
        "GRAPH_DATABASE_PROVIDER": "postgres",
        "GRAPH_DATASET_DATABASE_HANDLER": "postgres_graph",
        "ENABLE_BACKEND_ACCESS_CONTROL": "true",
        "DATA_ROOT_DIRECTORY": str(root / "data"),
        "SYSTEM_ROOT_DIRECTORY": str(root / "system"),
        "LLM_API_KEY": "not-used",
        "MOCK_EMBEDDING": "true",
        "COGNEE_SKIP_CONNECTION_TEST": "true",
        "SUMMARY_METHOD": "from_extraction",
        "TELEMETRY_DISABLED": "true",
    }
    for name in ("GRAPH_DATABASE_URL", "VECTOR_DB_URL", "EARLY_CHUNK_INDEXING", "GRAPH_EXTRACTOR"):
        environment.pop(name, None)
    try:
        run = subprocess.run(
            [sys.executable, str(driver)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        results = [line for line in run.stdout.splitlines() if line.startswith("RESULT ")]
        assert results, (
            f"the driver ended with code {run.returncode} and no result:\n"
            f"{run.stdout[-2000:]}\n{run.stderr[-6000:]}"
        )
        yield json.loads(results[-1].removeprefix("RESULT "))
    finally:
        created = asyncio.run(_databases(connect)) - existing
        asyncio.run(
            _execute(
                connect, [f'drop database if exists "{name}" with (force)' for name in created]
            )
        )
