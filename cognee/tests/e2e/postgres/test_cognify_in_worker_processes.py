"""A cognify run through worker processes stores what a run in one process stores.

On the full PostgreSQL setup, the only one workers are supported on: the
relational store, pgvector and the Postgres graph adapter, with databases of
its own for every dataset. ``worker_processes/driver.py`` does the runs in a
process of its own, so they get their own store configuration, with real worker
processes (``ProcessWorkerBackend``). Graph extraction is a stand-in and
embeddings are mocked, so no model and no key is needed.

Skipped unless COGNEE_PG_TEST_URL names a PostgreSQL server with pgvector whose
user may create databases, e.g.:

    COGNEE_PG_TEST_URL=postgres://cognee:cognee@localhost:5432/cognee_db
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy.engine import make_url

from cognee.shared.pipeline_names import COGNIFY_PIPELINE_NAME, INDEXING_PIPELINE_NAME

PG_URL = os.getenv("COGNEE_PG_TEST_URL")

pytestmark = pytest.mark.skipif(
    not PG_URL, reason="COGNEE_PG_TEST_URL not set; skipping the worker processes test"
)

DRIVER = Path(__file__).resolve().parent / "worker_processes" / "driver.py"
NOTHING_WITHOUT_A_SOURCE = {"graph_edge": 0, "graph_node": 0}


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


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """What the driver reports, from runs on a database of their own."""
    url = make_url(PG_URL)
    connect = {
        "host": url.host,
        "port": url.port or 5432,
        "user": url.username,
        "password": url.password,
        "database": url.database or "postgres",
    }
    root = tmp_path_factory.mktemp("worker_processes")
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
        driver = subprocess.run(
            [sys.executable, str(DRIVER)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        results = [line for line in driver.stdout.splitlines() if line.startswith("RESULT ")]
        assert results, (
            f"the driver ended with code {driver.returncode} and no result:\n"
            f"{driver.stdout[-2000:]}\n{driver.stderr[-6000:]}"
        )
        yield json.loads(results[-1].removeprefix("RESULT "))
    finally:
        created = asyncio.run(_databases(connect)) - existing
        asyncio.run(
            _execute(
                connect, [f'drop database if exists "{name}" with (force)' for name in created]
            )
        )


def _shards(run: dict, pipeline_name: str) -> list[tuple]:
    return [
        (shard["index"], shard["status"], shard["items"])
        for shard in run["shards"]
        if shard["pipeline"] == pipeline_name
    ]


def test_one_process_stores_a_graph_with_sources(runs):
    """What the other runs are compared with has to be a real result."""
    one_process = runs["one_process"]
    assert one_process["run_status"] == "PipelineRunCompleted"
    assert one_process["stamped"] == 25
    assert one_process["shards"] == []
    assert one_process["stored"]["rows"]["graph_node"] > 25
    assert one_process["stored"]["without_sources"] == NOTHING_WITHOUT_A_SOURCE


def test_a_run_through_workers_stores_what_one_process_stores(runs):
    workers, one_process = runs["workers"], runs["one_process"]
    assert workers["run_status"] == "PipelineRunCompleted"
    assert workers["item_statuses"] == {"PipelineRunCompleted": 25}
    assert workers["stamped"] == 25
    assert workers["stored"] == one_process["stored"]
    assert workers["graph"] == one_process["graph"]


def test_both_runs_of_a_cognify_send_their_documents_to_workers(runs):
    """With early chunk indexing on, the indexing run has a round of workers of
    its own before the cognify run's."""
    finished = [
        (0, "finished", {"completed": 10}),
        (1, "finished", {"completed": 10}),
        (2, "finished", {"completed": 5}),
    ]
    assert _shards(runs["workers"], INDEXING_PIPELINE_NAME) == finished
    assert _shards(runs["workers"], COGNIFY_PIPELINE_NAME) == finished


def test_workers_that_start_on_an_empty_graph_all_store_their_sources(runs):
    """Without early chunk indexing the workers are the first to write to the
    dataset's graph, several at once. Every node and edge they store has to
    name the documents it is from, which no row count shows."""
    on_empty_graph, one_process = runs["workers_on_an_empty_graph"], runs["one_process"]
    assert on_empty_graph["run_status"] == "PipelineRunCompleted"
    assert _shards(on_empty_graph, INDEXING_PIPELINE_NAME) == []
    assert [status for _, status, _ in _shards(on_empty_graph, COGNIFY_PIPELINE_NAME)] == [
        "finished"
    ] * 3
    assert on_empty_graph["stored"]["without_sources"] == NOTHING_WITHOUT_A_SOURCE
    assert on_empty_graph["graph"] == one_process["graph"]


def test_a_lost_worker_costs_only_the_documents_it_had_not_finished(runs):
    lost = runs["worker_lost"]
    assert lost["run_status"] == "PipelineRunErrored"
    assert lost["error_class"] == "ShardWorkerLost"
    assert lost["item_errors"] == ["ShardWorkerLost"]
    completed = lost["item_statuses"].get("PipelineRunCompleted", 0)
    assert completed + lost["item_statuses"].get("PipelineRunErrored", 0) == 25
    # The other two shards whole, and what the lost worker finished before it ended.
    assert 15 <= completed < 25
    assert lost["stamped"] == completed
    statuses = {index: status for index, status, _ in _shards(lost, COGNIFY_PIPELINE_NAME)}
    assert statuses == {0: "finished", 1: "lost", 2: "finished"}
    assert lost["stored"]["without_sources"] == NOTHING_WITHOUT_A_SOURCE


def test_the_next_cognify_finishes_what_a_lost_worker_left(runs):
    again, one_process = runs["worker_lost_again"], runs["one_process"]
    assert again["run_status"] == "PipelineRunCompleted"
    assert again["stamped"] == 25
    assert again["stored"] == one_process["stored"]
    assert again["graph"] == one_process["graph"]
