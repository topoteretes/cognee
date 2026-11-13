"""Cognify on PostgreSQL in one process and through worker processes.

Run by ``test_cognify_in_worker_processes.py`` in a process of its own, with the
stores configured in its environment. Prints one JSON object (``RESULT ...``)
the test asserts on.

The datasets hold the same 25 documents:

- ``one_process``: cognified without workers.
- ``workers``: cognified by worker processes, in shards of 10, two workers at a
  time. With early chunk indexing on (the default) a cognify is two runs, the
  indexing run and the cognify run, and each sends its documents to workers.
- ``workers_on_an_empty_graph``: the same without early chunk indexing, so
  nothing is in the dataset's graph when the workers start writing to it.
- ``worker_lost``: the worker of one shard of the cognify run ends part way.
  The dataset is then cognified again.
"""

import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

import asyncpg
import stand_in_extraction
from sqlalchemy import select

import cognee
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.methods import get_dataset_data, get_datasets_by_name
from cognee.modules.engine.operations.setup import setup
from cognee.modules.pipelines.execution import PipelineRunSpec, ShardedExecutor
from cognee.modules.pipelines.execution.process_backend import ProcessWorkerBackend
from cognee.modules.pipelines.models import PipelineRunShard, PipelineRunShardItem
from cognee.modules.pipelines.models.DataItemStatus import is_item_stamped
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import DatasetDatabase
from cognee.shared.pipeline_names import COGNIFY_PIPELINE_NAME

WORKER = str(Path(__file__).resolve().parent / "worker.py")
DOCUMENTS = 25
SHARD_SIZE = 10
WORKERS_AT_A_TIME = 2
TOPICS = ["harbours", "rivers", "bridges", "markets", "mills", "orchards", "quarries"]


def texts() -> list[str]:
    return [
        f"Document number{index:02d} describes {TOPICS[index % 7]} and {TOPICS[(index + 1) % 7]}. "
        f"Travellers crossing {TOPICS[index % 7]} often reach {TOPICS[(index + 1) % 7]} in winter."
        for index in range(DOCUMENTS)
    ]


async def dataset_databases(dataset_id) -> set[str]:
    async with get_relational_engine().get_async_session() as session:
        row = await session.get(DatasetDatabase, dataset_id)
    return {row.vector_database_name, row.graph_database_name}


async def stored(dataset) -> dict:
    """Rows per table in the dataset's own databases, and what there names no source."""
    rows, without_sources = {}, {}
    for database in sorted(await dataset_databases(dataset.id)):
        connection = await asyncpg.connect(
            host=os.environ["DB_HOST"],
            port=int(os.environ["DB_PORT"]),
            user=os.environ["DB_USERNAME"],
            password=os.environ["DB_PASSWORD"],
            database=database,
        )
        try:
            tables = await connection.fetch(
                "select table_name from information_schema.tables where table_schema = 'public'"
            )
            for table in sorted(record["table_name"] for record in tables):
                rows[table] = await connection.fetchval(f'select count(*) from "{table}"')
                # A graph that keeps its sources in the graph: every node and
                # edge names the documents it is from.
                if table in ("graph_node", "graph_edge"):
                    without_sources[table] = await connection.fetchval(
                        f"select count(*) from {table} where cardinality(source_ref_keys) = 0"
                    )
        finally:
            await connection.close()
    return {"rows": rows, "without_sources": without_sources}


async def graph(dataset) -> dict:
    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        nodes, edges = await (await get_graph_engine()).get_graph_data()
    return {
        "nodes": sorted(f"{node[1].get('type')}:{node[1].get('name')}" for node in nodes),
        "edges": dict(sorted(Counter(str(edge[2]) for edge in edges).items())),
    }


async def shards(dataset) -> list[dict]:
    async with get_relational_engine().get_async_session() as session:
        rows = (
            (
                await session.execute(
                    select(PipelineRunShard)
                    .where(PipelineRunShard.dataset_id == dataset.id)
                    .order_by(PipelineRunShard.created_at, PipelineRunShard.shard_index)
                )
            )
            .scalars()
            .all()
        )
        described = []
        for row in rows:
            items = (
                (
                    await session.execute(
                        select(PipelineRunShardItem).where(PipelineRunShardItem.shard_id == row.id)
                    )
                )
                .scalars()
                .all()
            )
            described.append(
                {
                    "pipeline": PipelineRunSpec.from_json(row.spec).pipeline_name,
                    "index": row.shard_index,
                    "status": row.status,
                    "items": dict(sorted(Counter(item.status for item in items).items())),
                    "item_errors": sorted({i.error_class for i in items if i.error_class}),
                }
            )
    return described


async def cognify(name: str, user, **kwargs) -> dict:
    result = await cognee.cognify(datasets=[name], user=user, raise_on_error=False, **kwargs)
    [dataset] = await get_datasets_by_name([name], user.id)
    run = result[dataset.id]
    entries = run.data_ingestion_info or []
    data = await get_dataset_data(dataset.id)
    return {
        "run_status": run.status,
        "error_class": getattr(run, "error_class", None),
        "item_statuses": dict(sorted(Counter(e["run_info"].status for e in entries).items())),
        "item_errors": sorted(
            {
                e["run_info"].error_class
                for e in entries
                if getattr(e["run_info"], "error_class", None)
            }
        ),
        "stamped": sum(is_item_stamped(item, COGNIFY_PIPELINE_NAME, dataset.id) for item in data),
        "stored": await stored(dataset),
        "graph": await graph(dataset),
        "shards": await shards(dataset),
    }


async def main() -> None:
    stand_in_extraction.install()
    await setup()
    user = await get_default_user()
    for name in ("one_process", "workers", "workers_on_an_empty_graph", "worker_lost"):
        await cognee.add(texts(), dataset_name=name, user=user)

    executor = ShardedExecutor(
        ProcessWorkerBackend(WORKERS_AT_A_TIME, [sys.executable, WORKER]), SHARD_SIZE
    )
    out = {"one_process": await cognify("one_process", user)}
    out["workers"] = await cognify("workers", user, data_item_executor=executor)
    out["workers_on_an_empty_graph"] = await cognify(
        "workers_on_an_empty_graph", user, data_item_executor=executor, early_chunk_indexing=False
    )
    os.environ["WORKER_E2E_LOST_SHARD"] = "1"
    out["worker_lost"] = await cognify("worker_lost", user, data_item_executor=executor)
    del os.environ["WORKER_E2E_LOST_SHARD"]
    out["worker_lost_again"] = await cognify("worker_lost", user, data_item_executor=executor)
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    asyncio.run(main())
