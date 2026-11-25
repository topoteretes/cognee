"""Three things through worker processes, each also in one process to compare with.

Run by ``test_any_pipeline_in_worker_processes.py`` in a process of its own,
with the stores configured in its environment. Prints one JSON object
(``RESULT ...``) the test asserts on.

- ``notes_*``: a pipeline of its own, registered by name, over 60 items that are
  JSON values and no ``Data`` rows (``run_custom_pipeline``).
- ``model_*``: cognify of 25 documents with a custom graph model, built from its
  JSON schema as a model that arrives over the API is. ``model_in_python``: the
  same with the class written in Python, which keeps a run in its own process.
- ``csv_*``: cognify of twelve CSV files. Each is a DLT source of its own, one
  item of the run, so the workers share them like documents.
  ``csv_worker_lost``: the worker of one shard ends before it takes it; the
  dataset is then cognified again.
- ``mixed_*``: cognify of a dataset that holds the documents and the CSV files.

Workers take shards of 2 items (2 documents, 2 notes, 2 CSV files), two at a
time.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import any_pipeline
import stand_in_extraction
from driver import cognify, graph, shards, texts

import cognee
from cognee.modules.data.methods import get_datasets_by_name
from cognee.modules.engine.operations.setup import setup
from cognee.modules.pipelines.execution import (
    ShardedExecutor,
    registered_tasks,
    set_data_item_executor,
)
from cognee.modules.pipelines.execution.process_backend import ProcessWorkerBackend
from cognee.modules.users.methods import get_default_user
from cognee.shared.graph_model_utils import graph_model_to_graph_schema, graph_schema_to_graph_model

WORKER = str(Path(__file__).resolve().parent / "worker_any_pipeline.py")
SHARD_SIZE = 2
WORKERS_AT_A_TIME = 2


async def notes(name: str, user) -> dict:
    result = await cognee.run_custom_pipeline(
        tasks=await registered_tasks(any_pipeline.NOTES, user, label="kept"),
        data=any_pipeline.note_items(),
        dataset=name,
        user=user,
        pipeline_name="notes_pipeline",
    )
    [dataset] = await get_datasets_by_name([name], user.id)
    run = result[dataset.id]
    return {
        "run_status": run.status,
        "graph": await graph(dataset),
        "shards": await shards(dataset),
    }


async def main() -> None:
    stand_in_extraction.install(any_pipeline.extract)
    await setup()
    user = await get_default_user()
    executor = ShardedExecutor(
        ProcessWorkerBackend(WORKERS_AT_A_TIME, [sys.executable, WORKER]), SHARD_SIZE
    )
    out = {}

    out["notes_one_process"] = await notes("notes_one_process", user)
    # A custom pipeline has no argument for an executor: the process's is used.
    set_data_item_executor(executor)
    out["notes_workers"] = await notes("notes_workers", user)
    set_data_item_executor(None)

    for name in ("model_one_process", "model_workers", "model_in_python"):
        await cognee.add(texts(), dataset_name=name, user=user)
    page = graph_schema_to_graph_model(graph_model_to_graph_schema(any_pipeline.Page))
    out["model_one_process"] = await cognify("model_one_process", user, graph_model=page)
    out["model_workers"] = await cognify(
        "model_workers", user, graph_model=page, data_item_executor=executor
    )
    out["model_in_python"] = await cognify(
        "model_in_python", user, graph_model=any_pipeline.Page, data_item_executor=executor
    )

    csv_files = any_pipeline.write_csv_files(Path(tempfile.mkdtemp(prefix="csv_sources_")))
    for name in ("csv_one_process", "csv_workers", "csv_worker_lost"):
        await cognee.add(csv_files, dataset_name=name, user=user)
    out["csv_one_process"] = await cognify("csv_one_process", user)
    out["csv_workers"] = await cognify("csv_workers", user, data_item_executor=executor)
    os.environ["WORKER_E2E_LOST_SHARD"] = "1"
    out["csv_worker_lost"] = await cognify("csv_worker_lost", user, data_item_executor=executor)
    del os.environ["WORKER_E2E_LOST_SHARD"]
    out["csv_worker_lost_again"] = await cognify(
        "csv_worker_lost", user, data_item_executor=executor
    )

    for name in ("mixed_one_process", "mixed_workers"):
        await cognee.add(texts(), dataset_name=name, user=user)
        await cognee.add(csv_files, dataset_name=name, user=user)
    out["mixed_one_process"] = await cognify("mixed_one_process", user)
    out["mixed_workers"] = await cognify("mixed_workers", user, data_item_executor=executor)
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    asyncio.run(main())
