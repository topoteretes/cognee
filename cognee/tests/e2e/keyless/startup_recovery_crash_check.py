"""Keyless end to end: a cognify run killed mid-way, a second run, then startup recovery.

Proves what startup recovery does to real data. Run A is SIGKILLed the moment
its second document is recorded complete (no terminal row, no inline
rollback), run B in a fresh process finishes the rest and skips A's two
completed documents, then ``recover_stale_pipeline_runs_on_startup`` runs the
way the API lifespan runs it. Afterwards every document must still be
complete, the graph must hold exactly what it held before recovery, search
must still answer, and A must be closed as ERRORED with
``AbandonedPipelineRunError``. A whole-run rollback would instead delete A's
two completed documents' artifacts and clear their marks.

Needs ``cognee[gliner]`` (no LLM key). Run:
``python cognee/tests/e2e/keyless/startup_recovery_crash_check.py``
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

for key in list(os.environ):
    if key.startswith(("LLM_", "EMBEDDING_", "OPENAI_", "GRAPH_EXTRACTOR", "BAML_")):
        os.environ.pop(key)
for key in ("COGNEE_SKIP_PREFLIGHT", "COGNEE_SKIP_CONNECTION_TEST", "MOCK_EMBEDDING"):
    os.environ.pop(key, None)

ROOT = Path(__file__).resolve().parent / ".crash_run"
ROOT.mkdir(exist_ok=True)
os.chdir(ROOT)
os.environ["DATA_ROOT_DIRECTORY"] = str(ROOT / "data")
os.environ["SYSTEM_ROOT_DIRECTORY"] = str(ROOT / "system")
os.environ["AUTO_FEEDBACK"] = "false"
os.environ["TELEMETRY_DISABLED"] = "1"
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"  # one graph, so counts are comparable
os.environ["COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS"] = "0"  # the run is minutes old here

import dotenv  # noqa: E402

dotenv.load_dotenv = lambda *args, **kwargs: False

DATASET = "startup_recovery_crash"
DOCS = [
    "Northwind Robotics is headquartered in Rotterdam. Mara Lindqvist is its chief executive.",
    "Tomas Reyes leads engineering at Northwind Robotics and reports to Mara Lindqvist.",
    "Priya Nair is head of sales at Northwind Robotics and works with Tomas Reyes on demos.",
    "Northwind Robotics opened an office in Lisbon, managed by Joao Carvalho.",
    "Joao Carvalho hired Elena Petrova as the first Lisbon engineer in 2024.",
    "Elena Petrova and Tomas Reyes co-authored the Northwind Robotics safety standard.",
]
CRASH_AFTER_DOCUMENTS = 2
STATUS_PREFIX = "STATUS "


async def stage(name: str) -> None:
    from sqlalchemy import select

    import cognee
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data
    from cognee.modules.pipelines.models import PipelineRun
    from cognee.modules.pipelines.models.DataItemStatus import is_data_item_completed

    if name == "seed":
        await cognee.add(DOCS, dataset_name=DATASET)
        return
    if name in ("run", "run_crash"):
        if name == "run_crash":
            from cognee.modules.pipelines.operations import run_tasks as run_tasks_module

            real_progress = run_tasks_module.log_pipeline_run_progress
            done = {"count": 0}

            async def crash_after_n_documents(**kwargs):
                await real_progress(**kwargs)
                done["count"] += 1
                if done["count"] == CRASH_AFTER_DOCUMENTS:
                    os.kill(os.getpid(), signal.SIGKILL)

            run_tasks_module.log_pipeline_run_progress = crash_after_n_documents
        await cognee.cognify(datasets=[DATASET], data_per_batch=1)
        return
    if name == "recover":
        from cognee.modules.cognify.recovery import recover_stale_pipeline_runs_on_startup

        await recover_stale_pipeline_runs_on_startup()
        return

    # "snapshot": everything the assertions compare
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        records = (await session.execute(select(Data))).scalars().all()
        runs = (
            (
                await session.execute(
                    select(PipelineRun)
                    .filter(PipelineRun.pipeline_name == "cognify_pipeline")
                    .order_by(PipelineRun.created_at)
                )
            )
            .scalars()
            .all()
        )
    dataset_id = str(runs[0].dataset_id) if runs else None
    completed = sorted(
        str(record.id)
        for record in records
        if is_data_item_completed(
            ((record.pipeline_status or {}).get("cognify_pipeline") or {}).get(dataset_id)
        )
    )
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph = await get_graph_engine()
    nodes, edges = await graph.get_graph_data()
    search_result = await cognee.search(
        query_type=cognee.SearchType.CHUNKS, query_text="Lisbon office", datasets=[DATASET]
    )
    print(
        STATUS_PREFIX
        + json.dumps(
            {
                "completed": completed,
                "runs": [
                    [str(run.pipeline_run_id), run.status.name, run.error_class] for run in runs
                ],
                "node_ids": sorted(str(node_id) for node_id, _ in nodes),
                "edge_count": len([edge for edge in edges if edge[0] != edge[1]]),
                "search_hits": len(search_result) if isinstance(search_result, list) else 1,
            }
        )
    )


def run_stage(name: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, __file__, name],
        env=os.environ,
        capture_output=True,
        text=True,
        check=False,
    )


def snapshot() -> dict:
    result = run_stage("snapshot")
    lines = [line for line in result.stdout.splitlines() if line.startswith(STATUS_PREFIX)]
    assert lines, f"snapshot stage printed no status:\n{result.stderr[-2000:]}"
    return json.loads(lines[-1][len(STATUS_PREFIX) :])


def main() -> None:
    import shutil

    shutil.rmtree(ROOT / "data", ignore_errors=True)
    shutil.rmtree(ROOT / "system", ignore_errors=True)

    seed = run_stage("seed")
    assert seed.returncode == 0, f"seed failed:\n{seed.stderr[-2000:]}"

    crashed = run_stage("run_crash")
    assert crashed.returncode == -signal.SIGKILL, (
        f"run A should have been SIGKILLed mid-run, exited {crashed.returncode}:\n"
        f"{crashed.stderr[-2000:]}"
    )
    after_crash = snapshot()
    assert len(after_crash["completed"]) == CRASH_AFTER_DOCUMENTS, after_crash["completed"]
    assert [run[1] for run in after_crash["runs"]] == ["DATASET_PROCESSING_STARTED"], after_crash[
        "runs"
    ]
    abandoned_run_id = after_crash["runs"][0][0]

    second = run_stage("run")
    assert second.returncode == 0, f"run B failed:\n{second.stderr[-2000:]}"
    before = snapshot()
    assert len(before["completed"]) == len(DOCS), before["completed"]
    assert before["search_hits"] > 0

    recovered = run_stage("recover")
    assert recovered.returncode == 0, f"recovery failed:\n{recovered.stderr[-2000:]}"
    after = snapshot()

    lost = set(before["node_ids"]) - set(after["node_ids"])
    assert not lost, f"recovery deleted {len(lost)} node(s) a completed document owned"
    assert after["edge_count"] == before["edge_count"], (before["edge_count"], after["edge_count"])
    assert after["completed"] == before["completed"], "recovery cleared a completed document"
    assert after["search_hits"] == before["search_hits"]
    closing = [run for run in after["runs"] if run[0] == abandoned_run_id]
    assert closing[-1][1] == "DATASET_PROCESSING_ERRORED", closing
    assert closing[-1][2] == "AbandonedPipelineRunError", closing

    print(
        "Startup recovery crash check passed: "
        f"{len(after['node_ids'])} nodes and {after['edge_count']} edges kept, "
        f"{len(after['completed'])}/{len(DOCS)} documents complete, run A closed as abandoned"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        asyncio.run(stage(sys.argv[1]))
    else:
        main()
