"""Worker processes run more than cognify's files, and store what one process stores.

Three things go through real worker processes (``ProcessWorkerBackend``) on the
full PostgreSQL setup, each next to a run in one process to compare with
(``worker_processes/driver_any_pipeline.py``):

- a pipeline of its own, registered by name, over items that are JSON values;
- cognify with a custom graph model;
- cognify of twelve CSV files, each a DLT source of its own and one item of the
  run, which the workers share like documents.

Graph extraction is a stand-in and embeddings are mocked, so no model and no
key is needed.

Skipped unless COGNEE_PG_TEST_URL names a PostgreSQL server with pgvector whose
user may create databases, e.g.:

    COGNEE_PG_TEST_URL=postgres://cognee:cognee@localhost:5432/cognee_db
"""

import os
from collections import Counter
from pathlib import Path

import pytest

from cognee.shared.pipeline_names import COGNIFY_PIPELINE_NAME
from cognee.tests.e2e.postgres.worker_processes.run_driver import driver_results

PG_URL = os.getenv("COGNEE_PG_TEST_URL")

pytestmark = pytest.mark.skipif(
    not PG_URL, reason="COGNEE_PG_TEST_URL not set; skipping the worker processes test"
)

DRIVER = Path(__file__).resolve().parent / "worker_processes" / "driver_any_pipeline.py"
NOTHING_WITHOUT_A_SOURCE = {"graph_edge": 0, "graph_node": 0}
NOTES, DOCUMENTS, CSV_FILES, ROWS_PER_FILE = 60, 25, 12, 30


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """What the driver reports, from runs on a database of their own."""
    with driver_results(DRIVER, PG_URL, tmp_path_factory.mktemp("any_pipeline")) as results:
        yield results


def _shards(run: dict, pipeline_name: str) -> list[tuple]:
    """``(status, items by how they ended)`` of the run's shards of a pipeline,
    in an order of their own: workers end in any order."""
    return sorted(
        (
            (shard["status"], shard["items"])
            for shard in run["shards"]
            if shard["pipeline"] == pipeline_name
        ),
        key=str,
    )


def _nodes_by_type(run: dict) -> dict:
    return dict(Counter(node.split(":", 1)[0] for node in run["graph"]["nodes"]))


def _graph(run: dict, dataset: str) -> dict:
    """A run's graph with its dataset's name taken out of the node names, which
    a DLT source's nodes carry: the same sources in two datasets then compare."""
    return {
        "nodes": sorted(node.replace(dataset, "dataset") for node in run["graph"]["nodes"]),
        "edges": run["graph"]["edges"],
    }


# ---- a pipeline of its own, over JSON values ----


def test_a_registered_pipeline_runs_its_json_items_in_workers(runs):
    one_process, workers = runs["notes_one_process"], runs["notes_workers"]
    assert one_process["run_status"] == "PipelineRunCompleted"
    assert one_process["shards"] == []
    assert _nodes_by_type(one_process) == {"Note": NOTES}

    assert workers["run_status"] == "PipelineRunCompleted"
    # 60 values in shards of 2: every one answered for by a worker.
    assert _shards(workers, "notes_pipeline") == [("finished", {"completed": 2})] * 30
    assert workers["graph"] == one_process["graph"]


# ---- cognify with a custom graph model ----


def test_a_graph_model_built_from_a_schema_is_used_by_the_workers(runs):
    one_process, workers = runs["model_one_process"], runs["model_workers"]
    assert one_process["run_status"] == "PipelineRunCompleted"
    assert _shards(one_process, COGNIFY_PIPELINE_NAME) == []
    # The custom model's own classes, not the default model's entities.
    assert _nodes_by_type(one_process)["Page"] == DOCUMENTS
    assert _nodes_by_type(one_process)["Topic"] > DOCUMENTS

    assert workers["run_status"] == "PipelineRunCompleted"
    assert workers["item_statuses"] == {"PipelineRunCompleted": DOCUMENTS}
    assert _shards(workers, COGNIFY_PIPELINE_NAME) == (
        [("finished", {"completed": 1})] + [("finished", {"completed": 2})] * 12
    )
    assert workers["graph"] == one_process["graph"]
    assert workers["stored"]["without_sources"] == NOTHING_WITHOUT_A_SOURCE


def test_a_graph_model_written_in_python_keeps_the_cognify_run_in_its_process(runs):
    """Its schema would not build the same class in a worker, so no worker is
    asked: the run stores what it stores without workers."""
    in_python = runs["model_in_python"]
    assert in_python["run_status"] == "PipelineRunCompleted"
    assert _shards(in_python, COGNIFY_PIPELINE_NAME) == []
    assert in_python["graph"] == runs["model_one_process"]["graph"]


# ---- cognify of many DLT sources ----


def test_one_process_stores_the_csv_files(runs):
    """What the other runs are compared with has to be a real result."""
    one_process = runs["csv_one_process"]
    assert one_process["run_status"] == "PipelineRunCompleted"
    assert one_process["shards"] == []
    # Every file is a source of its own, and every row of it a node.
    assert _nodes_by_type(one_process)["DltSourceDocument"] == CSV_FILES
    assert _nodes_by_type(one_process)["DltRow"] == CSV_FILES * ROWS_PER_FILE


def test_csv_files_are_shared_among_workers_like_documents(runs):
    workers, one_process = runs["csv_workers"], runs["csv_one_process"]
    assert workers["run_status"] == "PipelineRunCompleted"
    assert workers["item_statuses"] == {"PipelineRunCompleted": CSV_FILES}
    assert workers["stamped"] == CSV_FILES
    # Twelve sources, two to a worker, each stored whole by the worker given it.
    assert _shards(workers, COGNIFY_PIPELINE_NAME) == [("finished", {"completed": 2})] * 6
    assert _graph(workers, "csv_workers") == _graph(one_process, "csv_one_process")
    assert workers["stored"]["rows"] == one_process["stored"]["rows"]
    assert workers["stored"]["without_sources"] == NOTHING_WITHOUT_A_SOURCE


def test_a_lost_worker_loses_its_own_csv_files_only_and_the_next_run_stores_them(runs):
    lost, again = runs["csv_worker_lost"], runs["csv_worker_lost_again"]
    assert lost["run_status"] == "PipelineRunErrored"
    assert lost["error_class"] == "ShardWorkerLost"
    assert lost["item_statuses"] == {"PipelineRunCompleted": 10, "PipelineRunErrored": 2}
    assert lost["stamped"] == 10
    assert _shards(lost, COGNIFY_PIPELINE_NAME) == (
        [("finished", {"completed": 2})] * 5 + [("lost", {"errored": 2})]
    )
    # What the other workers stored stays.
    assert _nodes_by_type(lost)["DltSourceDocument"] == 10

    assert again["run_status"] == "PipelineRunCompleted"
    assert again["item_statuses"] == {"PipelineRunAlreadyCompleted": 10, "PipelineRunCompleted": 2}
    assert again["stamped"] == CSV_FILES
    assert _graph(again, "csv_worker_lost") == _graph(runs["csv_one_process"], "csv_one_process")
    assert again["stored"]["without_sources"] == NOTHING_WITHOUT_A_SOURCE


def test_one_run_shares_out_documents_and_csv_files_together(runs):
    workers, one_process = runs["mixed_workers"], runs["mixed_one_process"]
    assert one_process["run_status"] == "PipelineRunCompleted"
    assert _shards(one_process, COGNIFY_PIPELINE_NAME) == []

    assert workers["run_status"] == "PipelineRunCompleted"
    assert workers["item_statuses"] == {"PipelineRunCompleted": DOCUMENTS + CSV_FILES}
    assert workers["stamped"] == DOCUMENTS + CSV_FILES
    # 37 items, two to a worker, whatever they are.
    assert _shards(workers, COGNIFY_PIPELINE_NAME) == (
        [("finished", {"completed": 1})] + [("finished", {"completed": 2})] * 18
    )
    assert _graph(workers, "mixed_workers") == _graph(one_process, "mixed_one_process")
    assert workers["stored"]["without_sources"] == NOTHING_WITHOUT_A_SOURCE
