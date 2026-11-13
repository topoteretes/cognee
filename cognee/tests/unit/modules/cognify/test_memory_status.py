"""The memory-state derivation over a dataset's pipeline runs (SDK-397)."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from cognee.modules.cognify.memory_status import (
    MEMORY_EMPTY,
    MEMORY_EXTRACTING,
    MEMORY_EXTRACTION_FAILED,
    MEMORY_INDEXED,
    MEMORY_INDEXING,
    MEMORY_INDEXING_FAILED,
    MEMORY_READY,
    MEMORY_STAGED,
    SEARCH_LEVEL_CHUNKS,
    SEARCH_LEVEL_GRAPH,
    SEARCH_LEVEL_NONE,
    derive_memory_status,
)
from cognee.modules.pipelines.models.PipelineRun import PipelineRunStatus
from cognee.shared.pipeline_names import (
    ADD_PIPELINE_NAME,
    CODE_GRAPH_PIPELINE_NAME,
    COGNIFY_PIPELINE_NAME,
    INDEXING_PIPELINE_NAME,
)

STARTED = PipelineRunStatus.DATASET_PROCESSING_STARTED
COMPLETED = PipelineRunStatus.DATASET_PROCESSING_COMPLETED
ERRORED = PipelineRunStatus.DATASET_PROCESSING_ERRORED
INITIATED = PipelineRunStatus.DATASET_PROCESSING_INITIATED
BASE = datetime(2026, 10, 20, tzinfo=timezone.utc)


def _run(status, at=0, error_class=None, error_message=None):
    return SimpleNamespace(
        status=status,
        created_at=BASE + timedelta(seconds=at),
        error_class=error_class,
        error_message=error_message,
    )


def test_nothing_ran_is_empty_and_staged_after_add():
    empty = derive_memory_status({})
    assert (empty.state, empty.search_level, empty.searchable) == (
        MEMORY_EMPTY,
        SEARCH_LEVEL_NONE,
        False,
    )

    staged = derive_memory_status({ADD_PIPELINE_NAME: _run(COMPLETED)})
    assert staged.state == MEMORY_STAGED
    assert staged.searchable is False
    assert "cognify()" in staged.message


def test_a_cognify_build_reads_indexing_then_extracting_then_ready():
    """cognify() opens its extraction run first, then runs the indexing run inside it."""
    runs = {ADD_PIPELINE_NAME: _run(COMPLETED, 0), COGNIFY_PIPELINE_NAME: _run(STARTED, 1)}
    opened = derive_memory_status(runs)
    assert (opened.state, opened.search_level) == (MEMORY_EXTRACTING, SEARCH_LEVEL_NONE)
    assert "nothing is searchable yet" in opened.message

    runs[INDEXING_PIPELINE_NAME] = _run(STARTED, 2)
    indexing = derive_memory_status(runs)
    assert (indexing.state, indexing.search_level) == (MEMORY_INDEXING, SEARCH_LEVEL_NONE)
    assert indexing.is_active

    runs[INDEXING_PIPELINE_NAME] = _run(COMPLETED, 3)
    extracting = derive_memory_status(runs)
    assert (extracting.state, extracting.search_level) == (MEMORY_EXTRACTING, SEARCH_LEVEL_CHUNKS)
    assert (extracting.searchable, extracting.graph_ready) == (True, False)
    assert "chunk searches" in extracting.message

    runs[COGNIFY_PIPELINE_NAME] = _run(COMPLETED, 4)
    ready = derive_memory_status(runs)
    assert (ready.state, ready.search_level, ready.graph_ready) == (
        MEMORY_READY,
        SEARCH_LEVEL_GRAPH,
        True,
    )
    assert ready.indexing_status == COMPLETED.value
    assert ready.cognify_status == COMPLETED.value


def test_a_failed_extraction_keeps_the_indexed_chunks():
    runs = {
        COGNIFY_PIPELINE_NAME: _run(ERRORED, 3, "AuthenticationError", "invalid api key"),
        INDEXING_PIPELINE_NAME: _run(COMPLETED, 2),
    }
    status = derive_memory_status(runs)
    assert status.state == MEMORY_EXTRACTION_FAILED
    assert status.search_level == SEARCH_LEVEL_CHUNKS
    assert status.searchable is True
    assert status.is_failure
    assert status.error_class == "AuthenticationError"
    assert "invalid api key" in status.message
    assert "chunk searches" in status.message


def test_an_indexing_run_without_extraction_after_it():
    """A custom pipeline ran only the indexing run; nothing extracted since."""
    indexed = derive_memory_status({INDEXING_PIPELINE_NAME: _run(COMPLETED, 1)})
    assert (indexed.state, indexed.search_level) == (MEMORY_INDEXED, SEARCH_LEVEL_CHUNKS)
    assert "Entities have not been extracted" in indexed.message

    failed = derive_memory_status({INDEXING_PIPELINE_NAME: _run(ERRORED, 1, "EmbeddingError", "q")})
    assert (failed.state, failed.search_level) == (MEMORY_INDEXING_FAILED, SEARCH_LEVEL_NONE)
    assert failed.error_class == "EmbeddingError"


def test_an_earlier_completed_run_keeps_the_level_of_a_failed_rerun():
    runs = {COGNIFY_PIPELINE_NAME: _run(ERRORED, 9, "RuntimeError", "outage")}
    status = derive_memory_status(runs, completed_pipelines={COGNIFY_PIPELINE_NAME})
    assert status.state == MEMORY_EXTRACTION_FAILED
    assert status.graph_ready is True
    assert status.search_level == SEARCH_LEVEL_GRAPH

    status = derive_memory_status(
        {COGNIFY_PIPELINE_NAME: _run(ERRORED, 9)}, completed_pipelines={INDEXING_PIPELINE_NAME}
    )
    assert status.search_level == SEARCH_LEVEL_CHUNKS


def test_new_data_being_indexed_after_a_build_keeps_the_graph_level():
    runs = {COGNIFY_PIPELINE_NAME: _run(STARTED, 3), INDEXING_PIPELINE_NAME: _run(STARTED, 4)}
    status = derive_memory_status(
        runs, completed_pipelines={COGNIFY_PIPELINE_NAME, INDEXING_PIPELINE_NAME}
    )
    assert status.state == MEMORY_INDEXING
    assert status.search_level == SEARCH_LEVEL_GRAPH


def test_any_other_named_pipeline_counts_as_a_graph_build():
    status = derive_memory_status({"memify_pipeline": _run(COMPLETED, 1)})
    assert (status.state, status.search_level) == (MEMORY_READY, SEARCH_LEVEL_GRAPH)


def test_to_dict_carries_every_field():
    status = derive_memory_status({INDEXING_PIPELINE_NAME: _run(COMPLETED)})
    assert status.to_dict() == {
        "state": MEMORY_INDEXED,
        "search_level": SEARCH_LEVEL_CHUNKS,
        "searchable": True,
        "graph_ready": False,
        "message": status.message,
        "indexing_status": COMPLETED.value,
        "cognify_status": None,
        "error_class": None,
        "error_message": None,
    }


def test_legacy_initiated_rows_are_not_runs_in_progress():
    status = derive_memory_status({COGNIFY_PIPELINE_NAME: _run(INITIATED, 1)})
    assert status.state == MEMORY_EMPTY

    status = derive_memory_status(
        {INDEXING_PIPELINE_NAME: _run(COMPLETED, 1), COGNIFY_PIPELINE_NAME: _run(INITIATED, 2)}
    )
    assert status.state == MEMORY_INDEXED


def test_a_failed_enrichment_after_a_build_keeps_the_memory_ready():
    """improve() and memify write the graph too, but they do not build it."""
    runs = {
        COGNIFY_PIPELINE_NAME: _run(COMPLETED, 1),
        "memify_pipeline": _run(ERRORED, 2, "RuntimeError", "enrichment failed"),
    }
    status = derive_memory_status(runs)
    assert (status.state, status.graph_ready, status.error_class) == (MEMORY_READY, True, None)


def test_the_code_graph_pipeline_builds_a_graph():
    status = derive_memory_status({CODE_GRAPH_PIPELINE_NAME: _run(ERRORED, 1, "EnolaError")})
    assert status.state == MEMORY_EXTRACTION_FAILED
    assert status.error_class == "EnolaError"


def test_naive_and_aware_timestamps_order_together():
    """SQLite returns naive UTC datetimes; nothing may compare them with aware ones and raise."""
    naive = SimpleNamespace(
        status=COMPLETED,
        created_at=BASE.replace(tzinfo=None) + timedelta(seconds=5),
        error_class=None,
        error_message=None,
    )
    runs = {COGNIFY_PIPELINE_NAME: _run(COMPLETED, 1), INDEXING_PIPELINE_NAME: naive}
    assert derive_memory_status(runs).state == MEMORY_INDEXED
