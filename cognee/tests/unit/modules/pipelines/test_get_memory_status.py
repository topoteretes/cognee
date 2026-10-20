"""The memory-state derivation over a dataset's pipeline runs (SDK-397)."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from cognee.modules.pipelines.models.PipelineRun import PipelineRunStatus
from cognee.modules.pipelines.operations.get_memory_status import (
    MEMORY_CACHED,
    MEMORY_CACHING,
    MEMORY_CACHING_FAILED,
    MEMORY_EMPTY,
    MEMORY_EXTRACTING,
    MEMORY_EXTRACTION_FAILED,
    MEMORY_INDEXED,
    MEMORY_INDEXING,
    MEMORY_INDEXING_FAILED,
    MEMORY_READY,
    MEMORY_STAGED,
    SEARCH_LEVEL_CHUNKS,
    SEARCH_LEVEL_DOCUMENTS,
    SEARCH_LEVEL_GRAPH,
    SEARCH_LEVEL_NONE,
    derive_memory_status,
)
from cognee.shared.pipeline_names import (
    ADD_PIPELINE_NAME,
    CACHING_PIPELINE_NAME,
    COGNIFY_PIPELINE_NAME,
    INDEXING_PIPELINE_NAME,
)

STARTED = PipelineRunStatus.DATASET_PROCESSING_STARTED
COMPLETED = PipelineRunStatus.DATASET_PROCESSING_COMPLETED
ERRORED = PipelineRunStatus.DATASET_PROCESSING_ERRORED
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


def test_the_stages_move_the_state_and_the_search_level_up():
    runs = {ADD_PIPELINE_NAME: _run(COMPLETED, 0), CACHING_PIPELINE_NAME: _run(STARTED, 1)}
    caching = derive_memory_status(runs)
    assert (caching.state, caching.search_level) == (MEMORY_CACHING, SEARCH_LEVEL_NONE)

    runs[CACHING_PIPELINE_NAME] = _run(COMPLETED, 2)
    cached = derive_memory_status(runs)
    assert (cached.state, cached.search_level, cached.searchable, cached.graph_ready) == (
        MEMORY_CACHED,
        SEARCH_LEVEL_DOCUMENTS,
        True,
        False,
    )
    assert "DOCUMENTS_LEXICAL" in cached.message

    runs[INDEXING_PIPELINE_NAME] = _run(STARTED, 3)
    indexing = derive_memory_status(runs)
    assert (indexing.state, indexing.search_level) == (MEMORY_INDEXING, SEARCH_LEVEL_DOCUMENTS)

    runs[INDEXING_PIPELINE_NAME] = _run(COMPLETED, 4)
    indexed = derive_memory_status(runs)
    assert (indexed.state, indexed.search_level) == (MEMORY_INDEXED, SEARCH_LEVEL_CHUNKS)
    assert "Entity extraction has not completed" in indexed.message

    runs[COGNIFY_PIPELINE_NAME] = _run(STARTED, 5)
    extracting = derive_memory_status(runs)
    assert (extracting.state, extracting.search_level) == (MEMORY_EXTRACTING, SEARCH_LEVEL_CHUNKS)

    runs[COGNIFY_PIPELINE_NAME] = _run(COMPLETED, 6)
    ready = derive_memory_status(runs)
    assert (ready.state, ready.search_level, ready.graph_ready) == (
        MEMORY_READY,
        SEARCH_LEVEL_GRAPH,
        True,
    )
    assert ready.caching_status == COMPLETED.value
    assert ready.indexing_status == COMPLETED.value
    assert ready.cognify_status == COMPLETED.value


def test_a_failed_later_stage_keeps_what_the_earlier_ones_built():
    runs = {
        CACHING_PIPELINE_NAME: _run(COMPLETED, 1),
        INDEXING_PIPELINE_NAME: _run(COMPLETED, 2),
        COGNIFY_PIPELINE_NAME: _run(ERRORED, 3, "AuthenticationError", "invalid api key"),
    }
    status = derive_memory_status(runs)
    assert status.state == MEMORY_EXTRACTION_FAILED
    assert status.search_level == SEARCH_LEVEL_CHUNKS
    assert status.searchable is True
    assert status.error_class == "AuthenticationError"
    assert "invalid api key" in status.message
    assert "chunk searches" in status.message

    runs = {
        CACHING_PIPELINE_NAME: _run(COMPLETED, 1),
        INDEXING_PIPELINE_NAME: _run(ERRORED, 2, "EmbeddingError", "quota"),
    }
    status = derive_memory_status(runs)
    assert status.state == MEMORY_INDEXING_FAILED
    assert status.search_level == SEARCH_LEVEL_DOCUMENTS

    status = derive_memory_status({CACHING_PIPELINE_NAME: _run(ERRORED, 1, "OSError", "disk")})
    assert status.state == MEMORY_CACHING_FAILED
    assert status.searchable is False
    assert "nothing is searchable yet" in status.message


def test_an_earlier_completed_run_keeps_the_level_of_a_failed_rerun():
    runs = {COGNIFY_PIPELINE_NAME: _run(ERRORED, 9, "RuntimeError", "outage")}
    status = derive_memory_status(runs, completed_pipelines={COGNIFY_PIPELINE_NAME})
    assert status.state == MEMORY_EXTRACTION_FAILED
    assert status.graph_ready is True
    assert status.search_level == SEARCH_LEVEL_GRAPH

    status = derive_memory_status(
        {INDEXING_PIPELINE_NAME: _run(ERRORED, 9)}, completed_pipelines={CACHING_PIPELINE_NAME}
    )
    assert status.search_level == SEARCH_LEVEL_DOCUMENTS


def test_new_data_being_cached_after_a_build_keeps_the_graph_level():
    runs = {COGNIFY_PIPELINE_NAME: _run(COMPLETED, 1), CACHING_PIPELINE_NAME: _run(STARTED, 2)}
    status = derive_memory_status(runs)
    assert status.state == MEMORY_CACHING
    assert status.search_level == SEARCH_LEVEL_GRAPH
    assert status.graph_ready is True


def test_any_other_named_pipeline_counts_as_a_graph_build():
    status = derive_memory_status({"memify_pipeline": _run(COMPLETED, 1)})
    assert (status.state, status.search_level) == (MEMORY_READY, SEARCH_LEVEL_GRAPH)


def test_the_newest_terminal_run_decides_between_stages():
    # Cognify completed long ago; a fresh caching run for new data completed since.
    runs = {COGNIFY_PIPELINE_NAME: _run(COMPLETED, 1), CACHING_PIPELINE_NAME: _run(COMPLETED, 5)}
    status = derive_memory_status(runs)
    assert status.state == MEMORY_CACHED
    assert status.search_level == SEARCH_LEVEL_GRAPH


def test_to_dict_carries_every_field():
    status = derive_memory_status({CACHING_PIPELINE_NAME: _run(COMPLETED)})
    assert status.to_dict() == {
        "state": MEMORY_CACHED,
        "search_level": SEARCH_LEVEL_DOCUMENTS,
        "searchable": True,
        "graph_ready": False,
        "message": status.message,
        "caching_status": COMPLETED.value,
        "indexing_status": None,
        "cognify_status": None,
        "error_class": None,
        "error_message": None,
    }
