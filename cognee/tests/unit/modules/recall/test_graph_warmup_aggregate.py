"""Folding per-dataset memory states into recall's readiness verdict (SDK-397)."""

from cognee.modules.pipelines.operations.get_memory_status import (
    MEMORY_CACHED,
    MEMORY_CACHING,
    MEMORY_CACHING_FAILED,
    MEMORY_EMPTY,
    MEMORY_EXTRACTING,
    MEMORY_EXTRACTION_FAILED,
    MEMORY_INDEXED,
    MEMORY_READY,
    MEMORY_STAGED,
    SEARCH_LEVEL_CHUNKS,
    SEARCH_LEVEL_DOCUMENTS,
    SEARCH_LEVEL_GRAPH,
    SEARCH_LEVEL_NONE,
    MemoryStatus,
)
from cognee.modules.recall.methods.graph_warmup import (
    STATE_BUILD_FAILED,
    STATE_NEVER_BUILT,
    STATE_PARTIAL,
    STATE_WARM,
    aggregate_memory_statuses,
)


def _status(state, level, **kwargs):
    return MemoryStatus(
        state=state,
        search_level=level,
        searchable=level != SEARCH_LEVEL_NONE,
        graph_ready=level == SEARCH_LEVEL_GRAPH,
        message=f"{state} message",
        **kwargs,
    )


def test_one_warm_dataset_makes_the_set_warm():
    probe = aggregate_memory_statuses(
        [_status(MEMORY_CACHED, SEARCH_LEVEL_DOCUMENTS), _status(MEMORY_READY, SEARCH_LEVEL_GRAPH)]
    )
    assert probe.state == STATE_WARM
    assert probe.is_warm and probe.is_searchable
    assert probe.datapoint_count > 0


def test_partial_prefers_chunks_over_documents_and_carries_the_message():
    probe = aggregate_memory_statuses(
        [
            _status(MEMORY_CACHED, SEARCH_LEVEL_DOCUMENTS),
            _status(MEMORY_INDEXED, SEARCH_LEVEL_CHUNKS),
        ]
    )
    assert probe.state == STATE_PARTIAL
    assert probe.search_level == SEARCH_LEVEL_CHUNKS
    assert probe.memory_state == MEMORY_INDEXED
    assert probe.message == "indexed message"
    assert probe.is_searchable and not probe.is_warm


def test_a_failed_extraction_with_chunks_is_partial_with_its_cause():
    failed = _status(
        MEMORY_EXTRACTION_FAILED,
        SEARCH_LEVEL_CHUNKS,
        error_class="AuthenticationError",
        error_message="bad key",
    )
    probe = aggregate_memory_statuses([failed])
    assert probe.state == STATE_PARTIAL
    assert probe.error_class == "AuthenticationError"


def test_failure_counts_only_when_nothing_is_searchable():
    failed = _status(MEMORY_CACHING_FAILED, SEARCH_LEVEL_NONE, error_class="OSError")
    probe = aggregate_memory_statuses([failed])
    assert probe.state == STATE_BUILD_FAILED
    assert probe.error_class == "OSError"
    assert probe.datapoint_count == 0


def test_a_stage_in_flight_is_partial_so_recall_searches_what_exists():
    probe = aggregate_memory_statuses([_status(MEMORY_CACHING, SEARCH_LEVEL_NONE)])
    assert probe.state == STATE_PARTIAL
    assert probe.memory_state == MEMORY_CACHING


def test_staged_only_and_extracting_without_stages_keep_the_warm_failsafe():
    assert (
        aggregate_memory_statuses([_status(MEMORY_STAGED, SEARCH_LEVEL_NONE)]).state == STATE_WARM
    )
    assert (
        aggregate_memory_statuses([_status(MEMORY_EXTRACTING, SEARCH_LEVEL_NONE)]).state
        == STATE_WARM
    )


def test_nothing_at_all_is_never_built():
    assert aggregate_memory_statuses([]).state == STATE_NEVER_BUILT
    assert aggregate_memory_statuses([_status(MEMORY_EMPTY, SEARCH_LEVEL_NONE)]).state == (
        STATE_NEVER_BUILT
    )
