"""Derive a dataset's memory state from its pipeline runs.

A dataset's memory is built by up to three pipelines run in order (see
``cognee.modules.cognify.memory_pipelines``): ``caching_pipeline`` makes the
documents searchable by keyword, ``indexing_pipeline`` makes the chunks
searchable by vector, then ``cognify_pipeline`` (``code_graph_pipeline`` for
code) builds the knowledge graph. The per-pipeline statuses the status
endpoints report say what each run did; this folds them into the one answer
users ask for — *can I search this memory, and how good are the answers right
now?*

It reads the relational ``pipeline_runs`` table only, never a graph or vector
engine, so it is cheap enough for every status poll and for recall's readiness
probe. Every named pipeline counts: anything that is neither staging, caching
nor indexing is a graph build, which keeps custom graph-building pipelines
(memify, presort, user pipelines) in the picture.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import aliased

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.pipeline_names import (
    ADD_PIPELINE_NAME,
    CACHING_PIPELINE_NAME,
    INDEXING_PIPELINE_NAME,
    NON_GRAPH_PIPELINE_NAMES,
)

from ..models import PipelineRun, PipelineRunStatus

# The states, from nothing to everything.
MEMORY_EMPTY = "empty"  # no pipeline has touched the dataset
MEMORY_STAGED = "staged"  # add() ran; nothing built yet
MEMORY_CACHING = "caching"  # the caching pipeline is running
MEMORY_CACHED = "cached"  # documents answer keyword (BM25) searches
MEMORY_CACHING_FAILED = "caching_failed"
MEMORY_INDEXING = "indexing"  # the indexing pipeline is running
MEMORY_INDEXED = "indexed"  # chunks answer vector searches; extraction not completed yet
MEMORY_INDEXING_FAILED = "indexing_failed"
MEMORY_EXTRACTING = "extracting"  # a graph-building pipeline is running
MEMORY_READY = "ready"  # the graph is built
MEMORY_EXTRACTION_FAILED = "extraction_failed"  # the graph build errored

# The best search a dataset answers, from nothing to everything.
SEARCH_LEVEL_NONE = "none"
SEARCH_LEVEL_DOCUMENTS = "documents"  # DOCUMENTS_LEXICAL over the cached text
SEARCH_LEVEL_CHUNKS = "chunks"  # CHUNKS, CHUNKS_LEXICAL, RAG_COMPLETION, HYBRID_COMPLETION
SEARCH_LEVEL_GRAPH = "graph"  # everything, graph lanes included

_ACTIVE = (
    PipelineRunStatus.DATASET_PROCESSING_STARTED,
    PipelineRunStatus.DATASET_PROCESSING_INITIATED,
)
_COMPLETED = PipelineRunStatus.DATASET_PROCESSING_COMPLETED
_ERRORED = PipelineRunStatus.DATASET_PROCESSING_ERRORED

_FAILED_STATES = (MEMORY_CACHING_FAILED, MEMORY_INDEXING_FAILED, MEMORY_EXTRACTION_FAILED)
_ACTIVE_STATES = (MEMORY_CACHING, MEMORY_INDEXING, MEMORY_EXTRACTING)


@dataclass(frozen=True)
class MemoryStatus:
    """What a dataset's memory can do right now, and why."""

    state: str
    # The best search the dataset answers: "none", "documents", "chunks" or "graph".
    search_level: str
    # Some search answers (search_level is not "none").
    searchable: bool
    # A graph build completed at some point: graph-based searches answer.
    graph_ready: bool
    message: str
    caching_status: str | None = None
    indexing_status: str | None = None
    cognify_status: str | None = None
    # The classified cause when the state is a failure.
    error_class: str | None = None
    error_message: str | None = None

    @property
    def is_failure(self) -> bool:
        return self.state in _FAILED_STATES

    @property
    def is_active(self) -> bool:
        return self.state in _ACTIVE_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "search_level": self.search_level,
            "searchable": self.searchable,
            "graph_ready": self.graph_ready,
            "message": self.message,
            "caching_status": self.caching_status,
            "indexing_status": self.indexing_status,
            "cognify_status": self.cognify_status,
            "error_class": self.error_class,
            "error_message": self.error_message,
        }


def _status_value(run) -> str | None:
    if run is None:
        return None
    status = run.status
    return getattr(status, "value", status)


def _is(run, *statuses) -> bool:
    return run is not None and run.status in statuses


def _created_at(run) -> float:
    """A sortable key for a run's newest row; rows of one table are all naive or all aware."""
    value: datetime | None = getattr(run, "created_at", None)
    return value.timestamp() if value is not None else 0.0


def _failure_suffix(run) -> str:
    cause = (run.error_class if run is not None else None) or "unknown error"
    if run is not None and run.error_message:
        cause = f"{cause}: {run.error_message}"
    return cause


_LEVEL_ANSWERS = {
    SEARCH_LEVEL_DOCUMENTS: "keyword (DOCUMENTS_LEXICAL) searches over the cached documents answer",
    SEARCH_LEVEL_CHUNKS: (
        "chunk searches (CHUNKS, CHUNKS_LEXICAL, RAG_COMPLETION, HYBRID_COMPLETION) answer"
    ),
    SEARCH_LEVEL_GRAPH: "chunk and graph searches answer",
}


def _answers_now(level: str) -> str:
    return _LEVEL_ANSWERS.get(level, "nothing is searchable yet")


def _message(state: str, level: str, failed_run=None) -> str:
    answers = _answers_now(level)
    if state == MEMORY_EMPTY:
        return "No data has been added to this dataset yet."
    if state == MEMORY_STAGED:
        return (
            "Data is added but not processed: nothing is searchable yet. "
            "Run cognify() (or remember()) to build memory."
        )
    if state == MEMORY_CACHING:
        return (
            f"Memory is being cached: documents become searchable by keyword as they are "
            f"stored; chunking, embedding and entity extraction follow. Right now {answers}."
        )
    if state == MEMORY_CACHED:
        return (
            "Memory is cached: keyword (DOCUMENTS_LEXICAL, BM25) searches answer from the "
            "stored documents. Chunking and embedding come next, then entity extraction, "
            "and answers improve at each step."
        )
    if state == MEMORY_CACHING_FAILED:
        return (
            f"Caching failed ({_failure_suffix(failed_run)}); right now {answers}. The later "
            "stages still run; re-run cognify() to cache the documents."
        )
    if state == MEMORY_INDEXING:
        return (
            f"Memory is being indexed: chunks become searchable as they are embedded, and "
            f"entity extraction follows. Right now {answers}."
        )
    if state == MEMORY_INDEXED:
        return (
            "Memory is indexed: chunk searches (CHUNKS, CHUNKS_LEXICAL, RAG_COMPLETION, "
            "HYBRID_COMPLETION) answer. Entity extraction has not completed yet, so "
            "graph-based answers improve once it does."
        )
    if state == MEMORY_INDEXING_FAILED:
        return (
            f"Indexing failed ({_failure_suffix(failed_run)}); right now {answers}. Entity "
            "extraction still runs and stores the chunks itself; re-run cognify() if it "
            "did not complete."
        )
    if state == MEMORY_EXTRACTING:
        return (
            f"Entities and relationships are being extracted; answers improve when "
            f"extraction completes. Right now {answers}."
        )
    if state == MEMORY_READY:
        return (
            "Memory is fully built: documents, chunks, entities and relationships are searchable."
        )
    if state == MEMORY_EXTRACTION_FAILED:
        return (
            f"Entity extraction failed ({_failure_suffix(failed_run)}); right now {answers}. "
            "Fix the cause and re-run cognify() to build the graph."
        )
    return state


def derive_memory_status(
    latest_runs: dict[str, Any],
    *,
    completed_pipelines: set[str] | None = None,
) -> MemoryStatus:
    """Fold a dataset's latest run per pipeline into one memory state.

    ``latest_runs`` maps a pipeline name to that pipeline's newest run row
    (anything with ``status``, ``created_at``, ``error_class``,
    ``error_message``). ``completed_pipelines`` names the pipelines an older
    run of which completed: a failed re-run of a built dataset keeps what the
    earlier run built.
    """
    completed = set(completed_pipelines or ())
    caching = latest_runs.get(CACHING_PIPELINE_NAME)
    indexing = latest_runs.get(INDEXING_PIPELINE_NAME)
    add = latest_runs.get(ADD_PIPELINE_NAME)
    graph_runs = [run for name, run in latest_runs.items() if name not in NON_GRAPH_PIPELINE_NAMES]
    graph = max(graph_runs, key=_created_at) if graph_runs else None

    graph_ready = _is(graph, _COMPLETED) or any(
        name not in NON_GRAPH_PIPELINE_NAMES for name in completed
    )
    chunks_ready = graph_ready or _is(indexing, _COMPLETED) or INDEXING_PIPELINE_NAME in completed
    documents_ready = chunks_ready or _is(caching, _COMPLETED) or CACHING_PIPELINE_NAME in completed
    if graph_ready:
        level = SEARCH_LEVEL_GRAPH
    elif chunks_ready:
        level = SEARCH_LEVEL_CHUNKS
    elif documents_ready:
        level = SEARCH_LEVEL_DOCUMENTS
    else:
        level = SEARCH_LEVEL_NONE

    failed_run = None
    if _is(graph, *_ACTIVE):
        state = MEMORY_EXTRACTING
    elif _is(indexing, *_ACTIVE):
        state = MEMORY_INDEXING
    elif _is(caching, *_ACTIVE):
        state = MEMORY_CACHING
    else:
        terminal = [run for run in (caching, indexing, graph) if run is not None]
        newest = max(terminal, key=_created_at) if terminal else None
        if newest is None:
            state = MEMORY_STAGED if add is not None else MEMORY_EMPTY
        elif newest is graph:
            state = MEMORY_READY if _is(graph, _COMPLETED) else MEMORY_EXTRACTION_FAILED
        elif newest is indexing:
            state = MEMORY_INDEXED if _is(indexing, _COMPLETED) else MEMORY_INDEXING_FAILED
        else:
            state = MEMORY_CACHED if _is(caching, _COMPLETED) else MEMORY_CACHING_FAILED
        if state in _FAILED_STATES:
            failed_run = newest

    return MemoryStatus(
        state=state,
        search_level=level,
        searchable=level != SEARCH_LEVEL_NONE,
        graph_ready=graph_ready,
        message=_message(state, level, failed_run),
        caching_status=_status_value(caching),
        indexing_status=_status_value(indexing),
        cognify_status=_status_value(graph),
        error_class=failed_run.error_class if failed_run is not None else None,
        error_message=failed_run.error_message if failed_run is not None else None,
    )


async def get_memory_status(dataset_ids: list[UUID]) -> dict[str, MemoryStatus]:
    """The memory state of each dataset, keyed by dataset id as a string.

    Every requested dataset gets an entry (``empty`` when no pipeline has run
    on it), unlike ``get_pipeline_status`` which omits datasets without runs.
    Two queries for any number of datasets: the newest row per (dataset,
    pipeline) and the set of (dataset, pipeline) pairs that ever completed.
    """
    if not dataset_ids:
        return {}

    ranked = (
        select(
            PipelineRun,
            func.row_number()
            .over(
                partition_by=(PipelineRun.dataset_id, PipelineRun.pipeline_name),
                order_by=PipelineRun.created_at.desc(),
            )
            .label("rn"),
        )
        .filter(PipelineRun.dataset_id.in_(dataset_ids))
        # Operation records share the table but carry no pipeline name; they
        # are activity evidence, not memory state.
        .filter(PipelineRun.pipeline_name.isnot(None))
        .subquery()
    )
    latest_run = aliased(PipelineRun, ranked)
    latest_query = select(latest_run).filter(ranked.c.rn == 1)

    completed_query = (
        select(PipelineRun.dataset_id, PipelineRun.pipeline_name)
        .filter(PipelineRun.dataset_id.in_(dataset_ids))
        .filter(PipelineRun.pipeline_name.isnot(None))
        .filter(PipelineRun.status == _COMPLETED)
        .distinct()
    )

    async with get_relational_engine().get_async_session() as session:
        latest_rows = (await session.execute(latest_query)).scalars().all()
        completed_rows = (await session.execute(completed_query)).all()

    latest_by_dataset: dict[UUID, dict[str, PipelineRun]] = {}
    for run in latest_rows:
        latest_by_dataset.setdefault(run.dataset_id, {})[run.pipeline_name] = run
    completed_by_dataset: dict[UUID, set[str]] = {}
    for dataset_id, pipeline_name in completed_rows:
        completed_by_dataset.setdefault(dataset_id, set()).add(pipeline_name)

    return {
        str(dataset_id): derive_memory_status(
            latest_by_dataset.get(dataset_id, {}),
            completed_pipelines=completed_by_dataset.get(dataset_id, set()),
        )
        for dataset_id in dataset_ids
    }
