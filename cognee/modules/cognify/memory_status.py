"""Derive a dataset's memory state from its pipeline runs (SDK-397).

``cognify()`` builds a dataset's memory in two runs (see
``cognee.modules.cognify.memory_pipelines``): ``indexing_pipeline`` makes the
chunks searchable, then ``cognify_pipeline`` (``code_graph_pipeline`` for
code) builds the knowledge graph. The per-pipeline statuses the status
endpoints report say what each run did; this folds them into the one answer
users ask for: *can I search this memory, and how good are the answers right
now?* Every reader of that answer calls this function: the HTTP route, the
SDK, the CLI, ``remember()``'s result and recall's readiness probe.

It reads the relational ``pipeline_runs`` table only, never a graph or vector
engine, so it is cheap enough for every status poll and for every recall.
Every named pipeline counts: anything that is neither staging nor indexing is
a graph build, which keeps custom graph-building pipelines (memify, presort,
user pipelines) in the picture.

During a ``cognify()`` build the state goes ``indexing`` → ``extracting`` →
``ready`` (or ``extraction_failed``), with the search level rising from
``none`` to ``chunks`` when the indexing run completes and to ``graph`` when
extraction does.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import aliased

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.shared.pipeline_names import (
    ADD_PIPELINE_NAME,
    CODE_GRAPH_PIPELINE_NAME,
    COGNIFY_PIPELINE_NAME,
    INDEXING_PIPELINE_NAME,
)

# Pipelines that stage or index without building the knowledge graph. Every
# other named pipeline that completes counts as a graph build.
NON_GRAPH_PIPELINE_NAMES = (ADD_PIPELINE_NAME, INDEXING_PIPELINE_NAME)

# The runs that build a dataset's graph from its data. Other graph-writing
# pipelines (memify, improve's stages, custom pipelines) enrich a graph: they
# count for "the graph exists", but their failures do not make the dataset's
# extraction failed, and they decide the state only for a dataset no build
# run ever touched (a graph built by a custom pipeline).
_GRAPH_BUILD_PIPELINE_NAMES = (COGNIFY_PIPELINE_NAME, CODE_GRAPH_PIPELINE_NAME)

# The states, from nothing to everything.
MEMORY_EMPTY = "empty"  # no pipeline has touched the dataset
MEMORY_STAGED = "staged"  # add() ran; nothing built yet
MEMORY_INDEXING = "indexing"  # the indexing pipeline is running
MEMORY_INDEXED = "indexed"  # an indexing run completed and no extraction followed it
MEMORY_INDEXING_FAILED = "indexing_failed"  # the same, but the indexing run failed
MEMORY_EXTRACTING = "extracting"  # a graph-building pipeline is running
MEMORY_READY = "ready"  # the graph is built
MEMORY_EXTRACTION_FAILED = "extraction_failed"  # the graph build errored

# The best search a dataset answers, from nothing to everything.
SEARCH_LEVEL_NONE = "none"
SEARCH_LEVEL_CHUNKS = "chunks"  # CHUNKS, CHUNKS_LEXICAL, RAG_COMPLETION, HYBRID_COMPLETION
SEARCH_LEVEL_GRAPH = "graph"  # everything, graph lanes included

# INITIATED is no longer written; a row that still carries it is a legacy
# leftover, not a run in progress, and is left out of the state.
_ACTIVE = PipelineRunStatus.DATASET_PROCESSING_STARTED
_LEGACY = PipelineRunStatus.DATASET_PROCESSING_INITIATED
_COMPLETED = PipelineRunStatus.DATASET_PROCESSING_COMPLETED

_FAILED_STATES = (MEMORY_INDEXING_FAILED, MEMORY_EXTRACTION_FAILED)
_ACTIVE_STATES = (MEMORY_INDEXING, MEMORY_EXTRACTING)


@dataclass(frozen=True)
class MemoryStatus:
    """What a dataset's memory can do right now, and why."""

    state: str
    # The best search the dataset answers: "none", "chunks" or "graph".
    search_level: str
    message: str
    indexing_status: str | None = None
    cognify_status: str | None = None
    # The classified cause when the state is a failure.
    error_class: str | None = None
    error_message: str | None = None

    @property
    def searchable(self) -> bool:
        """Some search answers."""
        return self.search_level != SEARCH_LEVEL_NONE

    @property
    def graph_ready(self) -> bool:
        """A graph build completed at some point: graph-based searches answer."""
        return self.search_level == SEARCH_LEVEL_GRAPH

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
            "indexing_status": self.indexing_status,
            "cognify_status": self.cognify_status,
            "error_class": self.error_class,
            "error_message": self.error_message,
        }


def _status_value(run) -> str | None:
    if run is None:
        return None
    return getattr(run.status, "value", run.status)


def _is(run, status) -> bool:
    return run is not None and run.status == status


def _created_at(run) -> datetime:
    """A sortable key for a run's newest row. SQLite returns naive UTC datetimes."""
    value: datetime | None = getattr(run, "created_at", None)
    if value is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _failure(run) -> str:
    cause = run.error_class or "unknown error"
    if run.error_message:
        cause = f"{cause}: {run.error_message}"
    return cause


_ANSWERS = {
    SEARCH_LEVEL_NONE: "nothing is searchable yet",
    SEARCH_LEVEL_CHUNKS: (
        "chunk searches (CHUNKS, CHUNKS_LEXICAL, RAG_COMPLETION, HYBRID_COMPLETION) answer"
    ),
    SEARCH_LEVEL_GRAPH: "chunk and graph searches answer",
}


def _message(state: str, level: str, failed_run=None) -> str:
    answers = _ANSWERS[level]
    if state == MEMORY_EMPTY:
        return "No data has been added to this dataset yet."
    if state == MEMORY_STAGED:
        return (
            "Data is added but not processed: nothing is searchable yet. "
            "Run cognify() (or remember()) to build memory."
        )
    if state == MEMORY_INDEXING:
        return (
            "Memory is being indexed: the documents are chunked and embedded, and entity "
            f"extraction follows. Right now {answers}."
        )
    if state == MEMORY_INDEXED:
        return (
            "Memory is indexed: chunk searches (CHUNKS, CHUNKS_LEXICAL, RAG_COMPLETION, "
            "HYBRID_COMPLETION) answer. Entities have not been extracted yet; run cognify() "
            "to build the graph, and answers improve once it completes."
        )
    if state == MEMORY_INDEXING_FAILED:
        return (
            f"Indexing failed ({_failure(failed_run)}); right now {answers}. Re-run cognify(): "
            "its extraction stores the chunks itself."
        )
    if state == MEMORY_EXTRACTING:
        return (
            "Entities and relationships are being extracted; answers improve when extraction "
            f"completes. Right now {answers}."
        )
    if state == MEMORY_READY:
        return "Memory is fully built: chunks, entities and relationships are searchable."
    if state == MEMORY_EXTRACTION_FAILED:
        return (
            f"Entity extraction failed ({_failure(failed_run)}); right now {answers}. "
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
    ``error_message``). ``completed_pipelines`` names the pipelines whose
    work stands even if their latest run failed: an older run of which
    completed (a failed re-run of a built dataset keeps what the earlier run
    built), or, for the indexing pipeline, items a failed run completed (it
    rolls back per item, so their chunks stay searchable).
    """
    completed = set(completed_pipelines or ())
    latest_runs = {
        name: run for name, run in latest_runs.items() if getattr(run, "status", None) != _LEGACY
    }
    indexing = latest_runs.get(INDEXING_PIPELINE_NAME)
    build_runs = [run for name, run in latest_runs.items() if name in _GRAPH_BUILD_PIPELINE_NAMES]
    other_graph_runs = [
        run
        for name, run in latest_runs.items()
        if name not in NON_GRAPH_PIPELINE_NAMES and name not in _GRAPH_BUILD_PIPELINE_NAMES
    ]
    # The state follows the build runs; enrichment runs only where no build ran.
    graph_runs = build_runs or other_graph_runs
    graph = max(graph_runs, key=_created_at) if graph_runs else None

    if _is(graph, _COMPLETED) or any(name not in NON_GRAPH_PIPELINE_NAMES for name in completed):
        level = SEARCH_LEVEL_GRAPH
    elif _is(indexing, _COMPLETED) or INDEXING_PIPELINE_NAME in completed:
        level = SEARCH_LEVEL_CHUNKS
    else:
        level = SEARCH_LEVEL_NONE

    failed_run = None
    # cognify() opens its extraction run before the indexing run (callers
    # follow the extraction run), so while the indexing run is active the
    # extraction row is open too; the indexing run says where the build is.
    if _is(indexing, _ACTIVE):
        state = MEMORY_INDEXING
    elif _is(graph, _ACTIVE):
        state = MEMORY_EXTRACTING
    else:
        terminal = [run for run in (indexing, graph) if run is not None]
        newest = max(terminal, key=_created_at) if terminal else None
        if newest is None:
            state = MEMORY_STAGED if ADD_PIPELINE_NAME in latest_runs else MEMORY_EMPTY
        elif newest is graph:
            state = MEMORY_READY if _is(graph, _COMPLETED) else MEMORY_EXTRACTION_FAILED
        else:
            state = MEMORY_INDEXED if _is(indexing, _COMPLETED) else MEMORY_INDEXING_FAILED
        if state in _FAILED_STATES:
            failed_run = newest

    return MemoryStatus(
        state=state,
        search_level=level,
        message=_message(state, level, failed_run),
        indexing_status=_status_value(indexing),
        cognify_status=_status_value(graph),
        error_class=failed_run.error_class if failed_run is not None else None,
        error_message=failed_run.error_message if failed_run is not None else None,
    )


async def _datasets_with_indexed_items(dataset_ids: list[UUID]) -> set[UUID]:
    """The datasets among ``dataset_ids`` with an item the indexing pipeline completed."""
    if not dataset_ids:
        return set()
    # Deferred like the data package's own readers of this stamp: the pipeline
    # layer imports the data package at module scope.
    from cognee.modules.data.models import Data
    from cognee.modules.pipelines.models.DataItemStatus import is_item_stamped

    query = select(Data.dataset_id, Data.pipeline_status).filter(Data.dataset_id.in_(dataset_ids))
    found: set[UUID] = set()
    async with get_relational_engine().get_async_session() as session:
        for row in (await session.execute(query)).all():
            dataset_id = row.dataset_id
            if dataset_id in found:
                continue
            if is_item_stamped(row, INDEXING_PIPELINE_NAME, dataset_id):
                found.add(dataset_id)
    return found


async def get_memory_status(dataset_ids: list[UUID]) -> dict[str, MemoryStatus]:
    """The memory state of each dataset, keyed by dataset id as a string.

    Every requested dataset gets an entry (``empty`` when no pipeline has run
    on it), unlike ``get_pipeline_status`` which omits datasets without runs.
    Two queries for any number of datasets: the newest row per (dataset,
    pipeline) and the set of (dataset, pipeline) pairs that ever completed.
    A third reads the items' completion stamps, only for datasets whose
    indexing runs all failed, since such a run keeps the items it completed.
    Callers check the user's access to the datasets first.
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

    partly_indexed = [
        dataset_id
        for dataset_id, runs in latest_by_dataset.items()
        if INDEXING_PIPELINE_NAME in runs
        and INDEXING_PIPELINE_NAME not in completed_by_dataset.get(dataset_id, set())
    ]
    for dataset_id in await _datasets_with_indexed_items(partly_indexed):
        completed_by_dataset.setdefault(dataset_id, set()).add(INDEXING_PIPELINE_NAME)

    return {
        str(dataset_id): derive_memory_status(
            latest_by_dataset.get(dataset_id, {}),
            completed_pipelines=completed_by_dataset.get(dataset_id, set()),
        )
        for dataset_id in dataset_ids
    }
