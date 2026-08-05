"""Orchestrate one memory-accuracy-score run over a single dataset.

The dataset is named by the caller and entered as a database context here — see
:func:`run_memory_score` for why it cannot be inferred. The caller must hold READ
permission on it; see :func:`resolve_memory_score_dataset`.

Pipeline, in order:

1. Register the run (``INITIATED`` -> ``RUNNING``) and enter the dataset context.
2. Collect the REAL past questions of the user the run is attributed to (via the
   existing per-user :func:`cognee.modules.search.operations.get_queries`).
3. Cluster the graph into topics weighted by that real traffic, and evaluate the
   DATA FLOOR GATE (:func:`build_topics`). A gated run is persisted as
   ``SKIPPED_INSUFFICIENT_DATA`` and returns immediately — nothing past this
   point is generated, answered or judged, so no LLM completion token is spent.
   (Evaluating the topic half of the gate does cost one embedding batch; see
   ``build_topics``.)
4. Generate synthetic (question, expected_answer) pairs per topic, then answer
   every synthetic AND real question through cognee recall.
5. Judge. Synthetic questions have a golden answer, so they get a CORRECTNESS
   score. Real questions have none, so they get a GROUNDEDNESS boolean and
   nothing else. The two are stored in separate columns and aggregated
   separately — they are never averaged into one number.
6. Aggregate per topic and overall. ``overall_accuracy`` is computed from
   SYNTHETIC questions only.

Everything persisted here is a RAW SIGNAL. ``below_data_floor``,
``floor_reason``, ``schema_defined``, per-topic accuracy and the ungrounded real
questions are reported as-is; this module never decides "upload more data" vs
"define a schema". Thresholds and copy are the UI's job.

Why not ``run_tasks``: this run is now dataset-scoped, so the shape objection is
gone — ``run_tasks`` does ``session.get(Dataset, dataset_id)`` and reads
``dataset.id`` / ``dataset.owner_id``, which a single dataset can satisfy. What
remains is that ``run_tasks`` models ingestion: its status vocabulary is
``DATASET_PROCESSING_*`` and it writes ``pipeline_runs`` rows, neither of which
describes an evaluation. Scoring also needs per-question rows that
``PipelineRun.run_info`` JSON cannot be queried for. So this module keeps its own
async orchestration and its state in ``memory_score_runs``. Porting it onto
``run_tasks`` to inherit the progress queue and (once cognee#4291 lands) activity
cost attribution is a reasonable follow-up, not a prerequisite.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from fastapi import status
from sqlalchemy import select, update

from cognee.context_global_variables import set_database_global_context_variables
from cognee.eval_framework.answer_generation.answer_generation_executor import (
    AnswerGeneratorExecutor,
    retriever_options,
)
from cognee.eval_framework.evaluation.direct_llm_eval_adapter import DirectLLMEvalAdapter
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Dataset
from cognee.modules.memory_score.judges.groundedness_adapter import GroundednessAdapter
from cognee.modules.memory_score.methods.build_topics import TopicPlan, build_topics
from cognee.modules.memory_score.methods.generate_questions import (
    GeneratedQuestion,
    generate_questions,
)
from cognee.modules.memory_score.models import (
    MemoryScoreRun,
    MemoryScoreRunStatus,
    ScoredQuestion,
)

# Submodule, not the package: cognee.modules.search.operations does not re-export
# get_queries, so importing it from the package binds the MODULE and every call
# raises "'module' object is not callable".
from cognee.modules.search.operations.get_queries import get_queries
from cognee.modules.users.methods import get_user
from cognee.modules.users.permissions.methods import (
    get_all_user_permission_datasets,
    get_specific_user_permission_datasets,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("memory_score.run")

# ``ScoredQuestion.source`` values — treated as an enum at the app layer.
SOURCE_SYNTHETIC = "synthetic"
SOURCE_REAL = "real"

# PINNED RUN CONFIGURATION.
# The retriever and its top_k are fixed here instead of being inherited from the
# tenant's search defaults. Two reasons, both about the number this run
# produces: cost (top_k drives how much context every one of ~120 recalls pulls)
# and comparability (a tenant that flipped its default retriever or top_k would
# otherwise see its accuracy move without its memory having changed at all, and
# two tenants' scores would not be comparable). Changing these constants is a
# deliberate, repo-wide re-baselining of the score.
MEMORY_SCORE_RETRIEVER = "cognee_graph_completion"
MEMORY_SCORE_TOP_K = 5

# Maximum answer+judge pipelines in flight. Each one is 1-2 LLM calls, so this
# is the ceiling on concurrent LLM work for a run — a 120-question run must not
# open 120 sockets at once.
MEMORY_SCORE_CONCURRENCY = 5

# HARD CEILINGS ON WHAT ONE RUN MAY SPEND.
# Every question costs ~2,300 tokens end to end (generation + recall + judge), so
# the request parameters are a direct spend dial: at 500 synthetic questions a run
# is ~1.15M tokens (~$2.90) and that is the most a single call may authorise. The
# caps are enforced in BOTH places on purpose — the API rejects an out-of-range
# value outright (400, via the app-wide RequestValidationError handler) so the
# caller learns it was refused, and this module clamps
# whatever it is handed, so a scheduler or an SDK caller reaching past the HTTP
# layer cannot commit unbounded spend either.
MAX_SYNTHETIC_TARGET = 500
MAX_REAL_QUESTION_LIMIT = 200

# A run whose process died mid-flight would otherwise sit in RUNNING forever and
# block the tenant from ever scoring again. Past this age an INITIATED/RUNNING
# run stops counting as active.
ACTIVE_RUN_STALE_AFTER = timedelta(hours=2)

_ACTIVE_STATUSES = (MemoryScoreRunStatus.INITIATED, MemoryScoreRunStatus.RUNNING)

# Node types every ingestion produces on its own, as they appear in the schema
# inventory. There is no schema_defined field anywhere, so the signal is derived:
# a schema is "defined" when the graph carries a node type beyond these AND
# beyond the entity-type taxonomy (see _entity_type_names). A custom graph model
# stores its nodes under their own class name (get_graph_from_model writes
# ``type: type(data_point).__name__``), which is exactly what shows up here.
_INGESTION_TYPES = frozenset(
    {
        "DocumentChunk",
        "TextDocument",
        "PdfDocument",
        "AudioDocument",
        "ImageDocument",
        "CsvDocument",
        "DltRowDocument",
        "UnstructuredDocument",
        "TextSummary",
        "Entity",
        "NodeSet",
    }
)


class MemoryScoreRunInProgressError(CogneeValidationError):
    """A memory score run is already active for this tenant."""

    def __init__(
        self,
        message: str = "A memory score run is already in progress for this tenant.",
        name: str = "MemoryScoreRunInProgressError",
        status_code: int = status.HTTP_409_CONFLICT,
    ):
        super().__init__(message, name, status_code, log_level="WARNING")


class MemoryScoreDatasetNotFoundError(CogneeValidationError):
    """The requested dataset does not exist, or is not this tenant's."""

    def __init__(
        self,
        message: str = "Dataset not found for this tenant.",
        name: str = "MemoryScoreDatasetNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code, log_level="WARNING")


async def resolve_memory_score_dataset(
    tenant_id: UUID | None, dataset_id: UUID, requesting_user_id: UUID | None = None
) -> Dataset:
    """Load the dataset to score, verifying tenancy and then READ permission.

    Two checks, in this order, because the order is what keeps each answer from
    leaking more than it should:

    1. Existence and tenancy -> ``MemoryScoreDatasetNotFoundError`` (404).
       Whether some other tenant's dataset id exists is not this caller's
       business, so a cross-tenant id is indistinguishable from a made-up one.
    2. ``requesting_user_id``'s READ permission on the dataset ->
       ``PermissionDeniedError`` (403), via the repo's standard
       ``get_specific_user_permission_datasets``. Belonging to the tenant is NOT
       permission to read: a run over this dataset returns ``expected_answer``
       values lifted verbatim out of its chunk text plus full recall answers over
       it, so scoring it is a read of its contents and is gated like one.

    The ACL check cannot be left to the database layer, because the run enters
    the dataset context as ``dataset.owner_id`` (below) rather than as the
    caller — per-user database isolation would therefore never see the caller at
    all. This is the only place the caller's permission is consulted.

    ``requesting_user_id`` is None only for a run with no acting user (a
    scheduler), where there is no caller to authorise and the trigger itself is
    the trusted party. API callers always pass one.

    ``Dataset.owner_id`` is what makes a system-triggered run possible — the
    dataset context needs a user id, and the owner is the right one to use when
    no acting user started the run.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        dataset = await session.get(Dataset, dataset_id)

    if dataset is None or dataset.tenant_id != tenant_id:
        raise MemoryScoreDatasetNotFoundError(
            message=f"Dataset {dataset_id} not found for this tenant."
        )

    if requesting_user_id is not None:
        # Raises PermissionDeniedError (403) when the caller cannot read it.
        await get_specific_user_permission_datasets(requesting_user_id, "read", [dataset_id])

    return dataset


async def readable_dataset_ids(user_id: UUID) -> set[str]:
    """Ids of the datasets ``user_id`` may READ, as strings.

    The non-raising counterpart to the check in
    :func:`resolve_memory_score_dataset`, for the read endpoints: they answer 404
    rather than 403 for a run they may not see, so they need a predicate rather
    than an exception.

    Ids are stringified because a UUID column comes back as ``UUID`` or ``str``
    depending on the configured driver, and this set is compared against run rows.
    """
    user = await get_user(user_id)
    if user is None:
        return set()

    datasets = await get_all_user_permission_datasets(user, "read")
    return {str(dataset.id) for dataset in datasets}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _tenant_filter(tenant_id: UUID | None):
    """Tenant predicate that also covers the NULL-tenant (OSS) case."""
    if tenant_id is None:
        return MemoryScoreRun.tenant_id.is_(None)
    return MemoryScoreRun.tenant_id == tenant_id


async def create_memory_score_run(
    tenant_id: UUID | None, dataset_id: UUID, triggered_by_user_id: UUID | None = None
) -> UUID:
    """Register a run in ``INITIATED`` and return its id.

    Split out of :func:`run_memory_score` so a caller that runs the score in the
    background can hand a run id back to its client before the work starts —
    ``run_memory_score`` then claims this row instead of creating a second one.
    """
    run_id = uuid4()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(
            MemoryScoreRun(
                id=run_id,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                triggered_by_user_id=triggered_by_user_id,
                status=MemoryScoreRunStatus.INITIATED,
                below_data_floor=False,
                schema_defined=False,
                synthetic_question_count=0,
                real_question_count=0,
            )
        )
        await session.commit()

    return run_id


async def find_active_memory_score_run(tenant_id: UUID | None) -> UUID | None:
    """Return the tenant's active run id, or None.

    Active means ``INITIATED`` or ``RUNNING`` and younger than
    ``ACTIVE_RUN_STALE_AFTER``. Best-effort: there is no unique constraint
    backing this, so two requests racing on an empty table can both pass. It
    exists to stop the ordinary double-click / double-schedule case from
    spending a second run's worth of LLM tokens, not to be a lock.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        run = (
            await session.scalars(
                select(MemoryScoreRun)
                .where(_tenant_filter(tenant_id))
                .where(MemoryScoreRun.status.in_(_ACTIVE_STATUSES))
                .where(MemoryScoreRun.created_at >= _now() - ACTIVE_RUN_STALE_AFTER)
                .order_by(MemoryScoreRun.created_at.desc())
                .limit(1)
            )
        ).first()

    return run.id if run is not None else None


async def _claim_initiated_run(tenant_id: UUID | None) -> UUID | None:
    """Take over the tenant's newest ``INITIATED`` run, flipping it to ``RUNNING``.

    The flip is a conditional UPDATE rather than a read-modify-write, so two
    runners racing on the same row cannot both claim it: the loser's UPDATE
    matches nothing and it returns None (its caller then reports the run already
    in progress instead of starting a second one).

    Any OTHER ``INITIATED`` row for the tenant is retired in the same
    transaction. Such a row is one nobody will ever execute — a racing
    double-POST, or a process that died between registering and claiming — and
    left alone it would keep counting as active for ``ACTIVE_RUN_STALE_AFTER``,
    locking the tenant out while the client holding its id polled ``INITIATED``
    forever.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        run_id = (
            await session.scalars(
                select(MemoryScoreRun.id)
                .where(_tenant_filter(tenant_id))
                .where(MemoryScoreRun.status == MemoryScoreRunStatus.INITIATED)
                .where(MemoryScoreRun.created_at >= _now() - ACTIVE_RUN_STALE_AFTER)
                .order_by(MemoryScoreRun.created_at.desc())
                .limit(1)
            )
        ).first()

        if run_id is None:
            return None

        claimed = await session.execute(
            update(MemoryScoreRun)
            .where(MemoryScoreRun.id == run_id)
            .where(MemoryScoreRun.status == MemoryScoreRunStatus.INITIATED)
            .values(status=MemoryScoreRunStatus.RUNNING)
        )
        if not claimed.rowcount:
            await session.rollback()
            return None

        await session.execute(
            update(MemoryScoreRun)
            .where(_tenant_filter(tenant_id))
            .where(MemoryScoreRun.status == MemoryScoreRunStatus.INITIATED)
            .where(MemoryScoreRun.id != run_id)
            .values(
                status=MemoryScoreRunStatus.ERRORED,
                error=f"Superseded by memory score run {run_id}.",
                completed_at=_now(),
            )
        )
        await session.commit()

    return run_id


async def _update_run(run_id: UUID, **fields: Any) -> None:
    """Patch a run row. Never raises for a missing row — the run id is ours."""
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        run = await session.get(MemoryScoreRun, run_id)
        if run is None:
            logger.warning("memory score: run %s disappeared before it could be updated", run_id)
            return
        for key, value in fields.items():
            setattr(run, key, value)
        await session.commit()


async def _persist_questions(run_id: UUID, rows: list[dict[str, Any]]) -> None:
    """Write one ``ScoredQuestion`` row per answered question."""
    if not rows:
        return

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        for row in rows:
            session.add(ScoredQuestion(id=uuid4(), run_id=run_id, **row))
        await session.commit()


async def _entity_type_names() -> set[str]:
    """Names of the graph's ``EntityType`` nodes.

    Needed to read the schema inventory correctly. ``get_schema_inventory``
    reports an extracted Entity under the NAME of the EntityType its ``is_a``
    edge points at, and drops the literal ``EntityType`` type from the inventory
    altogether, so a plain cognify (which always has an LLM invent those names)
    yields inventory types like "person" / "organization". Without excluding
    them, every cognified graph would look like it had a schema defined.
    """
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()
    nodes, _ = await graph_engine.get_filtered_graph_data([{"type": ["EntityType"]}])

    names: set[str] = set()
    for node in nodes:
        try:
            _, properties = node
        except (TypeError, ValueError):
            continue
        name = properties.get("name") if isinstance(properties, dict) else None
        if isinstance(name, str) and name:
            names.add(name)
    return names


async def _detect_schema_defined() -> bool:
    """Raw ``schema_defined`` signal, derived from the schema inventory.

    True when the graph carries a node type that ingestion does not produce on
    its own and that is not part of the extracted entity-type taxonomy — i.e. a
    user-defined graph model. ``samples_per_type=0`` because only the type names
    matter here.

    Deliberately conservative, and in one known direction: an ontology whose
    classes land as EntityType names is indistinguishable from LLM-invented
    entity types and reports False. Best-effort too — a read failure reports
    False rather than failing the run, since the score does not depend on it.
    """
    from cognee.api.v1.visualize.get_schema_inventory import get_schema_inventory

    try:
        inventory = await get_schema_inventory(samples_per_type=0)
        entity_type_names = await _entity_type_names()
    except Exception as error:
        logger.warning("memory score: could not read the schema inventory: %s", error)
        return False

    return any(
        record.get("type")
        and record["type"] not in _INGESTION_TYPES
        and record["type"] not in entity_type_names
        for record in inventory
    )


def _build_retriever():
    """A fresh retriever pinned to the run configuration.

    One per question: the constructor only assigns fields, and the retrievers
    cache an engine handle on ``self`` during retrieval, so a per-question
    instance keeps the bounded fan-out free of shared mutable state.
    """
    return retriever_options[MEMORY_SCORE_RETRIEVER](top_k=MEMORY_SCORE_TOP_K)


def _as_text(value: Any) -> str:
    """Flatten a retriever's answer/context (str, list, or dict) into text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n\n".join(_as_text(item) for item in value)
    return str(value)


async def _answer(
    executor: AnswerGeneratorExecutor, question: str, golden_answer: str
) -> dict[str, Any]:
    """Answer one question through cognee recall.

    Reuses ``AnswerGeneratorExecutor`` one question at a time — its own loop is
    strictly sequential, and this run needs bounded concurrency instead. The
    executor reads ``instance["answer"]`` unconditionally, so a real question
    (which has no golden answer) must still pass ``golden_answer=""``.
    """
    try:
        records = await executor.question_answering_non_parallel(
            questions=[{"question": question, "answer": golden_answer}],
            retriever=_build_retriever(),
        )
    except Exception as error:
        logger.warning("memory score: recall failed for %r: %s", question[:80], error)
        return {"answer": "", "context": "", "error": f"Recall failed: {error}"}

    if not records:
        return {"answer": "", "context": "", "error": "Recall returned no answer."}

    record = records[0]
    return {
        "answer": _as_text(record.get("answer")),
        "context": _as_text(record.get("retrieval_context")),
        "error": None,
    }


async def _score_synthetic_question(
    executor: AnswerGeneratorExecutor,
    judge: DirectLLMEvalAdapter,
    question: GeneratedQuestion,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Answer + correctness-judge one synthetic question.

    ``grounded`` stays NULL: a synthetic question has a golden answer, so it is
    scored for correctness and never mixed into the groundedness signal.

    ``score`` distinguishes two outcomes that both leave the judge unconsulted,
    because they mean opposite things for the headline number:

    * recall or the judge FAILED -> ``score`` stays NULL. Unmeasured. The row is
      excluded from ``overall_accuracy``'s denominator, because an outage is not
      evidence about the memory.
    * recall SUCCEEDED and returned an empty answer -> ``score`` is 0.0. This
      question was generated from a chunk that is definitely in the graph, so
      recall finding nothing for it is precisely the failure this score exists to
      measure. Dropping it from the denominator instead would inflate
      ``overall_accuracy`` on exactly the thin or degraded datasets the feature is
      meant to flag.
    """
    row: dict[str, Any] = {
        "topic": question.topic,
        "source": SOURCE_SYNTHETIC,
        "text": question.text,
        "expected_answer": question.expected_answer,
        "answer": None,
        "score": None,
        "grounded": None,
        # Coverage is not knowable from a synthetic question: it was generated from a
        # chunk that is in the graph, so it can never evidence a gap.
        "answered": None,
        "reason": None,
        "source_query_id": None,
    }

    async with semaphore:
        answered = await _answer(executor, question.text, question.expected_answer)
        row["answer"] = answered["answer"]

        if answered["error"]:
            row["reason"] = answered["error"]
            return row

        if not answered["answer"].strip():
            row["score"] = 0.0
            row["reason"] = "Recall produced an empty answer."
            return row

        try:
            verdict = await judge.evaluate_correctness(
                question=question.text,
                answer=answered["answer"],
                golden_answer=question.expected_answer,
            )
        except Exception as error:
            logger.warning(
                "memory score: correctness judge failed for %r: %s", question.text[:80], error
            )
            row["reason"] = f"Correctness judge failed: {error}"
            return row

    row["score"] = verdict.get("score")
    row["reason"] = verdict.get("reason")
    return row


async def _score_real_question(
    executor: AnswerGeneratorExecutor,
    judge: GroundednessAdapter,
    query_id: UUID | None,
    question: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Answer + judge one real question for COVERAGE and groundedness.

    ``expected_answer`` and ``score`` stay NULL: a question the tenant actually
    asked has no golden answer, so correctness is not knowable and is never
    guessed at. What is knowable without one is two independent booleans:

    * ``answered`` — did the memory supply what was asked, or decline? This is the
      only place a COVERAGE gap can surface. Synthetic questions cannot reveal one,
      because they are generated from chunks that exist by construction, so a
      question the tenant actually asked is the sole input that can show the memory
      holding nothing relevant.
    * ``grounded`` — was what the answer asserted supported by the retrieved
      context? The hallucination signal.

    Both NULL means unmeasured (recall or judge failure), never "no".
    """
    row: dict[str, Any] = {
        "topic": None,
        "source": SOURCE_REAL,
        "text": question,
        "expected_answer": None,
        "answer": None,
        "score": None,
        "grounded": None,
        "answered": None,
        "reason": None,
        "source_query_id": query_id,
    }

    async with semaphore:
        answered = await _answer(executor, question, "")
        row["answer"] = answered["answer"]

        if answered["error"]:
            # UNMEASURED: an infrastructure failure is not a verdict about the
            # memory, and neither the coverage nor the hallucination list may be
            # polluted by a recall outage. Both booleans stay NULL, exactly as
            # ``score`` does on the synthetic path for the same class of failure.
            row["reason"] = answered["error"]
            return row

        if not answered["answer"].strip():
            # Recall succeeded and produced nothing to say: a COVERAGE gap, and the
            # clearest one there is. Not a groundedness verdict — an empty answer
            # asserts nothing, so there is nothing to be unsupported.
            row["answered"] = False
            row["reason"] = "Recall produced an empty answer."
            return row

        if not answered["context"].strip():
            # Nothing was retrieved, so the memory held nothing relevant. That is a
            # coverage gap decided without paying for a judge call. ``grounded``
            # stays NULL rather than False: with no context, "supported by the
            # context" has no truth value, and forcing False here would file a
            # coverage gap as a hallucination.
            row["answered"] = False
            row["reason"] = "No context was retrieved for this question."
            return row

        try:
            verdict = await judge.evaluate_groundedness(
                question=question,
                answer=answered["answer"],
                context=answered["context"],
            )
        except Exception as error:
            logger.warning(
                "memory score: groundedness judge failed for %r: %s", question[:80], error
            )
            row["reason"] = f"Groundedness judge failed: {error}"
            return row

    row["answered"] = verdict.get("answered")
    row["grounded"] = verdict.get("grounded")
    row["reason"] = verdict.get("reason")
    return row


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _synthetic_scores(rows: list[dict[str, Any]]) -> list[float]:
    """Scores of the synthetic rows that produced a verdict.

    A row is included whenever ``score`` is not NULL, which covers both a judged
    answer and the 0.0 recorded for an answer recall could not produce at all (see
    :func:`_score_synthetic_question`). Only rows left UNMEASURED by a recall or
    judge failure are excluded, so ``overall_accuracy``'s denominator is exactly
    the questions the memory was actually given a fair chance at.
    """
    return [row["score"] for row in rows if row["score"] is not None]


def _aggregate_topics(
    topic_plan: TopicPlan, synthetic_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Per-topic aggregate for ``MemoryScoreRun.topics``.

    ``accuracy`` is the mean CORRECTNESS score of the topic's synthetic
    questions and is None when none of them could be judged — never 0.0, which
    would read as "wrong" instead of "unmeasured".

    ``real_count`` / ``from_real_traffic`` come from the topic plan: clustering
    assigns real questions to their nearest topic centroid, so the count is how
    much real traffic hits the topic. Real questions themselves are stored with
    ``topic=NULL`` because that assignment is a weight, not a per-question label,
    and they carry no per-topic accuracy of any kind.
    """
    scores: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for row in synthetic_rows:
        label = row["topic"]
        counts[label] = counts.get(label, 0) + 1
        if row["score"] is not None:
            scores.setdefault(label, []).append(row["score"])

    # Distinct cluster labels can collide; merge them into one row rather than
    # reporting the same synthetic_count twice.
    merged: dict[str, int] = {}
    for topic in topic_plan.topics:
        merged[topic.label] = merged.get(topic.label, 0) + topic.real_question_count

    return [
        {
            "topic": label,
            "accuracy": _mean(scores.get(label, [])),
            "synthetic_count": counts.get(label, 0),
            "real_count": real_count,
            "from_real_traffic": real_count > 0,
        }
        for label, real_count in merged.items()
    ]


async def run_memory_score(
    tenant_id: UUID | None,
    dataset_id: UUID,
    triggered_by_user_id: UUID | None = None,
    synthetic_target: int = 100,
    real_question_limit: int = 20,
    user=None,
) -> UUID:
    """Score one dataset's memory accuracy and return the run id.

    Args:
        tenant_id: tenant that owns the dataset. ``User.tenant_id`` is nullable,
            so None is accepted and means the NULL-tenant (OSS) case.
        dataset_id: the dataset to score. REQUIRED and never inferred — see the
            scope note below. ``triggered_by_user_id`` must hold READ permission
            on it.
        triggered_by_user_id: acting user, or None for a scheduled run. Two jobs
            beyond attribution: it is the identity whose dataset READ permission
            is checked, and it is whose real question history gets replayed.
        synthetic_target: total synthetic questions to aim for, split across
            topics by real-traffic weight. Clamped to ``MAX_SYNTHETIC_TARGET``.
        real_question_limit: how many of the acting user's most recent real
            questions to replay. Clamped to ``MAX_REAL_QUESTION_LIMIT``.
        user: the acting ``User``, when the caller already has it. Only used to
            fill in ``triggered_by_user_id``.

    Returns:
        The ``MemoryScoreRun.id``, always — including for a gated
        (``SKIPPED_INSUFFICIENT_DATA``) or failed (``ERRORED``) run. Callers read
        the outcome off the run document, which is why the run id is the return
        value rather than a score.

    Raises:
        MemoryScoreRunInProgressError: the tenant already has an active run.
            This is the only exception that escapes — it is raised before any
            row is written, so there is no run to attribute it to. Every later
            failure is recorded on the run as ``ERRORED`` with the message in
            ``error`` and is NOT re-raised, matching how the repo handles
            background pipeline failures (see ``_perform_background_sync`` in
            cognee/api/v1/sync/sync.py): the run row is the channel for the
            outcome, and a background task raising into the event loop would
            report the failure nowhere.

    Scope note: the graph is scored for ONE dataset, and the caller names it.
    Under ``ENABLE_BACKEND_ACCESS_CONTROL`` (on by default whenever the
    configured graph/vector DBs support it) every user+dataset pair has its own
    graph database, so there is no single "tenant graph" to score and no
    defensible way to infer which dataset was meant. Callers state it:
    ``default_dataset`` on Cloud, ``main_dataset`` in OSS. Clustering, the chunk
    floor and recall all run inside that dataset's context, entered here via
    ``set_database_global_context_variables``.

    Real questions are read PER USER, not tenant-wide. ``queries`` rows are one
    member's search history, and the run document hands their verbatim text back
    to whoever reads it (in ``questions[].text`` and
    ``ungrounded_real_questions``), so a tenant-wide read would show every member
    what their colleagues have been searching for. The run therefore replays only
    the questions of the user it is attributed to — the trigger, or the dataset
    owner for a scheduled run. The cost is that the real-traffic topic weighting
    reflects one member's usage rather than the tenant's, so two members
    triggering a run over the same dataset can get slightly different topic
    splits; ``overall_accuracy`` itself stays comparable, since it comes from
    synthetic questions and the retriever and top_k are pinned.

    Real questions still carry no dataset attribution — ``queries`` records
    ``text``, ``query_type``, ``user_id`` and ``created_at``, no dataset — so a
    user active in several datasets can have their questions replayed against a
    dataset they were never asked against, which shows up as ungrounded answers.
    The run reports the dataset it scored so a caller can caveat that; fixing it
    properly needs a ``dataset_id`` on ``Query`` at log time, which would only
    help questions logged after it ships.
    """
    if triggered_by_user_id is None and user is not None:
        triggered_by_user_id = getattr(user, "id", None)

    # Clamped, not rejected: the HTTP layer already refuses an out-of-range value
    # with a 400, so anything arriving here past a cap came from a scheduler or an
    # in-process caller, and a run capped at the documented ceiling is a better
    # outcome for those than either a crash or unbounded spend.
    synthetic_target = max(0, min(synthetic_target, MAX_SYNTHETIC_TARGET))
    real_question_limit = max(0, min(real_question_limit, MAX_REAL_QUESTION_LIMIT))

    # Resolved before any run row is written, so a bad dataset id 404s (and an
    # unpermitted one 403s) without leaving an INITIATED row behind to block the
    # tenant's next attempt.
    dataset = await resolve_memory_score_dataset(tenant_id, dataset_id, triggered_by_user_id)

    # A caller that pre-registered the run (background mode) left an INITIATED
    # row behind; claim it so the run id it already handed out is the one that
    # gets executed. Otherwise register a fresh run.
    run_id = await _claim_initiated_run(tenant_id)
    if run_id is None:
        active_run_id = await find_active_memory_score_run(tenant_id)
        if active_run_id is not None:
            raise MemoryScoreRunInProgressError(
                message=(
                    "A memory score run is already in progress for this tenant "
                    f"(run {active_run_id})."
                )
            )
        run_id = await create_memory_score_run(tenant_id, dataset_id, triggered_by_user_id)
        await _update_run(run_id, status=MemoryScoreRunStatus.RUNNING)
    else:
        # Claiming matches on tenant, not dataset, so re-stamp the row: it must
        # name the dataset actually scored, not whichever one it was registered
        # with.
        await _update_run(run_id, dataset_id=dataset_id)

    logger.info("memory score: run %s started for tenant %s", run_id, tenant_id)

    try:
        # Every graph/vector read below — the chunk floor, clustering and every
        # recall — must happen inside the dataset's own database context, or under
        # ENABLE_BACKEND_ACCESS_CONTROL it would read an empty ambient graph.
        # owner_id, not the triggering user: a scheduled run has no user.
        async with set_database_global_context_variables(dataset.id, dataset.owner_id):
            schema_defined = await _detect_schema_defined()

            # Per user, never tenant-wide — see the docstring. The dataset owner
            # stands in for a scheduled run, which has no acting user but does
            # have someone the report is for.
            question_owner_id = triggered_by_user_id or dataset.owner_id
            queries = (
                await get_queries(question_owner_id, real_question_limit)
                if question_owner_id is not None and real_question_limit > 0
                else []
            )
            real_questions = [
                (query.id, query.text.strip())
                for query in queries
                if query.text and query.text.strip()
            ]

            topic_plan = await build_topics([text for _, text in real_questions])

            if topic_plan.below_data_floor:
                # Gate hit. Persist the raw signals and stop: nothing is generated,
                # nothing is answered, nothing is judged, no LLM token is spent.
                await _update_run(
                    run_id,
                    status=MemoryScoreRunStatus.SKIPPED_INSUFFICIENT_DATA,
                    below_data_floor=True,
                    floor_reason=topic_plan.floor_reason,
                    schema_defined=schema_defined,
                    overall_accuracy=None,
                    synthetic_question_count=0,
                    real_question_count=0,
                    topics=_aggregate_topics(topic_plan, []),
                    completed_at=_now(),
                )
                logger.info(
                    "memory score: run %s skipped below the data floor (%s)",
                    run_id,
                    topic_plan.floor_reason,
                )
                return run_id

            generated = await generate_questions(topic_plan, synthetic_target)

            executor = AnswerGeneratorExecutor()
            correctness_judge = DirectLLMEvalAdapter()
            groundedness_judge = GroundednessAdapter()

            # One semaphore across BOTH question sets, so it caps the run's total
            # in-flight LLM work rather than each set's.
            semaphore = asyncio.Semaphore(MEMORY_SCORE_CONCURRENCY)

            synthetic_rows, real_rows = await asyncio.gather(
                asyncio.gather(
                    *(
                        _score_synthetic_question(executor, correctness_judge, question, semaphore)
                        for question in generated
                    )
                ),
                asyncio.gather(
                    *(
                        _score_real_question(
                            executor, groundedness_judge, query_id, question, semaphore
                        )
                        for query_id, question in real_questions
                    )
                ),
            )

            await _persist_questions(run_id, synthetic_rows + real_rows)

            await _update_run(
                run_id,
                status=MemoryScoreRunStatus.COMPLETED,
                below_data_floor=False,
                floor_reason=None,
                schema_defined=schema_defined,
                # SYNTHETIC ONLY. Real questions have no golden answer, so folding
                # their groundedness in here would invent a correctness number.
                overall_accuracy=_mean(_synthetic_scores(synthetic_rows)),
                synthetic_question_count=len(synthetic_rows),
                real_question_count=len(real_rows),
                topics=_aggregate_topics(topic_plan, synthetic_rows),
                completed_at=_now(),
            )

            logger.info(
                "memory score: run %s completed with %d synthetic and %d real question(s)",
                run_id,
                len(synthetic_rows),
                len(real_rows),
            )
    except Exception as error:
        logger.error("memory score: run %s failed: %s", run_id, error, exc_info=True)
        await _update_run(
            run_id,
            status=MemoryScoreRunStatus.ERRORED,
            error=str(error),
            completed_at=_now(),
        )

    return run_id


async def get_memory_score_run(run_id: UUID) -> MemoryScoreRun | None:
    """Return one run row, or None when it does not exist."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        return await session.get(MemoryScoreRun, run_id)


async def get_latest_memory_score_run(
    tenant_id: UUID | None, dataset_ids: set[str] | None = None
) -> MemoryScoreRun | None:
    """Return the most recent run, or None when there is none to return.

    Args:
        tenant_id: tenant to scope to. None means the NULL-tenant (OSS) case.
        dataset_ids: when given, only runs over one of these datasets count —
            the caller's readable set. "Latest" then means the latest run the
            caller is allowed to see, not the tenant's latest, so a member cannot
            read a run over a dataset they have no permission on just because a
            colleague scored it more recently. An empty set matches nothing.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        statement = select(MemoryScoreRun).where(_tenant_filter(tenant_id))

        if dataset_ids is not None:
            # Cast to the column's own type: a UUID column may round-trip as
            # UUID or str depending on the driver, and IN needs the driver's form.
            statement = statement.where(
                MemoryScoreRun.dataset_id.in_([UUID(dataset_id) for dataset_id in dataset_ids])
            )

        return (
            await session.scalars(statement.order_by(MemoryScoreRun.created_at.desc()).limit(1))
        ).first()


async def get_memory_score_questions(run_id: UUID) -> list[ScoredQuestion]:
    """Return a run's scored questions, oldest first."""
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        questions = (
            await session.scalars(
                select(ScoredQuestion)
                .where(ScoredQuestion.run_id == run_id)
                .order_by(ScoredQuestion.created_at.asc())
            )
        ).all()

    return list(questions)


def build_memory_score_document(
    run: MemoryScoreRun, questions: list[ScoredQuestion]
) -> dict[str, Any]:
    """Assemble the full run document served by the score API.

    Raw signals only. There is deliberately no call-to-action field: the
    backend does not decide "upload more data" vs "define a schema", it reports
    ``below_data_floor``, ``floor_reason``, ``schema_defined``, the per-topic
    accuracies and the ungrounded real questions, and the UI applies its own
    thresholds.

    ``score`` is populated for synthetic questions only and ``grounded`` for
    real ones only; both keys are always present (null where they do not apply)
    so the shape never changes between runs.

    ``dataset_id`` names the dataset actually scored, so a caller never has to
    infer what the number covers.

    ``judged_synthetic_question_count`` is ``overall_accuracy``'s real
    DENOMINATOR, and it is reported because it is not derivable from
    ``synthetic_question_count`` — that one counts every synthetic row asked,
    including the ones a recall or judge failure left unmeasured. A consumer
    comparing the two can see how much of the run actually landed, and can
    distinguish "82% over 100 questions" from "82% over 9 of them".

    ``coverage`` is a SECOND KPI, deliberately not folded into
    ``overall_accuracy``. The two answer different questions — accuracy is "of what
    the memory attempted, how much was right", coverage is "of what was asked, how
    much could it attempt at all" — and a thin memory scores HIGH on the first and
    LOW on the second. Blending them would produce a number where 0.6 could mean
    either wrong answers or absent data, which are opposite remedies. Kept apart,
    the pair reads directly: low coverage means the data is missing, while high
    coverage with low accuracy means the data is there and recall is failing.

    Coverage comes only from real questions. A synthetic question is generated from a
    chunk that is in the graph, so it can never evidence a gap; a question the tenant
    actually asked is the only input that can. It is therefore null when the run
    replayed no real questions, and it inherits their sample size — see
    ``real_question_limit``.
    """
    status_value = run.status.value if isinstance(run.status, MemoryScoreRunStatus) else run.status

    judged_synthetic_question_count = sum(
        1
        for question in questions
        if question.source == SOURCE_SYNTHETIC and question.score is not None
    )

    # COVERAGE, over the real questions that produced a verdict. Rows left NULL by a
    # recall or judge failure are excluded from both sides of the fraction, so an
    # outage lowers confidence in the number rather than the number itself.
    measured_real = [
        question
        for question in questions
        if question.source == SOURCE_REAL and question.answered is not None
    ]
    answered_real = [question for question in measured_real if question.answered]
    coverage = len(answered_real) / len(measured_real) if measured_real else None

    return {
        "run_id": str(run.id),
        "status": status_value,
        # Always name the dataset scored: a tenant with several datasets needs to
        # know which one this number describes, and the replayed real questions
        # are tenant-wide so some may not belong to it.
        "dataset_id": str(run.dataset_id) if run.dataset_id else None,
        "below_data_floor": bool(run.below_data_floor),
        "floor_reason": run.floor_reason,
        "schema_defined": bool(run.schema_defined),
        "overall_accuracy": run.overall_accuracy,
        "synthetic_question_count": run.synthetic_question_count or 0,
        "judged_synthetic_question_count": judged_synthetic_question_count,
        "real_question_count": run.real_question_count or 0,
        "coverage": coverage,
        "measured_real_question_count": len(measured_real),
        "answered_real_question_count": len(answered_real),
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "topics": run.topics if isinstance(run.topics, list) else [],
        "questions": [
            {
                "text": question.text,
                "topic": question.topic,
                "source": question.source,
                "answer": question.answer,
                "expected_answer": question.expected_answer,
                "score": question.score,
                "grounded": question.grounded,
                "answered": question.answered,
                "reason": question.reason,
            }
            for question in questions
        ],
        # Two lists, because they are two different failures with two different
        # remedies. Ungrounded means the memory answered and the answer was not
        # supported — a hallucination. Unanswerable means the memory had nothing to
        # say — a coverage gap. The second is the "questions your users asked that
        # could not be answered" list; the first must never be read as that.
        "ungrounded_real_questions": [
            question.text
            for question in questions
            if question.source == SOURCE_REAL and question.grounded is False
        ],
        "unanswerable_real_questions": [
            question.text
            for question in questions
            if question.source == SOURCE_REAL and question.answered is False
        ],
    }
