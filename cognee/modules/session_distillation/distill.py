"""Distill a finished session's learnings into the persistent knowledge graph.

Flow (curator calls parallel by batch; accept/write calls parallel by lesson):

1. LOAD    session QA turns + distillable session-context entries + recoverable evidence.
2. CURATE  pack the session timeline into batches; one curator LLM call per batch.
3. ACCEPT  per proposed lesson: search prior lessons/entities, then writer/rejecter LLM.
4. PERSIST render accepted lessons as documents; add + cognify them in one pass.

Everything is fail-open per unit: a failed curator batch or writer call drops only its own
work, never the whole run. Evidence recovered from abandoned maintenance (see
``evidence.py``) is the one exception: a batch built from it is consumed only when every
call in that batch succeeded, so a partial failure leaves it retryable.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Union
from uuid import UUID

from cognee.context_global_variables import session_user, set_database_global_context_variables
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_context_builder import coerce_active_context_entries
from cognee.infrastructure.session.session_context_models import SessionContextEntry
from cognee.infrastructure.session.session_maintenance_worker import get_tracked_evidence_ids
from cognee.infrastructure.session.session_search_models import SessionTurnEvidence
from cognee.modules.data.models import Dataset
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.truth_subspace.constants import truth_session_node_set
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.async_utils import gather_with_concurrency_limit
from cognee.shared.logging_utils import get_logger

from .evidence import (
    build_evidence_batches,
    load_eligible_evidence,
    mark_evidence_distilled,
)
from .models import (
    CURATOR_BLOCKS_PER_BATCH,
    CURATOR_CONCURRENCY,
    GLOSSARY_ENTITIES_PER_LESSON,
    MAX_CANDIDATE_CHARS,
    MAX_QA_ANSWER_CHARS,
    MAX_QA_QUESTION_CHARS,
    MIN_GATE_CONFIDENCE,
    NOVELTY_LESSONS_PER_LESSON,
    WRITER_CONCURRENCY,
    CuratorBatchOutcome,
    CuratorBatchOutput,
    DistillationInputBatch,
    DistillationResult,
    ProposedLesson,
    WrittenLesson,
)

logger = get_logger("session_distillation")

CURATOR_PROMPT_FILE = "session_distillation_curator_system.txt"
WRITER_PROMPT_FILE = "session_distillation_writer_system.txt"

# Node set marking distillate documents in the graph: used to tag them on write and to
# scope the novelty search to previously persisted lessons.
DISTILLATE_NODE_SET = ["session_learnings"]


@dataclass(frozen=True, slots=True)
class SessionDistillationScope:
    """Resolved identity for one session distillation run."""

    session_id: str
    user: User
    dataset: Dataset

    @property
    def user_id(self) -> str:
        return str(self.user.id)

    @property
    def dataset_id(self) -> str:
        return str(self.dataset.id)

    def result(self, status: str, documents: Optional[List[str]] = None) -> DistillationResult:
        return DistillationResult(
            session_id=self.session_id,
            dataset_id=self.dataset_id,
            status=status,
            documents=documents or [],
        )


async def resolve_distillation_scope(
    *,
    session_id: str,
    dataset: Union[str, UUID],
    user: Optional[User],
) -> SessionDistillationScope:
    resolved_user = user if user is not None else session_user.get()
    if resolved_user is None or getattr(resolved_user, "id", None) is None:
        resolved_user = await get_default_user()

    if dataset is None:
        raise CogneeValidationError(
            message=(
                "dataset is required so the distilled learnings land in the graph "
                "they should connect to."
            ),
            name="SessionDistillationError",
        )

    writable_datasets = await get_authorized_existing_datasets([dataset], "write", resolved_user)
    if not writable_datasets:
        raise CogneeValidationError(
            message=f"Dataset '{dataset}' not found or not writable for this user.",
            name="SessionDistillationError",
        )

    return SessionDistillationScope(
        session_id=session_id,
        user=resolved_user,
        dataset=writable_datasets[0],
    )


async def load_distillable_session_inputs(
    scope: SessionDistillationScope,
    tracked_evidence_ids: set[str],
) -> tuple[List[dict], List[SessionContextEntry], List[SessionTurnEvidence]]:
    """Load QA turns, context entries worth distilling, and recoverable turn evidence."""
    session_manager = get_session_manager()
    context_rows = await session_manager.get_session_context_entries_strict(
        user_id=scope.user_id,
        session_id=scope.session_id,
    )
    raw_qa = await session_manager.get_session(
        user_id=scope.user_id,
        session_id=scope.session_id,
        formatted=False,
    )

    qa_rows = [
        entry.model_dump() if hasattr(entry, "model_dump") else dict(entry)
        for entry in (raw_qa if isinstance(raw_qa, list) else [])
    ]
    context_entries = [
        entry
        for entry in coerce_active_context_entries(context_rows)
        if entry.harmful_count == 0 and entry.confidence >= MIN_GATE_CONFIDENCE
    ]
    evidence = load_eligible_evidence(
        context_rows,
        dataset_id=scope.dataset_id,
        tracked_evidence_ids=tracked_evidence_ids,
    )
    return qa_rows, context_entries, evidence


def build_curator_batches(
    qa_rows: List[dict],
    context_entries: List[SessionContextEntry],
) -> List[str]:
    """Pack the session timeline into coarse, size-safe chronological batches."""
    timeline: List[tuple[str, str]] = []
    for row in qa_rows:
        question = " ".join((row.get("question") or "").split())[:MAX_QA_QUESTION_CHARS]
        answer = " ".join((row.get("answer") or "").split())[:MAX_QA_ANSWER_CHARS]
        if not question and not answer:
            continue
        block = f"User: {question}\nAssistant: {answer}"
        timeline.append((row.get("time") or "", block))

    for entry in context_entries:
        content = " ".join(entry.content.split())[:MAX_CANDIDATE_CHARS]
        block = f"Candidate {entry.id} [{entry.context_profile}/{entry.section}]: {content}"
        timeline.append((entry.created_at or "", block))

    timeline.sort(key=lambda item: item[0])
    blocks = [block for _timestamp, block in timeline]

    return [
        "\n\n".join(blocks[index : index + CURATOR_BLOCKS_PER_BATCH])
        for index in range(0, len(blocks), CURATOR_BLOCKS_PER_BATCH)
    ]


async def curate_batch(batch: DistillationInputBatch) -> CuratorBatchOutcome:
    """Run one curator call while preserving failure versus successful-empty output."""
    system_prompt = read_query_prompt(CURATOR_PROMPT_FILE)
    if not system_prompt:
        logger.warning("Distillation curator prompt not found: %s", CURATOR_PROMPT_FILE)
        return CuratorBatchOutcome(batch=batch, succeeded=False)
    try:
        result = await LLMGateway.acreate_structured_output(
            text_input=batch.text,
            system_prompt=system_prompt,
            response_model=CuratorBatchOutput,
        )
        return CuratorBatchOutcome(
            batch=batch,
            succeeded=True,
            lessons=tuple(result.lessons),
        )
    except Exception as error:
        logger.warning("Distillation curator batch failed open: %s", error)
        return CuratorBatchOutcome(batch=batch, succeeded=False)


async def propose_lesson_batches(
    batches: List[DistillationInputBatch],
) -> List[CuratorBatchOutcome]:
    """Curate every batch concurrently, keeping each batch's outcome separate."""
    curator_calls = [lambda batch=batch: curate_batch(batch) for batch in batches]
    return await gather_with_concurrency_limit(curator_calls, CURATOR_CONCURRENCY)


async def search_payload_texts(
    vector_engine,
    collection: str,
    limit: int,
    *,
    query_text: str | None = None,
    query_vector: list | None = None,
    node_name: Optional[List[str]] = None,
) -> List[str]:
    """Vector-search one collection and return de-duplicated payload texts; [] on failure."""
    try:
        results = await vector_engine.search(
            collection,
            query_text=query_text,
            query_vector=query_vector,
            limit=limit,
            include_payload=True,
            node_name=node_name,
        )
    except Exception as error:
        logger.debug("Distillation search on %s failed open: %s", collection, error)
        return []

    texts: List[str] = []
    seen = set()
    for result in results or []:
        payload = getattr(result, "payload", None)
        if not isinstance(payload, dict):
            continue
        text = payload.get("text") or payload.get("name")
        if not text:
            continue
        text = str(text).strip()
        key = text.casefold()
        if text and key not in seen:
            texts.append(text)
            seen.add(key)
    return texts


def build_writer_input(
    lesson: ProposedLesson,
    members: List[SessionContextEntry],
    prior_lessons: List[str],
    glossary: List[str],
) -> str:
    sections = [f"PROPOSED LESSON:\n{lesson.working_statement}"]
    if members:
        sections.append(
            "MEMBER ENTRIES:\n" + "\n".join(f"- {member.content}" for member in members)
        )
    if prior_lessons:
        sections.append(
            "SIMILAR EXISTING LESSONS:\n" + "\n".join(f"- {prior}" for prior in prior_lessons)
        )
    if glossary:
        sections.append("ENTITY GLOSSARY:\n" + "\n".join(f"- {name}" for name in glossary))
    return "\n\n".join(sections)


async def write_or_reject(
    lesson: ProposedLesson,
    members: List[SessionContextEntry],
    prior_lessons: List[str],
    glossary: List[str],
) -> Optional[WrittenLesson]:
    """One writer/rejecter call for one proposed lesson. A failed call returns None."""
    system_prompt = read_query_prompt(WRITER_PROMPT_FILE)
    if not system_prompt:
        logger.warning("Distillation writer prompt not found: %s", WRITER_PROMPT_FILE)
        return None

    text_input = build_writer_input(lesson, members, prior_lessons, glossary)
    try:
        return await LLMGateway.acreate_structured_output(
            text_input=text_input,
            system_prompt=system_prompt,
            response_model=WrittenLesson,
        )
    except Exception as error:
        logger.warning("Distillation writer call failed open: %s", error)
        return None


async def evaluate_proposed_lesson(
    vector_engine,
    lesson: ProposedLesson,
    entries_by_id: dict,
) -> Optional[WrittenLesson]:
    members = [
        entries_by_id[entry_id] for entry_id in lesson.member_entry_ids if entry_id in entries_by_id
    ]
    prior_lessons, glossary = await asyncio.gather(
        search_payload_texts(
            vector_engine,
            "DocumentChunk_text",
            NOVELTY_LESSONS_PER_LESSON,
            query_text=lesson.working_statement,
            node_name=DISTILLATE_NODE_SET,
        ),
        search_payload_texts(
            vector_engine,
            "Entity_name",
            GLOSSARY_ENTITIES_PER_LESSON,
            query_text=lesson.working_statement,
        ),
    )
    return await write_or_reject(lesson, members, prior_lessons, glossary)


async def decide_lesson_batches(
    scope: SessionDistillationScope,
    curated_batches: List[CuratorBatchOutcome],
    context_entries: List[SessionContextEntry],
) -> tuple[List[WrittenLesson], set[str]]:
    """Judge every proposal, keeping only batches whose curator and writer calls all won.

    A batch containing any failure is dropped whole — its accepted siblings included — so
    its source evidence stays retryable instead of being half-consumed. Returns the
    accepted lessons and the evidence IDs those fully decided batches consumed.
    """
    entries_by_id = {entry.id: entry for entry in context_entries}
    curated_batches = [curated for curated in curated_batches if curated.succeeded]
    proposals = [lesson for curated in curated_batches for lesson in curated.lessons]

    decisions: List[Optional[WrittenLesson]] = []
    if proposals:
        async with set_database_global_context_variables(scope.dataset.id, scope.dataset.owner_id):
            vector_engine = await get_vector_engine_async()
            writer_calls = [
                lambda lesson=lesson: evaluate_proposed_lesson(
                    vector_engine,
                    lesson,
                    entries_by_id,
                )
                for lesson in proposals
            ]
            decisions = await gather_with_concurrency_limit(writer_calls, WRITER_CONCURRENCY)

    # Decisions come back in proposal order, so each batch takes its own slice back.
    remaining = iter(decisions)
    accepted: List[WrittenLesson] = []
    consumed_evidence_ids: set[str] = set()
    for curated in curated_batches:
        batch_decisions = [next(remaining) for _lesson in curated.lessons]
        if any(decision is None for decision in batch_decisions):
            continue
        accepted.extend(
            decision
            for decision in batch_decisions
            if decision.accept and decision.statement.strip()
        )
        consumed_evidence_ids.update(curated.batch.source_evidence_ids)
    return accepted, consumed_evidence_ids


def render_lesson_document(
    lesson: WrittenLesson,
    *,
    session_id: str,
    distilled_on: str,
) -> str:
    """Render ONE accepted lesson as a standalone markdown document.

    The template — not the LLM — controls the format. One document per lesson, so each
    learning is an independently identifiable unit in the graph.
    """
    statement = lesson.statement.strip()
    why = lesson.why_learned.strip().rstrip(".")
    body = f"{statement} ({why}.)" if why else statement
    return f"# Session learning — {distilled_on} (session {session_id})\n\n{body}\n"


async def publish_distilled_lessons(
    scope: SessionDistillationScope,
    accepted: List[WrittenLesson],
) -> List[str]:
    distilled_on = datetime.utcnow().strftime("%Y-%m-%d")
    documents = [
        render_lesson_document(lesson, session_id=scope.session_id, distilled_on=distilled_on)
        for lesson in accepted
    ]

    # Imported lazily to avoid a circular import through the cognee package root.
    from cognee.api.v1.add import add
    from cognee.api.v1.cognify import cognify

    node_set = [*DISTILLATE_NODE_SET, truth_session_node_set(scope.session_id)]
    await add(documents, dataset_id=scope.dataset.id, user=scope.user, node_set=node_set)
    await cognify(datasets=[scope.dataset.id], user=scope.user)
    return documents


async def distill_session(
    session_id: str,
    dataset: Union[str, UUID],
    user: Optional[User] = None,
) -> DistillationResult:
    """Distill one finished session's distillable learnings into its dataset's knowledge graph."""
    scope = await resolve_distillation_scope(session_id=session_id, dataset=dataset, user=user)

    qa_rows, context_entries, evidence = await load_distillable_session_inputs(
        scope,
        get_tracked_evidence_ids(),
    )
    if not context_entries and not evidence:
        return scope.result("no_gated_entries")

    batches = [
        DistillationInputBatch(text=text)
        for text in build_curator_batches(qa_rows, context_entries)
    ] + build_evidence_batches(evidence)
    curated_batches = await propose_lesson_batches(batches)
    accepted, consumed_evidence_ids = await decide_lesson_batches(
        scope, curated_batches, context_entries
    )

    # Publish first: a publication failure must leave every source evidence retryable.
    documents = await publish_distilled_lessons(scope, accepted) if accepted else []
    if consumed_evidence_ids:
        await mark_evidence_distilled(
            get_session_manager(),
            user_id=scope.user_id,
            session_id=scope.session_id,
            dataset_id=scope.dataset_id,
            evidence_ids=consumed_evidence_ids,
        )

    if not any(curated.lessons for curated in curated_batches):
        return scope.result("no_proposed_lessons")
    if not accepted:
        return scope.result("no_accepted_lessons")
    return scope.result("completed", documents=documents)
