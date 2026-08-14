"""Distill a finished session's learnings into the persistent knowledge graph.

Flow (curator calls parallel by batch; accept/write calls parallel by lesson):

1. LOAD    session QA turns + distillable session-context entries.
2. CURATE  pack the session timeline into batches; one curator LLM call per batch.
3. ACCEPT  per proposed lesson: search prior lessons/entities, then writer/rejecter LLM.
4. PERSIST render accepted lessons as documents; add + cognify them in one pass.

Everything is fail-open per unit: a failed curator batch or writer call drops only its own
work, never the whole run.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
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
from cognee.infrastructure.session.session_persist_watermark import (
    load_state_row,
    save_state_row,
)
from cognee.modules.data.models import Dataset
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.truth_subspace.constants import truth_session_node_set
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.async_utils import gather_with_concurrency_limit
from cognee.shared.logging_utils import get_logger

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
    CuratorBatchOutput,
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

# Watermark row (same pattern as agent_context_extraction's trace watermark): which
# gated entry ids a previous run already considered. Keyed per dataset so the same
# session distilled into two datasets tracks progress independently.
DISTILLATION_STATE_KIND = "distillation_state"


def _distillation_state_id(dataset_id: str) -> str:
    return f"__distillation_state__:{dataset_id}"


async def _load_processed_entry_ids(scope: "SessionDistillationScope") -> set:
    """Gated entry ids a previous distillation run already considered. Missing -> empty."""
    # No kind fallback: the state id is per dataset while the kind is shared,
    # so a kind match could return another dataset's watermark.
    row = await load_state_row(
        get_session_manager(),
        user_id=scope.user_id,
        session_id=scope.session_id,
        state_id=_distillation_state_id(scope.dataset_id),
    )
    values = (row or {}).get("processed_entry_ids")
    return {value for value in (values or []) if isinstance(value, str)}


async def _save_processed_entry_ids(scope: "SessionDistillationScope", entry_ids: set) -> None:
    """Persist the watermark; advances only after a run completes without raising."""
    await save_state_row(
        get_session_manager(),
        user_id=scope.user_id,
        session_id=scope.session_id,
        payload={
            "id": _distillation_state_id(scope.dataset_id),
            "kind": DISTILLATION_STATE_KIND,
            "processed_entry_ids": sorted(entry_ids),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


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
) -> tuple[List[dict], List[SessionContextEntry]]:
    """Load QA turns and keep context entries worth distilling."""
    session_manager = get_session_manager()
    context_rows = await session_manager.get_session_context_entries(
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

    # Net-helpfulness gate (same clamped signal the serving-time ranker uses):
    # entries rated harmful more often than helpful are excluded, but a single
    # misread harmful rating no longer disqualifies an entry for life, and
    # helpful ratings count as positive evidence.
    context_entries = [
        entry
        for entry in coerce_active_context_entries(context_rows)
        if (entry.helpful_count - entry.harmful_count) >= 0
        and entry.confidence >= MIN_GATE_CONFIDENCE
    ]
    return qa_rows, context_entries


def _timeline_sort_key(stamp: str) -> float:
    """Chronological key for mixed timestamp formats; unparseable stamps sort first.

    Lexical string comparison mis-orders mixed formats (offset vs naive, differing
    precision), so parse to epoch seconds and fall back cleanly.
    """
    try:
        return datetime.fromisoformat(str(stamp)).timestamp()
    except (TypeError, ValueError):
        return float("-inf")


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
        # Explicit feedback is the strongest durable signal a turn carries; without
        # this line the curator never saw it anywhere in the loop.
        feedback = " ".join((row.get("feedback_text") or "").split())[:MAX_CANDIDATE_CHARS]
        if feedback:
            block += f"\nUser feedback on this answer: {feedback}"
        timeline.append((row.get("time") or "", block))

    for entry in context_entries:
        content = " ".join(entry.content.split())[:MAX_CANDIDATE_CHARS]
        block = f"Candidate {entry.id} [{entry.context_profile}/{entry.section}]: {content}"
        timeline.append((entry.created_at or "", block))

    timeline.sort(key=lambda item: _timeline_sort_key(item[0]))
    blocks = [block for _timestamp, block in timeline]

    return [
        "\n\n".join(blocks[index : index + CURATOR_BLOCKS_PER_BATCH])
        for index in range(0, len(blocks), CURATOR_BLOCKS_PER_BATCH)
    ]


async def curate_batch(batch_text: str) -> List[ProposedLesson]:
    """One curator call over one batch slice. Fail-open -> []."""
    system_prompt = read_query_prompt(CURATOR_PROMPT_FILE)
    if not system_prompt:
        logger.warning("Distillation curator prompt not found: %s", CURATOR_PROMPT_FILE)
        return []
    try:
        result = await LLMGateway.acreate_structured_output(
            text_input=batch_text,
            system_prompt=system_prompt,
            response_model=CuratorBatchOutput,
        )
        return list(result.lessons)
    except Exception as error:
        logger.warning("Distillation curator batch failed open: %s", error)
        return []


async def propose_lessons(
    qa_rows: List[dict],
    context_entries: List[SessionContextEntry],
) -> List[ProposedLesson]:
    """Pack session inputs into curator batches, then flatten proposed lessons."""
    batches = build_curator_batches(qa_rows, context_entries)
    if not batches:
        return []

    curator_calls = [lambda batch=batch: curate_batch(batch) for batch in batches]
    per_batch = await gather_with_concurrency_limit(curator_calls, CURATOR_CONCURRENCY)

    proposed = [lesson for batch_lessons in per_batch for lesson in batch_lessons]
    # Batches are curated independently, so one theme restated across the session
    # comes back as near-identical proposals; dedupe before paying writer calls.
    return _dedupe_by_statement(proposed, lambda lesson: lesson.working_statement)


def _dedupe_by_statement(lessons: list, statement_of) -> list:
    """Keep the first lesson per normalized statement text."""
    seen = set()
    unique = []
    for lesson in lessons:
        key = " ".join(str(statement_of(lesson)).casefold().split())
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(lesson)
    return unique


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
    """One writer/rejecter call for one proposed lesson. Fail-open -> None."""
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


async def accept_proposed_lessons(
    scope: SessionDistillationScope,
    proposed: List[ProposedLesson],
    context_entries: List[SessionContextEntry],
) -> List[WrittenLesson]:
    entries_by_id = {entry.id: entry for entry in context_entries}
    async with set_database_global_context_variables(scope.dataset.id, scope.dataset.owner_id):
        vector_engine = await get_vector_engine_async()

        def write_lesson(lesson: ProposedLesson):
            return lambda: evaluate_proposed_lesson(
                vector_engine,
                lesson,
                entries_by_id,
            )

        writer_calls = [write_lesson(lesson) for lesson in proposed]
        decisions = await gather_with_concurrency_limit(writer_calls, WRITER_CONCURRENCY)

    accepted = [
        lesson
        for lesson in decisions
        if lesson is not None and lesson.accept and lesson.statement.strip()
    ]
    # Writers run per lesson and can converge on the same final statement.
    return _dedupe_by_statement(accepted, lambda lesson: lesson.statement)


def render_lesson_document(
    lesson: WrittenLesson,
    *,
    session_id: str,
) -> str:
    """Render ONE accepted lesson as a standalone markdown document.

    The template — not the LLM — controls the format. One document per lesson, so each
    learning is an independently identifiable unit in the graph. Deliberately no run
    date in the text: the same lesson re-accepted on a different day must produce an
    identical document so ingestion dedups it instead of storing a near-duplicate.
    """
    statement = lesson.statement.strip()
    why = lesson.why_learned.strip().rstrip(".")
    body = f"{statement} ({why}.)" if why else statement
    document = f"# Session learning — (session {session_id})\n\n{body}\n"
    # Entity provenance (stable across re-acceptance, so dedup still holds).
    # Member entry ids are deliberately NOT embedded: they are cache-internal and
    # differ per run, which would defeat content-hash dedup of identical lessons.
    entities = [name.strip() for name in lesson.entities if isinstance(name, str) and name.strip()]
    if entities:
        document += f"\nEntities: {', '.join(entities)}\n"
    return document


async def publish_distilled_lessons(
    scope: SessionDistillationScope,
    accepted: List[WrittenLesson],
) -> List[str]:
    documents = [render_lesson_document(lesson, session_id=scope.session_id) for lesson in accepted]

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
    """Distill one finished session's distillable learnings into its dataset's knowledge graph.

    A per-(session, dataset) watermark tracks which gated entries earlier runs already
    considered: re-running improve() on an unchanged session performs zero curator or
    writer LLM calls. The watermark advances only after a run completes without raising.
    """
    scope = await resolve_distillation_scope(session_id=session_id, dataset=dataset, user=user)

    qa_rows, context_entries = await load_distillable_session_inputs(scope)
    if not context_entries:
        return scope.result("no_gated_entries")

    processed_entry_ids = await _load_processed_entry_ids(scope)
    gated_entry_ids = {entry.id for entry in context_entries}
    if not (gated_entry_ids - processed_entry_ids):
        return scope.result("no_new_entries")

    async def finish(status: str, documents: Optional[List[str]] = None) -> DistillationResult:
        # Every gated entry was considered this run (curation sees the full timeline
        # for context), so the watermark covers them all.
        await _save_processed_entry_ids(scope, processed_entry_ids | gated_entry_ids)
        return scope.result(status, documents=documents)

    proposed = await propose_lessons(qa_rows, context_entries)
    if not proposed:
        return await finish("no_proposed_lessons")

    accepted = await accept_proposed_lessons(scope, proposed, context_entries)
    if not accepted:
        return await finish("no_accepted_lessons")

    documents = await publish_distilled_lessons(scope, accepted)
    return await finish("completed", documents=documents)
