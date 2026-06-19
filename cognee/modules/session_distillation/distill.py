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
from datetime import datetime
from typing import Awaitable, Callable, List, Optional, Union
from uuid import UUID

from cognee.context_global_variables import session_user, set_database_global_context_variables
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_context_builder import coerce_active_context_entries
from cognee.infrastructure.session.session_context_models import SessionContextEntry
from cognee.modules.data.models import Dataset
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

from .models import (
    BATCH_CHAR_BUDGET,
    CURATOR_CONCURRENCY,
    GLOSSARY_ENTITIES_PER_LESSON,
    MAX_QA_ANSWER_CHARS,
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


# -- Public entry point -------------------------------------------------------


async def distill_session(
    session_id: str,
    dataset: Union[str, UUID],
    user: Optional[User] = None,
) -> DistillationResult:
    """Distill one finished session's distillable learnings into its dataset's knowledge graph."""
    scope = await _resolve_distillation_scope(session_id=session_id, dataset=dataset, user=user)

    qa_rows, context_entries = await _load_distillable_session_inputs(scope)
    if not context_entries:
        return scope.result("no_gated_entries")

    proposed = await _propose_lessons(qa_rows, context_entries)
    if not proposed:
        return scope.result("no_proposed_lessons")

    accepted = await _accept_proposed_lessons(scope, proposed, context_entries)
    if not accepted:
        return scope.result("no_accepted_lessons")

    documents = await _publish_distilled_lessons(scope, accepted)
    return scope.result("completed", documents=documents)


# -- Load session inputs ------------------------------------------------------


async def _load_distillable_session_inputs(
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

    context_entries = [
        entry
        for entry in coerce_active_context_entries(context_rows)
        if entry.harmful_count == 0 and entry.confidence >= MIN_GATE_CONFIDENCE
    ]
    return qa_rows, context_entries


# -- Curate proposed lessons --------------------------------------------------


def build_batches(qa_rows: List[dict], context_entries: List[SessionContextEntry]) -> List[str]:
    """Pack the session timeline into size-bounded batches of turns + candidates.

    Turns and candidates are interleaved in chronological order, then greedily packed so
    each batch stays under the char budget. A single oversized block gets its own batch
    rather than being dropped. Each candidate carries its id so the curator can cite it.
    """
    timeline: List[tuple[str, str]] = []
    for row in qa_rows:
        question = " ".join((row.get("question") or "").split())
        answer = " ".join((row.get("answer") or "").split())[:MAX_QA_ANSWER_CHARS]
        if not question and not answer:
            continue
        timeline.append((row.get("time") or "", f"User: {question}\nAssistant: {answer}"))
    for entry in context_entries:
        block = f"Candidate {entry.id} ({entry.section}): {entry.content}"
        timeline.append((entry.created_at or "", block))

    timeline.sort(key=lambda item: item[0])

    batches: List[str] = []
    current: List[str] = []
    current_chars = 0
    for _timestamp, block in timeline:
        if current and current_chars + len(block) > BATCH_CHAR_BUDGET:
            batches.append("\n\n".join(current))
            current, current_chars = [], 0
        current.append(block)
        current_chars += len(block)
    if current:
        batches.append("\n\n".join(current))
    return batches


async def curate_batches(batches: List[str]) -> List[ProposedLesson]:
    """Run one curator call per batch in parallel; flatten the proposed lessons.

    Cross-batch duplicates are accepted here; the per-lesson writer/rejecter handles
    already-known lessons against the persisted graph.
    """
    per_batch = await _gather_bounded(
        [lambda b=batch: _curate_batch(b) for batch in batches], CURATOR_CONCURRENCY
    )
    return [lesson for batch_lessons in per_batch for lesson in batch_lessons]


async def _propose_lessons(
    qa_rows: List[dict],
    context_entries: List[SessionContextEntry],
) -> List[ProposedLesson]:
    return await curate_batches(build_batches(qa_rows, context_entries))


async def _curate_batch(batch_text: str) -> List[ProposedLesson]:
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


# -- Accept proposed lessons --------------------------------------------------


async def _judge_and_write(
    vector_engine,
    proposed: List[ProposedLesson],
    entries_by_id: dict,
) -> List[WrittenLesson]:
    """For each proposed lesson in parallel: gather evidence, then write-or-reject."""

    async def judge_one(lesson: ProposedLesson) -> Optional[WrittenLesson]:
        members = [
            entries_by_id[entry_id]
            for entry_id in lesson.member_entry_ids
            if entry_id in entries_by_id
        ]
        # The two searches are independent; run them concurrently, then write once both land.
        prior_lessons, glossary = await asyncio.gather(
            _search_payload_texts(
                vector_engine,
                "DocumentChunk_text",
                NOVELTY_LESSONS_PER_LESSON,
                query_text=lesson.working_statement,
                node_name=DISTILLATE_NODE_SET,
            ),
            _search_payload_texts(
                vector_engine,
                "Entity_name",
                GLOSSARY_ENTITIES_PER_LESSON,
                query_text=lesson.working_statement,
            ),
        )
        return await _write_or_reject(lesson, members, prior_lessons, glossary)

    decisions = await _gather_bounded(
        [lambda lesson=lesson: judge_one(lesson) for lesson in proposed], WRITER_CONCURRENCY
    )
    return [decision for decision in decisions if decision is not None]


async def _accept_proposed_lessons(
    scope: SessionDistillationScope,
    proposed: List[ProposedLesson],
    context_entries: List[SessionContextEntry],
) -> List[WrittenLesson]:
    entries_by_id = {entry.id: entry for entry in context_entries}
    async with set_database_global_context_variables(scope.dataset.id, scope.dataset.owner_id):
        vector_engine = get_vector_engine()
        decisions = await _judge_and_write(vector_engine, proposed, entries_by_id)
    return [lesson for lesson in decisions if lesson.accept and lesson.statement.strip()]


async def _write_or_reject(
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

    try:
        return await LLMGateway.acreate_structured_output(
            text_input="\n\n".join(sections),
            system_prompt=system_prompt,
            response_model=WrittenLesson,
        )
    except Exception as error:
        logger.warning("Distillation writer call failed open: %s", error)
        return None


# -- Publish accepted lessons -------------------------------------------------


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


async def _publish_distilled_lessons(
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

    await add(documents, dataset_id=scope.dataset.id, user=scope.user, node_set=DISTILLATE_NODE_SET)
    await cognify(datasets=[scope.dataset.id], user=scope.user)
    return documents


# -- Shared helpers -----------------------------------------------------------


async def _gather_bounded(factories: List[Callable[[], Awaitable]], limit: int) -> list:
    """Run async factories concurrently, capped at ``limit`` in flight at once."""
    semaphore = asyncio.Semaphore(limit)

    async def run(factory: Callable[[], Awaitable]):
        async with semaphore:
            return await factory()

    return list(await asyncio.gather(*(run(factory) for factory in factories)))


async def _search_payload_texts(
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


# -- Resolve distillation scope ----------------------------------------------


async def _resolve_distillation_scope(
    *,
    session_id: str,
    dataset: Union[str, UUID],
    user: Optional[User],
) -> SessionDistillationScope:
    if user is not None and getattr(user, "id", None) is not None:
        resolved_user = user
    else:
        ctx_user = session_user.get()
        resolved_user = (
            ctx_user
            if ctx_user is not None and getattr(ctx_user, "id", None) is not None
            else await get_default_user()
        )

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
