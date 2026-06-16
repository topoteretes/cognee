"""Distill a finished session's learnings into the persistent knowledge graph.

Flow (curator calls parallel by batch; judge/write calls parallel by lesson):

1. LOAD    gated session-context entries + the session's QA turns.
2. BATCH   interleave turns and candidates chronologically, packed into size-bounded
           batches (each batch is a contiguous session slice: its turns + its candidates).
3. CURATE  one curator LLM call per batch, in parallel -> proposed lessons.
4. JUDGE   per proposed lesson, in parallel: search prior lessons + entities, then one
           writer/rejecter LLM call that either writes a standalone lesson or rejects it
           (already_known | not_durable | unsupported).
5. PERSIST render each accepted lesson as its own document; add + cognify them in one pass.

Everything is fail-open per unit: a failed curator batch or writer call drops only its own
work, never the whole run.
"""

import asyncio
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
from cognee.modules.data.methods import get_dataset, get_datasets_by_name
from cognee.modules.session_lifecycle.metrics import get_session_row
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


# -- Entry point --------------------------------------------------------------


async def distill_session(
    session_id: str,
    dataset: Union[str, UUID, None] = None,
    user: Optional[User] = None,
) -> DistillationResult:
    """Distill one finished session's gated learnings into its dataset's knowledge graph."""
    resolved_user = await _resolve_user(user)
    dataset_obj = await _resolve_dataset(session_id, dataset, resolved_user)
    dataset_id = str(dataset_obj.id)

    # 1. LOAD
    session_manager = get_session_manager()
    context_rows = await session_manager.get_session_context_entries(
        user_id=str(resolved_user.id), session_id=session_id
    )
    gated = gate_context_entries(context_rows)
    if not gated:
        return DistillationResult(
            session_id=session_id, dataset_id=dataset_id, status="no_gated_entries"
        )
    qa_rows = await _load_qa_rows(session_manager, str(resolved_user.id), session_id)

    # 2. BATCH
    batches = build_batches(qa_rows, gated)
    entries_by_id = {entry.id: entry for entry in gated}

    async with set_database_global_context_variables(dataset_obj.id, dataset_obj.owner_id):
        vector_engine = get_vector_engine()

        # 3. CURATE (parallel by batch)
        proposed = await curate_batches(batches)
        if not proposed:
            return DistillationResult(
                session_id=session_id,
                dataset_id=dataset_id,
                status="no_proposed_lessons",
                gated_entry_count=len(gated),
                batch_count=len(batches),
            )

        # 4. JUDGE + WRITE (parallel by lesson)
        decisions = await judge_and_write(vector_engine, proposed, entries_by_id)

    accepted = [lesson for lesson in decisions if lesson.accept and lesson.statement.strip()]
    rejected_count = len(decisions) - len(accepted)
    if not accepted:
        return DistillationResult(
            session_id=session_id,
            dataset_id=dataset_id,
            status="no_accepted_lessons",
            gated_entry_count=len(gated),
            batch_count=len(batches),
            proposed_lesson_count=len(proposed),
            rejected_lesson_count=rejected_count,
        )

    # 5. PERSIST
    distilled_on = datetime.utcnow().strftime("%Y-%m-%d")
    documents = [
        render_lesson_document(lesson, session_id=session_id, distilled_on=distilled_on)
        for lesson in accepted
    ]
    await _persist_lessons(documents, dataset_obj, resolved_user)

    return DistillationResult(
        session_id=session_id,
        dataset_id=dataset_id,
        status="completed",
        documents=documents,
        gated_entry_count=len(gated),
        batch_count=len(batches),
        proposed_lesson_count=len(proposed),
        accepted_lesson_count=len(accepted),
        rejected_lesson_count=rejected_count,
    )


# -- 1. Load + gate -----------------------------------------------------------


def gate_context_entries(raw_entries: list) -> List[SessionContextEntry]:
    """Keep entries worth distilling: never rated harmful, confidence above the gate.

    Deterministic — no search, no LLM. ``coerce_active_context_entries`` first drops
    non-context rows (feedback entries, garbage); ``harmful_count == 0`` is stricter than
    "net helpfulness >= 0" (one harmful rating drops the entry).
    """
    gated = []
    for entry in coerce_active_context_entries(raw_entries):
        if entry.harmful_count > 0:
            continue
        if entry.confidence < MIN_GATE_CONFIDENCE:
            continue
        gated.append(entry)
    return gated


async def _load_qa_rows(session_manager, user_id: str, session_id: str) -> List[dict]:
    raw_qa = await session_manager.get_session(
        user_id=user_id, session_id=session_id, formatted=False
    )
    return [
        entry.model_dump() if hasattr(entry, "model_dump") else dict(entry)
        for entry in (raw_qa if isinstance(raw_qa, list) else [])
    ]


# -- 2. Batch -----------------------------------------------------------------


def build_batches(qa_rows: List[dict], gated: List[SessionContextEntry]) -> List[str]:
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
    for entry in gated:
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


# -- 3. Curate (parallel by batch) -------------------------------------------


async def curate_batches(batches: List[str]) -> List[ProposedLesson]:
    """Run one curator call per batch in parallel; flatten the proposed lessons.

    Cross-batch duplicates are accepted here; the per-lesson writer/rejecter handles
    already-known lessons against the persisted graph.
    """
    per_batch = await _gather_bounded(
        [lambda b=batch: _curate_batch(b) for batch in batches], CURATOR_CONCURRENCY
    )
    return [lesson for batch_lessons in per_batch for lesson in batch_lessons]


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


# -- 4. Judge + write (parallel by lesson) -----------------------------------


async def judge_and_write(
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
            existing_lessons(vector_engine, lesson.working_statement),
            anchor_entities(vector_engine, lesson.working_statement),
        )
        return await _write_or_reject(lesson, members, prior_lessons, glossary)

    decisions = await _gather_bounded(
        [lambda lesson=lesson: judge_one(lesson) for lesson in proposed], WRITER_CONCURRENCY
    )
    return [decision for decision in decisions if decision is not None]


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


async def existing_lessons(vector_engine, statement: str) -> List[str]:
    """Previously persisted lessons most similar to this statement — the novelty signal.

    Scoped to the session-learnings node set, so it compares against prior distilled
    lessons rather than all graph content.
    """
    return await _search_payload_texts(
        vector_engine,
        "DocumentChunk_text",
        NOVELTY_LESSONS_PER_LESSON,
        query_text=statement,
        node_name=DISTILLATE_NODE_SET,
    )


async def anchor_entities(vector_engine, statement: str) -> List[str]:
    """Existing entity names similar to a lesson statement, for verbatim anchoring."""
    return await _search_payload_texts(
        vector_engine, "Entity_name", GLOSSARY_ENTITIES_PER_LESSON, query_text=statement
    )


# -- 5. Persist ---------------------------------------------------------------


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


async def _persist_lessons(documents: List[str], dataset_obj, user: User) -> None:
    # Imported lazily to avoid a circular import through the cognee package root.
    from cognee.api.v1.add import add
    from cognee.api.v1.cognify import cognify

    await add(documents, dataset_id=dataset_obj.id, user=user, node_set=DISTILLATE_NODE_SET)
    await cognify(datasets=[dataset_obj.id], user=user)


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


# -- Resolution ---------------------------------------------------------------


async def _resolve_user(user: Optional[User]) -> User:
    if user is not None and getattr(user, "id", None) is not None:
        return user
    ctx_user = session_user.get()
    if ctx_user is not None and getattr(ctx_user, "id", None) is not None:
        return ctx_user
    return await get_default_user()


async def _resolve_dataset(session_id: str, dataset: Union[str, UUID, None], user: User):
    """Explicit arg wins, then the SessionRecord's dataset_id, else a clear error."""
    if dataset is not None:
        resolved = await _lookup_dataset(dataset, user)
        if resolved is None:
            raise CogneeValidationError(
                message=f"Dataset '{dataset}' not found for this user.",
                name="SessionDistillationError",
            )
        return resolved

    record = await get_session_row(session_id=session_id, user_id=user.id)
    if record is not None and record.dataset_id is not None:
        resolved = await get_dataset(user.id, record.dataset_id)
        if resolved is not None:
            return resolved

    raise CogneeValidationError(
        message=(
            f"Session '{session_id}' has no associated dataset. Pass dataset=<name or id> "
            "so the distilled learnings land in the graph they should connect to."
        ),
        name="SessionDistillationError",
    )


async def _lookup_dataset(dataset: Union[str, UUID], user: User):
    if isinstance(dataset, UUID):
        return await get_dataset(user.id, dataset)
    try:
        return await get_dataset(user.id, UUID(str(dataset)))
    except (ValueError, TypeError):
        matches = await get_datasets_by_name(str(dataset), user.id)
        return matches[0] if matches else None
