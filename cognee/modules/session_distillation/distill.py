"""Distill a finished session's learnings into the persistent knowledge graph.

Pipeline (one curator LLM call, then one writer LLM call per surviving lesson):

1. Gate: keep session-context entries that were never rated harmful and clear the
   confidence threshold. Section labels travel along as metadata only.
2. Novelty search: for each gated entry, fetch similar existing chunks from the
   dataset's graph so the curator can drop lessons the graph already knows.
3. Curator: one LLM call merges duplicate entries, drops session-local trivia,
   judges novelty, and classifies each surviving lesson.
4. Anchoring search: for each surviving lesson, fetch existing entity names so the
   writer can use them verbatim and extraction connects to existing nodes.
5. Writers: one LLM call per lesson rewrites it as standalone, entity-anchored prose.
6. Render + cognify: each lesson is rendered (by a template, never freehand) as its own
   standalone document, and all of them are added + cognified into the session's dataset
   in a single pass — one document per learning, so each is an independent graph unit.

A curator failure aborts distillation; a writer failure drops only that lesson.
"""

import asyncio
from datetime import datetime
from typing import List, Optional, Union
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
    GLOSSARY_ENTITIES_PER_LESSON,
    MAX_DIGEST_QUESTION_CHARS,
    MAX_SOURCE_MESSAGE_CHARS,
    MIN_GATE_CONFIDENCE,
    NOVELTY_SNIPPETS_PER_ENTRY,
    CuratedLesson,
    CurationPlan,
    DistillationResult,
    DistilledLesson,
)

logger = get_logger("session_distillation")

CURATOR_PROMPT_FILE = "session_distillation_curator_system.txt"
WRITER_PROMPT_FILE = "session_distillation_writer_system.txt"

# Node set tag marking distillate documents in the graph, for provenance/debugging.
DISTILLATE_NODE_SET = ["session_learnings"]


# -- Deterministic helpers (pure, unit-testable) -----------------------------


def gate_context_entries(raw_entries: list) -> List[SessionContextEntry]:
    """Keep entries worth distilling: never rated harmful, confidence above the gate.

    All sections pass through — the curator judges durability by content, not by the
    (unreliable) section label. ``harmful_count == 0`` also guarantees net helpfulness
    is non-negative.
    """
    gated = []
    for entry in coerce_active_context_entries(raw_entries):
        if entry.harmful_count > 0:
            continue
        if entry.confidence < MIN_GATE_CONFIDENCE:
            continue
        gated.append(entry)
    return gated


def build_session_digest(qa_rows: List[dict]) -> str:
    """One line per turn (questions only) so the curator sees the session's shape."""
    lines = []
    for row in qa_rows:
        question = " ".join((row.get("question") or "").split())
        if not question:
            continue
        lines.append(f"- {question[:MAX_DIGEST_QUESTION_CHARS]}")
    return "\n".join(lines)


def render_lesson_document(
    distilled: DistilledLesson,
    *,
    session_id: str,
    distilled_on: str,
) -> str:
    """Render ONE lesson as a standalone markdown document for the graph.

    The template — not the LLM — controls the format. One document per lesson, so each
    learning is an independently identifiable unit in the graph (its own data item, its
    own provenance) rather than being bundled with the rest of the session.
    """
    statement = distilled.statement.strip()
    why = distilled.why_learned.strip().rstrip(".")
    body = f"{statement} ({why}.)" if why else statement
    return f"# Session learning — {distilled_on} (session {session_id})\n\n{body}\n"


# -- Search helpers (fail-open per item) -------------------------------------


async def _search_snippets(
    vector_engine,
    collection: str,
    limit: int,
    *,
    query_text: str | None = None,
    query_vector: list | None = None,
) -> List[str]:
    """Vector-search one collection and return payload texts; [] on any failure.

    Pass ``query_vector`` to reuse a stored embedding (no re-embedding); ``query_text`` is
    the fallback the engine embeds itself when no vector is given.
    """
    try:
        results = await vector_engine.search(
            collection,
            query_text=query_text,
            query_vector=query_vector,
            limit=limit,
            include_payload=True,
        )
    except Exception as error:
        logger.debug("Distillation search on %s failed open: %s", collection, error)
        return []

    snippets = []
    for result in results or []:
        payload = getattr(result, "payload", None)
        if not isinstance(payload, dict):
            continue
        text = payload.get("text") or payload.get("name")
        if text and str(text).strip():
            snippets.append(str(text).strip())
    return snippets


async def existing_knowledge(vector_engine, entry: SessionContextEntry) -> List[str]:
    """Graph chunks already similar to this entry — the curator's novelty signal.

    Reuses the entry's stored embedding (computed when the entry was created); falls back
    to embedding its content only when no stored vector is available.
    """
    return await _search_snippets(
        vector_engine,
        "DocumentChunk_text",
        NOVELTY_SNIPPETS_PER_ENTRY,
        query_vector=entry.embedding,
        query_text=entry.content,
    )


async def anchor_entities(vector_engine, statement: str) -> List[str]:
    """Existing entity names similar to a lesson statement, for verbatim anchoring.

    The statement is freshly written by the curator, so there is no stored vector to reuse.
    """
    return await _search_snippets(
        vector_engine,
        "Entity_name",
        GLOSSARY_ENTITIES_PER_LESSON,
        query_text=statement,
    )


# -- LLM calls ----------------------------------------------------------------


def build_source_messages_by_entry(
    gated: List[SessionContextEntry], context_rows: List[dict]
) -> dict:
    """Map each gated entry id to the user messages (feedback raw_text) that created it.

    This is the curator's provenance signal: teaching turns store no QA entry, so the
    user's assertions are only visible through the feedback entries referenced by
    ``source_feedback_ids``.
    """
    raw_text_by_feedback_id = {}
    for row in context_rows:
        if isinstance(row, dict) and row.get("kind") == "feedback" and row.get("id"):
            raw_text = " ".join(str(row.get("raw_text") or "").split())
            if raw_text:
                raw_text_by_feedback_id[row["id"]] = raw_text[:MAX_SOURCE_MESSAGE_CHARS]

    messages_by_entry = {}
    for entry in gated:
        messages = [
            raw_text_by_feedback_id[feedback_id]
            for feedback_id in entry.source_feedback_ids
            if feedback_id in raw_text_by_feedback_id
        ]
        if messages:
            messages_by_entry[entry.id] = messages
    return messages_by_entry


async def _run_curator(
    gated: List[SessionContextEntry],
    qa_rows: List[dict],
    known_by_entry_id: dict,
    source_messages_by_entry: dict,
) -> Optional[CurationPlan]:
    """The single wide-view call: merge, drop, judge novelty, classify."""
    system_prompt = read_query_prompt(CURATOR_PROMPT_FILE)
    if not system_prompt:
        logger.warning("Distillation curator prompt not found: %s", CURATOR_PROMPT_FILE)
        return None

    sections = ["GATED SESSION ENTRIES:"]
    for entry in gated:
        sections.append(f"- id={entry.id} section={entry.section} content={entry.content}")
        for message in source_messages_by_entry.get(entry.id, []):
            sections.append(f"  SOURCE USER MESSAGE: {message}")
        for snippet in known_by_entry_id.get(entry.id, []):
            sections.append(f"  EXISTING KNOWLEDGE: {snippet}")

    digest = build_session_digest(qa_rows)
    if digest:
        sections.append("")
        sections.append("SESSION DIGEST:")
        sections.append(digest)

    try:
        return await LLMGateway.acreate_structured_output(
            text_input="\n".join(sections),
            system_prompt=system_prompt,
            response_model=CurationPlan,
        )
    except Exception as error:
        logger.warning("Distillation curator call failed: %s", error)
        return None


async def _write_lesson(
    curated: CuratedLesson,
    member_entries: List[SessionContextEntry],
    source_messages: List[str],
    glossary: List[str],
) -> Optional[DistilledLesson]:
    """One narrow call per lesson: rewrite as standalone, entity-anchored prose.

    Grounding is the user's own words (member entries + source messages), never the
    assistant's prior answers — those are exactly the unsupported claims the curator drops.
    """
    system_prompt = read_query_prompt(WRITER_PROMPT_FILE)
    if not system_prompt:
        logger.warning("Distillation writer prompt not found: %s", WRITER_PROMPT_FILE)
        return None

    sections = [f"WORKING STATEMENT:\n{curated.working_statement}"]
    if member_entries:
        sections.append(
            "MEMBER ENTRIES:\n" + "\n".join(f"- {entry.content}" for entry in member_entries)
        )
    if source_messages:
        sections.append(
            "SOURCE USER MESSAGES:\n" + "\n".join(f"- {message}" for message in source_messages)
        )
    if glossary:
        sections.append("ENTITY GLOSSARY:\n" + "\n".join(f"- {name}" for name in glossary))
    if curated.overlap_note:
        sections.append(f"OVERLAP NOTE:\n{curated.overlap_note}")

    try:
        return await LLMGateway.acreate_structured_output(
            text_input="\n\n".join(sections),
            system_prompt=system_prompt,
            response_model=DistilledLesson,
        )
    except Exception as error:
        logger.warning("Distillation writer call failed for one lesson: %s", error)
        return None


# -- Resolution helpers --------------------------------------------------------


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


# -- Entry point ----------------------------------------------------------------


async def distill_session(
    session_id: str,
    dataset: Union[str, UUID, None] = None,
    user: Optional[User] = None,
) -> DistillationResult:
    """Distill one session's gated learnings into its dataset's knowledge graph.

    Returns a DistillationResult whose ``status`` explains what happened and whose
    ``document`` carries the rendered markdown when distillation completed.
    """
    resolved_user = await _resolve_user(user)
    dataset_obj = await _resolve_dataset(session_id, dataset, resolved_user)

    session_manager = get_session_manager()
    context_rows = await session_manager.get_session_context_entries(
        user_id=str(resolved_user.id), session_id=session_id
    )
    gated = gate_context_entries(context_rows)
    if not gated:
        return DistillationResult(
            session_id=session_id,
            dataset_id=str(dataset_obj.id),
            status="no_gated_entries",
        )

    raw_qa = await session_manager.get_session(
        user_id=str(resolved_user.id), session_id=session_id, formatted=False
    )
    qa_rows = [
        entry.model_dump() if hasattr(entry, "model_dump") else dict(entry)
        for entry in (raw_qa if isinstance(raw_qa, list) else [])
    ]

    async with set_database_global_context_variables(dataset_obj.id, dataset_obj.owner_id):
        vector_engine = get_vector_engine()

        known_by_entry_id = {
            entry.id: await existing_knowledge(vector_engine, entry) for entry in gated
        }

        source_messages_by_entry = build_source_messages_by_entry(gated, context_rows)
        plan = await _run_curator(gated, qa_rows, known_by_entry_id, source_messages_by_entry)
        if plan is None:
            return DistillationResult(
                session_id=session_id,
                dataset_id=str(dataset_obj.id),
                status="curator_failed",
                gated_entry_count=len(gated),
            )

        new_lessons = [lesson for lesson in plan.lessons if lesson.novelty == "new"]
        skipped_already_known = len(plan.lessons) - len(new_lessons)
        if not new_lessons:
            return DistillationResult(
                session_id=session_id,
                dataset_id=str(dataset_obj.id),
                status="nothing_new",
                gated_entry_count=len(gated),
                skipped_already_known=skipped_already_known,
            )

        entries_by_id = {entry.id: entry for entry in gated}

        async def write_one(curated: CuratedLesson):
            members = [
                entries_by_id[entry_id]
                for entry_id in curated.member_entry_ids
                if entry_id in entries_by_id
            ]
            source_messages = []
            for member in members:
                for message in source_messages_by_entry.get(member.id, []):
                    if message not in source_messages:
                        source_messages.append(message)
            glossary = await anchor_entities(vector_engine, curated.working_statement)
            distilled = await _write_lesson(curated, members, source_messages, glossary)
            return (curated, distilled) if distilled is not None else None

        written = await asyncio.gather(
            *(write_one(lesson) for lesson in new_lessons), return_exceptions=True
        )
        lessons = [item for item in written if isinstance(item, tuple)]

    if not lessons:
        return DistillationResult(
            session_id=session_id,
            dataset_id=str(dataset_obj.id),
            status="no_lessons_written",
            gated_entry_count=len(gated),
            skipped_already_known=skipped_already_known,
        )

    distilled_on = datetime.utcnow().strftime("%Y-%m-%d")
    documents = [
        render_lesson_document(distilled, session_id=session_id, distilled_on=distilled_on)
        for _curated, distilled in lessons
    ]

    # Imported lazily to avoid a circular import through the cognee package root.
    from cognee.api.v1.add import add
    from cognee.api.v1.cognify import cognify

    # One document per lesson (each its own data item), ingested in a single cognify pass.
    await add(
        documents,
        dataset_id=dataset_obj.id,
        user=resolved_user,
        node_set=DISTILLATE_NODE_SET,
    )
    await cognify(datasets=[dataset_obj.id], user=resolved_user)

    return DistillationResult(
        session_id=session_id,
        dataset_id=str(dataset_obj.id),
        status="completed",
        documents=documents,
        session_summary=plan.session_summary,
        gated_entry_count=len(gated),
        lesson_count=len(lessons),
        skipped_already_known=skipped_already_known,
    )
