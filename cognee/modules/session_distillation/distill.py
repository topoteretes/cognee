"""Distill a finished session's learnings into the persistent knowledge graph.

Flow (curator calls parallel by batch; accept/write calls parallel by lesson):

1. LOAD    distillable session-context entries the watermark does not cover, plus the
           QA turns from the turn that produced the earliest such entry onward (older
           turns were already curated with the entries they produced).
2. CURATE  pack them into batches; one curator LLM call per batch. Every new entry is
           curated unless ``IMPROVE_DISTILL_MAX_BATCHES_PER_RUN`` caps the run.
3. ACCEPT  per proposed lesson: search prior lessons/entities, then writer/rejecter LLM.
4. PERSIST render accepted lessons as documents; add + cognify them in one pass.

An entry is sealed (added to the watermark) once its curator batch completed and the
run's surviving lessons are published, whatever the writer did with the lessons:
processed once is processed. A batch whose curator call failed seals nothing, so only its
entries come back next run; the run still finishes by RAISING after publishing what
survived, so the failure is reported. A run cut short by an exhausted LLM budget, in the
curator or the writer phase, seals nothing and re-raises, so improve can stop.

With ``DISTILLATION_MODE=coding`` (SDK-1335) the same flow runs with the coding
prompts under ``prompts/coding/``, the typed ``Coding*`` models, the unit's request and
final answer pinned to every curator batch, the writer's compact STYLE line, evidence
rules on the lesson type, and typed lesson documents with ``State-Key`` replacement
(``lesson_store``). Unset, nothing here reads differently.
"""

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from cognee.context_global_variables import session_user, set_database_global_context_variables
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.session.coding_unit_extraction import parse_qa_context
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_context_builder import (
    clamped_net_helpfulness,
    coerce_active_context_entries,
)
from cognee.infrastructure.session.session_context_models import (
    MIN_GATE_CONFIDENCE,
    SessionContextEntry,
)
from cognee.infrastructure.session.session_node_set import (
    bridge_node_set,
    get_session_node_set,
)
from cognee.infrastructure.session.session_persist_watermark import (
    get_distilled_entry_ids,
    save_distilled_entry_ids,
)
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.models import Dataset
from cognee.modules.improve.config import coding_distillation_enabled, get_improve_config
from cognee.modules.improve.constants import SESSION_LEARNINGS_NODE_SET
from cognee.modules.pipelines.exceptions import AfterRunHookError
from cognee.modules.truth_subspace.constants import truth_session_node_set
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.async_utils import gather_with_concurrency_limit
from cognee.shared.logging_utils import get_logger

from .lesson_store import (
    LessonDocument,
    delete_superseded_lessons,
    find_lessons_by_state_key,
    lesson_data_id,
    mentioned_files,
    render_coding_lesson_document,
    seen_in_sessions,
)
from .models import (
    CODING_WRITER_STYLE_LINE,
    CURATOR_BLOCKS_PER_BATCH,
    CURATOR_CONCURRENCY,
    GLOSSARY_ENTITIES_PER_LESSON,
    MAX_CANDIDATE_CHARS,
    MAX_QA_ANSWER_CHARS,
    MAX_QA_FEEDBACK_CHARS,
    MAX_QA_QUESTION_CHARS,
    NOVELTY_LESSONS_PER_LESSON,
    PITFALL_STATE_KEY_PREFIX,
    WRITER_CONCURRENCY,
    CodingCuratorBatchOutput,
    CodingWrittenLesson,
    CuratorBatchOutput,
    DistillationResult,
    ProposedLesson,
    WrittenLesson,
)

logger = get_logger("session_distillation")


class DistillationCallsFailedError(RuntimeError):
    """Some curator/writer LLM calls failed.

    Entries of completed curator batches are sealed; entries of failed batches are
    left for the next run. Lessons lost to failed writer calls are not retried.
    """


@dataclass
class CuratorBatch:
    """One curator call's input and the gated entries it covers."""

    text: str
    entry_ids: list[str]


@dataclass
class CuratorRun:
    """What the curator phase produced and which entries it got through."""

    proposed: list[ProposedLesson]
    failed_batches: int
    curated_entry_ids: set[str]
    budget_exhausted: bool = False


CURATOR_PROMPT_FILE = "session_distillation_curator_system.txt"
WRITER_PROMPT_FILE = "session_distillation_writer_system.txt"
CODING_PROMPT_DIR = "coding"

# A candidate or statement that opens with this names a state key (coding mode).
STATE_KEY_RE = re.compile(r"^\s*\[state-key:\s*([^\]]+?)\s*\]\s*")


def distillation_prompt_file(name: str, coding: bool) -> str:
    """The prompt file for the mode: the coding copy lives under ``prompts/coding/``."""
    return f"{CODING_PROMPT_DIR}/{name}" if coding else name


def candidate_state_key(content: str) -> str:
    match = STATE_KEY_RE.match(content or "")
    return match.group(1) if match else ""


def strip_state_key(text: str) -> str:
    return STATE_KEY_RE.sub("", text or "", count=1).strip()


# Node set marking distillate documents in the graph: used to tag them on write and to
# scope the novelty search to previously persisted lessons.
DISTILLATE_NODE_SET = [SESSION_LEARNINGS_NODE_SET]


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

    def result(
        self,
        status: str,
        documents: list[str] | None = None,
        uncurated_entries: int = 0,
    ) -> DistillationResult:
        return DistillationResult(
            session_id=self.session_id,
            dataset_id=self.dataset_id,
            status=status,
            documents=documents or [],
            uncurated_entries=uncurated_entries,
        )


async def resolve_distillation_scope(
    *,
    session_id: str,
    dataset: str | UUID,
    user: User | None,
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


def is_entry_distillable(entry: SessionContextEntry) -> bool:
    """Distillation gate: net helpfulness is not negative and confidence clears the threshold.

    Net helpfulness is helpful_count minus harmful_count, clamped exactly as the session
    ranker scores it. One misread harmful rating no longer disqualifies an entry forever:
    an entry rated harmful once and helpful three times is distillable.
    """
    return clamped_net_helpfulness(entry) >= 0 and entry.confidence >= MIN_GATE_CONFIDENCE


async def load_distillable_context_entries(
    scope: SessionDistillationScope,
) -> list[SessionContextEntry]:
    """Load the session's context entries and keep the distillable ones.

    Split from the QA load on purpose: this read decides the two cheap exits
    (``no_gated_entries``, ``no_new_entries``), and the default debounce runs
    distillation after every ``remember(session_id=...)`` — the full QA history
    must not be loaded and dumped just to conclude there is nothing to do.
    """
    session_manager = get_session_manager()
    context_rows = await session_manager.get_session_context_entries(
        user_id=scope.user_id,
        session_id=scope.session_id,
    )
    return [
        entry
        for entry in coerce_active_context_entries(context_rows)
        if is_entry_distillable(entry)
    ]


async def load_latest_qa_row(scope: SessionDistillationScope) -> list[dict]:
    """The session's newest QA turn as a one-row list: a coding unit's request and answer."""
    session_manager = get_session_manager()
    raw_qa = await session_manager.get_session(
        user_id=scope.user_id, session_id=scope.session_id, formatted=False, last_n=1
    )
    return [
        entry.model_dump() if hasattr(entry, "model_dump") else dict(entry)
        for entry in (raw_qa if isinstance(raw_qa, list) else [])
    ]


async def load_session_qa_rows(
    scope: SessionDistillationScope, since: datetime | None
) -> list[dict]:
    """Load the session's QA turns for curator batching — only after the exits.

    With ``since``, only the turns from the one that produced the earliest new
    entry onward are read: the ones stored at or after it plus the one just
    before it. The analysis that writes an entry runs while a turn is answered,
    before that turn's QA row is stored, and it judges the turn before, so the
    entry's time falls after the turn it is about. Older turns were already sent
    with the entries they produced; the cache filters them out instead of this
    process reading the whole session every run.
    """
    session_manager = get_session_manager()
    raw_qa = await session_manager.get_session(
        user_id=scope.user_id,
        session_id=scope.session_id,
        formatted=False,
        since=since,
        lead_in=1 if since is not None else 0,
    )
    return [
        entry.model_dump() if hasattr(entry, "model_dump") else dict(entry)
        for entry in (raw_qa if isinstance(raw_qa, list) else [])
    ]


def earliest_entry_time(entries: list[SessionContextEntry]) -> datetime | None:
    """The earliest ``created_at`` among the entries, aware (naive is UTC); None if unset."""
    times = []
    for entry in entries:
        if not entry.created_at:
            continue
        parsed = datetime.fromisoformat(entry.created_at)
        times.append(parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc))
    return min(times, default=None)


def select_curator_inputs(
    qa_rows: list[dict],
    new_entries: list[SessionContextEntry],
    max_batches: int,
) -> tuple[list[dict], list[SessionContextEntry]]:
    """Pick what one run curates.

    With ``max_batches`` 0 everything goes: every new entry and every turn given.
    With a cap, the run packs into at most that many batches: new entries first,
    highest confidence then most recent; QA turns fill the remaining blocks, most
    recent first. Entries left out are still sealed by the watermark when the run
    completes, so a capped run trades the overflow for a bounded run.
    """
    if max_batches == 0:
        return qa_rows, list(new_entries)
    budget = max_batches * CURATOR_BLOCKS_PER_BATCH
    entries = sorted(
        new_entries, key=lambda entry: (entry.confidence, entry.created_at or ""), reverse=True
    )[:budget]
    qa_budget = budget - len(entries)
    recent_qa = sorted(qa_rows, key=lambda row: row.get("time") or "")[-qa_budget:]
    return (recent_qa if qa_budget else []), entries


def build_curator_batches(
    qa_rows: list[dict],
    context_entries: list[SessionContextEntry],
    *,
    pin_turns: bool = False,
) -> list[CuratorBatch]:
    """Pack the session timeline into coarse, size-safe chronological batches.

    With ``pin_turns`` (coding mode) the turns are not part of the timeline: every
    batch opens with all of them, so each curator call judges its candidates against
    the unit's request and final answer, and the candidates alone fill the batches.
    """
    timeline: list[tuple[str, str, str | None]] = []
    for row in qa_rows:
        question = " ".join((row.get("question") or "").split())[:MAX_QA_QUESTION_CHARS]
        answer = " ".join((row.get("answer") or "").split())[:MAX_QA_ANSWER_CHARS]
        if not question and not answer:
            continue
        block = f"User: {question}\nAssistant: {answer}"
        feedback_text = " ".join((row.get("feedback_text") or "").split())[:MAX_QA_FEEDBACK_CHARS]
        if feedback_text:
            block += f"\nUser feedback on this answer: {feedback_text}"
        timeline.append((row.get("time") or "", block, None))

    for entry in context_entries:
        content = " ".join(entry.content.split())[:MAX_CANDIDATE_CHARS]
        block = f"Candidate {entry.id} [{entry.context_profile}/{entry.section}]: {content}"
        timeline.append((entry.created_at or "", block, entry.id))

    timeline.sort(key=lambda item: item[0])

    pinned: list[str] = []
    if pin_turns:
        pinned = [block for _timestamp, block, entry_id in timeline if entry_id is None]
        timeline = [item for item in timeline if item[2] is not None]

    batches: list[CuratorBatch] = []
    for index in range(0, len(timeline), CURATOR_BLOCKS_PER_BATCH):
        chunk = timeline[index : index + CURATOR_BLOCKS_PER_BATCH]
        blocks = pinned + [block for _timestamp, block, _entry_id in chunk]
        batches.append(
            CuratorBatch(
                text="\n\n".join(blocks),
                entry_ids=[entry_id for _timestamp, _block, entry_id in chunk if entry_id],
            )
        )
    return batches


async def curate_batch(batch_text: str, *, coding: bool = False) -> list[ProposedLesson] | None:
    """One curator call over one batch slice.

    ``None`` means the call FAILED (missing prompt, LLM error) — distinct from
    ``[]``, the curator deciding nothing in the batch is durable. The caller
    must not seal entries behind a failed call as if they were curated. An
    exhausted LLM budget is raised, not swallowed: every later call would fail
    the same way, and improve stops the run on it.
    """
    prompt_file = distillation_prompt_file(CURATOR_PROMPT_FILE, coding)
    system_prompt = read_query_prompt(prompt_file)
    if not system_prompt:
        logger.warning("Distillation curator prompt not found: %s", prompt_file)
        return None
    try:
        result = await LLMGateway.acreate_structured_output(
            text_input=batch_text,
            system_prompt=system_prompt,
            response_model=CodingCuratorBatchOutput if coding else CuratorBatchOutput,
        )
        return list(result.lessons)
    except LLMPaymentRequiredError:
        raise
    except Exception as error:
        logger.warning("Distillation curator batch failed: %s", error, exc_info=True)
        return None


async def propose_lessons(
    qa_rows: list[dict],
    context_entries: list[SessionContextEntry],
    *,
    coding: bool = False,
) -> CuratorRun:
    """Pack session inputs into curator batches, then flatten proposed lessons.

    ``curated_entry_ids`` holds the entries of every batch whose call completed;
    a failed curator call is not "nothing durable found", so its entries are
    left out and ``failed_batches`` counts it. Once one call reports an
    exhausted budget the remaining batches are skipped, not called, and
    ``budget_exhausted`` is set.
    """
    batches = build_curator_batches(qa_rows, context_entries, pin_turns=coding)
    if not batches:
        return CuratorRun(proposed=[], failed_batches=0, curated_entry_ids=set())

    budget_exhausted = False

    async def run_batch(batch: CuratorBatch) -> list[ProposedLesson] | None:
        nonlocal budget_exhausted
        if budget_exhausted:
            return None
        try:
            return await curate_batch(batch.text, coding=coding)
        except LLMPaymentRequiredError:
            budget_exhausted = True
            return None

    curator_calls = [lambda batch=batch: run_batch(batch) for batch in batches]
    per_batch = await gather_with_concurrency_limit(curator_calls, CURATOR_CONCURRENCY)

    proposed = [
        lesson
        for batch_lessons in per_batch
        if batch_lessons is not None
        for lesson in batch_lessons
    ]
    curated_entry_ids = {
        entry_id
        for batch, batch_lessons in zip(batches, per_batch)
        if batch_lessons is not None
        for entry_id in batch.entry_ids
    }
    failed_batches = sum(1 for batch_lessons in per_batch if batch_lessons is None)
    return CuratorRun(
        proposed=proposed,
        failed_batches=failed_batches,
        curated_entry_ids=curated_entry_ids,
        budget_exhausted=budget_exhausted,
    )


async def search_payload_texts(
    vector_engine,
    collection: str,
    limit: int,
    *,
    query_text: str | None = None,
    query_vector: list | None = None,
    node_name: list[str] | None = None,
) -> list[str]:
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
        logger.debug("Distillation search on %s failed open: %s", collection, error, exc_info=True)
        return []

    texts: list[str] = []
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


def recase_glossary(glossary: list[str], texts: list[str]) -> list[str]:
    """Glossary names as the session's own text spells them, where it spells them.

    The graph stores entity names in lower case (``marktaskcommand``); the lesson must
    name symbols as the repository does (``MarkTaskCommand``). A case-insensitive match
    in the proposed statement or its member candidates supplies the casing.
    """
    corpus = "\n".join(texts)
    recased = []
    for name in glossary:
        match = re.search(re.escape(name), corpus, re.IGNORECASE) if name else None
        recased.append(match.group(0) if match else name)
    return recased


def build_writer_input(
    lesson: ProposedLesson,
    members: list[SessionContextEntry],
    prior_lessons: list[str],
    glossary: list[str],
    *,
    coding: bool = False,
) -> str:
    sections = [f"PROPOSED LESSON:\n{lesson.working_statement}"]
    state_key = getattr(lesson, "state_key", "") if coding else ""
    if state_key:
        sections.append(f"PROPOSED STATE KEY:\n{state_key}")
    if members:
        sections.append(
            "MEMBER ENTRIES:\n" + "\n".join(f"- {member.content}" for member in members)
        )
    if prior_lessons:
        sections.append(
            "SIMILAR EXISTING LESSONS:\n" + "\n".join(f"- {prior}" for prior in prior_lessons)
        )
    if coding and glossary:
        glossary = recase_glossary(
            glossary, [lesson.working_statement, *(member.content for member in members)]
        )
    if glossary:
        sections.append("ENTITY GLOSSARY:\n" + "\n".join(f"- {name}" for name in glossary))
    if coding:
        sections.append(CODING_WRITER_STYLE_LINE)
    return "\n\n".join(sections)


async def write_or_reject(
    lesson: ProposedLesson,
    members: list[SessionContextEntry],
    prior_lessons: list[str],
    glossary: list[str],
    *,
    coding: bool = False,
) -> WrittenLesson | None:
    """One writer/rejecter call for one proposed lesson.

    ``None`` means the call FAILED (missing prompt, LLM error); a rejection is
    a ``WrittenLesson`` with ``accept=False``. The caller counts failures so a
    writer outage is never mistaken for "every lesson rejected".
    """
    prompt_file = distillation_prompt_file(WRITER_PROMPT_FILE, coding)
    system_prompt = read_query_prompt(prompt_file)
    if not system_prompt:
        logger.warning("Distillation writer prompt not found: %s", prompt_file)
        return None

    text_input = build_writer_input(lesson, members, prior_lessons, glossary, coding=coding)
    try:
        return await LLMGateway.acreate_structured_output(
            text_input=text_input,
            system_prompt=system_prompt,
            response_model=CodingWrittenLesson if coding else WrittenLesson,
        )
    except LLMPaymentRequiredError:
        raise
    except Exception as error:
        logger.warning("Distillation writer call failed: %s", error, exc_info=True)
        return None


def is_failed_trace_candidate(entry: SessionContextEntry) -> bool:
    """A ``failure_lessons`` candidate built from an errored trace (it links the trace)."""
    return entry.section == "failure_lessons" and bool(entry.source_trace_ids)


def keep_state_key_update(
    proposed: ProposedLesson, written: CodingWrittenLesson
) -> CodingWrittenLesson:
    """``already_known`` never applies to a state-key update (coding mode).

    A state lesson replaces the earlier lesson with its key, so a changed state is new
    by definition. When the writer still rejects one as known, the curator's working
    statement is written as the state lesson instead of dropping the update.
    """
    state_key = str(getattr(proposed, "state_key", "") or "").strip()
    if written.accept or written.reason != "already_known" or not state_key:
        return written
    logger.info(
        "Coding distillation: keeping state-key update '%s' the writer called known", state_key
    )
    return CodingWrittenLesson(
        accept=True,
        statement=strip_state_key(proposed.working_statement),
        lesson_type="state",
        state_key=state_key,
    )


def apply_evidence_rules(
    lesson: CodingWrittenLesson, members: list[SessionContextEntry]
) -> CodingWrittenLesson:
    """Make the lesson type match its evidence (coding mode).

    A lesson backed by a repeated-error candidate (one carrying a ``pitfall:`` state
    key) is a pitfall with that key, whatever the writer typed; a ``pitfall`` with no
    failed-trace candidate behind it is a ``change``. Only state and pitfall lessons
    keep a state key, and a key the writer echoed into the statement is stripped.
    """
    if not lesson.accept:
        return lesson
    pitfall_keys = [
        key
        for key in (candidate_state_key(member.content) for member in members)
        if key.startswith(PITFALL_STATE_KEY_PREFIX)
    ]
    lesson_type = lesson.lesson_type
    state_key = lesson.state_key.strip()
    if pitfall_keys:
        lesson_type, state_key = "pitfall", pitfall_keys[0]
    elif lesson_type == "pitfall" and not any(is_failed_trace_candidate(m) for m in members):
        lesson_type, state_key = "change", ""
    if lesson_type not in ("state", "pitfall"):
        state_key = ""
    return lesson.model_copy(
        update={
            "lesson_type": lesson_type,
            "state_key": state_key,
            "statement": strip_state_key(lesson.statement),
        }
    )


async def evaluate_proposed_lesson(
    vector_engine,
    lesson: ProposedLesson,
    entries_by_id: dict,
    *,
    coding: bool = False,
) -> WrittenLesson | None:
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
    written = await write_or_reject(lesson, members, prior_lessons, glossary, coding=coding)
    if coding and isinstance(written, CodingWrittenLesson):
        return apply_evidence_rules(keep_state_key_update(lesson, written), members)
    return written


async def accept_proposed_lessons(
    scope: SessionDistillationScope,
    proposed: list[ProposedLesson],
    context_entries: list[SessionContextEntry],
    *,
    coding: bool = False,
) -> tuple[list[WrittenLesson], int]:
    """Returns ``(accepted lessons, failed writer calls)`` — see write_or_reject.

    Raises ``LLMPaymentRequiredError`` once a writer call reports an exhausted
    budget; the remaining calls are skipped, not made.
    """
    entries_by_id = {entry.id: entry for entry in context_entries}
    budget_exhausted = False

    async def write_lesson(lesson: ProposedLesson, vector_engine) -> WrittenLesson | None:
        nonlocal budget_exhausted
        if budget_exhausted:
            return None
        try:
            return await evaluate_proposed_lesson(
                vector_engine, lesson, entries_by_id, coding=coding
            )
        except LLMPaymentRequiredError:
            budget_exhausted = True
            return None

    async with set_database_global_context_variables(scope.dataset.id, scope.dataset.owner_id):
        vector_engine = await get_vector_engine_async()
        writer_calls = [
            lambda lesson=lesson: write_lesson(lesson, vector_engine) for lesson in proposed
        ]
        decisions = await gather_with_concurrency_limit(writer_calls, WRITER_CONCURRENCY)

    if budget_exhausted:
        raise LLMPaymentRequiredError(
            f"session {scope.session_id}: LLM budget exhausted while writing lessons"
        )

    accepted = [
        lesson
        for lesson in decisions
        if lesson is not None and lesson.accept and lesson.statement.strip()
    ]
    failed_calls = sum(1 for lesson in decisions if lesson is None)
    return accepted, failed_calls


def render_lesson_document(
    lesson: WrittenLesson,
    *,
    session_id: str,
) -> str:
    """Render ONE accepted lesson as a standalone markdown document.

    The template — not the LLM — controls the format. One document per lesson, so each
    learning is an independently identifiable unit in the graph. The document carries no
    run date: an identical lesson re-accepted on a later run must hash identically so
    add()'s content dedup absorbs it instead of storing a second copy.
    """
    statement = lesson.statement.strip()
    why = lesson.why_learned.strip().rstrip(".")
    body = f"{statement} ({why}.)" if why else statement
    return f"# Session learning (session {session_id})\n\n{body}\n"


def lesson_metadata(lesson: CodingWrittenLesson, scope, qa_context: dict, seen: int | None) -> dict:
    """The ``Data.external_metadata`` of one coding lesson (SDK-1336 reads it from the listing)."""
    metadata = {
        "lesson_type": lesson.lesson_type,
        "state_key": lesson.state_key,
        "session_id": scope.session_id,
        "session_date": str(qa_context.get("date") or ""),
        "git_commit": str(qa_context.get("end_commit") or qa_context.get("start_commit") or ""),
        "git_branch": str(qa_context.get("branch") or ""),
        "files": mentioned_files(lesson.statement),
    }
    if seen is not None:
        metadata["seen_in_sessions"] = seen
    return metadata


async def build_coding_lesson_documents(
    scope: SessionDistillationScope,
    accepted: list[CodingWrittenLesson],
    qa_context: dict,
) -> list[LessonDocument]:
    documents = []
    for lesson in accepted:
        seen = None
        if lesson.lesson_type == "pitfall" and lesson.state_key:
            rows = await find_lessons_by_state_key(scope.dataset.id, lesson.state_key)
            seen = seen_in_sessions(rows, scope.session_id)
        text = render_coding_lesson_document(
            lesson.statement,
            session_id=scope.session_id,
            lesson_type=lesson.lesson_type,
            state_key=lesson.state_key,
            seen_in_sessions=seen,
        )
        documents.append(
            LessonDocument(
                text=text,
                metadata=lesson_metadata(lesson, scope, qa_context, seen),
                data_id=lesson_data_id(scope.dataset.id, text),
            )
        )
    return documents


async def _lesson_node_set(scope: SessionDistillationScope) -> list[str]:
    # Lessons inherit the session's pinned node set, like the Q&A and trace
    # bridges do, so a node_name-scoped recall also sees what the session taught.
    session_node_set = await get_session_node_set(
        get_session_manager(), scope.user_id, scope.session_id
    )
    return bridge_node_set(
        [*DISTILLATE_NODE_SET, truth_session_node_set(scope.session_id)], session_node_set
    )


async def publish_distilled_lessons(
    scope: SessionDistillationScope,
    accepted: list[WrittenLesson],
    *,
    qa_context: dict | None = None,
) -> list[str]:
    """Store the accepted lessons as documents; ``qa_context`` given means coding mode.

    Coding lessons are added as ``DataItem``s with their metadata and a deterministic
    id, and once they are in the graph every older lesson of the dataset with one of
    their state keys is deleted, so a state key always names its newest lesson.
    """
    # Imported lazily to avoid a circular import through the cognee package root.
    from cognee.api.v1.add import add
    from cognee.api.v1.cognify import cognify

    node_set = await _lesson_node_set(scope)
    if qa_context is None:
        documents = [
            render_lesson_document(lesson, session_id=scope.session_id) for lesson in accepted
        ]
        await add(documents, dataset_id=scope.dataset.id, user=scope.user, node_set=node_set)
        await cognify(datasets=[scope.dataset.id], user=scope.user)
        return documents

    from cognee.tasks.ingestion.data_item import DataItem

    if not all(isinstance(lesson, CodingWrittenLesson) for lesson in accepted):
        raise TypeError("coding distillation publishes CodingWrittenLesson items only")
    coding_lessons = [lesson for lesson in accepted if isinstance(lesson, CodingWrittenLesson)]
    lessons = await build_coding_lesson_documents(scope, coding_lessons, qa_context)
    items = [
        DataItem(
            data=lesson.text,
            external_metadata=lesson.metadata,
            data_id=lesson.data_id,
            literal_text=True,
        )
        for lesson in lessons
    ]
    await add(items, dataset_id=scope.dataset.id, user=scope.user, node_set=node_set)
    await cognify(datasets=[scope.dataset.id], user=scope.user)
    for state_key in sorted({lesson.metadata["state_key"] for lesson in lessons} - {""}):
        keep = {lesson.data_id for lesson in lessons if lesson.metadata["state_key"] == state_key}
        await delete_superseded_lessons(
            dataset_id=scope.dataset.id, state_key=state_key, keep=keep, user=scope.user
        )
    return [lesson.text for lesson in lessons]


async def select_undistilled_entries(
    scope: SessionDistillationScope,
    context_entries: list[SessionContextEntry],
) -> list[SessionContextEntry]:
    """Gated entries not yet covered by this (session, dataset)'s distillation watermark."""
    distilled_ids = await get_distilled_entry_ids(
        get_session_manager(), scope.user_id, scope.session_id, scope.dataset_id
    )
    return [entry for entry in context_entries if entry.id not in distilled_ids]


async def advance_distillation_watermark(
    scope: SessionDistillationScope,
    context_entries: list[SessionContextEntry],
) -> None:
    """Record every gated entry this run saw as distilled into the scope's dataset.

    Called only once the run is complete: lessons published, or the LLM decided there
    was nothing durable to keep. A run that raised never reaches this point, so its
    window is retried next time.
    """
    await save_distilled_entry_ids(
        get_session_manager(),
        scope.user_id,
        scope.session_id,
        scope.dataset_id,
        [entry.id for entry in context_entries],
    )


async def distill_session(
    session_id: str,
    dataset: str | UUID,
    user: User | None = None,
    max_batches: int | None = None,
) -> DistillationResult:
    """Distill one finished session's distillable learnings into its dataset's knowledge graph.

    A per-(session, dataset) watermark records which gated entries a completed run
    already covered; when none are new the call returns ``no_new_entries`` and makes
    zero LLM calls. ``max_batches`` caps the curator batches of this run (0 = no
    cap); ``None`` reads ``IMPROVE_DISTILL_MAX_BATCHES_PER_RUN``. A failed step after
    the lessons' cognify run (``AfterRunHookError``) is raised once the watermark moved.
    """
    scope = await resolve_distillation_scope(session_id=session_id, dataset=dataset, user=user)

    context_entries = await load_distillable_context_entries(scope)
    if not context_entries:
        return scope.result("no_gated_entries")

    new_entries = await select_undistilled_entries(scope, context_entries)
    if not new_entries:
        logger.info(
            "Session %s: all %d gated entries already distilled into dataset %s",
            scope.session_id,
            len(context_entries),
            scope.dataset_id,
        )
        return scope.result("no_new_entries")

    if max_batches is None:
        max_batches = get_improve_config().distill_max_batches_per_run
    coding = coding_distillation_enabled()
    if coding:
        # A coding unit is judged against its own request and final answer only.
        qa_rows = await load_latest_qa_row(scope)
        qa_context = parse_qa_context(qa_rows[-1].get("context") if qa_rows else "")
    else:
        qa_rows = await load_session_qa_rows(scope, earliest_entry_time(new_entries))
        qa_context = None
    qa_rows, curated_entries = select_curator_inputs(qa_rows, new_entries, max_batches)
    uncurated = len(new_entries) - len(curated_entries)
    if uncurated:
        logger.warning(
            "Session %s: curating %d of %d new gated entries (cap %d batches per run); "
            "the other %d are sealed without curation",
            scope.session_id,
            len(curated_entries),
            len(new_entries),
            max_batches,
            uncurated,
        )
    run = await propose_lessons(qa_rows, curated_entries, coding=coding)
    if run.budget_exhausted:
        # Nothing was written for the completed batches either, and every later
        # call would fail the same way: seal nothing, let improve stop the run.
        # A budget rejection in the writer phase raises out of
        # accept_proposed_lessons below with the same effect.
        raise LLMPaymentRequiredError(
            f"session {scope.session_id}: LLM budget exhausted during curation; "
            f"{len(run.curated_entry_ids)} curated entries left unsealed for the next run"
        )
    if not run.proposed and not run.failed_batches:
        await advance_distillation_watermark(scope, context_entries)
        return scope.result("no_proposed_lessons", uncurated_entries=uncurated)

    accepted: list[WrittenLesson] = []
    writer_failures = 0
    if run.proposed:
        accepted, writer_failures = await accept_proposed_lessons(
            scope, run.proposed, curated_entries, coding=coding
        )

    documents: list[str] = []
    hook_error: AfterRunHookError | None = None
    if accepted:
        # Publish what survived before reporting a failure: the lesson documents
        # are date-free, so a re-run re-accepting the same lesson dedups in add()
        # instead of duplicating.
        try:
            documents = await publish_distilled_lessons(scope, accepted, qa_context=qa_context)
        except AfterRunHookError as error:
            # The lessons landed; only the step after their cognify failed. The
            # watermark still moves below, so the curator and writer calls are not
            # repeated, and the error is raised after it.
            hook_error = error

    if run.failed_batches:
        # Seal the batches that completed, now that their lessons have landed:
        # processed once is processed. The watermark is the set of gated entries
        # covered, replaced on each write, so carry the earlier seals along.
        # Failed batches and a cap's overflow stay pending for the next run.
        new_ids = {entry.id for entry in new_entries}
        await advance_distillation_watermark(
            scope,
            [
                entry
                for entry in context_entries
                if entry.id not in new_ids or entry.id in run.curated_entry_ids
            ],
        )
        raise DistillationCallsFailedError(
            f"session {scope.session_id}: {run.failed_batches} curator batch(es) and "
            f"{writer_failures} writer call(s) failed; {len(run.curated_entry_ids)} entries "
            f"of completed batches sealed, the rest left for the next run "
            f"({len(accepted)} lesson document(s) from surviving calls were published)"
        ) from hook_error

    await advance_distillation_watermark(scope, context_entries)
    if writer_failures:
        # The curator processed every entry, so they are sealed; the lessons the
        # writer lost are not retried. Raise so the stage reports the loss.
        raise DistillationCallsFailedError(
            f"session {scope.session_id}: {writer_failures} writer call(s) failed; every "
            f"curated entry sealed, the lost lessons are not retried "
            f"({len(accepted)} lesson document(s) from surviving calls were published)"
        ) from hook_error
    if hook_error is not None:
        raise hook_error
    if not accepted:
        return scope.result("no_accepted_lessons", uncurated_entries=uncurated)
    return scope.result("completed", documents=documents, uncurated_entries=uncurated)
