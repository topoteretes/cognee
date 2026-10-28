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
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from cognee.context_global_variables import session_user, set_database_global_context_variables
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
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
from cognee.modules.improve.config import get_improve_config
from cognee.modules.improve.constants import SESSION_LEARNINGS_NODE_SET
from cognee.modules.pipelines.exceptions import AfterRunHookError
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
    MAX_QA_FEEDBACK_CHARS,
    MAX_QA_QUESTION_CHARS,
    NOVELTY_LESSONS_PER_LESSON,
    WRITER_CONCURRENCY,
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
) -> list[CuratorBatch]:
    """Pack the session timeline into coarse, size-safe chronological batches."""
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

    batches: list[CuratorBatch] = []
    for index in range(0, len(timeline), CURATOR_BLOCKS_PER_BATCH):
        chunk = timeline[index : index + CURATOR_BLOCKS_PER_BATCH]
        batches.append(
            CuratorBatch(
                text="\n\n".join(block for _timestamp, block, _entry_id in chunk),
                entry_ids=[entry_id for _timestamp, _block, entry_id in chunk if entry_id],
            )
        )
    return batches


async def curate_batch(batch_text: str) -> list[ProposedLesson] | None:
    """One curator call over one batch slice.

    ``None`` means the call FAILED (missing prompt, LLM error) — distinct from
    ``[]``, the curator deciding nothing in the batch is durable. The caller
    must not seal entries behind a failed call as if they were curated. An
    exhausted LLM budget is raised, not swallowed: every later call would fail
    the same way, and improve stops the run on it.
    """
    system_prompt = read_query_prompt(CURATOR_PROMPT_FILE)
    if not system_prompt:
        logger.warning("Distillation curator prompt not found: %s", CURATOR_PROMPT_FILE)
        return None
    try:
        result = await LLMGateway.acreate_structured_output(
            text_input=batch_text,
            system_prompt=system_prompt,
            response_model=CuratorBatchOutput,
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
) -> CuratorRun:
    """Pack session inputs into curator batches, then flatten proposed lessons.

    ``curated_entry_ids`` holds the entries of every batch whose call completed;
    a failed curator call is not "nothing durable found", so its entries are
    left out and ``failed_batches`` counts it. Once one call reports an
    exhausted budget the remaining batches are skipped, not called, and
    ``budget_exhausted`` is set.
    """
    batches = build_curator_batches(qa_rows, context_entries)
    if not batches:
        return CuratorRun(proposed=[], failed_batches=0, curated_entry_ids=set())

    budget_exhausted = False

    async def run_batch(batch: CuratorBatch) -> list[ProposedLesson] | None:
        nonlocal budget_exhausted
        if budget_exhausted:
            return None
        try:
            return await curate_batch(batch.text)
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


def build_writer_input(
    lesson: ProposedLesson,
    members: list[SessionContextEntry],
    prior_lessons: list[str],
    glossary: list[str],
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
    members: list[SessionContextEntry],
    prior_lessons: list[str],
    glossary: list[str],
) -> WrittenLesson | None:
    """One writer/rejecter call for one proposed lesson.

    ``None`` means the call FAILED (missing prompt, LLM error); a rejection is
    a ``WrittenLesson`` with ``accept=False``. The caller counts failures so a
    writer outage is never mistaken for "every lesson rejected".
    """
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
    except LLMPaymentRequiredError:
        raise
    except Exception as error:
        logger.warning("Distillation writer call failed: %s", error, exc_info=True)
        return None


async def evaluate_proposed_lesson(
    vector_engine,
    lesson: ProposedLesson,
    entries_by_id: dict,
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
    return await write_or_reject(lesson, members, prior_lessons, glossary)


async def accept_proposed_lessons(
    scope: SessionDistillationScope,
    proposed: list[ProposedLesson],
    context_entries: list[SessionContextEntry],
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
            return await evaluate_proposed_lesson(vector_engine, lesson, entries_by_id)
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


async def publish_distilled_lessons(
    scope: SessionDistillationScope,
    accepted: list[WrittenLesson],
) -> list[str]:
    documents = [render_lesson_document(lesson, session_id=scope.session_id) for lesson in accepted]

    # Imported lazily to avoid a circular import through the cognee package root.
    from cognee.api.v1.add import add
    from cognee.api.v1.cognify import cognify

    # Lessons inherit the session's pinned node set, like the Q&A and trace
    # bridges do, so a node_name-scoped recall also sees what the session taught.
    session_node_set = await get_session_node_set(
        get_session_manager(), scope.user_id, scope.session_id
    )
    node_set = bridge_node_set(
        [*DISTILLATE_NODE_SET, truth_session_node_set(scope.session_id)], session_node_set
    )
    await add(documents, dataset_id=scope.dataset.id, user=scope.user, node_set=node_set)
    await cognify(datasets=[scope.dataset.id], user=scope.user)
    return documents


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
    qa_rows = await load_session_qa_rows(scope, earliest_entry_time(new_entries))
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
    run = await propose_lessons(qa_rows, curated_entries)
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
            scope, run.proposed, curated_entries
        )

    documents: list[str] = []
    hook_error: AfterRunHookError | None = None
    if accepted:
        # Publish what survived before reporting a failure: the lesson documents
        # are date-free, so a re-run re-accepting the same lesson dedups in add()
        # instead of duplicating.
        try:
            documents = await publish_distilled_lessons(scope, accepted)
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
