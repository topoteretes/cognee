"""The improve stages, wrapping the helpers that already exist.

Each class is a gate, a call into code that lives elsewhere
(``memify_pipelines/``, ``infrastructure/session/``, ``modules/user_preferences/``,
``modules/truth_subspace/``, ``modules/session_distillation/``) and a mapping of
that code's return onto ``StageResult``. No stage changes how it computes
anything; the bodies below are the former private helpers of
``cognee/api/v1/improve/improve.py``, moved here as-is.

Every stage receives the resolved dataset id, never a name (plan Part 5.3).
Heavy imports stay inside ``gate``/``run`` so importing this package pulls in
nothing but pydantic and the config modules.
"""

from cognee.modules.preflight import llm_available
from cognee.shared.logging_utils import get_logger

from .constants import AGENT_TRACE_FEEDBACKS_NODE_SET
from .inputs import ImproveRunInputs
from .result import LEGACY_CUSTOM_TASKS_STAGE_NAME, REASON_BACKEND_UNSUPPORTED, StageResult
from .stage import BaseStage

logger = get_logger("improve")

# Stage-specific gate reasons.
REASON_SESSION_MANAGER_UNAVAILABLE = "session_manager_unavailable"
REASON_AUTO_FEEDBACK_DISABLED = "auto_feedback_disabled"
REASON_PERSONALIZATION_DISABLED = "personalization_disabled"
REASON_OPT_IN_DISABLED = "opt_in_disabled"
REASON_TURNED_OFF = "turned_off"
REASON_NO_WRITES_SINCE_LAST_IMPROVE = "no_writes_since_last_improve"
REASON_NO_NEW_SESSION_ENTRIES = "no_new_entries"
REASON_NO_NEW_TRACE_STEPS = "no_new_trace_steps"
REASON_EDGE_EVIDENCE_DISABLED = "edge_evidence_disabled"
REASON_NO_NEW_FACTS = "no_new_facts"
# The stage drafts text with an LLM and no usable one is configured (keyless
# installs run on GLiNER + fastembed); the stages that only cognify or embed
# are not gated on this — cognify resolves its own extractor.
REASON_NO_LLM_CONFIGURED = "no_llm_configured"


def _already_completed(stage_name: str, reason: str) -> StageResult:
    result = StageResult(stage=stage_name, status="already_completed", reason=reason)
    result._raw_run = {}
    return result


async def _watermarks_show_nothing_new(inputs: ImproveRunInputs, *, kind: str) -> bool:
    """Pre-pipeline check for the persist stages: is every session fully covered?

    True only when the stage's own persist watermarks cover every given
    session, so the memify pipeline need not run at all — an unconditional run
    would log a completed ``memify_pipeline`` row even with nothing new.
    Fail-open: any error here means
    "run the pipeline"; the extraction tasks keep their own watermark logic,
    so this check is an optimization, never the correctness gate.
    """
    try:
        from cognee.infrastructure.session.get_session_manager import get_session_manager

        session_manager = get_session_manager()
        if not session_manager.is_available:
            return False
        user_id = str(inputs.user.id)
        if kind == "qa":
            from cognee.tasks.memify.extract_user_sessions import has_new_session_qa

            return not await has_new_session_qa(session_manager, user_id, inputs.session_id_list)
        from cognee.tasks.memify.extract_agent_trace_feedbacks import has_new_trace_steps

        return not await has_new_trace_steps(session_manager, user_id, inputs.session_id_list)
    except Exception as error:
        logger.debug(
            "improve: persist pre-check could not decide, running the pipeline: %s",
            error,
            exc_info=True,
        )
        return False


async def _pending_trace_watermarks(
    inputs: ImproveRunInputs, backlog: dict[str, int] | None = None
) -> dict[str, tuple[int, int]] | None:
    """``(persisted, total)`` step counts of each session with unpersisted steps.

    ``persisted`` is the effective watermark (a stale one, above the LIVE
    count, reads as 0). With ``backlog``, the step counts a run started with,
    only sessions still below their starting count are pending: steps written
    while the run goes on wait for the next run, and a watermark another
    persister moved past the starting count reads as done, not stale. None
    when the session store cannot be read.
    """
    try:
        from cognee.infrastructure.session.get_session_manager import get_session_manager
        from cognee.infrastructure.session.session_persist_watermark import (
            TRACE_PERSIST_WATERMARK,
        )
        from cognee.tasks.memify.extract_agent_trace_feedbacks import resolve_trace_window

        session_manager = get_session_manager()
        if not session_manager.is_available:
            return None
        user_id = str(inputs.user.id)
        pending = {}
        for session_id in inputs.session_id_list:
            total = await session_manager.get_agent_trace_count(
                user_id=user_id, session_id=session_id
            )
            persisted = await TRACE_PERSIST_WATERMARK.read_count(
                session_manager, user_id, session_id
            )
            step_limit = None
            if backlog is not None:
                if session_id not in backlog:
                    continue
                step_limit = backlog[session_id]
            # The live total drives the stale check; the limit only bounds the reach.
            effective, pending_steps = resolve_trace_window(
                total, persisted, None, session_id=session_id, step_limit=step_limit
            )
            if total and pending_steps:
                pending[session_id] = (effective, total)
        return pending
    except Exception as error:
        logger.debug("improve: trace watermarks unreadable: %s", error, exc_info=True)
        return None


class FeedbackWeightsStage(BaseStage):
    """Stage 1: move ``feedback_weight`` on the graph elements scored answers used."""

    name = "feedback_weights"
    needs_sessions = True

    def gate(self, inputs: ImproveRunInputs) -> str | None:
        # No influence gate: feedback_influence is a read-time ranking knob that
        # recall()/search() take per call, so weights must be written even when
        # the global default is 0 (the only value base_config doesn't warn about).
        if not inputs.capabilities.supports_feedback_weights:
            return REASON_BACKEND_UNSUPPORTED
        return None

    async def run(self, inputs: ImproveRunInputs) -> StageResult:
        from cognee.memify_pipelines.apply_feedback_weights import (
            apply_feedback_weights_pipeline,
        )

        result = await apply_feedback_weights_pipeline(
            user=inputs.user,
            session_ids=inputs.session_id_list,
            dataset=inputs.dataset_id,
            alpha=inputs.feedback_alpha,
            run_in_background=False,
        )
        logger.info("improve: feedback weights applied from %d session(s)", len(inputs.session_ids))
        return StageResult.from_pipeline_run(self.name, result, sessions=len(inputs.session_ids))


class PersistSessionQAStage(BaseStage):
    """Stage 2: cognify session Q&A into the graph (``user_sessions_from_cache``).

    The single fail-closed stage (decision D2): losing Q&A would be data loss,
    so an error here stops the run instead of being swallowed. A failed step
    after the completed cognify (``AfterRunHookError``) is recorded as errored
    without stopping it: the Q&A is stored.
    """

    name = "persist_session_qa"
    needs_sessions = True
    fatal = True

    async def run(self, inputs: ImproveRunInputs) -> StageResult:
        from cognee.memify_pipelines.persist_sessions_in_knowledge_graph import (
            persist_sessions_in_knowledge_graph_pipeline,
        )

        if await _watermarks_show_nothing_new(inputs, kind="qa"):
            return _already_completed(self.name, REASON_NO_NEW_SESSION_ENTRIES)

        result = await persist_sessions_in_knowledge_graph_pipeline(
            user=inputs.user,
            session_ids=inputs.session_id_list,
            dataset=inputs.dataset_id,
            run_in_background=False,
        )
        logger.info("improve: session Q&A persisted from %d session(s)", len(inputs.session_ids))
        return StageResult.from_pipeline_run(self.name, result, sessions=len(inputs.session_ids))


# Sessions one persist_agent_traces pass cognifies, one window each. A pass is
# one memify run, which holds the dataset lock until it ends, so this bounds how
# long a remember() or cognify() on the dataset waits behind a draining run.
TRACE_SESSIONS_PER_PASS = 10


class PersistAgentTracesStage(BaseStage):
    """Stage 3: cognify per-step agent trace feedback (``agent_trace_feedbacks``)."""

    name = "persist_agent_traces"
    needs_sessions = True

    async def run(self, inputs: ImproveRunInputs) -> StageResult:
        from cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph import (
            persist_agent_trace_feedbacks_in_knowledge_graph_pipeline,
        )

        if await _watermarks_show_nothing_new(inputs, kind="traces"):
            return _already_completed(self.name, REASON_NO_NEW_TRACE_STEPS)

        # Each pipeline call cognifies every pending step of each pending session
        # as one document (the OLDEST trace_window_steps of them when that cap is
        # set) and advances their watermarks, so a long session costs one cognify
        # run, not one per window. The loop drains the backlog the run started
        # with, unless trace_windows_per_run caps the run; steps an agent writes
        # meanwhile wait for the next run (the window is cut at the starting
        # count, persist_up_to), so a live session cannot keep the run (and the
        # dataset's improve lock) going. A pass's windows build in one cognify
        # run; a failed build raises with its cause and ends the run, nothing
        # built, every watermark in place. A pass takes at most
        # TRACE_SESSIONS_PER_PASS sessions, those with the fewest windows so far,
        # so the dataset lock is released between passes and sessions late in
        # the list get their turn before earlier ones drain.
        config = inputs.config
        max_windows = config.trace_windows_per_run
        sessions = len(inputs.session_ids)
        windows = 0
        stuck: list[str] = []
        served: dict[str, int] = {}
        stage_result = None
        pending = await _pending_trace_watermarks(inputs)
        backlog = {sid: total for sid, (_mark, total) in (pending or {}).items()}
        while pending:
            limit = TRACE_SESSIONS_PER_PASS
            if max_windows:
                limit = max(0, min(limit, max_windows - windows))
            candidates = sorted(
                (sid for sid in pending if sid not in stuck), key=lambda sid: served.get(sid, 0)
            )
            batch = {sid: pending[sid][0] for sid in candidates[:limit]}
            if not batch:
                break
            result = await persist_agent_trace_feedbacks_in_knowledge_graph_pipeline(
                user=inputs.user,
                session_ids=list(batch),
                dataset=inputs.dataset_id,
                node_set_name=AGENT_TRACE_FEEDBACKS_NODE_SET,
                raw_trace_content=False,
                last_n_steps=config.trace_window_steps or None,
                run_in_background=False,
                persist_up_to=backlog,
            )
            windows += len(batch)
            for sid in batch:
                served[sid] = served.get(sid, 0) + 1
            stage_result = StageResult.from_pipeline_run(
                self.name, result, sessions=sessions, windows=windows
            )
            if stage_result.status == "errored":
                break
            # A failed build raises and ended the run above. A session can still
            # come back unmoved when extraction skipped it (its context could not
            # be read): set it aside for the rest of the run and report it.
            pending = await _pending_trace_watermarks(inputs, backlog)
            if pending is None:
                break
            stuck.extend(
                sid for sid, mark in batch.items() if sid in pending and pending[sid][0] == mark
            )
        logger.info(
            "improve: persisted %d agent trace window(s) from %d session(s)", windows, sessions
        )
        if stage_result is not None and stage_result.status == "errored":
            return stage_result
        if pending is None:
            return StageResult.errored(
                self.name,
                "agent trace watermarks could not be read",
                sessions=sessions,
                windows=windows,
            )
        if stuck:
            return StageResult.errored(
                self.name,
                f"{len(stuck)} session(s) did not advance their trace persist watermark: "
                + ", ".join(stuck),
                sessions=sessions,
                windows=windows,
                sessions_stuck=len(stuck),
            )
        if stage_result is not None:
            return stage_result
        return _already_completed(self.name, REASON_NO_NEW_TRACE_STEPS)


class ExtractAgentContextStage(BaseStage):
    """Stage 4: flush pending trace windows into agent-profile lessons.

    Delegates to ``agent_context_extraction.extract_pending_agent_context`` per
    session, sharing the watermark mid-session extraction uses.
    ``min_new_traces=1`` flushes whatever is still unprocessed before
    distillation. Fail-open per session: one failing session never blocks the
    others; the stage reports ``errored`` when any session failed.
    """

    name = "extract_agent_context"
    needs_sessions = True

    def gate(self, inputs: ImproveRunInputs) -> str | None:
        from cognee.infrastructure.session.get_session_manager import get_session_manager

        session_manager = get_session_manager()
        if not session_manager.is_available:
            return REASON_SESSION_MANAGER_UNAVAILABLE
        if not session_manager.is_auto_feedback_enabled():
            return REASON_AUTO_FEEDBACK_DISABLED
        if not llm_available():
            return REASON_NO_LLM_CONFIGURED
        return None

    async def run(self, inputs: ImproveRunInputs) -> StageResult:
        from cognee.infrastructure.session.agent_context_extraction import (
            extract_pending_agent_context,
        )
        from cognee.infrastructure.session.get_session_manager import get_session_manager

        session_manager = get_session_manager()
        user_id = str(inputs.user.id)
        touched = 0
        failed = 0
        last_error: BaseException | None = None
        for session_id in inputs.session_ids:
            try:
                ids = await extract_pending_agent_context(
                    session_manager=session_manager,
                    user_id=user_id,
                    session_id=session_id,
                    min_new_traces=1,
                )
                touched += len(ids)
            except Exception as e:
                failed += 1
                last_error = e
                logger.warning(
                    "improve: agent-context extraction failed for '%s' (non-fatal): %s",
                    session_id,
                    e,
                    exc_info=True,
                )
        counts = {"lessons": touched, "sessions_failed": failed}
        if failed:
            return StageResult.errored(self.name, last_error, **counts)
        return StageResult.completed(self.name, **counts)


class DistillSessionsStage(BaseStage):
    """Stage 5: distill each session's gated guidance into lessons (``session_learnings``).

    Delegates to ``session_distillation.distill_session`` per session. A
    session with no gated guidance yields no lessons; an error on one session
    never blocks the others. ``distill_session`` runs its own add/cognify and
    never calls ``improve``, so there is no recursion.
    """

    name = "distill_sessions"
    needs_sessions = True

    def gate(self, inputs: ImproveRunInputs) -> str | None:
        if not llm_available():
            return REASON_NO_LLM_CONFIGURED
        return None

    async def run(self, inputs: ImproveRunInputs) -> StageResult:
        from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
        from cognee.modules.session_distillation import distill_session

        distilled = 0
        completed = 0
        failed = 0
        uncurated = 0
        last_error: BaseException | None = None
        for session_id in inputs.session_ids:
            try:
                result = await distill_session(
                    session_id,
                    dataset=inputs.dataset_id,
                    user=inputs.user,
                    max_batches=inputs.config.distill_max_batches_per_run,
                )
                distilled += len(result.documents)
                uncurated += result.uncurated_entries
                if result.status == "completed":
                    completed += 1
                logger.info(
                    "improve: distilled session '%s' -> status=%s documents=%d",
                    session_id,
                    result.status,
                    len(result.documents),
                )
            except LLMPaymentRequiredError as e:
                # Every later session would fail the same way: stop here, and keep
                # this error so the stage result reports the exhausted budget.
                failed += 1
                last_error = e
                logger.warning(
                    "improve: LLM budget exhausted distilling session '%s'; "
                    "skipping the remaining sessions: %s",
                    session_id,
                    e,
                )
                break
            except Exception as e:
                failed += 1
                last_error = e
                logger.warning(
                    "improve: session distillation failed for '%s' (non-fatal): %s",
                    session_id,
                    e,
                    exc_info=True,
                )
        counts = {
            "documents": distilled,
            "sessions_completed": completed,
            "sessions_failed": failed,
            "entries_uncurated": uncurated,
        }
        if failed:
            return StageResult.errored(self.name, last_error, **counts)
        return StageResult.completed(self.name, **counts)


class UpdateUserPreferencesStage(BaseStage):
    """Stage 6: fold rated turns and stated preferences into the user's ``prefers`` subgraph."""

    name = "update_user_preferences"
    needs_sessions = True

    def gate(self, inputs: ImproveRunInputs) -> str | None:
        from cognee.base_config import get_base_config

        if not get_base_config().personalization_enabled:
            return REASON_PERSONALIZATION_DISABLED
        return None

    async def run(self, inputs: ImproveRunInputs) -> StageResult:
        from cognee.modules.user_preferences.update import update_user_preferences

        result = await update_user_preferences(
            session_ids=inputs.session_id_list,
            dataset=inputs.dataset_id,
            user=inputs.user,
        )
        if result.status == "personalization_disabled":
            logger.debug("improve: user preference stage skipped (PERSONALIZATION_ENABLED is off)")
            return StageResult.skipped(self.name, REASON_PERSONALIZATION_DISABLED)
        logger.info(
            "improve: user preferences updated -> status=%s turns=%d edges=%d "
            "pruned=%d text_lines=%d",
            result.status,
            result.turns_applied,
            result.edges_written,
            result.edges_pruned,
            result.text_lines_added,
        )
        return StageResult(
            stage=self.name,
            status="completed",
            reason=result.status if result.status != "completed" else None,
            counts={
                "turns_applied": result.turns_applied,
                "edges_written": result.edges_written,
                "edges_pruned": result.edges_pruned,
                "text_lines_added": result.text_lines_added,
            },
        )


class BuildTruthSubspaceStage(BaseStage):
    """Stage 7: build the truth subspace from distilled learnings (opt-in)."""

    name = "build_truth_subspace"
    needs_sessions = True

    def gate(self, inputs: ImproveRunInputs) -> str | None:
        if not inputs.build_truth_subspace:
            return REASON_OPT_IN_DISABLED
        if not inputs.capabilities.supports_truth_state:
            return REASON_BACKEND_UNSUPPORTED
        return None

    async def run(self, inputs: ImproveRunInputs) -> StageResult:
        from cognee.modules.truth_subspace.build import build_truth_subspace

        result_ts = await build_truth_subspace(
            dataset=inputs.dataset_id,
            session_ids=inputs.session_id_list,
            user=inputs.user,
        )
        logger.info("improve: truth subspace built -> %s", result_ts)
        counts = {
            key: int(value)
            for key, value in (result_ts or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        # The build fails open — a failed centroid commit or chunk write comes
        # back as {"status": "errored", "error": ...} instead of a raise — so
        # translate that status: reporting it completed would hide the failure
        # from the ImproveResult AND from the improve operation row (the
        # "errored stage records a failed run" rule keys off the stage status).
        status = (result_ts or {}).get("status")
        if status == "errored":
            return StageResult.errored(
                self.name,
                (result_ts or {}).get("error") or "truth subspace build errored",
                **counts,
            )
        if status == "skipped":
            # Normally unreachable: this stage's gate checks the same
            # capability first. Reachable by a mid-run capability change; the
            # mapping should not lie about it.
            return StageResult.skipped(self.name, (result_ts or {}).get("reason") or "skipped")
        return StageResult.completed(self.name, **counts)


class ReviewConflictsStage(BaseStage):
    """Review changed facts and repair conflicts with missing citations or pending writes."""

    name = "review_conflicts"

    def gate(self, inputs: ImproveRunInputs) -> str | None:
        if not inputs.review_conflicts:
            return REASON_TURNED_OFF
        if not llm_available():
            return REASON_NO_LLM_CONFIGURED
        from cognee.modules.provenance.edge_evidence.config import get_provenance_config

        if not get_provenance_config().edge_evidence_enabled:
            return REASON_EDGE_EVIDENCE_DISABLED
        return None

    async def run(self, inputs: ImproveRunInputs) -> StageResult:
        from datetime import datetime, timezone

        from .review_conflicts import build_stamp_fields, review_conflicts
        from .watermarks import REVIEW_WATERMARK_KEY, last_improve_watermark, watermark_stamp

        started_at = datetime.now(timezone.utc)
        watermark = await last_improve_watermark(
            inputs.dataset_id,
            REVIEW_WATERMARK_KEY,
            exclude_operation_id=inputs.improve_operation_id,
        )
        review = await review_conflicts(watermark, dataset=inputs.dataset, user=inputs.user)
        if review is None:
            result = _already_completed(self.name, REASON_NO_NEW_FACTS)
        else:
            result = StageResult.from_pipeline_run(
                self.name,
                review.run_result,
                entities_failed=len(review.failed_entity_ids),
                entities_skipped=len(review.skipped_entity_ids),
            )
            # Only a completed review moves the watermark.
            if result.status != "completed":
                return result
        stamp_fields = build_stamp_fields(review)
        if stamp_fields is not None:
            result._run_info_stamp = watermark_stamp(
                REVIEW_WATERMARK_KEY, result.status, started_at, **stamp_fields
            )
        return result


class LegacyCustomTasksStage(BaseStage):
    """Deprecated (SDK-1115): ``improve(extraction_tasks=..., enrichment_tasks=...)``.

    Not in ``DEFAULT_STAGES``. ``improve()`` appends it for a run whose caller
    gave custom tasks, so they keep running for one release, as the removed
    triplet-enrichment stage ran them. Its raw return fills the deprecated
    ``ImproveResult.memify_run``. Removed in the next release, with those options.
    """

    name = LEGACY_CUSTOM_TASKS_STAGE_NAME

    async def run(self, inputs: ImproveRunInputs) -> StageResult:
        from cognee.api.v1.memify.memify import run_legacy_memify

        run_result = await run_legacy_memify(
            **dict(inputs.legacy_memify or {}),
            dataset=inputs.dataset_id,
            user=inputs.user,
            run_in_background=False,
        )
        return StageResult.from_pipeline_run(self.name, run_result)


class GlobalContextIndexStage(BaseStage):
    """Stage 9: build retrieval-ready bucket and root summaries (opt-in).

    The pipeline's default method counts entities instead of prompting, so this
    stage runs on the keyless path too. ``already_completed`` when only reads
    ran on the dataset since its last build, read from ``pipeline_runs``: a
    build reads the whole dataset.
    """

    name = "global_context_index"

    def gate(self, inputs: ImproveRunInputs) -> str | None:
        if not inputs.build_global_context_index:
            return REASON_TURNED_OFF
        return None

    async def run(self, inputs: ImproveRunInputs) -> StageResult:
        from datetime import datetime, timezone

        from cognee.context_global_variables import backend_access_control_enabled
        from cognee.memify_pipelines.global_context_index import global_context_index_pipeline

        from .watermarks import (
            GLOBAL_CONTEXT_WATERMARK_KEY,
            has_dataset_changed_since_last_improve,
            watermark_stamp,
        )

        # Captured before the change check and the build, so a write racing
        # this run stays visible to the next run's check.
        started_at = datetime.now(timezone.utc)
        if await has_dataset_changed_since_last_improve(
            inputs.dataset_id,
            GLOBAL_CONTEXT_WATERMARK_KEY,
            exclude_operation_id=inputs.improve_operation_id,
            # Without access control datasets share one graph, and relation and
            # type lines read edges that any of them may write.
            any_dataset=not backend_access_control_enabled(),
        ):
            run_result = await global_context_index_pipeline(
                user=inputs.user,
                dataset=inputs.dataset_id,
                run_in_background=False,
            )
            logger.info("improve: global context index updated")
            result = StageResult.from_pipeline_run(self.name, run_result)
        else:
            result = _already_completed(self.name, REASON_NO_WRITES_SINCE_LAST_IMPROVE)
        if result.status in ("completed", "already_completed"):
            result._run_info_stamp = watermark_stamp(
                GLOBAL_CONTEXT_WATERMARK_KEY, result.status, started_at
            )
        return result
