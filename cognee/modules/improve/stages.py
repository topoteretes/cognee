"""The nine improve stages, wrapping the helpers that already exist.

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

from cognee.shared.logging_utils import get_logger

from .constants import AGENT_TRACE_FEEDBACKS_NODE_SET
from .inputs import ImproveRunInputs
from .result import REASON_BACKEND_UNSUPPORTED, StageResult
from .stage import BaseStage

logger = get_logger("improve")

# Stage-specific gate reasons.
REASON_SESSION_MANAGER_UNAVAILABLE = "session_manager_unavailable"
REASON_AUTO_FEEDBACK_DISABLED = "auto_feedback_disabled"
REASON_PERSONALIZATION_DISABLED = "personalization_disabled"
REASON_OPT_IN_DISABLED = "opt_in_disabled"
REASON_TRIPLET_EMBEDDING_DISABLED = "triplet_embedding_disabled"
REASON_NO_WRITES_SINCE_LAST_IMPROVE = "no_writes_since_last_improve"
REASON_NO_NEW_SESSION_ENTRIES = "no_new_entries"
REASON_NO_NEW_TRACE_STEPS = "no_new_trace_steps"


def _already_completed(stage_name: str, reason: str) -> StageResult:
    result = StageResult(stage=stage_name, status="already_completed", reason=reason)
    result._raw_run = {}
    return result


async def _watermarks_show_nothing_new(inputs: ImproveRunInputs, *, kind: str) -> bool:
    """Pre-pipeline check for the persist stages: is every session fully covered?

    True only when the stage's own persist watermarks cover every given
    session, so the memify pipeline need not run at all — an unconditional run
    logs a completed ``memify_pipeline`` row even with nothing new, which the
    enrichment change-check counts as a graph write (a repeat session improve
    would then re-embed the whole dataset). Fail-open: any error here means
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
    so an error here stops the run instead of being swallowed.
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

        result = await persist_agent_trace_feedbacks_in_knowledge_graph_pipeline(
            user=inputs.user,
            session_ids=inputs.session_id_list,
            dataset=inputs.dataset_id,
            node_set_name=AGENT_TRACE_FEEDBACKS_NODE_SET,
            raw_trace_content=False,
            last_n_steps=None,  # persist all stored steps on demand
            run_in_background=False,
        )
        logger.info(
            "improve: agent trace steps persisted from %d session(s)", len(inputs.session_ids)
        )
        return StageResult.from_pipeline_run(self.name, result, sessions=len(inputs.session_ids))


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

    async def run(self, inputs: ImproveRunInputs) -> StageResult:
        from cognee.modules.session_distillation import distill_session

        distilled = 0
        completed = 0
        failed = 0
        last_error: BaseException | None = None
        for session_id in inputs.session_ids:
            try:
                result = await distill_session(
                    session_id, dataset=inputs.dataset_id, user=inputs.user
                )
                distilled += len(result.documents)
                if result.status == "completed":
                    completed += 1
                logger.info(
                    "improve: distilled session '%s' -> status=%s documents=%d",
                    session_id,
                    result.status,
                    len(result.documents),
                )
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


class TripletEnrichmentStage(BaseStage):
    """Stage 8: default memify enrichment (triplet embeddings).

    Skipped when ``triplet_embedding`` is off and the caller supplied no tasks
    of their own (the default task list would be empty). ``already_completed``
    when no write pipeline has completed for this dataset since its last
    improve, read from ``pipeline_runs`` — unless the caller scoped the run
    with ``node_name`` or supplied custom tasks: the watermark's unit is the
    dataset, and "nothing changed" does not imply that narrower or different
    work was already done.
    """

    name = "triplet_enrichment"

    def gate(self, inputs: ImproveRunInputs) -> str | None:
        if inputs.has_custom_memify_tasks:
            return None
        from cognee.modules.cognify.config import get_cognify_config

        if not get_cognify_config().triplet_embedding:
            return REASON_TRIPLET_EMBEDDING_DISABLED
        return None

    async def run(self, inputs: ImproveRunInputs) -> StageResult:
        from datetime import datetime, timezone

        from cognee.modules.memify import memify

        from .graph_changes import (
            enrichment_watermark_stamp,
            has_graph_changed_since_last_improve,
        )

        # Only a full, unscoped enrichment stamps the watermark: the stamp
        # says "the whole dataset was enriched as of started_at", which
        # narrower or different work cannot claim. started_at is captured
        # before the change check and the memify run, so a write racing the
        # row close stays visible to the next run's gate.
        full_scope = not inputs.has_custom_memify_tasks and not inputs.node_name
        started_at = datetime.now(timezone.utc)

        if full_scope and not await has_graph_changed_since_last_improve(
            inputs.dataset_id, exclude_operation_id=inputs.improve_operation_id
        ):
            result = StageResult(
                stage=self.name,
                status="already_completed",
                reason=REASON_NO_WRITES_SINCE_LAST_IMPROVE,
            )
            result._raw_run = {}
            result._run_info_stamp = enrichment_watermark_stamp(result.status, started_at)
            return result

        kwargs = dict(inputs.memify_kwargs)
        if kwargs.get("node_type") is None:
            from cognee.modules.engine.models.node_set import NodeSet

            kwargs["node_type"] = NodeSet

        # The default memify tasks never read the projected graph: they stream
        # triplets straight from the graph DB (or no-op). Pass the non-empty
        # sentinel the other improve stages already use so memify skips the
        # full-graph projection. Custom tasks/data keep the projection, since a
        # caller-supplied task may consume it.
        if not inputs.has_custom_memify_tasks:
            kwargs["data"] = [{}]

        run_result = await memify(
            dataset=inputs.dataset_id,
            node_name=inputs.node_name,
            user=inputs.user,
            run_in_background=False,
            **kwargs,
        )
        result = StageResult.from_pipeline_run(self.name, run_result)
        if full_scope and result.status in ("completed", "already_completed"):
            result._run_info_stamp = enrichment_watermark_stamp(result.status, started_at)
        return result


class GlobalContextIndexStage(BaseStage):
    """Stage 9: build retrieval-ready bucket and root summaries (opt-in)."""

    name = "global_context_index"

    def gate(self, inputs: ImproveRunInputs) -> str | None:
        if not inputs.build_global_context_index:
            return REASON_OPT_IN_DISABLED
        return None

    async def run(self, inputs: ImproveRunInputs) -> StageResult:
        from cognee.memify_pipelines.global_context_index import global_context_index_pipeline

        result = await global_context_index_pipeline(
            user=inputs.user,
            dataset=inputs.dataset_id,
            run_in_background=False,
            bucketing_strategy="graph",
            max_bucket_size=4,
        )
        logger.info("improve: global context index updated")
        return StageResult.from_pipeline_run(self.name, result)
