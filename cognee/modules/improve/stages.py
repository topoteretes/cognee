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

from typing import Any, Dict, List, Optional

from cognee.shared.logging_utils import get_logger

from .constants import (
    AGENT_TRACE_FEEDBACKS_NODE_SET,
    SESSION_LEARNINGS_NODE_SET,
    USER_PREFERENCES_NODE_SET,
    USER_SESSIONS_NODE_SET,
)
from .inputs import ImproveRunInputs
from .result import REASON_BACKEND_UNSUPPORTED, StageResult
from .stage import BaseStage

logger = get_logger("improve")

# Stage-specific gate reasons.
REASON_FEEDBACK_INFLUENCE_ZERO = "feedback_influence_zero"
REASON_SESSION_MANAGER_UNAVAILABLE = "session_manager_unavailable"
REASON_AUTO_FEEDBACK_DISABLED = "auto_feedback_disabled"
REASON_PERSONALIZATION_DISABLED = "personalization_disabled"
REASON_OPT_IN_DISABLED = "opt_in_disabled"
REASON_TRIPLET_EMBEDDING_DISABLED = "triplet_embedding_disabled"
REASON_NO_WRITES_SINCE_LAST_IMPROVE = "no_writes_since_last_improve"


class FeedbackWeightsStage(BaseStage):
    """Stage 1: move ``feedback_weight`` on the graph elements scored answers used."""

    name = "feedback_weights"
    kind = "session"
    pipeline_name = "memify_pipeline"
    label = "feedback weighting"
    summary = "Re-weights used nodes/edges from session feedback (feedback_weight)."
    effects = [
        {"effect": "modifies", "target_type": "Entity", "property": "feedback_weight"},
        {"effect": "modifies", "target_type": "EntityType", "property": "feedback_weight"},
    ]

    def gate(self, inputs: ImproveRunInputs) -> Optional[str]:
        from cognee.base_config import get_base_config

        if get_base_config().default_feedback_influence <= 0:
            return REASON_FEEDBACK_INFLUENCE_ZERO
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
    so an error here stops the chain instead of being swallowed.
    """

    name = "persist_session_qa"
    kind = "session"
    fatal = True
    pipeline_name = "memify_pipeline"
    label = "persist sessions"
    summary = "Cognifies cached user Q&A sessions into the graph."
    effects = [
        {"effect": "produces", "target_type": "Session", "target_node_set": USER_SESSIONS_NODE_SET},
        {"effect": "produces", "target_type": "Entity", "target_node_set": USER_SESSIONS_NODE_SET},
    ]

    async def run(self, inputs: ImproveRunInputs) -> StageResult:
        from cognee.memify_pipelines.persist_sessions_in_knowledge_graph import (
            persist_sessions_in_knowledge_graph_pipeline,
        )

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
    kind = "session"
    pipeline_name = "memify_pipeline"
    label = "persist agent traces"
    summary = "Cognifies agent trace feedback into the graph."
    effects = [
        {
            "effect": "produces",
            "target_type": "Entity",
            "target_node_set": AGENT_TRACE_FEEDBACKS_NODE_SET,
        },
    ]

    async def run(self, inputs: ImproveRunInputs) -> StageResult:
        from cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph import (
            persist_agent_trace_feedbacks_in_knowledge_graph_pipeline,
        )

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
    kind = "session"
    after = ("persist_agent_traces",)
    label = "extract agent context"
    summary = "Turns pending tool-call traces into agent-profile lessons (session context)."
    effects: List[Dict[str, Any]] = []

    def gate(self, inputs: ImproveRunInputs) -> Optional[str]:
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
        last_error: Optional[BaseException] = None
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
    kind = "session"
    after = ("extract_agent_context",)
    pipeline_name = "cognify_pipeline"
    label = "distill sessions"
    summary = "Curates gated session guidance into entity-anchored lessons."
    effects = [
        {
            "effect": "produces",
            "target_type": "Entity",
            "target_node_set": SESSION_LEARNINGS_NODE_SET,
        },
    ]

    async def run(self, inputs: ImproveRunInputs) -> StageResult:
        from cognee.modules.session_distillation import distill_session

        distilled = 0
        completed = 0
        failed = 0
        last_error: Optional[BaseException] = None
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
    kind = "session"
    label = "user preferences"
    summary = "Folds ratings and stated preferences into per-user prefers weights."
    effects = [
        {
            "effect": "produces",
            "target_type": "UserPreference",
            "target_node_set": USER_PREFERENCES_NODE_SET,
        },
    ]

    def gate(self, inputs: ImproveRunInputs) -> Optional[str]:
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
    kind = "session"
    after = ("distill_sessions",)
    label = "truth subspace"
    summary = "Scores chunks against accepted lessons (truth_alignment coordinates)."
    effects = [
        {"effect": "modifies", "target_type": "DocumentChunk", "property": "truth_alignment"},
    ]

    def gate(self, inputs: ImproveRunInputs) -> Optional[str]:
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
        return StageResult.completed(self.name, **counts)


class TripletEnrichmentStage(BaseStage):
    """Stage 8: default memify enrichment (triplet embeddings).

    Skipped when ``triplet_embedding`` is off and the caller supplied no tasks
    of their own (the default task list would be empty). ``already_completed``
    when no write pipeline has completed for this dataset since its last
    improve, read from ``pipeline_runs``.
    """

    name = "triplet_enrichment"
    kind = "graph"
    after = ("build_truth_subspace",)
    pipeline_name = "memify_pipeline"
    label = "memify (triplets)"
    summary = "Default enrichment: builds triplet embeddings over the graph."
    effects = [
        {"effect": "enriches", "target_type": "Entity"},
    ]

    def gate(self, inputs: ImproveRunInputs) -> Optional[str]:
        if inputs.has_custom_memify_tasks:
            return None
        from cognee.modules.cognify.config import get_cognify_config

        if not get_cognify_config().triplet_embedding:
            return REASON_TRIPLET_EMBEDDING_DISABLED
        return None

    async def run(self, inputs: ImproveRunInputs) -> StageResult:
        from cognee.modules.memify import memify

        from .graph_changes import has_graph_changed_since_last_improve

        if not inputs.has_custom_memify_tasks:
            if not await has_graph_changed_since_last_improve(inputs.dataset_id):
                result = StageResult(
                    stage=self.name,
                    status="already_completed",
                    reason=REASON_NO_WRITES_SINCE_LAST_IMPROVE,
                )
                result._raw_run = {}
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

        result = await memify(
            dataset=inputs.dataset_id,
            node_name=inputs.node_name,
            user=inputs.user,
            run_in_background=False,
            **kwargs,
        )
        return StageResult.from_pipeline_run(self.name, result)


class GlobalContextIndexStage(BaseStage):
    """Stage 9: build retrieval-ready bucket and root summaries (opt-in)."""

    name = "global_context_index"
    kind = "graph"
    after = ("triplet_enrichment",)
    pipeline_name = "memify_pipeline"
    label = "global context index"
    summary = "Builds hierarchical context summaries for retrieval."
    effects = [
        {"effect": "produces", "target_type": "GlobalContextSummary"},
        {"effect": "enriches", "target_type": "TextSummary"},
    ]

    def gate(self, inputs: ImproveRunInputs) -> Optional[str]:
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
