"""Tool implementations for the 3-tool job-finding flow."""

from __future__ import annotations

import cognee
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.memify_pipelines.apply_feedback_weights import apply_feedback_weights_pipeline

from .agent.agent_models import (
    FormattedJobOutput,
    RecommendationDecision,
    RecommendationOutput,
    ToolName,
)
from .agent.agent_state import JobAgentState
from .agent.tool_contracts import RunnerContext, ToolExecutionResult
from .memory_models import (
    AgentAction,
    FeedbackRecord,
    FormattedJob,
    JobRecommendation,
    persist_data_points,
)
from .skill_logic import update_skill_with_feedback


async def process_job_agent_tool(
    state: JobAgentState,
    context: RunnerContext,
) -> ToolExecutionResult:
    """Process one job into structured format + recommendation and persist it."""
    formatted = await LLMGateway.acreate_structured_output(
        state.job.job_description,
        (
            "Extract structured job information. Keep values concise and literal. "
            "If unknown, return empty strings/lists."
        ),
        FormattedJobOutput,
    )
    state.formatted_job = formatted.model_dump()

    recommendation = await LLMGateway.acreate_structured_output(
        (
            f"skill.md:\n{context.skill_text}\n\n"
            f"job_description:\n{state.job.job_description}\n\n"
            f"formatted_job:\n{formatted.model_dump_json(indent=2)}"
        ),
        (
            "Decide if the candidate should APPLY or DONT_APPLY. "
            "Provide short rationale and confidence between 0 and 1."
        ),
        RecommendationOutput,
    )
    state.recommendation = recommendation

    formatted_dp = FormattedJob(
        job_id=state.job.job_id,
        raw_text=state.job.job_description,
        **state.formatted_job,
    )
    recommendation_dp = JobRecommendation(
        job_id=state.job.job_id,
        decision=recommendation.decision.value,
        rationale=recommendation.rationale,
        confidence=recommendation.confidence,
    )
    await persist_data_points(
        data_points=[formatted_dp, recommendation_dp],
        dataset_name=context.dataset_name,
        user=context.user,
        pipeline_name="job_find_process_pipeline",
    )

    return ToolExecutionResult(
        observation=f"Recommendation: {recommendation.decision.value}",
        continue_loop=True,
    )


async def update_process_job_agent_skill_tool(
    state: JobAgentState,
    context: RunnerContext,
) -> ToolExecutionResult:
    """Apply feedback weights then update skill text."""
    if not context.runtime_data.get("pending_feedbacks"):
        return ToolExecutionResult(
            observation="No pending feedback available for skill update.",
            continue_loop=True,
        )

    await apply_feedback_weights_pipeline(
        user=context.user,
        session_ids=[context.session_id],
        dataset=context.dataset_name,
        alpha=0.1,
        batch_size=100,
        run_in_background=False,
    )

    last_feedback = context.runtime_data["pending_feedbacks"][-1]
    updated_skill = update_skill_with_feedback(context.skill_text, last_feedback)
    context.skill_text = updated_skill
    context.skill_md_path.write_text(updated_skill, encoding="utf-8")
    context.runtime_data["pending_feedbacks"] = []

    return ToolExecutionResult(
        observation="skill.md updated from latest feedback.",
        continue_loop=True,
    )


async def request_feedback_tool(
    state: JobAgentState,
    context: RunnerContext,
) -> ToolExecutionResult:
    """Choose mocked feedback branch and end process for this job."""
    if not state.recommendation:
        return ToolExecutionResult(
            observation="Cannot request feedback without recommendation.",
            should_end_process=True,
            continue_loop=False,
            stop_reason="MISSING_RECOMMENDATION",
        )

    decision = state.recommendation.decision
    if decision == RecommendationDecision.APPLY:
        feedback_text = state.job.feedback_if_recommended
    else:
        feedback_text = state.job.feedback_if_not_recommended

    state.feedback_text = feedback_text
    context.runtime_data.setdefault("pending_feedbacks", []).append(feedback_text)

    existing = await cognee.session.get_session(
        session_id=context.session_id,
        user=context.user,
        last_n=1,
    )
    if existing:
        latest = existing[-1]
        qa_id = getattr(latest, "qa_id", None)
        if qa_id:
            score = 5 if decision == RecommendationDecision.APPLY else 2
            await cognee.session.add_feedback(
                session_id=context.session_id,
                qa_id=qa_id,
                feedback_text=feedback_text,
                feedback_score=score,
                user=context.user,
            )

    feedback_dp = FeedbackRecord(
        job_id=state.job.job_id,
        decision=decision.value,
        feedback_text=feedback_text,
    )
    await persist_data_points(
        data_points=[feedback_dp],
        dataset_name=context.dataset_name,
        user=context.user,
        pipeline_name="job_find_feedback_pipeline",
    )

    return ToolExecutionResult(
        observation=f"Feedback captured for {decision.value}.",
        should_end_process=True,
        continue_loop=False,
        stop_reason="REQUEST_FEEDBACK_COMPLETED",
    )


async def store_agent_action(
    state: JobAgentState,
    context: RunnerContext,
    thought: str,
    tool_name: ToolName,
    observation: str,
    stop_reason: str | None,
) -> None:
    """Persist one loop action trace element."""
    action_dp = AgentAction(
        job_id=state.job.job_id,
        iteration=state.iteration,
        thought=thought,
        tool_name=tool_name.value,
        observation=observation,
        stop_reason=stop_reason or "",
    )
    await persist_data_points(
        data_points=[action_dp],
        dataset_name=context.dataset_name,
        user=context.user,
        pipeline_name="job_find_agent_action_pipeline",
    )
