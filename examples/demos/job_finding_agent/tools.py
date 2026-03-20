"""Tool implementations for the 3-tool job-finding flow."""

from __future__ import annotations

import re

import cognee
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.memify_pipelines.apply_feedback_weights import apply_feedback_weights_pipeline

from examples.demos.job_finding_agent.agent.agent_models import (
    FormattedJobOutput,
    RecommendationDecision,
    RecommendationOutput,
    ToolName,
)
from examples.demos.job_finding_agent.agent.agent_state import JobAgentState
from examples.demos.job_finding_agent.agent.tool_contracts import (
    RunnerContext,
    ToolExecutionResult,
)
from examples.demos.job_finding_agent.memory_models import (
    ActionTaskJobNode,
    ActionChainRoot,
    AgentAction,
    JobSequenceNode,
    build_node_id_from_text,
    persist_with_low_level_pipeline,
)
from examples.demos.job_finding_agent.skill_logic import update_skill_with_feedback

PROCESS_PIPELINE_NAME = "job_find_process_pipeline"
ACTION_PIPELINE_NAME = "job_find_agent_action_pipeline"


def _fallback_role_title(job_description: str) -> str:
    """Best-effort local fallback so role_title is always present."""
    lines = [line.strip(" -*\t") for line in job_description.splitlines() if line.strip()]
    for line in lines[:12]:
        lower = line.lower()
        if lower.startswith("title:") or lower.startswith("role:"):
            return line.split(":", 1)[1].strip() or "Unspecified Role"

    title_keywords = (
        "engineer",
        "developer",
        "scientist",
        "manager",
        "architect",
        "specialist",
        "analyst",
        "lead",
        "director",
    )
    for line in lines[:12]:
        if any(keyword in line.lower() for keyword in title_keywords):
            return re.sub(r"\s+", " ", line).strip()[:120]

    return "Unspecified Role"


async def process_job_agent_tool(
    state: JobAgentState,
    context: RunnerContext,
) -> ToolExecutionResult:
    """Process one job into structured format + recommendation and persist it."""
    job_sequence_index = int(state.metadata.get("job_sequence_index", 0))
    sequence_text = f"Job{job_sequence_index}"
    previous_job_sequence_node = context.runtime_data.get("last_job_sequence_node")
    job_sequence_node = JobSequenceNode(
        id=build_node_id_from_text(sequence_text),
        position=job_sequence_index,
        previous=previous_job_sequence_node,
        text=sequence_text,
    )
    context.runtime_data["last_job_sequence_node"] = job_sequence_node

    formatted: FormattedJobOutput = await LLMGateway.acreate_structured_output(
        state.job.job_description,
        (
            "Extract a concise normalized job summary. "
            "role_title is mandatory and must never be empty. "
            "Put the retrieval-ready summary in the `text` field."
        ),
        FormattedJobOutput,
    )
    if not formatted.role_title.strip():
        formatted.role_title = _fallback_role_title(state.job.job_description)
    state.formatted_job = formatted.model_dump()

    recommendation: RecommendationOutput = await LLMGateway.acreate_structured_output(
        (
            f"skill.md:\n{context.skill_text}\n\n"
            f"job_description:\n{state.job.job_description}\n\n"
            f"formatted_job:\n{formatted.model_dump_json(indent=2)}"
        ),
        (
            "Decide if the candidate should APPLY or DONT_APPLY. "
            "Provide short rationale. Put a retrieval-ready summary in `text`."
        ),
        RecommendationOutput,
    )
    state.recommendation = recommendation

    if not formatted.text.strip():
        formatted.text = state.job.job_description.strip()
    formatted.id = build_node_id_from_text(formatted.text)

    if not recommendation.text.strip():
        recommendation.text = (
            f"Recommendation: {recommendation.decision.value}. "
            f"Rationale: {recommendation.rationale}"
        ).strip()
    if recommendation.text.strip() == formatted.text.strip():
        recommendation.text = (
            f"Recommendation: {recommendation.decision.value}. "
            f"Rationale: {recommendation.rationale}"
        ).strip()
    recommendation.id = build_node_id_from_text(recommendation.text)
    formatted.job_sequence = job_sequence_node
    # Keep explicit relation in the datapoint model
    formatted.recommendation = recommendation
    state.metadata["job_sequence_node"] = job_sequence_node
    await persist_with_low_level_pipeline(
        data_points=[formatted],
        dataset_name=context.dataset_name,
        user=context.user,
        pipeline_name=PROCESS_PIPELINE_NAME,
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

    pending_feedbacks = list(context.runtime_data.get("pending_feedbacks", []))
    updated_skill = await update_skill_with_feedback(context.skill_text, pending_feedbacks)
    context.skill_text = updated_skill
    context.skill_md_path.write_text(updated_skill, encoding="utf-8")
    context.runtime_data["pending_feedbacks"] = []

    return ToolExecutionResult(
        observation="skill.md updated from latest feedback.",
        should_end_process=True,
        continue_loop=False,
        stop_reason="SKILL_UPDATED_FOR_JOB",
    )


async def request_feedback_tool(
    state: JobAgentState,
    context: RunnerContext,
) -> ToolExecutionResult:
    """Choose feedback branch and end process for this job."""
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

    return ToolExecutionResult(
        observation=f"Feedback captured in session for {decision.value}.",
        should_end_process=False,
        continue_loop=True,
        stop_reason="FEEDBACK_CAPTURED",
    )


async def store_agent_action(
    state: JobAgentState,
    context: RunnerContext,
    iteration: int,
    thought: str,
    tool_name: ToolName,
    observation: str,
    stop_reason: str | None,
    prev_action: AgentAction | None = None,
) -> AgentAction:
    """Persist one loop action trace element."""
    action_text = (
        f"Job {state.job.job_id} | Iteration {iteration} | Tool {tool_name.value} | "
        f"Thought: {thought} | Observation: {observation} | Stop: {stop_reason or ''}"
    ).strip()
    chain_root_text = f"Agent action chain root for job {state.job.job_id}"
    action_job_node = state.metadata.get("action_job_node")
    if action_job_node is None:
        job_sequence_index = int(state.metadata.get("job_sequence_index", 0))
        action_sequence_text = f"Agent Task Job{job_sequence_index}"
        previous_action_job_node = context.runtime_data.get("last_action_task_job_node")
        action_job_node = ActionTaskJobNode(
            id=build_node_id_from_text(action_sequence_text),
            position=job_sequence_index,
            previous=previous_action_job_node,
            text=action_sequence_text,
        )
        context.runtime_data["last_action_task_job_node"] = action_job_node
        state.metadata["action_job_node"] = action_job_node

    chain_root_dp = ActionChainRoot(
        id=build_node_id_from_text(chain_root_text),
        action_job_node=action_job_node,
        job_id=state.job.job_id,
        text=chain_root_text,
    )
    action_dp = AgentAction(
        id=build_node_id_from_text(action_text),
        chain_root=chain_root_dp,
        iteration=iteration,
        prev_action=prev_action,
        text=action_text,
    )
    await persist_with_low_level_pipeline(
        data_points=[action_dp],
        dataset_name=context.runtime_data.get("action_dataset_name", context.dataset_name),
        user=context.user,
        pipeline_name=ACTION_PIPELINE_NAME,
    )
    return action_dp
