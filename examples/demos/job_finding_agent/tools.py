"""Tool implementations for the 3-tool job-finding flow."""

from __future__ import annotations

import difflib
import re

import cognee
from cognee import SearchType
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from pydantic import BaseModel, Field

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
    SkillStateSnapshot,
    build_node_id_from_text,
    persist_with_low_level_pipeline,
)
from examples.demos.job_finding_agent.skill_logic import update_skill_with_feedback

PROCESS_PIPELINE_NAME = "job_find_process_pipeline"
ACTION_PIPELINE_NAME = "job_find_agent_action_pipeline"


class ResearchStepDecision(BaseModel):
    """One retrieval-planning step before recommendation."""

    thought: str = Field(min_length=1)
    query: str = Field(min_length=1)
    continue_research: bool = True


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


def _stringify_search_result(search_result: object) -> str:
    """Normalize Cognee search output into compact text."""
    if search_result is None:
        return "No results."
    if isinstance(search_result, str):
        return search_result.strip() or "No results."
    if isinstance(search_result, list):
        chunks: list[str] = []
        for item in search_result[:5]:
            if isinstance(item, str):
                chunks.append(item.strip())
            elif hasattr(item, "model_dump_json"):
                chunks.append(item.model_dump_json(indent=2))
            else:
                chunks.append(str(item))
        joined = "\n".join(part for part in chunks if part)
        return joined.strip() or "No results."
    return str(search_result)


def _get_or_create_action_job_node(
    state: JobAgentState,
    context: RunnerContext,
) -> ActionTaskJobNode:
    """Get or create the action task job node for current job."""
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
    return action_job_node


def _build_skill_change_text(
    previous_skill: str,
    current_skill: str,
    job_id: str,
    version: int,
    feedbacks: list[str],
) -> str:
    """Build compact dynamic change-only summary between two skill states."""
    prev_lines = previous_skill.splitlines()
    curr_lines = current_skill.splitlines()
    diff_lines = list(
        difflib.unified_diff(
            prev_lines,
            curr_lines,
            fromfile="previous_skill.md",
            tofile="current_skill.md",
            lineterm="",
            n=0,
        )
    )
    change_lines = [
        line
        for line in diff_lines
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith("+++")
        and not line.startswith("---")
    ]
    diff_preview = "\n".join(change_lines[:120]) if change_lines else "No textual changes."
    feedback_block = (
        "\n".join(f"- {item}" for item in feedbacks if item.strip()) if feedbacks else "- (none)"
    )
    return (
        f"Skill snapshot version {version} for {job_id}\n"
        f"Dynamic changes only:\n{diff_preview}\n\n"
        f"Feedback used for this update:\n{feedback_block}"
    ).strip()


async def _run_pre_decision_research(
    state: JobAgentState,
    context: RunnerContext,
    role_title: str,
    max_steps: int = 1,
) -> str:
    """Run a tiny research loop over Cognee memory before final decision."""
    notes: list[str] = []
    for step in range(1, max_steps + 1):
        decision = await LLMGateway.acreate_structured_output(
            (
                f"Skill:\n{context.skill_text}\n\n"
                f"Current job id: {state.job.job_id}\n"
                f"Current role title: {role_title}\n"
                f"Current job description:\n{state.job.job_description}\n\n"
                "Previous research notes:\n"
                + ("\n".join(notes) if notes else "(none)")
            ),
            (
                "You are planning one memory-research step before job recommendation.\n"
                "Create a high-signal retrieval query for graph-completion search that finds:\n"
                "1) candidate preferences learned from feedback,\n"
                "2) similar accepted/rejected job patterns,\n"
                "3) constraints (remote/onsite, role scope, seniority, domain fit).\n"
                "Avoid vague or generic queries. Avoid repeating previous queries. "
                "Prefer concrete phrases from the current role title and responsibilities. "
                "Set continue_research=false only if evidence is already sufficient for a decision."
            ),
            ResearchStepDecision,
        )

        search_result = await cognee.search(
            query_text=decision.query,
            query_type=SearchType.GRAPH_COMPLETION,
            datasets=[context.dataset_name],
            user=context.user,
            top_k=5,
            system_prompt=(
                "You are a job-fit memory researcher. Return concise research notes only.\n"
                "Use retrieved graph memory to extract:\n"
                "- stable candidate preferences\n"
                "- prior accept/reject patterns\n"
                "- explicit constraints (remote/onsite, seniority, scope, domain)\n"
                "Do not answer as a final recommender. Do not invent facts. "
                "If evidence is weak, say so."
            ),
        )
        observation = _stringify_search_result(search_result)
        notes.append(
            (
                f"[Research step {step}] Thought: {decision.thought}\n"
                f"Query: {decision.query}\n"
                f"Notes:\n{observation}\n"
            )
        )

        if not decision.continue_research:
            break

    return "\n".join(notes).strip() or "No memory evidence collected."


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

    memory_research_context = await _run_pre_decision_research(
        state=state,
        context=context,
        role_title=formatted.role_title,
    )
    state.metadata["memory_research_context"] = memory_research_context
    state.formatted_job = formatted.model_dump()

    recommendation: RecommendationOutput = await LLMGateway.acreate_structured_output(
        (
            f"skill.md:\n{context.skill_text}\n\n"
            f"memory_research_context:\n{memory_research_context}\n\n"
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
    """Update skill text from pending feedback and persist snapshot."""
    if not context.runtime_data.get("pending_feedbacks"):
        return ToolExecutionResult(
            observation="No pending feedback available for skill update.",
            continue_loop=True,
        )

    previous_skill_text = context.skill_text
    pending_feedbacks = list(context.runtime_data.get("pending_feedbacks", []))
    updated_skill = await update_skill_with_feedback(context.skill_text, pending_feedbacks)
    context.skill_text = updated_skill
    context.skill_md_path.write_text(updated_skill, encoding="utf-8")
    context.runtime_data["pending_feedbacks"] = []

    action_job_node = _get_or_create_action_job_node(state=state, context=context)
    previous_snapshot = context.runtime_data.get("last_skill_snapshot")
    snapshot_version = int(context.runtime_data.get("skill_snapshot_version", 0)) + 1
    snapshot_text = _build_skill_change_text(
        previous_skill=previous_skill_text,
        current_skill=updated_skill,
        job_id=state.job.job_id,
        version=snapshot_version,
        feedbacks=pending_feedbacks,
    )
    skill_snapshot = SkillStateSnapshot(
        id=build_node_id_from_text(snapshot_text),
        action_job_node=action_job_node,
        job_id=state.job.job_id,
        version=snapshot_version,
        feedbacks=[item for item in pending_feedbacks if item.strip()],
        previous_snapshot=previous_snapshot,
        text=snapshot_text,
    )
    await persist_with_low_level_pipeline(
        data_points=[skill_snapshot],
        dataset_name=context.runtime_data.get("action_dataset_name", context.dataset_name),
        user=context.user,
        pipeline_name=ACTION_PIPELINE_NAME,
    )
    context.runtime_data["last_skill_snapshot"] = skill_snapshot
    context.runtime_data["skill_snapshot_version"] = snapshot_version

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
    action_job_node = _get_or_create_action_job_node(state=state, context=context)

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
