"""Record SkillRun feedback through the remember() typed-entry path."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional
from uuid import NAMESPACE_URL, UUID, uuid5

from cognee.memory.entries import SkillRunEntry
from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.engine.models import (
    CandidateSkill,
    NodeSet,
    SkillRun,
    ToolCall,
    UNSCORED_SKILL_RUN_SCORE,
)
from cognee.modules.engine.operations.setup import setup
from cognee.modules.engine.utils.generate_node_id import generate_node_id
from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
    resolve_authorized_user_datasets,
)
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.tools.resolve_skills import resolve_skills
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage.add_data_points import add_data_points


logger = get_logger("cognee.tools.skill_runs")


def _skill_run_data_id(dataset_id: UUID, run_id: str) -> UUID:
    """Stable pseudo data id used to attach SkillRun writes to dataset ACL tables."""
    return uuid5(NAMESPACE_URL, f"cognee:skill-runs:{dataset_id}:{run_id}")


def _make_storage_context(user, dataset, run_id: str) -> PipelineContext:
    return PipelineContext(
        user=user,
        dataset=dataset,
        data_item=SimpleNamespace(id=_skill_run_data_id(dataset.id, run_id)),
        pipeline_name="skill_runs_pipeline",
    )


def _coerce_tool_trace(raw_trace: list[dict]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index, item in enumerate(raw_trace):
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict tool_trace item at index %d", index)
            continue
        calls.append(ToolCall(**item))
    return calls


def _candidate_skill_for_id(skill_id: str, selected_skill) -> CandidateSkill:
    is_selected = str(skill_id) == str(selected_skill.id)
    if not is_selected:
        return CandidateSkill(skill_id=str(skill_id))

    return CandidateSkill(
        skill_id=str(selected_skill.id),
        skill_name=selected_skill.name,
        skill_description=selected_skill.description,
        skill_text=selected_skill.skill_text or selected_skill.search_text,
    )


async def remember_skill_run_entry(
    entry: SkillRunEntry,
    *,
    dataset_name: str,
    session_id: Optional[str],
    user=None,
) -> tuple[SkillRun, object]:
    """Persist a dataset-scoped SkillRun from the remember() typed-entry path."""
    await setup()

    if user is None:
        from cognee.modules.users.methods import get_default_user

        user = await get_default_user()

    user, authorized_datasets = await resolve_authorized_user_datasets(dataset_name, user)
    dataset = authorized_datasets[0]

    owner_id = getattr(dataset, "owner_id", None) or getattr(user, "id", None)
    if owner_id is None:
        raise ValueError("SkillRun persistence requires a dataset owner or user.")

    async with set_database_global_context_variables(dataset.id, owner_id):
        resolved_skills = await resolve_skills([entry.selected_skill_id], dataset_id=dataset.id)
        if not resolved_skills:
            raise ValueError(
                f"Skill '{entry.selected_skill_id}' was not found or is not visible "
                f"in dataset '{dataset.name}'"
            )

        selected_skill = resolved_skills[0]
        candidate_ids = entry.candidate_skill_ids or [str(selected_skill.id)]
        success_score = (
            UNSCORED_SKILL_RUN_SCORE if entry.success_score is None else entry.success_score
        )

        run = SkillRun(
            run_id=entry.run_id,
            selected_skill_id=str(selected_skill.id),
            selected_skill_name=selected_skill.name,
            selected_skill=selected_skill,
            dataset_scope=[str(dataset.id)],
            task_text=entry.task_text,
            result_summary=entry.result_summary,
            success_score=success_score,
            session_id=session_id or "agentic",
            candidate_skills=[
                _candidate_skill_for_id(skill_id, selected_skill) for skill_id in candidate_ids
            ],
            task_pattern_id=entry.task_pattern_id,
            router_version=entry.router_version,
            tool_trace=_coerce_tool_trace(entry.tool_trace),
            error_type=entry.error_type,
            error_message=entry.error_message,
            started_at_ms=entry.started_at_ms,
            latency_ms=entry.latency_ms,
            feedback=entry.feedback,
        )
        run.belongs_to_set = [
            NodeSet(id=generate_node_id(f"NodeSet:{entry.node_set}"), name=entry.node_set)
        ]

        await add_data_points([run], ctx=_make_storage_context(user, dataset, entry.run_id))

    return run, dataset
