"""Internal proposal-first skill improvement used by remember()."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import BaseModel, Field

from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.engine.models import NodeSet, Skill, SkillImprovementProposal, SkillRun
from cognee.modules.engine.utils.generate_node_id import generate_node_id
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.tools.resolve_skills import find_skill_by_id, find_skill_by_name
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage.add_data_points import add_data_points


logger = get_logger("cognee.skill_improvement")


class SkillImprovementDraft(BaseModel):
    proposed_procedure: str = Field(default="")
    rationale: str = Field(default="")
    confidence: float = Field(default=0.0)


def _dataset_scope(dataset) -> list[str]:
    dataset_id = getattr(dataset, "id", None)
    return [str(dataset_id)] if dataset_id is not None else []


def _skills_node_set() -> NodeSet:
    return NodeSet(id=generate_node_id("NodeSet:skills"), name="skills")


def _storage_context(user, dataset, key: str) -> Optional[PipelineContext]:
    dataset_id = getattr(dataset, "id", None)
    if user is None or dataset is None or dataset_id is None:
        return None
    return PipelineContext(
        user=user,
        dataset=dataset,
        data_item=SimpleNamespace(id=uuid5(NAMESPACE_URL, f"cognee:skill-improvement:{key}")),
        pipeline_name="skill_improvement_pipeline",
    )


def _format_skill_procedure(skill_name: str, procedure: str) -> str:
    procedure = (procedure or "").strip()
    if procedure.startswith("#"):
        return procedure
    return f"# {skill_name}\n\n{procedure}".strip()


async def improve_skill_from_config(
    config: dict[str, Any],
    *,
    dataset,
    user=None,
) -> Optional[SkillImprovementProposal]:
    """Run the internal skill-improvement operation requested by remember()."""
    if not isinstance(config, dict):
        raise ValueError("skill_improvement must be a configuration dictionary.")

    skill_name = config.get("skill_name") or config.get("name")
    proposal_id = config.get("proposal_id")
    apply = bool(config.get("apply", False))

    if not skill_name:
        raise ValueError("skill_improvement requires 'skill_name'.")

    return await improve_skill(
        skill_name,
        dataset=dataset,
        user=user,
        proposal_id=proposal_id,
        apply=apply,
        score_threshold=float(config.get("score_threshold", 0.5)),
        max_runs=int(config.get("max_runs", 5)),
    )


async def improve_skill(
    skill_name: str,
    *,
    dataset,
    user=None,
    proposal_id: Optional[str] = None,
    apply: bool = False,
    score_threshold: float = 0.5,
    max_runs: int = 5,
) -> Optional[SkillImprovementProposal]:
    """Create or apply a graph-only SkillImprovementProposal.

    This is intentionally internal. Callers opt in through ``cognee.remember``
    with ``skill_improvement={...}``; no top-level public API is exposed.
    """
    dataset_id = getattr(dataset, "id", None)
    if dataset_id is None:
        raise ValueError("Skill improvement requires one explicit dataset.")
    if apply and not proposal_id:
        raise ValueError("skill_improvement apply=True requires proposal_id.")

    owner_id = getattr(dataset, "owner_id", None) or getattr(user, "id", None)
    if owner_id is None:
        raise ValueError("Skill improvement requires a dataset owner or user.")

    async with set_database_global_context_variables(dataset_id, owner_id):
        if apply:
            return await _apply_proposal(
                proposal_id=proposal_id,
                skill_name=skill_name,
                dataset_id=dataset_id,
                dataset=dataset,
                user=user,
            )

        skill = await find_skill_by_name(skill_name, dataset_id=dataset_id)
        if skill is None:
            raise ValueError(f"Skill {skill_name!r} was not found in dataset {dataset.name!r}.")

        runs = await _find_recent_failure_runs(
            dataset_id=dataset_id,
            skill_id=str(skill.id),
            skill_name=skill.name,
            score_threshold=score_threshold,
            max_runs=max_runs,
        )
        if not runs:
            logger.info("No low-scoring or errored SkillRun records for %s", skill.name)
            return None

        draft = await _generate_proposal(skill, runs)
        try:
            from cognee.infrastructure.llm import get_llm_config

            model_name = get_llm_config().llm_model
        except Exception:
            model_name = ""
        proposal = SkillImprovementProposal(
            proposal_id=str(uuid4()),
            skill_id=str(skill.id),
            skill_name=skill.name,
            skill=skill,
            dataset_scope=_dataset_scope(dataset),
            old_procedure=skill.procedure,
            proposed_procedure=_format_skill_procedure(skill.name, draft.proposed_procedure),
            runs_used=[run.run_id for run in runs],
            runs=runs,
            model_name=model_name,
            confidence=draft.confidence,
            rationale=draft.rationale,
            status="proposed",
            belongs_to_set=[_skills_node_set()],
        )
        await add_data_points([proposal], ctx=_storage_context(user, dataset, proposal.proposal_id))
        return proposal


async def _apply_proposal(
    *,
    proposal_id: str,
    skill_name: str,
    dataset_id: UUID,
    dataset,
    user=None,
) -> SkillImprovementProposal:
    proposal = await _find_proposal(proposal_id=proposal_id, dataset_id=dataset_id)
    if proposal is None:
        raise ValueError(f"Proposal {proposal_id!r} was not found in dataset {dataset.name!r}.")
    if proposal.skill_name != skill_name:
        raise ValueError("Proposal does not target the requested skill.")

    skill = await find_skill_by_id(proposal.skill_id, dataset_id=dataset_id)
    if skill is None:
        skill = await find_skill_by_name(proposal.skill_name, dataset_id=dataset_id)
    if skill is None:
        raise ValueError(
            f"Skill {proposal.skill_name!r} was not found in dataset {dataset.name!r}."
        )

    skill.procedure = _format_skill_procedure(skill.name, proposal.proposed_procedure)
    skill.skill_text = "\n\n".join(
        part for part in (skill.name, skill.description, skill.procedure) if part
    )
    skill.search_text = skill.skill_text
    skill.belongs_to_set = [_skills_node_set()]
    proposal.status = "applied"
    proposal.belongs_to_set = [_skills_node_set()]

    await add_data_points(
        [skill, proposal],
        ctx=_storage_context(user, dataset, f"{proposal.proposal_id}:apply"),
    )
    return proposal


async def _generate_proposal(skill: Skill, runs: list[SkillRun]) -> SkillImprovementDraft:
    from cognee.modules.retrieval.utils.completion import generate_completion

    run_context = "\n\n".join(
        f"- run_id={run.run_id}; score={run.success_score}; "
        f"error={run.error_type or run.error_message or 'none'}; result={run.result_summary}"
        for run in runs
    )
    context = (
        f"# Skill\nName: {skill.name}\nDescription: {skill.description}\n\n"
        f"# Current Procedure\n{skill.procedure}\n\n# Failure Evidence\n{run_context}"
    )
    return await generate_completion(
        query=(
            "Propose a revised skill procedure. Return proposed_procedure as a complete "
            f"SKILL.md body that starts with '# {skill.name}'. Write direct instructions "
            "for the agent to follow, not prose about what the skill should do. "
            "Do not mutate state."
        ),
        context=context,
        user_prompt_path="context_for_question.txt",
        system_prompt_path="answer_simple_question.txt",
        response_model=SkillImprovementDraft,
    )


async def _find_recent_failure_runs(
    *,
    dataset_id: UUID,
    skill_id: str,
    skill_name: str,
    score_threshold: float,
    max_runs: int,
) -> list[SkillRun]:
    runs: list[SkillRun] = []
    for raw in await _load_nodes_by_type(SkillRun):
        run = _coerce_model(raw, SkillRun)
        if run is None:
            continue
        if str(dataset_id) not in (run.dataset_scope or []):
            continue
        if (
            run.selected_skill_id not in (skill_id, skill_name)
            and run.selected_skill_name != skill_name
        ):
            continue
        is_error = bool(run.error_type or run.error_message)
        is_low_score = run.success_score < score_threshold
        if is_error or is_low_score:
            runs.append(run)
    return sorted(runs, key=lambda run: run.started_at_ms, reverse=True)[:max_runs]


async def _find_proposal(
    *,
    proposal_id: str,
    dataset_id: UUID,
) -> Optional[SkillImprovementProposal]:
    for raw in await _load_nodes_by_type(SkillImprovementProposal):
        proposal = _coerce_model(raw, SkillImprovementProposal)
        if proposal is None:
            continue
        if proposal.proposal_id == proposal_id and str(dataset_id) in (
            proposal.dataset_scope or []
        ):
            return proposal
    return None


async def _load_nodes_by_type(model):
    try:
        from cognee.infrastructure.databases.graph import get_graph_engine
    except Exception:
        return []

    try:
        graph_engine = await get_graph_engine()
    except Exception:
        return []

    get_by_type = getattr(graph_engine, "get_nodes_by_type", None)
    if get_by_type is not None:
        try:
            return await get_by_type(node_type=model)
        except Exception as exc:
            logger.warning("Skill improvement lookup failed: %s", exc)
            return []

    get_nodeset = getattr(graph_engine, "get_nodeset_subgraph", None)
    if get_nodeset is not None:
        try:
            nodes, _ = await get_nodeset(node_type=model, node_name=["skills"])
            if nodes:
                return nodes
        except Exception as exc:
            logger.warning("Skill improvement nodeset lookup failed: %s", exc)

    get_graph_data = getattr(graph_engine, "get_graph_data", None)
    if get_graph_data is None:
        return []
    try:
        nodes, _ = await get_graph_data()
        return nodes
    except Exception as exc:
        logger.warning("Skill improvement full graph lookup failed: %s", exc)
        return []


def _coerce_model(raw, model):
    if isinstance(raw, model):
        return raw
    if isinstance(raw, (list, tuple)) and len(raw) > 1:
        raw = raw[1]
    data = raw.model_dump() if hasattr(raw, "model_dump") else raw
    if not isinstance(data, dict):
        return None
    data = {k: v for k, v in data.items() if k != "metadata"}
    try:
        return model.model_validate(data)
    except Exception:
        return None
