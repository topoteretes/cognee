"""List / fetch dataset-scoped Skill nodes for UI consumption.

Reads the knowledge graph scoped to one dataset and returns Skill nodes with
their publisher/provenance metadata. Uses the indexed ``get_nodes_by_type``
lookup (via ``resolve_skills``) and only falls back to a full graph scan when a
backend does not support it, so a Skills tab listing does not load the entire
graph per dataset.
"""

from typing import Any
from uuid import UUID

from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.engine.models import Skill
from cognee.modules.tools.resolve_skills import (
    _coerce_skill,
    _load_skill_nodes,
    find_skill_by_id,
)


def _skill_to_dict(skill: Skill, *, include_procedure: bool = False) -> dict[str, Any]:
    """Project a Skill model onto the UI skill shape."""
    data = {
        "id": str(skill.id),
        "name": skill.name or "",
        "description": skill.description or "",
        "maintainer": skill.maintainer or "",
        "maintainer_url": skill.maintainer_url or "",
        "version": skill.skill_version or "",
        "tags": list(skill.tags or []),
        "license": skill.license or "",
        "declared_tools": list(skill.declared_tools or []),
        "dataset_scope": list(skill.dataset_scope or []),
        "is_active": bool(skill.is_active),
        "source_repo_url": skill.source_repo_url or "",
        "source_dir": skill.source_dir or "",
    }
    if include_procedure:
        data["procedure"] = skill.procedure or ""
    return data


async def list_skills(
    dataset: str | UUID | None = None,
    include_inactive: bool = False,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return the Skill nodes for an authorized dataset.

    Parameters:
        dataset: dataset id/name used to scope the graph databases.
        include_inactive: when False (default) only ``is_active`` skills are returned.
        limit/offset: optional pagination over the name-sorted result.
    """
    if dataset is not None:
        owner_id = await _resolve_dataset_owner(dataset)
        if owner_id is not None:
            async with set_database_global_context_variables(dataset, owner_id):
                skills = await _collect_skills(dataset, include_inactive)
        else:
            skills = await _collect_skills(dataset, include_inactive)
    else:
        skills = await _collect_skills(dataset, include_inactive)

    if offset:
        skills = skills[offset:]
    if limit is not None:
        skills = skills[:limit]
    return skills


async def get_skill(skill_id: str, dataset: str | UUID) -> dict[str, Any] | None:
    """Return a single skill (including its full ``procedure``), or None."""
    if not isinstance(dataset, UUID):
        try:
            dataset = UUID(str(dataset))
        except (ValueError, TypeError):
            return None
    owner_id = await _resolve_dataset_owner(dataset)
    if owner_id is not None:
        async with set_database_global_context_variables(dataset, owner_id):
            skill = await find_skill_by_id(skill_id, dataset_id=dataset)
    else:
        skill = await find_skill_by_id(skill_id, dataset_id=dataset)
    return _skill_to_dict(skill, include_procedure=True) if skill else None


async def _resolve_dataset_owner(dataset: str | UUID) -> UUID | None:
    """Return the owner id for a dataset, or None when it cannot be resolved."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Dataset

    if not isinstance(dataset, UUID):
        return None

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        record = await session.get(Dataset, dataset)
        return record.owner_id if record else None


async def _collect_skills(
    dataset: str | UUID | None,
    include_inactive: bool,
) -> list[dict[str, Any]]:
    raw_nodes = await _load_skill_nodes()
    dataset_id = _as_uuid(dataset)

    skills: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_nodes:
        # The indexed lookup can return adjacent nodes (e.g. the "skills" NodeSet)
        # that coercion would accept; require the raw node's type to be "Skill".
        if not _is_skill_node(raw):
            continue
        skill = _coerce_skill(raw)
        if skill is None or str(skill.id) in seen:
            continue
        if not include_inactive and not skill.is_active:
            continue
        # Keep skills scoped to this dataset. Skills with an empty dataset_scope
        # (pre-scope ingestion) are kept rather than hidden.
        scope = skill.dataset_scope or []
        if dataset_id is not None and scope and str(dataset_id) not in scope:
            continue
        seen.add(str(skill.id))
        skills.append(_skill_to_dict(skill))

    skills.sort(key=lambda s: (s["name"] or "").lower())
    return skills


def _is_skill_node(raw: Any) -> bool:
    """True only for raw graph nodes whose ``type`` property is ``Skill``."""
    if isinstance(raw, Skill):
        return True
    node = raw[1] if isinstance(raw, (list, tuple)) and len(raw) > 1 else raw
    data = node.model_dump() if hasattr(node, "model_dump") else node
    return isinstance(data, dict) and data.get("type") == "Skill"


def _as_uuid(dataset: str | UUID | None) -> UUID | None:
    if dataset is None:
        return None
    if isinstance(dataset, UUID):
        return dataset
    try:
        return UUID(str(dataset))
    except (ValueError, TypeError):
        return None
