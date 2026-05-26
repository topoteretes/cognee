"""Dataset-scoped Skill lookup helpers for the agentic retriever."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union
from uuid import UUID

from cognee.modules.engine.models import Skill
from cognee.shared.logging_utils import get_logger


logger = get_logger("cognee.tools.resolve_skills")


def _skill_in_dataset_scope(skill: Skill, dataset_id: UUID) -> bool:
    return skill.is_active and str(dataset_id) in (skill.dataset_scope or [])


async def resolve_skills(
    skills: Optional[Sequence[Union[str, Skill]]] = None,
    *,
    dataset_id: Optional[UUID] = None,
) -> List[Skill]:
    """Resolve explicit skills inside one dataset.

    v1 intentionally rejects unscoped skill lookup and multi-dataset lookup.
    """
    if dataset_id is None:
        raise ValueError("Skill lookup requires one explicit dataset.")

    resolved: List[Skill] = []
    seen_ids = set()

    for item in skills or []:
        skill = item if isinstance(item, Skill) else None
        if isinstance(item, str):
            skill = await find_skill_by_name(item, dataset_id=dataset_id)
        elif skill is None:
            logger.warning("Skill entries must be Skill or str; got %s", type(item).__name__)

        if skill is None:
            logger.warning("Skill %r not found in dataset %s; skipping", item, dataset_id)
            continue
        if not _skill_in_dataset_scope(skill, dataset_id):
            logger.warning("Skill %r is outside dataset scope; skipping", skill.name)
            continue
        if skill.id not in seen_ids:
            resolved.append(skill)
            seen_ids.add(skill.id)

    return resolved


async def find_skill_by_id(skill_id: str, *, dataset_id: UUID) -> Optional[Skill]:
    raw_nodes = await _load_skill_nodes()
    for raw in raw_nodes:
        skill = _coerce_skill(raw)
        if skill is None:
            continue
        if str(skill.id) == str(skill_id) and _skill_in_dataset_scope(skill, dataset_id):
            return skill
    return None


async def find_skill_by_name(name: str, *, dataset_id: UUID) -> Optional[Skill]:
    raw_nodes = await _load_skill_nodes(name=name)
    for raw in raw_nodes:
        skill = _coerce_skill(raw)
        if skill is None or skill.name != name:
            continue
        if _skill_in_dataset_scope(skill, dataset_id):
            return skill
    return None


async def _load_skill_nodes(name: Optional[str] = None):
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
            return await get_by_type(node_type=Skill)
        except Exception as exc:
            logger.warning("Skill lookup by type failed: %s", exc)
            return []

    get_nodeset = getattr(graph_engine, "get_nodeset_subgraph", None)
    if get_nodeset is not None and name is not None:
        try:
            nodes, _ = await get_nodeset(node_type=Skill, node_name=[name])
            if nodes:
                return nodes
        except Exception as exc:
            logger.warning("Skill lookup by nodeset failed: %s", exc)

    get_graph_data = getattr(graph_engine, "get_graph_data", None)
    if get_graph_data is None:
        return []
    try:
        nodes, _ = await get_graph_data()
        return nodes
    except Exception as exc:
        logger.warning("Skill lookup by full graph scan failed: %s", exc)
        return []


def _coerce_skill(raw) -> Optional[Skill]:
    if isinstance(raw, Skill):
        return raw
    if isinstance(raw, (list, tuple)) and len(raw) > 1:
        raw = raw[1]
    data = raw.model_dump() if hasattr(raw, "model_dump") else raw
    if not isinstance(data, dict):
        return None
    data = {k: v for k, v in data.items() if k != "metadata"}
    try:
        return Skill.model_validate(data)
    except Exception:
        return None
