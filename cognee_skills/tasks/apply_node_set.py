"""Pipeline task that sets belongs_to_set on Skill DataPoints using Cognee's NodeSet."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.engine.utils.generate_node_id import generate_node_id

from cognee_skills.models.skill import Skill


async def apply_node_set(
    skills: List[Skill],
    context: Optional[Dict[str, Any]] = None,
    node_set: str = "skills",
) -> List[Skill]:
    """Tag all skills with belongs_to_set using Cognee's official NodeSet type."""
    ns = NodeSet(id=generate_node_id(f"NodeSet:{node_set}"), name=node_set)

    for skill in skills:
        existing = skill.belongs_to_set or []
        existing_names = {s.name if hasattr(s, "name") else s for s in existing}
        if node_set not in existing_names:
            skill.belongs_to_set = list(existing) + [ns]

    return skills
