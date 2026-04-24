"""Pipeline task that sets belongs_to_set on Skill DataPoints using Cognee's NodeSet."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cognee.low_level import DataPoint
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.engine.utils.generate_node_id import generate_node_id

from cognee.modules.engine.models.Skill import Skill


def _tag(dp: DataPoint, ns: NodeSet, node_set: str) -> None:
    """Add the NodeSet to a DataPoint's belongs_to_set if not already present."""
    existing = dp.belongs_to_set or []
    existing_names = {s.name if hasattr(s, "name") else s for s in existing}
    if node_set not in existing_names:
        dp.belongs_to_set = list(existing) + [ns]


async def apply_node_set(
    skills: List[Skill],
    context: Optional[Dict[str, Any]] = None,
    node_set: str = "skills",
) -> List[Skill]:
    """Tag all skills and their children with belongs_to_set."""
    ns = NodeSet(id=generate_node_id(f"NodeSet:{node_set}"), name=node_set)

    for skill in skills:
        _tag(skill, ns, node_set)
        for resource in skill.resources or []:
            _tag(resource, ns, node_set)
        for _edge, pattern in skill.solves or []:
            _tag(pattern, ns, node_set)

    return skills
