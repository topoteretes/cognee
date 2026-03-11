"""Pipeline task that parses a skills folder into Skill DataPoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cognee.cognee_skills.models.skill import Skill
from cognee.cognee_skills.parser.skill_parser import parse_skills_folder


def parse_skills_task(
    data: Any,
    context: Optional[Dict[str, Any]] = None,
    source_repo: str = "",
) -> List[Skill]:
    """Parse a skills folder path into Skill DataPoints.

    Accepts either a string path or a list whose first element is a string path.
    Returns a flat list of Skill DataPoints ready for add_data_points.
    """
    if isinstance(data, (list, tuple)):
        skills_path = data[0] if data else None
    else:
        skills_path = data

    if not skills_path or not isinstance(skills_path, str):
        raise ValueError(f"Expected a skills folder path, got: {skills_path!r}")

    return parse_skills_folder(skills_path, source_repo=source_repo)
