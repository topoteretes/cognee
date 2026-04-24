"""Ingest SKILL.md files as Skill DataPoints into the knowledge graph.

This is the bridge between the SKILL.md loader and Cognee's standard storage
path: parsed Skills are persisted through `add_data_points`, so they land in
both the graph engine and the vector index — which is exactly what
`resolve_skills()` queries at search time.
"""

from pathlib import Path
from typing import List, Union

from cognee.modules.engine.models import Skill
from cognee.modules.tools.loaders import (
    load_skill_from_md,
    load_skills_from_directory,
)
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage.add_data_points import add_data_points


logger = get_logger("cognee.tools.ingest_skills")


def looks_like_skill_source(data) -> bool:
    """Return True when `data` is a path to a SKILL.md file or a directory that
    contains at least one SKILL.md. Used for auto-dispatch in remember()/add()."""
    if not isinstance(data, str):
        return False
    path = Path(data)
    try:
        if path.is_file():
            return path.name.upper() == "SKILL.MD"
        if path.is_dir():
            return any(path.rglob("SKILL.md")) or any(path.rglob("skill.md"))
    except OSError:
        return False
    return False


async def add_skills(source: Union[str, Path]) -> List[Skill]:
    """Parse SKILL.md file(s) from `source` and persist them as Skill DataPoints.

    Args:
        source: Either a SKILL.md file path or a directory scanned recursively
            for SKILL.md files.

    Returns:
        The persisted Skill DataPoints (useful for inspection and testing).
    """
    path = Path(source)
    if path.is_dir():
        skills = load_skills_from_directory(path)
    elif path.is_file():
        skills = [load_skill_from_md(path)]
    else:
        raise FileNotFoundError(f"Skill source not found: {source}")

    if not skills:
        logger.warning("No SKILL.md files discovered under %s", source)
        return []

    await add_data_points(skills)
    logger.info("Ingested %d skill(s) from %s", len(skills), source)
    return skills
