"""SKILL.md loader following the Anthropic Agent Skills format.

A SKILL.md file has YAML frontmatter delimited by "---" lines and a markdown body:

    ---
    name: churn-analyst
    description: Find patterns in customer churn.
    allowed-tools: memory_search, load_skill
    version: "1"
    ---
    # Procedure
    1. ...

The `allowed-tools` field may be either a YAML list or a comma-separated string.
"""

from pathlib import Path
from typing import List, Tuple

import yaml

from cognee.modules.engine.models import Skill


_FRONTMATTER_DELIM = "---"


def _split_frontmatter(text: str) -> Tuple[dict, str]:
    """Split YAML frontmatter from body. Returns (metadata_dict, body_str).

    Raises ValueError if the frontmatter is malformed.
    """
    if not text.startswith(_FRONTMATTER_DELIM):
        return {}, text

    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_DELIM:
        return {}, text

    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == _FRONTMATTER_DELIM:
            end_idx = i
            break

    if end_idx is None:
        raise ValueError("SKILL.md frontmatter is not closed with '---'")

    frontmatter_text = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
    metadata = yaml.safe_load(frontmatter_text) or {}
    if not isinstance(metadata, dict):
        raise ValueError("SKILL.md frontmatter must be a YAML mapping")
    return metadata, body


def _normalize_tool_list(value) -> List[str]:
    """Accept list or comma-separated string for allowed-tools."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    raise ValueError(f"allowed-tools must be a list or string, got {type(value).__name__}")


def parse_skill_md(text: str) -> Skill:
    """Parse a SKILL.md string into a Skill DataPoint."""
    metadata, body = _split_frontmatter(text)
    if "name" not in metadata or "description" not in metadata:
        raise ValueError("SKILL.md frontmatter must contain 'name' and 'description'")

    return Skill(
        name=str(metadata["name"]),
        description=str(metadata["description"]),
        procedure=body,
        declared_tools=_normalize_tool_list(metadata.get("allowed-tools")),
        skill_version=str(metadata.get("version", "1")),
    )


def load_skill_from_md(path: Path) -> Skill:
    """Read a SKILL.md file from disk and return a Skill DataPoint."""
    text = Path(path).read_text(encoding="utf-8")
    return parse_skill_md(text)


def load_skills_from_directory(root: Path) -> List[Skill]:
    """Recursively load every SKILL.md under root."""
    root_path = Path(root)
    return [load_skill_from_md(p) for p in sorted(root_path.rglob("SKILL.md"))]
