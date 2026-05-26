"""Small SKILL.md parser for the v1 agentic-skills flow."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid5

import yaml

from cognee.modules.engine.models import Skill
from cognee.modules.tools.path_safety import (
    trusted_is_dir,
    trusted_is_file,
    trusted_read_text,
    trusted_rglob,
)
from cognee.shared.logging_utils import get_logger


logger = get_logger(__name__)

NAMESPACE = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

_DESCRIPTION_ALIASES = ("description", "summary", "short_description", "about")
_TOOLS_ALIASES = ("allowed-tools", "allowed_tools", "declared_tools", "tools")


def _deterministic_id(namespace_key: str) -> UUID:
    return uuid5(NAMESPACE, namespace_key)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _normalize_path(path: Path) -> str:
    return os.path.normpath(os.path.realpath(os.path.abspath(os.fspath(path))))


def _is_relative_to(path: Path, base_dir: Path) -> bool:
    try:
        path_str = _normalize_path(path)
        base_str = _normalize_path(base_dir)
    except (OSError, RuntimeError, ValueError):
        return False
    base_prefix = base_str if base_str.endswith(os.sep) else f"{base_str}{os.sep}"
    return path_str == base_str or path_str.startswith(base_prefix)


def _parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    match = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?(.*)", text, re.DOTALL)
    if not match:
        return {}, text.strip()

    try:
        frontmatter = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse SKILL.md frontmatter: %s", exc)
        frontmatter = {}

    if not isinstance(frontmatter, dict):
        frontmatter = {}

    return frontmatter, match.group(2).strip()


def _pop_first(data: Dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for key in aliases:
        if key in data:
            return data.pop(key)
    return None


def _extract_description(frontmatter: Dict[str, Any], body: str) -> str:
    raw = _pop_first(frontmatter, _DESCRIPTION_ALIASES)
    if raw:
        return str(raw).strip()

    for paragraph in re.split(r"\n{2,}", body):
        paragraph = paragraph.strip()
        if paragraph and not paragraph.startswith("#"):
            return re.sub(r"[`*_~]", "", paragraph)[:500]

    return ""


def _extract_tools(frontmatter: Dict[str, Any]) -> List[str]:
    raw = _pop_first(frontmatter, _TOOLS_ALIASES)
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(tool).strip() for tool in raw if str(tool).strip()]
    return [tool.strip() for tool in re.split(r"[\s,]+", str(raw)) if tool.strip()]


def _skill_slug(skill_file: Path) -> str:
    return skill_file.parent.name


def _build_search_text(name: str, description: str, procedure: str) -> str:
    return "\n\n".join(part for part in (name, description, procedure) if part)


def parse_skill_file(
    skill_md: Path,
    source_repo: str = "",
    skill_key: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Optional[Skill]:
    """Parse one concrete SKILL.md file into one Skill node."""
    skill_md = Path(skill_md)
    if base_dir is not None and not _is_relative_to(skill_md, base_dir):
        raise ValueError(f"Skill file is outside the allowed base directory: {skill_md}")
    if not trusted_is_file(skill_md) or skill_md.name.lower() != "skill.md":
        return None

    raw_text = trusted_read_text(skill_md, encoding="utf-8")
    if not raw_text.strip():
        return None

    frontmatter, body = _parse_frontmatter(raw_text)
    name = skill_key or _skill_slug(skill_md)
    description = _extract_description(frontmatter, body)
    declared_tools = _extract_tools(frontmatter)
    source_file = _normalize_path(skill_md)
    source_dir = _normalize_path(skill_md.parent)
    skill_text = _build_search_text(name, description, body)

    return Skill(
        id=_deterministic_id(f"skill:{source_dir}:{name}"),
        name=name,
        description=description,
        procedure=body,
        declared_tools=declared_tools,
        source_file=source_file,
        source_dir=source_dir,
        content_hash=_content_hash(raw_text),
        skill_text=skill_text,
        search_text=skill_text,
        belongs_to_set=["skills"],
    )


def parse_skills_folder(
    skills_root: str | Path,
    source_repo: str = "",
    base_dir: Optional[Path] = None,
) -> List[Skill]:
    """Parse every SKILL.md under a directory. Removed files are ignored in v1."""
    skills_root = Path(skills_root)
    if not trusted_is_dir(skills_root):
        raise FileNotFoundError(f"Skills directory not found: {skills_root}")
    base_dir = Path(base_dir) if base_dir is not None else skills_root

    skills: List[Skill] = []
    for skill_file in sorted(trusted_rglob(skills_root, "*")):
        if not trusted_is_file(skill_file) or skill_file.name.lower() != "skill.md":
            continue
        skill = parse_skill_file(skill_file, source_repo=source_repo, base_dir=base_dir)
        if skill is not None:
            skills.append(skill)
    return skills
