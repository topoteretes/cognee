"""Parser for SKILL.md-based skill folders.

Supports the Anthropic skills convention:
  skills/
    my-skill/
      SKILL.md          (YAML frontmatter + markdown body)
      references/       (optional)
      scripts/          (optional)
      assets/           (optional)
      *.md              (extra reference docs)
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid5

import yaml

from cognee.cognee_skills.models.skill import Skill, SkillResource

logger = logging.getLogger(__name__)

NAMESPACE = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

RESOURCE_TYPE_MAP = {
    "scripts": "script",
    "references": "reference",
    "assets": "asset",
}

SKILL_ENTRY_FILE = "SKILL.md"

BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".svg",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".dll",
}


def _deterministic_id(namespace_key: str) -> UUID:
    return uuid5(NAMESPACE, namespace_key)


def _parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    """Split YAML frontmatter and markdown body from a SKILL.md file.

    Uses yaml.safe_load for robust parsing of lists, nested objects,
    multi-line scalars, and special characters in values.
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", text, re.DOTALL)
    if not match:
        return {}, text

    raw_yaml = match.group(1)
    body = match.group(2)

    try:
        frontmatter = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse YAML frontmatter: %s", exc)
        frontmatter = {}

    if not isinstance(frontmatter, dict):
        frontmatter = {}

    return frontmatter, body.strip()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _is_binary(path: Path) -> bool:
    return path.suffix.lower() in BINARY_EXTENSIONS


def _read_text_safe(path: Path, max_chars: int = 50_000) -> Optional[str]:
    """Read text content, returning None for binary or unreadable files."""
    if _is_binary(path):
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"
        return text
    except Exception:
        return None


def _classify_resource(path: Path, skill_dir: Path) -> str:
    """Classify a resource by its parent directory name."""
    try:
        relative = path.relative_to(skill_dir)
        top_dir = relative.parts[0] if len(relative.parts) > 1 else ""
        return RESOURCE_TYPE_MAP.get(top_dir, "other")
    except ValueError:
        return "other"


def _scan_resources(skill_dir: Path) -> List[SkillResource]:
    """Scan a skill directory for bundled resources (everything except SKILL.md)."""
    resources: List[SkillResource] = []

    for item in sorted(skill_dir.rglob("*")):
        if not item.is_file():
            continue
        if item.name == SKILL_ENTRY_FILE:
            continue

        rel_path = str(item.relative_to(skill_dir))
        resource_type = _classify_resource(item, skill_dir)
        content = _read_text_safe(item)
        resource_hash = _content_hash(content) if content else ""

        resource = SkillResource(
            id=_deterministic_id(f"resource:{skill_dir.name}/{rel_path}"),
            name=item.name,
            path=rel_path,
            resource_type=resource_type,
            content=content,
            content_hash=resource_hash,
        )
        resources.append(resource)

    return resources


def _extract_tags(body: str, frontmatter: Dict[str, Any]) -> List[str]:
    """Extract tags from frontmatter or by scanning markdown headings."""
    if "tags" in frontmatter:
        raw = frontmatter["tags"]
        if isinstance(raw, list):
            return [str(t).strip() for t in raw if t is not None]
        return [t.strip() for t in str(raw).split(",") if t.strip()]

    tags: List[str] = []
    for match in re.finditer(r"^##\s+(.+)", body, re.MULTILINE):
        heading = match.group(1).strip().lower()
        heading = re.sub(r"[^a-z0-9\s-]", "", heading).strip()
        if heading and len(heading) < 40:
            tags.append(heading.replace(" ", "-"))
    return tags[:10]


def _detect_complexity(body: str) -> str:
    """Heuristic: classify skill complexity from its instructions."""
    lower = body.lower()
    if any(kw in lower for kw in ["subagent", "multi-step", "agent loop", "orchestrat"]):
        return "agent"
    if any(kw in lower for kw in ["workflow", "pipeline", "step 1", "step 2", "process"]):
        return "workflow"
    return "simple"


def _extract_triggers(description: str) -> List[str]:
    """Extract trigger phrases from the description field.

    Looks for patterns like:
      'Use when ... "phrase1", "phrase2"'
      'TRIGGER when: ...'
      Or falls back to splitting on commas/semicolons after "when".
    """
    triggers: List[str] = []

    quoted = re.findall(r'"([^"]+)"', description)
    if quoted:
        triggers.extend(quoted)
        return triggers

    when_match = re.search(r"(?:use when|trigger when|activate when)[:\s]*(.*)", description, re.I)
    if when_match:
        text = when_match.group(1)
        parts = re.split(r"[,;]|(?:\bor\b)", text)
        triggers.extend(p.strip().strip('"').strip("'") for p in parts if p.strip())

    return triggers


def parse_skill_file(
    skill_md: Path,
    source_repo: str = "",
    skill_key: Optional[str] = None,
) -> Optional[Skill]:
    """Parse a single SKILL.md file into a Skill DataPoint.

    Args:
        skill_md: Path to the SKILL.md file.
        source_repo: Provenance label.
        skill_key: Override for skill_id; defaults to parent directory name,
                   or the file stem if the file is at the skills root.
    """
    if not skill_md.is_file():
        return None

    if skill_key is None:
        skill_key = skill_md.parent.name

    raw_text = skill_md.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(raw_text)

    name = frontmatter.pop("name", skill_key)
    description = frontmatter.pop("description", "")

    triggers = _extract_triggers(description)
    tags = _extract_tags(body, frontmatter)
    complexity = _detect_complexity(body)

    skill_dir = skill_md.parent
    resources = _scan_resources(skill_dir) if skill_dir != skill_md else []

    known_keys = {"name", "description", "tags"}
    extra = {k: v for k, v in frontmatter.items() if k not in known_keys} or None

    skill = Skill(
        id=_deterministic_id(f"skill:{skill_key}"),
        skill_id=skill_key,
        name=name,
        description=description,
        instructions=body,
        description_raw=description,
        triggers_raw=triggers,
        tags_raw=tags,
        tools=[],
        triggers=triggers,
        tags=tags,
        source_path=str(skill_md.parent),
        source_repo=source_repo,
        content_hash=_content_hash(raw_text),
        complexity=complexity,
        is_active=True,
        extra_metadata=extra,
        resources=resources,
        related_skills=[],
    )
    return skill


def parse_skill_folder(
    skill_dir: Path,
    source_repo: str = "",
) -> Optional[Skill]:
    """Parse a single skill folder into a Skill DataPoint.

    Looks for SKILL.md inside the given directory.
    """
    skill_md = skill_dir / SKILL_ENTRY_FILE
    return parse_skill_file(skill_md, source_repo=source_repo)


def parse_skills_folder(
    skills_root: str | Path,
    source_repo: str = "",
) -> List[Skill]:
    """Parse all skill folders (and flat SKILL.md files) under a root directory.

    Supports two layouts:
      1. Subfolder convention: ``skills_root/my-skill/SKILL.md``
      2. Flat files: ``skills_root/SKILL.md`` or ``skills_root/my-skill.md``

    Args:
        skills_root: Path to the directory containing skill subdirectories.
        source_repo: Provenance label (e.g. "anthropics/skills", "my-org/skills").

    Returns:
        List of Skill DataPoints, one per valid skill folder/file found.
    """
    skills_root = Path(skills_root)
    if not skills_root.is_dir():
        raise FileNotFoundError(f"Skills directory not found: {skills_root}")

    if not source_repo:
        source_repo = skills_root.name

    seen_keys: set[str] = set()
    skills: List[Skill] = []

    for child in sorted(skills_root.iterdir()):
        if child.is_dir():
            skill = parse_skill_folder(child, source_repo=source_repo)
            if skill is not None:
                seen_keys.add(skill.skill_id)
                skills.append(skill)

    for child in sorted(skills_root.iterdir()):
        if not child.is_file():
            continue
        if child.suffix.lower() != ".md":
            continue

        if child.name.upper() == SKILL_ENTRY_FILE:
            skill_key = skills_root.name
        else:
            skill_key = child.stem

        if skill_key in seen_keys:
            continue

        skill = parse_skill_file(child, source_repo=source_repo, skill_key=skill_key)
        if skill is not None:
            seen_keys.add(skill_key)
            skills.append(skill)

    return skills
