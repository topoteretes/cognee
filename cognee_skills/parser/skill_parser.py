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
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid5

from cognee_skills.models.skill import Skill, SkillResource

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
    """Split YAML frontmatter and markdown body from a SKILL.md file."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", text, re.DOTALL)
    if not match:
        return {}, text

    raw_yaml = match.group(1)
    body = match.group(2)

    frontmatter: Dict[str, Any] = {}
    current_key: Optional[str] = None
    current_value_lines: list[str] = []

    for line in raw_yaml.split("\n"):
        kv_match = re.match(r"^(\w[\w-]*)\s*:\s*(.*)", line)
        if kv_match:
            if current_key is not None:
                frontmatter[current_key] = _collapse_value(current_value_lines)
            current_key = kv_match.group(1)
            current_value_lines = [kv_match.group(2)]
        elif current_key is not None:
            current_value_lines.append(line)

    if current_key is not None:
        frontmatter[current_key] = _collapse_value(current_value_lines)

    return frontmatter, body.strip()


def _collapse_value(lines: list[str]) -> str:
    """Join multi-line YAML scalar values into a single string."""
    joined = " ".join(line.strip() for line in lines if line.strip())
    for quote in ('"', "'"):
        if joined.startswith(quote) and joined.endswith(quote):
            joined = joined[1:-1]
    if joined.startswith(">"):
        joined = joined[1:].strip()
    return joined.strip()


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
            return raw
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


def parse_skill_folder(
    skill_dir: Path,
    source_repo: str = "",
) -> Optional[Skill]:
    """Parse a single skill folder into a Skill DataPoint."""
    skill_md = skill_dir / SKILL_ENTRY_FILE
    if not skill_md.is_file():
        return None

    raw_text = skill_md.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(raw_text)

    name = frontmatter.pop("name", skill_dir.name)
    description = frontmatter.pop("description", "")

    triggers = _extract_triggers(description)
    tags = _extract_tags(body, frontmatter)
    complexity = _detect_complexity(body)
    resources = _scan_resources(skill_dir)

    known_keys = {"name", "description", "tags"}
    extra = {k: v for k, v in frontmatter.items() if k not in known_keys} or None

    skill = Skill(
        id=_deterministic_id(f"skill:{skill_dir.name}"),
        skill_id=skill_dir.name,
        name=name,
        description=description,
        instructions=body,
        description_raw=description,
        triggers_raw=triggers,
        tags_raw=tags,
        tools=[],
        triggers=triggers,
        tags=tags,
        source_path=str(skill_dir),
        source_repo=source_repo,
        content_hash=_content_hash(raw_text),
        complexity=complexity,
        is_active=True,
        extra_metadata=extra,
        resources=resources,
        related_skills=[],
    )
    return skill


def parse_skills_folder(
    skills_root: str | Path,
    source_repo: str = "",
) -> List[Skill]:
    """Parse all skill folders under a root directory.

    Args:
        skills_root: Path to the directory containing skill subdirectories.
        source_repo: Provenance label (e.g. "anthropics/skills", "my-org/skills").

    Returns:
        List of Skill DataPoints, one per valid skill folder found.
    """
    skills_root = Path(skills_root)
    if not skills_root.is_dir():
        raise FileNotFoundError(f"Skills directory not found: {skills_root}")

    if not source_repo:
        source_repo = skills_root.name

    skills: List[Skill] = []
    for child in sorted(skills_root.iterdir()):
        if not child.is_dir():
            continue
        skill = parse_skill_folder(child, source_repo=source_repo)
        if skill is not None:
            skills.append(skill)

    return skills
