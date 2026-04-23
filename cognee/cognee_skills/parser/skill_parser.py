"""Parser for skill documentation files.

Supports multiple community formats:
  - Anthropic agent-skills spec (SKILL.md, name + description frontmatter)
  - OpenClaw skills (same spec + metadata.openclaw extensions)
  - Muratcankoylan context-engineering skills (body-heavy with When to Activate section)
  - Any markdown file with enough content to infer a skill

Entry file discovery order per folder:
  SKILL.md → skill.md → README.md (only if it looks like a skill)

Required for a valid skill: a non-empty name (or inferable from folder/file name)
  and non-empty body (description can be derived by LLM from body if missing).
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

# Ordered list of candidate entry file names to try per skill folder
SKILL_ENTRY_CANDIDATES = ["SKILL.md", "skill.md", "Skill.md"]

# README.md accepted only when the folder contains no other .md file that looks like a skill
README_CANDIDATES = ["README.md", "readme.md"]

RESOURCE_TYPE_MAP = {
    "scripts": "script",
    "references": "reference",
    "assets": "asset",
}

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

# Frontmatter key aliases → canonical field name
_NAME_ALIASES = ("name", "title", "skill_name", "skill-name")
_DESCRIPTION_ALIASES = ("description", "summary", "short_description", "about")
_TAGS_ALIASES = ("tags", "categories", "keywords", "labels")


def _deterministic_id(namespace_key: str) -> UUID:
    return uuid5(NAMESPACE, namespace_key)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _is_binary(path: Path) -> bool:
    return path.suffix.lower() in BINARY_EXTENSIONS


def _read_text_safe(path: Path, max_chars: int = 50_000) -> Optional[str]:
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
    try:
        relative = path.relative_to(skill_dir)
        top_dir = relative.parts[0] if len(relative.parts) > 1 else ""
        return RESOURCE_TYPE_MAP.get(top_dir, "other")
    except ValueError:
        return "other"


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


def _parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    """Split YAML frontmatter and markdown body.

    Handles standard --- delimiters. Returns ({}, full_text) if no frontmatter.
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", text, re.DOTALL)
    if not match:
        return {}, text.strip()

    try:
        frontmatter = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse YAML frontmatter: %s", exc)
        frontmatter = {}

    if not isinstance(frontmatter, dict):
        frontmatter = {}

    return frontmatter, match.group(2).strip()


def _pop_first(d: Dict[str, Any], aliases: tuple) -> Any:
    """Pop and return the first matching key from a dict."""
    for key in aliases:
        if key in d:
            return d.pop(key)
    return None


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------


def _extract_name(frontmatter: Dict[str, Any], fallback: str) -> str:
    val = _pop_first(frontmatter, _NAME_ALIASES)
    return str(val).strip() if val else fallback


def _extract_description(frontmatter: Dict[str, Any], body: str) -> str:
    """Extract description from frontmatter aliases, or infer from body.

    Falls back to the first non-heading paragraph so the LLM enrichment
    step has something to work with even for format-free files.
    """
    val = _pop_first(frontmatter, _DESCRIPTION_ALIASES)
    if val:
        return str(val).strip()

    # Try first non-heading paragraph of the body
    for para in re.split(r"\n{2,}", body):
        para = para.strip()
        if para and not para.startswith("#") and len(para) >= 30:
            # Strip inline markdown (bold/italic/code) for a clean hint
            clean = re.sub(r"[`*_~]", "", para)
            return clean[:500]

    return ""


def _extract_tags(frontmatter: Dict[str, Any], body: str) -> List[str]:
    """Extract tags from frontmatter (multiple alias forms + nested metadata)."""
    # Explicit tags fields
    raw = _pop_first(frontmatter, _TAGS_ALIASES)
    if raw is not None:
        if isinstance(raw, list):
            return [str(t).strip() for t in raw if t]
        return [t.strip() for t in str(raw).split(",") if t.strip()]

    # OpenClaw: metadata.openclaw.tags
    metadata = frontmatter.get("metadata")
    if isinstance(metadata, dict):
        openclaw = metadata.get("openclaw", {})
        if isinstance(openclaw, dict) and openclaw.get("tags"):
            raw = openclaw["tags"]
            if isinstance(raw, list):
                return [str(t).strip() for t in raw if t]

    # Fall back to ## headings as rough tags (max 8)
    tags: List[str] = []
    for m in re.finditer(r"^##\s+(.+)", body, re.MULTILINE):
        heading = re.sub(r"[^a-z0-9\s-]", "", m.group(1).strip().lower()).strip()
        if heading and len(heading) < 40:
            tags.append(heading.replace(" ", "-"))
    return tags[:8]


def _extract_triggers(frontmatter: Dict[str, Any], description: str, body: str) -> List[str]:
    """Extract trigger phrases from multiple sources.

    Priority:
      1. Frontmatter `triggers` / `activation` field
      2. `## When to Activate` section in body (muratcankoylan convention)
      3. Quoted phrases in description
      4. Comma/semicolon-split after "use when" / "trigger when" in description
    """
    # 1. Frontmatter field
    raw = frontmatter.pop("triggers", frontmatter.pop("activation", None))
    if raw:
        if isinstance(raw, list):
            return [str(t).strip() for t in raw if t]
        return [t.strip() for t in str(raw).split(",") if t.strip()]

    # 2. ## When to Activate section
    activate_match = re.search(
        r"##\s+When to Activate\s*\n(.*?)(?=\n##|\Z)", body, re.DOTALL | re.IGNORECASE
    )
    if activate_match:
        section = activate_match.group(1)
        bullets = re.findall(r"[-*]\s+(.+)", section)
        if bullets:
            return [b.strip() for b in bullets[:12]]

    # 3 & 4. From description
    if description:
        quoted = re.findall(r'"([^"]{5,})"', description)
        if quoted:
            return quoted[:8]

        when_match = re.search(
            r"(?:use when|trigger when|activate when|triggers?)[:\s]+(.*)", description, re.I
        )
        if when_match:
            parts = re.split(r"[,;]|(?:\bor\b)", when_match.group(1))
            return [p.strip().strip("\"'") for p in parts if p.strip()][:8]

    return []


def _extract_tools(frontmatter: Dict[str, Any]) -> List[str]:
    """Extract allowed tools from `allowed-tools` or `allowed_tools` (Anthropic spec)."""
    raw = frontmatter.pop("allowed-tools", frontmatter.pop("allowed_tools", None))
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if t]
    # Space-delimited string per the Anthropic spec
    return [t.strip() for t in str(raw).split() if t.strip()]


def _detect_complexity(body: str) -> str:
    lower = body.lower()
    if any(kw in lower for kw in ["subagent", "multi-step", "agent loop", "orchestrat"]):
        return "agent"
    if any(kw in lower for kw in ["workflow", "pipeline", "step 1", "step 2", "process"]):
        return "workflow"
    return "simple"


# ---------------------------------------------------------------------------
# File / folder discovery
# ---------------------------------------------------------------------------


def _find_skill_file(skill_dir: Path) -> Optional[Path]:
    """Return the skill entry file inside a directory, trying multiple names."""
    for name in SKILL_ENTRY_CANDIDATES:
        p = skill_dir / name
        if p.is_file():
            return p

    # Accept README.md only if there's no other .md file that could be a skill
    for name in README_CANDIDATES:
        p = skill_dir / name
        if p.is_file():
            return p

    return None


def _scan_resources(skill_dir: Path, entry_file: Path) -> List[SkillResource]:
    resources: List[SkillResource] = []
    for item in sorted(skill_dir.rglob("*")):
        if not item.is_file() or item == entry_file:
            continue
        rel_path = str(item.relative_to(skill_dir))
        resource_type = _classify_resource(item, skill_dir)
        content = _read_text_safe(item)
        resource = SkillResource(
            id=_deterministic_id(f"resource:{skill_dir.name}/{rel_path}"),
            name=item.name,
            path=rel_path,
            resource_type=resource_type,
            content=content,
            content_hash=_content_hash(content) if content else "",
        )
        resources.append(resource)
    return resources


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_skill_file(
    skill_md: Path,
    source_repo: str = "",
    skill_key: Optional[str] = None,
) -> Optional[Skill]:
    """Parse a skill markdown file into a Skill DataPoint.

    Tolerates missing or thin frontmatter. If `description` cannot be found
    in frontmatter or inferred from the body, it is left empty so that the
    LLM enrichment step (enrich_skills) can derive it from the full body text.

    Args:
        skill_md: Path to the skill file.
        source_repo: Provenance label.
        skill_key: Override for skill_id; defaults to parent directory name
                   or file stem for flat files.
    """
    if not skill_md.is_file():
        return None

    raw_text = skill_md.read_text(encoding="utf-8")
    if not raw_text.strip():
        return None

    if skill_key is None:
        # For ``SKILL.md`` / ``skill.md`` / ``README.md`` in a ``<slug>/``
        # folder, the slug is the parent directory name. For a flat
        # ``<slug>.md`` file the slug is the file stem. Matches
        # parse_skills_folder()'s Pass 1 vs Pass 2 split.
        _folder_entries = {c.upper() for c in SKILL_ENTRY_CANDIDATES} | {"README.MD"}
        if skill_md.name.upper() in _folder_entries:
            skill_key = skill_md.parent.name
        else:
            skill_key = skill_md.stem

    frontmatter, body = _parse_frontmatter(raw_text)

    name = _extract_name(frontmatter, fallback=skill_key)
    description = _extract_description(frontmatter, body)
    tags = _extract_tags(frontmatter, body)
    triggers = _extract_triggers(frontmatter, description, body)
    tools = _extract_tools(frontmatter)
    complexity = _detect_complexity(body)

    skill_dir = skill_md.parent
    is_flat = skill_dir == skill_md.parent and skill_md.parent == skill_md.parent
    resources = _scan_resources(skill_dir, skill_md) if skill_md.parent != skill_md else []

    # Preserve any remaining unknown frontmatter fields as extra_metadata
    known_keys = (
        set(_NAME_ALIASES)
        | set(_DESCRIPTION_ALIASES)
        | set(_TAGS_ALIASES)
        | {
            "triggers",
            "activation",
            "allowed-tools",
            "allowed_tools",
            "license",
            "compatibility",
            "homepage",
            "metadata",
        }
    )
    extra = {k: v for k, v in frontmatter.items() if k not in known_keys} or None

    return Skill(
        id=_deterministic_id(f"skill:{skill_key}"),
        # ``name`` holds the canonical slug identifier (folder name or
        # explicit frontmatter ``skill_id:``), NOT the human-readable
        # frontmatter ``name:`` value. This is what every downstream
        # reader — client.load, resolve_skills, inspect, amendify —
        # looks up by. The display name from frontmatter is preserved
        # in ``description_raw`` and the enrichment pipeline.
        name=skill_key,
        description=description,
        procedure=body,
        declared_tools=tools,
        description_raw=description,
        triggers_raw=triggers,
        tags_raw=tags,
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


def parse_skill_folder(
    skill_dir: Path,
    source_repo: str = "",
) -> Optional[Skill]:
    """Parse a single skill folder into a Skill DataPoint.

    Tries SKILL.md, skill.md, Skill.md, then README.md.
    """
    entry = _find_skill_file(skill_dir)
    if entry is None:
        return None
    return parse_skill_file(entry, source_repo=source_repo)


def parse_skills_folder(
    skills_root: str | Path,
    source_repo: str = "",
) -> List[Skill]:
    """Parse all skills under a root directory.

    Supports two layouts:
      1. Subfolder convention: skills_root/my-skill/SKILL.md
      2. Flat files:           skills_root/my-skill.md

    Args:
        skills_root: Path to the directory containing skill subdirectories or files.
        source_repo: Provenance label (e.g. "anthropics/skills").

    Returns:
        List of Skill DataPoints, one per valid skill found.
    """
    skills_root = Path(skills_root)
    if not skills_root.is_dir():
        raise FileNotFoundError(f"Skills directory not found: {skills_root}")

    if not source_repo:
        source_repo = skills_root.name

    seen_keys: set[str] = set()
    skills: List[Skill] = []

    # Pass 1: subfolder skills
    for child in sorted(skills_root.iterdir()):
        if not child.is_dir():
            continue
        skill = parse_skill_folder(child, source_repo=source_repo)
        if skill is not None:
            seen_keys.add(skill.name)
            skills.append(skill)

    # Pass 2: flat .md files at root (skip if already covered by a folder)
    for child in sorted(skills_root.iterdir()):
        if not child.is_file() or child.suffix.lower() != ".md":
            continue

        skill_key = (
            skills_root.name
            if child.name.upper() in {c.upper() for c in SKILL_ENTRY_CANDIDATES}
            else child.stem
        )

        if skill_key in seen_keys:
            continue

        skill = parse_skill_file(child, source_repo=source_repo, skill_key=skill_key)
        if skill is not None:
            seen_keys.add(skill_key)
            skills.append(skill)

    return skills
