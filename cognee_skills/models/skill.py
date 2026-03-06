from typing import Any, Dict, List, Optional
from cognee.low_level import DataPoint
from cognee.infrastructure.engine import Edge


class SkillResource(DataPoint):
    """A bundled file within a skill folder (reference doc, script, asset)."""

    name: str
    path: str
    resource_type: str  # "reference", "script", "asset", "other"
    content: Optional[str] = None
    content_hash: str = ""
    metadata: dict = {"index_fields": ["name"]}


class Skill(DataPoint):
    """An agentic skill parsed from a SKILL.md folder."""

    skill_id: str
    name: str
    description: str
    instructions: str  # full markdown body — stored but NOT indexed

    # Parser-derived originals (never overwritten by LLM)
    description_raw: str = ""
    triggers_raw: List[str] = []
    tags_raw: List[str] = []

    # LLM-enriched fields (filled by enrich_skills task)
    instruction_summary: str = ""
    triggers: List[str] = []
    tags: List[str] = []
    complexity: str = ""  # "simple", "workflow", "agent"
    task_pattern_candidates: List[str] = []

    # Parser-only fields (deterministic, never LLM-filled)
    tools: List[str] = []
    source_path: str = ""
    source_repo: str = ""
    content_hash: str = ""
    is_active: bool = True
    extra_metadata: Optional[Dict[str, Any]] = None

    # Enrichment provenance
    enrichment_model: str = ""
    enrichment_confidence: float = 0.0

    resources: List[SkillResource] = []
    related_skills: List["Skill"] = []
    solves: List[tuple[Edge, "TaskPattern"]] = []

    metadata: dict = {"index_fields": ["name", "instruction_summary", "description"]}


from .task_pattern import TaskPattern  # noqa: E402

Skill.model_rebuild()
