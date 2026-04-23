from typing import Annotated, Any, Dict, List, Optional

from cognee.infrastructure.engine import DataPoint, Edge, Embeddable, LLMContext, Dedup


class SkillResource(DataPoint):
    """A bundled file within a skill folder (reference doc, script, asset).

    Carried alongside the main Skill in rich-ingest flows. The agentic
    retriever ignores these; the self-improvement pipeline uses them for
    progressive disclosure and context injection.
    """

    name: str
    path: str
    resource_type: str  # "reference" | "script" | "asset" | "other"
    content: Optional[str] = None
    content_hash: str = ""
    metadata: dict = {"index_fields": ["name"]}


class TaskPattern(DataPoint):
    """A normalized intent/task category that skills can solve."""

    pattern_id: str
    name: str = ""
    pattern_key: str = ""
    text: str  # LLM-generated intent description
    category: str = ""

    # Evidence / provenance
    source_skill_ids: List[str] = []  # which skills proposed this pattern
    examples: List[str] = []  # trigger phrases that led to this pattern
    enrichment_model: str = ""
    enrichment_confidence: float = 0.0

    prefers: List[tuple[Edge, "Skill"]] = []
    metadata: dict = {"index_fields": ["text"]}


class Skill(DataPoint):
    """Procedural playbook stored alongside memory.

    One canonical Skill model serves both ingest paths:

    * ``cognee.remember("skills/")`` — frontmatter-only parse for the
      agentic retriever. Fills ``name``, ``description``, ``procedure``,
      ``declared_tools``, ``skill_version``.
    * ``cognee.remember("skills/", enrich=True)`` — parse + LLM enrichment
      for the self-improvement loop. Also fills the parser and enrichment
      fields below.

    Fields specific to one subsystem stay at their defaults when the other
    subsystem ingests; no code path misbehaves because an optional field
    is empty.

    Historical field names ``instructions`` (now ``procedure``), ``tools``
    (now ``declared_tools``), and ``skill_id`` (now ``name``) are gone —
    callers read the canonical fields directly, or for already-serialized
    graph nodes they read ``props.get("procedure")`` etc.
    """

    # --- canonical agentic-retriever fields -------------------------------
    name: Annotated[str, Dedup()]
    description: Annotated[str, Embeddable(), LLMContext()]
    procedure: Annotated[str, LLMContext("Loaded on skill selection")] = ""
    declared_tools: List[str] = []
    dataset_scope: Optional[List[str]] = None
    skill_version: str = "1"

    # --- parser-only fields (populated by cognee_skills.parser) -----------
    description_raw: str = ""
    triggers_raw: List[str] = []
    tags_raw: List[str] = []
    source_path: str = ""
    source_repo: str = ""
    content_hash: str = ""
    is_active: bool = True
    extra_metadata: Optional[Dict[str, Any]] = None

    # --- LLM-enriched fields (populated by enrich_skills task) ------------
    instruction_summary: str = ""
    triggers: List[str] = []
    tags: List[str] = []
    complexity: str = ""  # "simple" | "workflow" | "agent"
    task_pattern_candidates: List[str] = []
    enrichment_model: str = ""
    enrichment_confidence: float = 0.0

    # --- relationships ----------------------------------------------------
    resources: List[SkillResource] = []
    related_skills: List["Skill"] = []
    solves: List[tuple[Edge, TaskPattern]] = []

    metadata: dict = {"index_fields": ["name", "instruction_summary", "description"]}


# Resolve forward references now that all three classes are defined in
# this single module. No circular imports needed.
Skill.model_rebuild()
TaskPattern.model_rebuild()
