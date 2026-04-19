from typing import Annotated, List, Optional

from cognee.infrastructure.engine import DataPoint, Embeddable, LLMContext, Dedup


class Skill(DataPoint):
    """
    Procedural playbook stored alongside memory. Description is embedded and always
    loaded in the agentic retriever's system prompt; the full procedure body is
    loaded on demand via the load_skill tool (progressive disclosure).

    Instance attributes:
    - name: Unique skill identifier.
    - description: Short summary; embedded for vector retrieval and shown in the catalog.
    - procedure: Markdown body with step-by-step instructions.
    - declared_tools: Tool names the skill may invoke. Intersected with the ambient
      tool set at execution time; skills cannot escalate beyond the user's permissions.
    - dataset_scope: Optional dataset ID strings this skill applies to; None means all.
      Stored as strings so the type serializes cleanly into LanceDB-backed vector
      indexes (Arrow has no native UUID mapping).
    - skill_version: Free-form version string for tracking skill revisions.
    """

    name: Annotated[str, Dedup()]
    description: Annotated[str, Embeddable(), LLMContext()]
    procedure: Annotated[str, LLMContext("Loaded on skill selection")] = ""
    declared_tools: List[str] = []
    dataset_scope: Optional[List[str]] = None
    skill_version: str = "1"
