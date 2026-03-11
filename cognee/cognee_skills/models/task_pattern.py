from typing import List
from cognee.low_level import DataPoint
from cognee.infrastructure.engine import Edge


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


from .skill import Skill  # noqa: E402

TaskPattern.model_rebuild()
