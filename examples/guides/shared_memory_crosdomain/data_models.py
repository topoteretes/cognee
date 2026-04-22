"""Graph node type for agent findings.

Declaring `structural_tags` and `description` as index fields makes both
embeddable, which is what allows Agent N to retrieve Agent M's findings by
structural similarity even when the two agents share no natural-language
vocabulary.
"""

from cognee.infrastructure.engine import DataPoint


class StructuralFinding(DataPoint):
    description: str
    structural_tags: list[str]
    citations: list[str]
    metadata: dict = {"index_fields": ["description", "structural_tags"]}
