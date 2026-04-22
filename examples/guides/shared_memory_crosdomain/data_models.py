"""Graph node type for agent findings.

Both ``description`` and ``structural_tags`` are declared as index fields
so they are embedded by the vector engine. ``structural_tags`` is stored
as a comma-joined string because cognee's vector indexers assume index
fields are string-typed; ``tags()`` splits it back into a list for
analytics (cross-pollination, tag-vocabulary validation).
"""

from cognee.infrastructure.engine import DataPoint


class StructuralFinding(DataPoint):
    description: str
    structural_tags: str
    citations: str
    metadata: dict = {"index_fields": ["description", "structural_tags"]}

    def tags(self) -> list[str]:
        return [t.strip() for t in self.structural_tags.split(",") if t.strip()]

    def citation_list(self) -> list[str]:
        return [c.strip() for c in self.citations.split(",") if c.strip()]
