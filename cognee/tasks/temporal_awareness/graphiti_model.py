from cognee.infrastructure.engine import DataPoint
from typing import Optional


class GraphitiNode(DataPoint):
    content: Optional[str] = None
    name: Optional[str] = None
    summary: Optional[str] = None

    metadata: dict = {"index_fields": ["name", "summary", "content"]}
