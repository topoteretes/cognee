from cognee.infrastructure.engine import DataPoint
from typing import ClassVar, Optional


class GraphitiNode(DataPoint):
    __tablename__ = "graphitinode"
    content: Optional[str] = None
    name: Optional[str] = None
    summary: Optional[str] = None
    pydantic_type: str = "GraphitiNode"

    _metadata: dict = {"index_fields": ["name", "summary", "content"], "type": "GraphitiNode"}
