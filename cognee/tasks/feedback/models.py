from typing import List, Optional, Union
from uuid import UUID

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models import Entity, NodeSet
from cognee.tasks.temporal_graph.models import Event


class FeedbackEnrichment(DataPoint):
    """Minimal DataPoint for feedback enrichment that works with extract_graph_from_data."""

    text: str
    contains: Optional[List[Union[Entity, Event]]] = None
    metadata: dict = {"index_fields": ["text"]}

    question: str
    original_answer: str
    improved_answer: str
    feedback_id: UUID
    interaction_id: UUID
    belongs_to_set: Optional[List[NodeSet]] = None

    context: str = ""
    feedback_text: str = ""
    new_context: str = ""
    explanation: str = ""
