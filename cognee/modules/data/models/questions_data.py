from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import Column, DateTime, JSON, UUID

from cognee.modules.data.models.questions_base import QuestionsBase


class Questions(QuestionsBase):
    __tablename__ = "eval_questions"

    id = Column(UUID, primary_key=True, default=uuid4)

    payload = Column(JSON, nullable=False)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
