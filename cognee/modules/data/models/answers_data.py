from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import Column, DateTime, JSON, Uuid

from cognee.modules.data.models.answers_base import AnswersBase


class Answers(AnswersBase):
    __tablename__ = "eval_answers"

    id = Column(Uuid, primary_key=True, default=uuid4)

    payload = Column(JSON, nullable=False)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
