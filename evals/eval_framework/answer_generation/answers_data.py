from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import Column, DateTime, JSON, UUID

from evals.eval_framework.answer_generation.answers_base import AnswersBase


class Answers(AnswersBase):
    __tablename__ = "eval_answers"

    id = Column(UUID, primary_key=True, default=uuid4)

    payload = Column(JSON, nullable=False)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def to_json(self) -> dict:
        return {
            "id": str(self.id),
            "payload": self.payload,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }
