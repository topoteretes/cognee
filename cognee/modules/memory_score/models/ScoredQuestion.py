"""SQLAlchemy model for one scored question inside a memory-score run."""

from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, String, Text, UUID

from cognee.infrastructure.databases.relational import Base


class ScoredQuestion(Base):
    """A single question answered and judged during a memory-score run.

    ``source`` is ``"synthetic"`` or ``"real"`` and decides which judge
    columns are populated — they are mutually exclusive on purpose:

    * synthetic (generated from a real chunk, has a golden answer)
      -> ``expected_answer`` + ``score``, ``grounded`` stays NULL.
    * real (a question the tenant actually asked, no golden answer)
      -> ``grounded`` boolean only, ``score``/``expected_answer`` stay NULL.

    ``source_query_id`` links a real question back to its
    ``queries.id`` row; it is NULL for synthetic questions.
    """

    __tablename__ = "memory_score_questions"

    id = Column(UUID, primary_key=True, default=uuid4)

    run_id = Column(UUID, index=True)

    topic = Column(String, nullable=True)
    # "synthetic" | "real" — treated as an enum at the app layer.
    source = Column(String)

    text = Column(Text)
    expected_answer = Column(Text, nullable=True)
    answer = Column(Text, nullable=True)

    # Correctness score, synthetic questions only.
    score = Column(Float, nullable=True)
    # Groundedness boolean, real questions only.
    grounded = Column(Boolean, nullable=True)

    reason = Column(Text, nullable=True)
    source_query_id = Column(UUID, nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
