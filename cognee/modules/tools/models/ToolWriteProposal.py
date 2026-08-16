from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from cognee.infrastructure.databases.relational.ModelBase import Base


class ToolWriteProposal(Base):
    """A proposed correction (UPDATE) to an authorized external database.

    Write-back is approval-gated: proposals are drafted from evidence (a
    detected contradiction, feedback, or an explicit instruction), stored
    here with a dry-run affected-row estimate, and touch the source database
    only when an explicit apply call executes them — mirroring the
    skill-improvement proposal lifecycle.

    Status flow: ``proposed`` → ``applied`` | ``rejected`` | ``failed``
    (failed = apply ran but rolled back: affected-row cap exceeded, estimate
    mismatch, or a database error).
    """

    __tablename__ = "tool_write_proposals"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    user_id: Mapped[UUID] = mapped_column(Uuid, nullable=False, index=True)
    connection_name: Mapped[str] = mapped_column(String, nullable=False)

    status: Mapped[str] = mapped_column(String, nullable=False, default="proposed", index=True)

    # What the correction is meant to achieve, in natural language.
    instruction: Mapped[str] = mapped_column(Text, nullable=False)
    # Where the inaccuracy signal came from (e.g. a contradicts edge's
    # first_fact/second_fact/reason/confidence). Never secret material.
    evidence: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    sql: Mapped[str] = mapped_column(Text, nullable=False)
    target_table: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Dry-run rowcount captured at proposal time (UPDATE executed and rolled
    # back). Apply re-checks the actual rowcount against this.
    estimated_rows: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    applied_rows: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
