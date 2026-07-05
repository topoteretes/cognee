from typing import Literal
from uuid import UUID
from datetime import datetime, timezone
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational.get_async_session import get_async_session
from cognee.infrastructure.databases.relational.ModelBase import Base
from cognee.modules.governance.hash_chain import compute_row_hash

audit_event_table = sa.Table(
    "governance_audit_event",
    Base.metadata,
    sa.Column("id", sa.UUID, primary_key=True),
    sa.Column("actor_id", sa.UUID),
    sa.Column("action", sa.String),
    sa.Column("target_dataset_id", sa.UUID),
    sa.Column("outcome", sa.String),
    sa.Column("policy_id", sa.UUID),
    sa.Column("denial_reason", sa.Text),
    sa.Column("timestamp", sa.DateTime(timezone=True)),
    sa.Column("previous_hash", sa.String),
    sa.Column("row_hash", sa.String)
)

async def get_last_hash(dataset_id: UUID | None) -> str | None:
    """
    Returns the row_hash of the most recently inserted audit record
    for this dataset, ordered by timestamp DESC.
    Returns None if no records exist yet (first record in chain).
    """
    if not dataset_id:
        return None
        
    async with get_async_session() as session:
        stmt = (
            sa.select(audit_event_table.c.row_hash)
            .where(audit_event_table.c.target_dataset_id == dataset_id)
            .order_by(audit_event_table.c.timestamp.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        row = result.fetchone()
        return row[0] if row else None

async def insert_audit_event(
    actor_id: UUID | None,
    action: str,
    target_dataset_id: UUID | None,
    outcome: Literal["ALLOWED", "DENIED"],
    policy_id: UUID | None = None,
    denial_reason: str | None = None,
) -> None:
    """
    Computes the hash chain and inserts one governance_audit_event row.
    This is the only place row_hash and previous_hash are computed.
    Non-blocking: call with asyncio.create_task() from the hook.
    """
    try:
        previous_hash = await get_last_hash(target_dataset_id)
        timestamp = datetime.now(timezone.utc)
        
        fields = {
            "actor_id": str(actor_id) if actor_id else "",
            "action": action,
            "target_dataset_id": str(target_dataset_id) if target_dataset_id else "",
            "outcome": outcome,
            "timestamp": timestamp.isoformat(),
            "denial_reason": denial_reason or "",
            "previous_hash": previous_hash or "",
        }
        row_hash = compute_row_hash(fields)
        
        async with get_async_session(auto_commit=True) as session:
            stmt = sa.insert(audit_event_table).values(
                id=sa.func.gen_random_uuid() if hasattr(sa.func, 'gen_random_uuid') else None, # We rely on default or just generate in python
                actor_id=actor_id,
                action=action,
                target_dataset_id=target_dataset_id,
                outcome=outcome,
                policy_id=policy_id,
                denial_reason=denial_reason,
                timestamp=timestamp,
                previous_hash=previous_hash,
                row_hash=row_hash
            )
            import uuid
            stmt = sa.insert(audit_event_table).values(
                id=uuid.uuid4(),
                actor_id=actor_id,
                action=action,
                target_dataset_id=target_dataset_id,
                outcome=outcome,
                policy_id=policy_id,
                denial_reason=denial_reason,
                timestamp=timestamp,
                previous_hash=previous_hash,
                row_hash=row_hash
            )
            await session.execute(stmt)
    except Exception as e:
        from cognee.shared.logging_utils import get_logger
        logger = get_logger("governance.audit")
        logger.error("Failed to insert audit event: %s", e)

async def fetch_audit_events(
    dataset_id: UUID,
    outcome: Literal["ALLOWED", "DENIED"] | None = None,
) -> list[dict]:
    """
    Returns governance_audit_event rows for this dataset, ordered by
    timestamp ASC (chronological — required for hash chain verification).
    """
    async with get_async_session() as session:
        stmt = (
            sa.select(audit_event_table)
            .where(audit_event_table.c.target_dataset_id == dataset_id)
        )
        if outcome:
            stmt = stmt.where(audit_event_table.c.outcome == outcome)
        stmt = stmt.order_by(audit_event_table.c.timestamp.asc())
        
        result = await session.execute(stmt)
        return [dict(row._mapping) for row in result.all()]
