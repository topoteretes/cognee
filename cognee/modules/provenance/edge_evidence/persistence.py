"""Single-statement bulk persistence for edge evidence."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from cognee.infrastructure.databases.relational import get_relational_engine

from .buffer import ProvenanceBatch, ProvenanceBuffer
from .models import ProvenanceEdgeEvidence


async def persist_provenance_batch(batch: ProvenanceBatch) -> int:
    """Insert a batch idempotently with one executemany statement and commit."""
    if not batch.evidence_rows:
        return 0

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        bind = await session.connection()
        dialect = getattr(getattr(bind, "dialect", None), "name", "")
        rows = list(batch.evidence_rows)

        if dialect in {"sqlite", "postgresql"}:
            insert = sqlite_insert if dialect == "sqlite" else pg_insert
            statement = insert(ProvenanceEdgeEvidence).on_conflict_do_nothing(index_elements=["id"])
            await session.execute(statement, rows)
        else:
            ids = [row["id"] for row in rows]
            existing_ids = set(
                (
                    await session.execute(
                        select(ProvenanceEdgeEvidence.id).where(ProvenanceEdgeEvidence.id.in_(ids))
                    )
                )
                .scalars()
                .all()
            )
            session.add_all(
                ProvenanceEdgeEvidence(**row) for row in rows if row["id"] not in existing_ids
            )
        await session.commit()
    return len(rows)


async def flush_context_provenance(ctx: Any) -> int:
    """Flush a context's pending edge evidence in one transaction."""
    buffer = getattr(ctx, "provenance_buffer", None)
    if not isinstance(buffer, ProvenanceBuffer):
        return 0
    batch = buffer.snapshot()
    persisted = await persist_provenance_batch(batch)
    buffer.mark_persisted(batch)
    return persisted
