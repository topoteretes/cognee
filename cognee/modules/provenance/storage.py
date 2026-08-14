"""Async repository for the provenance ledger over the shared relational DB.

Hash-chain appends must be race-free under async SQLAlchemy on SQLite AND
Postgres, possibly multi-process. Three layers of defense:

1. A process-local per-event-loop ``asyncio.Lock`` (cheap serialization of the
   common case).
2. A Postgres advisory transaction lock (cross-process serialization; a no-op
   on SQLite, which is single-writer anyway).
3. The partial unique index on ``sequence_id`` plus a bounded retry — the
   correctness backstop: a lost race is an ``IntegrityError``, never a forked
   chain.

Unlike semantica, a failing ``get_chain_head`` is NOT swallowed (its silent
"restart at sequence_id=1" failure mode is removed) — the manager's
graceful-degradation catch lives one level up and writes nothing on failure.
"""

import asyncio
from typing import AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple
from weakref import WeakKeyDictionary

from sqlalchemy import delete, distinct, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import get_async_session

from .models import ProvenanceEntry, ProvenanceEntryRow

# Constant app-level advisory lock id for the provenance chain (Postgres only).
_PG_ADVISORY_KEY = 0x_C09_EE01

# asyncio.Lock binds to the event loop it is first awaited on, so a single
# module-level lock breaks when several loops exist (each pytest-asyncio test
# gets a fresh loop). Keep one lock per loop instead.
_chain_locks: "WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = WeakKeyDictionary()


def _get_chain_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _chain_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _chain_locks[loop] = lock
    return lock


async def _acquire_chain_write_lock(session: AsyncSession) -> None:
    if session.bind.dialect.name == "postgresql":
        # Released automatically at COMMIT/ROLLBACK; serializes across processes.
        await session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _PG_ADVISORY_KEY})


async def get_chain_head(session: AsyncSession) -> Optional[Tuple[int, Optional[str]]]:
    """Return (sequence_id, checksum) of the newest chained entry, or None if empty."""
    row = (
        await session.execute(
            select(ProvenanceEntryRow.sequence_id, ProvenanceEntryRow.checksum)
            .where(ProvenanceEntryRow.sequence_id.is_not(None))
            .order_by(ProvenanceEntryRow.sequence_id.desc())
            .limit(1)
        )
    ).first()
    return (row[0], row[1]) if row else None


_WriteFn = Callable[[AsyncSession, int, Optional[str]], Awaitable[Optional[ProvenanceEntry]]]


async def append_chained_many(write_fns: List[_WriteFn]) -> List[Optional[ProvenanceEntry]]:
    """Run many logical chained writes as ONE transaction claiming consecutive slots.

    This is the batching seam: one lock acquisition, one head read, and one
    COMMIT for the whole list — instead of a transaction per ledger entry —
    so pipeline-scale writes (thousands of entries per cognify batch) do not
    serialize a round-trip each behind the chain locks.

    Each ``write_fn(session, next_seq, prev_checksum)`` performs ALL statements
    for one logical track/invalidate call. A write_fn that claims its slot
    returns an entry with ``sequence_id == next_seq``, advancing the head for
    the next write_fn; a chain-preserving no-op (relationship/chunk re-track)
    returns the stored entry and consumes nothing. A lost cross-process race on
    the ``ux_provenance_sequence_id`` index rolls back the whole batch and
    retries it with a fresh head; the chain can never fork.
    """
    if not write_fns:
        return []
    async with _get_chain_lock():
        for attempt in range(5):
            async with get_async_session() as session:
                try:
                    async with session.begin():
                        await _acquire_chain_write_lock(session)
                        head = await get_chain_head(session)
                        next_seq = head[0] + 1 if head else 1
                        prev_checksum = head[1] if head else None
                        results: List[Optional[ProvenanceEntry]] = []
                        for write_fn in write_fns:
                            entry = await write_fn(session, next_seq, prev_checksum)
                            if entry is not None and entry.sequence_id == next_seq:
                                prev_checksum = entry.checksum
                                next_seq += 1
                            results.append(entry)
                        return results
                except IntegrityError:
                    # Lost a cross-process race on ux_provenance_sequence_id.
                    if attempt == 4:
                        raise
                    await asyncio.sleep(0.05 * (attempt + 1))


async def append_chained(write_fn: _WriteFn) -> ProvenanceEntry:
    """Run one logical chained write as a single transaction claiming one slot."""
    return (await append_chained_many([write_fn]))[0]


async def retrieve_row(session: AsyncSession, entity_id: str) -> Optional[ProvenanceEntryRow]:
    return await session.get(ProvenanceEntryRow, entity_id)


async def find_free_archive_id(session: AsyncSession, entity_id: str, last_updated) -> str:
    """Free ``"{entity_id}:v:{last_updated}"`` key (``:1``, ``:2``… on collision)."""
    base_id = f"{entity_id}:v:{last_updated}"
    candidate = base_id
    counter = 1
    while await retrieve_row(session, candidate) is not None:
        candidate = f"{base_id}:{counter}"
        counter += 1
    return candidate


async def retrieve(entity_id: str) -> Optional[ProvenanceEntry]:
    async with get_async_session() as session:
        row = await retrieve_row(session, entity_id)
        return ProvenanceEntry.from_row(row) if row else None


async def retrieve_many(entity_ids: List[str]) -> List[Optional[ProvenanceEntry]]:
    """Fetch several entries in one session, preserving input order (None gaps)."""
    if not entity_ids:
        return []
    async with get_async_session() as session:
        rows = (
            await session.execute(
                select(ProvenanceEntryRow).where(ProvenanceEntryRow.entity_id.in_(entity_ids))
            )
        ).scalars()
        by_id = {row.entity_id: ProvenanceEntry.from_row(row) for row in rows}
    return [by_id.get(entity_id) for entity_id in entity_ids]


async def retrieve_all() -> List[ProvenanceEntry]:
    """Materialize the whole ledger. Small ledgers / tests only — the audit
    walks (``verify_chain``/``check``) use the paged iterators below."""
    async with get_async_session() as session:
        rows = (await session.execute(select(ProvenanceEntryRow))).scalars()
        return [ProvenanceEntry.from_row(row) for row in rows]


async def iter_chained(page_size: int = 1000) -> AsyncIterator[ProvenanceEntry]:
    """Stream chained entries in ``sequence_id`` order, one page at a time.

    Keyset pagination on the unique partial index — constant memory no matter
    how large the append-only ledger has grown.
    """
    last_sequence_id: Optional[int] = None
    async with get_async_session() as session:
        while True:
            statement = select(ProvenanceEntryRow).where(
                ProvenanceEntryRow.sequence_id.is_not(None)
            )
            if last_sequence_id is not None:
                statement = statement.where(ProvenanceEntryRow.sequence_id > last_sequence_id)
            statement = statement.order_by(ProvenanceEntryRow.sequence_id.asc()).limit(page_size)
            rows = (await session.execute(statement)).scalars().all()
            if not rows:
                return
            for row in rows:
                yield ProvenanceEntry.from_row(row)
            last_sequence_id = rows[-1].sequence_id


async def iter_all(page_size: int = 1000) -> AsyncIterator[ProvenanceEntry]:
    """Stream every ledger entry (keyset pagination on the PK), constant memory."""
    last_entity_id: Optional[str] = None
    async with get_async_session() as session:
        while True:
            statement = select(ProvenanceEntryRow)
            if last_entity_id is not None:
                statement = statement.where(ProvenanceEntryRow.entity_id > last_entity_id)
            statement = statement.order_by(ProvenanceEntryRow.entity_id.asc()).limit(page_size)
            rows = (await session.execute(statement)).scalars().all()
            if not rows:
                return
            for row in rows:
                yield ProvenanceEntry.from_row(row)
            last_entity_id = rows[-1].entity_id


async def retrieve_all_ids() -> set:
    """Every entity_id in the ledger (id column only — no row materialization)."""
    async with get_async_session() as session:
        result = await session.execute(select(ProvenanceEntryRow.entity_id))
        return set(result.scalars())


async def retrieve_all_activity_ids() -> set:
    """Every distinct non-null activity_id in the ledger."""
    async with get_async_session() as session:
        result = await session.execute(
            select(distinct(ProvenanceEntryRow.activity_id)).where(
                ProvenanceEntryRow.activity_id.is_not(None)
            )
        )
        return set(result.scalars())


async def aggregate_statistics() -> Dict:
    """DB-side ledger statistics — no client-side full-table load."""
    async with get_async_session() as session:
        total = (
            await session.execute(select(func.count()).select_from(ProvenanceEntryRow))
        ).scalar()
        type_rows = await session.execute(
            select(ProvenanceEntryRow.entity_type, func.count()).group_by(
                ProvenanceEntryRow.entity_type
            )
        )
        unique_sources = (
            await session.execute(
                select(func.count(distinct(ProvenanceEntryRow.source_document))).where(
                    ProvenanceEntryRow.source_document.is_not(None),
                    ProvenanceEntryRow.source_document != "",
                )
            )
        ).scalar()
        invalidated_count = (
            await session.execute(
                select(func.count()).where(ProvenanceEntryRow.invalidated.is_(True))
            )
        ).scalar()
    return {
        "total_entries": total or 0,
        "entity_types": {entity_type: count for entity_type, count in type_rows},
        "unique_sources": unique_sources or 0,
        "invalidated_count": invalidated_count or 0,
    }


async def trace_lineage(entity_id: str, max_depth: Optional[int] = None) -> List[ProvenanceEntry]:
    """Batched BFS over parent_entity_id + used_entities links, seed at depth 0.

    One SELECT per level (``WHERE entity_id IN (frontier)``), output in BFS
    order. Matching semantica's depth semantics: with ``max_depth`` set, only
    nodes at depth < max_depth are visited.
    """
    lineage: List[ProvenanceEntry] = []
    visited: set = set()
    frontier: List[str] = [entity_id]
    depth = 0

    async with get_async_session() as session:
        while frontier:
            if max_depth is not None and depth >= max_depth:
                break

            rows = (
                await session.execute(
                    select(ProvenanceEntryRow).where(ProvenanceEntryRow.entity_id.in_(frontier))
                )
            ).scalars()
            by_id = {row.entity_id: ProvenanceEntry.from_row(row) for row in rows}

            next_frontier: List[str] = []
            for current_id in frontier:
                if current_id in visited:
                    continue
                visited.add(current_id)
                entry = by_id.get(current_id)
                if entry is None:
                    continue
                lineage.append(entry)
                if entry.parent_entity_id and entry.parent_entity_id not in visited:
                    next_frontier.append(entry.parent_entity_id)
                for used_id in entry.used_entities:
                    if used_id not in visited and used_id not in next_frontier:
                        next_frontier.append(used_id)

            frontier = next_frontier
            depth += 1

    return lineage


async def clear() -> int:
    """Delete every ledger row (dev/test teardown only). Returns rows removed."""
    async with get_async_session() as session:
        async with session.begin():
            result = await session.execute(delete(ProvenanceEntryRow))
            return result.rowcount or 0
