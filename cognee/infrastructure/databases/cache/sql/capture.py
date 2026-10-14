"""Atomic capture admission and retryable vector materialization for SQL caches."""

import asyncio
import hashlib
import json
import time
from uuid import NAMESPACE_URL, uuid4, uuid5

from sqlalchemy import insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .tables import cache_capture_receipts as receipts
from .tables import cache_qa_entries, cache_trace_entries


class CaptureConflictError(ValueError):
    """The same scoped client event ID was used for different content."""


def payload_hash(entry):
    canonical = json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


async def find_capture(
    adapter, *, tenant_id, user_id, dataset_id, dataset_name, session_id, client_event_id, entry
):
    """Read only the authenticated caller's receipt, without mutable dataset state.

    A name retry uses the name recorded at admission, even after rename/deletion.
    UUID callers always match the original UUID. No capture content is returned.
    """
    await adapter._ensure_initialized()
    predicate = [
        receipts.c.tenant_id == str(tenant_id or ""),
        receipts.c.user_id == str(user_id),
        receipts.c.session_id == session_id,
        receipts.c.client_event_id == str(client_event_id),
    ]
    predicate.append(
        receipts.c.dataset_id == str(dataset_id)
        if dataset_id is not None
        else receipts.c.receipt["dataset_name"].as_string() == dataset_name
    )
    async with adapter.sessionmaker() as session:
        rows = (await session.execute(select(receipts).where(*predicate))).mappings().all()
    if not rows:
        return None
    if len(rows) != 1:
        raise CaptureConflictError("Ambiguous dataset name; retry with the original dataset_id")
    existing = rows[0]
    if existing["payload_hash"] != payload_hash(entry):
        raise CaptureConflictError("client_event_id already belongs to another payload")
    return existing


async def resume_capture(adapter, existing, index_qa):
    predicate = [
        receipts.c[key] == existing[key]
        for key in ("tenant_id", "user_id", "dataset_id", "session_id", "client_event_id")
    ]
    await _finish_index(
        adapter,
        predicate,
        existing["user_id"],
        existing["session_id"],
        existing["receipt"]["entry_id"],
        index_qa,
    )
    return existing["receipt"]


async def store_capture(
    adapter,
    *,
    tenant_id,
    user_id,
    dataset_id,
    dataset_name,
    session_id,
    client_event_id,
    entry,
    build_payload,
    index_qa,
) -> dict:
    """Commit the entry and receipt together; serialize indexing across retries.

    A vector write cannot join the SQL transaction. Its stable ID makes replay an
    upsert if a process dies after indexing but before recording completion. This
    guarantees one stored vector, not exactly one external embedding API call in
    that crash window. An embedding failure remains retryable, never acknowledged.
    """
    await adapter._ensure_initialized()
    scope = {
        "tenant_id": str(tenant_id or ""),
        "user_id": str(user_id),
        "dataset_id": str(dataset_id),
        "session_id": session_id,
        "client_event_id": str(client_event_id),
    }
    predicate = [receipts.c[key] == value for key, value in scope.items()]
    digest = payload_hash(entry)
    entry_id = str(uuid5(NAMESPACE_URL, json.dumps(scope, sort_keys=True)))
    receipt = {
        "status": "session_stored",
        "dataset_name": dataset_name,
        "dataset_id": str(dataset_id),
        "session_ids": [session_id],
        "entry_type": entry["type"],
        "entry_id": entry_id,
        "client_event_id": str(client_event_id),
        "payload_hash": digest,
    }
    table = cache_qa_entries if entry["type"] == "qa" else cache_trace_entries
    dialect_insert = pg_insert if adapter._is_postgres else sqlite_insert
    async with adapter.sessionmaker() as session, session.begin():
        # Match normal cache writers' lock order before touching TTL rows.
        await adapter._lock_session_writes(session, table, user_id, session_id)
        admitted = await session.execute(
            dialect_insert(receipts)
            .values(
                **scope,
                payload_hash=digest,
                receipt=receipt,
                indexed=entry["type"] != "qa",
            )
            .on_conflict_do_nothing()
            .returning(receipts.c.client_event_id)
        )
        if admitted.scalar_one_or_none() is not None:
            payload = await build_payload(entry_id)
            values = {
                "user_id": user_id,
                "session_id": session_id,
                "payload": payload,
                "expires_at": adapter._session_expiry(),
            }
            if entry["type"] == "qa":
                values["qa_id"] = entry_id
            await session.execute(insert(table).values(**values))
            await adapter._refresh_session_ttl(session, table, user_id, session_id)
        else:
            existing = (await session.execute(select(receipts).where(*predicate))).mappings().one()
            if existing["payload_hash"] != digest:
                raise CaptureConflictError("client_event_id already belongs to another payload")
            receipt = existing["receipt"]

    if entry["type"] == "qa":
        await _finish_index(adapter, predicate, user_id, session_id, receipt["entry_id"], index_qa)
    await adapter._maybe_purge_expired()
    return receipt


async def _finish_index(adapter, predicate, user_id, session_id, entry_id, index_qa):
    # No database lock spans the embedding API: usage accounting can write to
    # the same SQL database and unrelated captures continue while indexing runs.
    owner = str(uuid4())
    deadline = time.monotonic() + 65
    while True:
        async with adapter.sessionmaker() as session, session.begin():
            claim = await session.execute(
                update(receipts)
                .where(
                    *predicate,
                    receipts.c.indexed.is_(False),
                    receipts.c.index_lease_until <= time.time(),
                )
                .values(index_owner=owner, index_lease_until=time.time() + 120)
                .returning(receipts.c.client_event_id)
            )
            acquired = claim.scalar_one_or_none() is not None
            indexed = (
                await session.execute(select(receipts.c.indexed).where(*predicate))
            ).scalar_one()
        if indexed:
            return
        if acquired:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("Capture indexing is in progress; retry this event ID")
        await asyncio.sleep(0.05)
    try:
        async with adapter.sessionmaker() as session:
            cached = (
                await session.execute(
                    select(cache_qa_entries.c.payload).where(
                        cache_qa_entries.c.user_id == user_id,
                        cache_qa_entries.c.session_id == session_id,
                        cache_qa_entries.c.qa_id == entry_id,
                        adapter._not_expired(cache_qa_entries),
                    )
                )
            ).scalar_one_or_none()
        # Forget/TTL wins over delayed replay; receipt tombstones don't retain text.
        if cached is not None:
            await asyncio.wait_for(index_qa(entry_id, cached), timeout=60)
        async with adapter.sessionmaker() as session, session.begin():
            await session.execute(
                update(receipts)
                .where(*predicate, receipts.c.index_owner == owner)
                .values(indexed=True, index_owner=None, index_lease_until=0)
            )
    except BaseException:
        async with adapter.sessionmaker() as session, session.begin():
            await session.execute(
                update(receipts)
                .where(*predicate, receipts.c.index_owner == owner)
                .values(index_owner=None, index_lease_until=0)
            )
        raise
