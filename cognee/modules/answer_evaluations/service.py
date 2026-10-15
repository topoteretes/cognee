"""Relational transactions own admission; encrypted warehouse is a reconciled replica."""

import secrets
import time

from sqlalchemy import delete, select, update

from .models import EvaluationRun, EvaluationScope, EvaluationTrace
from .privacy import canonical, sanitize

RETENTION = 30 * 86400
MAX_TRACES = 100
MAX_RECEIPTS = 10000


class EvaluationError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


async def lock_scope(session, identity, owner, dataset, *, create=False):
    # UPDATE obtains a transaction-held write lock on SQLite and PostgreSQL.
    await session.execute(
        update(EvaluationScope)
        .where(EvaluationScope.id == identity)
        .values(revision=EvaluationScope.revision + 1)
    )
    scope = await session.get(EvaluationScope, identity, populate_existing=True)
    if scope is None and create:
        scope = EvaluationScope(
            id=identity,
            owner_id=owner,
            dataset_id=dataset,
            enabled=True,
            revision=1,
            synced_revision=-1,
            last_run_at=0,
        )
        session.add(scope)
        await session.flush()  # Racing creation fails atomically; caller retries.
    return scope


async def prune(session, scope, now):
    await session.execute(
        delete(EvaluationTrace).where(
            EvaluationTrace.scope_id == scope.id, EvaluationTrace.expires_at <= now
        )
    )
    await session.execute(
        delete(EvaluationRun).where(
            EvaluationRun.scope_id == scope.id, EvaluationRun.created_at <= now - RETENTION
        )
    )


async def ingest(session, protection, owner, trace, *, now=None):
    now = time.time() if now is None else now
    dataset = str(trace.dataset_id)
    identity = protection.identity(owner, dataset)
    scope = await lock_scope(session, identity, owner, dataset, create=True)
    if not scope.enabled:
        raise EvaluationError(403, "Evaluation capture revoked; enable it explicitly")
    await prune(session, scope, now)
    key = protection.identity(identity, str(trace.event_id))
    # Compare original canonical payload via keyed digest. Never persist originals.
    body = trace.model_dump(mode="json")
    fingerprint = protection.identity(body)
    existing = await session.get(EvaluationTrace, key)
    if existing:
        if existing.fingerprint != fingerprint:
            raise EvaluationError(409, "Event ID already used with different content")
        return {"status": "accepted", "receipt": key}
    rows = list(
        (
            await session.scalars(
                select(EvaluationTrace).where(EvaluationTrace.scope_id == identity)
            )
        ).all()
    )
    if len(rows) >= MAX_RECEIPTS:
        raise EvaluationError(429, "Evaluation receipt capacity reached")
    safe = sanitize(body)
    safe.pop("event_id")
    safe.pop("dataset_id")
    row = EvaluationTrace(
        id=key,
        scope_id=identity,
        fingerprint=fingerprint,
        created_at=now,
        expires_at=trace.captured_at.timestamp() + RETENTION,
        priority=secrets.randbits(53) / 2**53,
        ciphertext=protection.encrypt(safe),
    )
    session.add(row)
    # Bottom-k random-priority sample; recent expiry makes this a rolling pool.
    candidates = sorted([r for r in rows if r.ciphertext] + [row], key=lambda r: r.priority)
    for evicted in candidates[MAX_TRACES:]:
        evicted.ciphertext = None
    return {"status": "accepted", "receipt": key}


async def erase(session, scope, receipt=None):
    if receipt:
        existing = await session.get(EvaluationTrace, receipt)
        if existing is None or existing.scope_id != scope.id:
            raise EvaluationError(404, "Evaluation trace not found")
    query = update(EvaluationTrace).where(EvaluationTrace.scope_id == scope.id)
    if receipt:
        query = query.where(EvaluationTrace.id == receipt)
    await session.execute(query.values(ciphertext=None))
    # Purge evaluation associations too; historical private identifiers aren't immortal.
    await session.execute(delete(EvaluationRun).where(EvaluationRun.scope_id == scope.id))
    return {"status": "deletion_pending", "warehouse_pending": True}


async def summary(session, identity, now=None):
    from .schema import score

    now = time.time() if now is None else now
    row = await session.scalar(
        select(EvaluationRun)
        .where(EvaluationRun.scope_id == identity, EvaluationRun.status == "complete")
        .order_by(EvaluationRun.created_at.desc())
        .limit(1)
    )
    base = {
        "schema": 2,
        "scope": "personal",
        "source": "live",
        "score": 47,
        "score_kind": "normalized_net_benefit",
        "score_floor": 47,
        "score_ceiling": 100,
        "status": "pending",
    }
    if row is None:
        return base
    from datetime import datetime, timezone

    result = row.result
    return {
        **base,
        **result["summary"],
        "run_id": row.id,
        "evaluated_at": datetime.fromtimestamp(row.created_at, timezone.utc).isoformat(),
        "status": "stale" if now - row.created_at > 48 * 3600 else "ready",
        "score": score(
            result["summary"]["better"], result["summary"]["worse"], result["summary"]["pairs"]
        ),
    }
