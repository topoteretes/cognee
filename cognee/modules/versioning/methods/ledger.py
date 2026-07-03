"""CRUD for the version-op write-ahead ledger.

The write-ahead rule: the inverse payload is committed here *before* the
destructive graph/vector operation executes. A crash between the two leaves a
``CAPTURED`` row whose payload can be replayed (restore is idempotent) or
inspected — never a destroyed artifact without its inverse.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import attributes as orm_attributes

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.storage.utils import JSONEncoder
from cognee.modules.versioning.methods.inverse import INVERSE_PAYLOAD_VERSION
from cognee.modules.versioning.methods.timeline import ensure_utc
from cognee.modules.versioning.models import VersionOp, VersionOpStatus

DEFAULT_RETENTION_DAYS = 30


def get_retention_days() -> int:
    return int(os.environ.get("VERSION_RETENTION_DAYS", DEFAULT_RETENTION_DAYS))


def sanitize_payload(payload: Any) -> Any:
    """Normalize a payload to JSON-native types (datetimes -> ISO strings).

    Vector-row payloads read back from storage can contain datetime objects;
    the relational JSON column serializer requires JSON-native values.
    """
    return json.loads(json.dumps(payload, cls=JSONEncoder))


async def create_version_op(
    dataset_id: UUID,
    op_type: str,
    *,
    steps: Optional[List[Dict[str, Any]]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> UUID:
    """Commit a new ledger row (status CAPTURED) and return its id."""
    now = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "version": INVERSE_PAYLOAD_VERSION,
        "steps": sanitize_payload(steps or []),
    }
    if extra:
        payload.update(sanitize_payload(extra))

    op = VersionOp(
        dataset_id=dataset_id,
        op_type=op_type,
        status=VersionOpStatus.CAPTURED,
        payload=payload,
        created_at=now,
        expires_at=now + timedelta(days=get_retention_days()),
    )

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(op)
        await session.commit()
        return op.id


async def append_op_step(op_id: UUID, step: Dict[str, Any]) -> None:
    """Append one captured inverse step to an op and commit before it executes."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        op = (
            (await session.execute(select(VersionOp).where(VersionOp.id == op_id)))
            .scalars()
            .first()
        )
        if op is None:
            raise ValueError(f"Version op {op_id} not found.")
        op.payload["steps"].append(sanitize_payload(step))
        orm_attributes.flag_modified(op, "payload")
        await session.commit()


async def set_op_status(op_id: UUID, status: str) -> None:
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        op = (
            (await session.execute(select(VersionOp).where(VersionOp.id == op_id)))
            .scalars()
            .first()
        )
        if op is None:
            raise ValueError(f"Version op {op_id} not found.")
        op.status = status
        await session.commit()


async def get_version_op(op_id: UUID) -> VersionOp:
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        op = (
            (await session.execute(select(VersionOp).where(VersionOp.id == op_id)))
            .scalars()
            .first()
        )
    if op is None:
        raise ValueError(f"Version op {op_id} not found.")
    return op


def assert_within_retention(op: VersionOp) -> None:
    """Raise when the op's retention window has lapsed (payload may be pruned)."""
    if op.expires_at is None:
        return
    if datetime.now(timezone.utc) > ensure_utc(op.expires_at):
        raise ValueError(
            f"Version op {op.id} is outside the retention window "
            f"(expired {op.expires_at}); it can no longer be undone."
        )
