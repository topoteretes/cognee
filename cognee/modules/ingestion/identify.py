from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models.Data import Data
from cognee.modules.users.models import User

from .data_types import IngestionData


def content_hash_predicates(content_hash: str, user: User, dataset_id: UUID) -> tuple:
    """The dedup predicate every ingestion lookup shares.

    Dedup is scoped to (dataset, owner, tenant): in a shared multi-writer
    dataset, two users adding the same bytes stay two rows — matching the
    old per-user identity semantics and keeping owner_id checks meaningful.
    ``identify``, ``identify_data`` and ``identify_many`` must agree on which
    row wins for a given hash, so they all build their filter here.
    """
    tenant_filter = Data.tenant_id == user.tenant_id if user.tenant_id else Data.tenant_id.is_(None)
    return (
        Data.dataset_id == dataset_id,
        Data.content_hash == content_hash,
        Data.owner_id == user.id,
        tenant_filter,
    )


async def identify(data: IngestionData, user: User, dataset_id: UUID) -> Optional[UUID]:
    """Resolve the existing ``Data`` row for this content in this dataset.

    Dedup is a lookup, not an identity: a hit returns the id of the row that
    already holds this content in this dataset for this owner/tenant; a miss
    returns ``None`` and the caller mints a fresh random id. The id itself
    carries no content information, so it stays stable when a document's
    content is updated. Rows are dataset-scoped (the startup migration
    backfills pre-refactor rows), so one scoped probe is sufficient.
    """
    content_hash: str = await data.aget_identifier()
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        return (
            await session.execute(
                select(Data.id)
                .filter(*content_hash_predicates(content_hash, user, dataset_id))
                .limit(1)
            )
        ).scalar_one_or_none()


async def identify_data(
    data: IngestionData,
    user: User,
    dataset_id: UUID,
    session: Optional[AsyncSession] = None,
) -> Optional[Data]:
    """:func:`identify`, but return the whole row instead of its id.

    Callers that need the row's columns right after resolving it — the
    incremental pipeline wrapper reads ``pipeline_status`` — used to pay a
    second session (and, under ``NullPool``, a second TLS+SCRAM connection)
    to fetch by the id ``identify`` had just returned. One scoped probe
    returns the same row. Pass ``session`` to run inside a session the
    caller already holds (e.g. the one that goes on to update the row).
    """
    content_hash: str = await data.aget_identifier()
    predicates = content_hash_predicates(content_hash, user, dataset_id)

    async def _lookup(active_session: AsyncSession) -> Optional[Data]:
        return (
            await active_session.execute(select(Data).filter(*predicates).limit(1))
        ).scalar_one_or_none()

    if session is not None:
        return await _lookup(session)

    async with get_relational_engine().get_async_session() as own_session:
        return await _lookup(own_session)
