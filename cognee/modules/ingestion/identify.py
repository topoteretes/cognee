from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models.Data import Data
from cognee.modules.users.models import User

from .data_types import IngestionData
from .node_set_identity import UNSCOPED, node_set_matches


def content_hash_predicates(content_hash: str, user: User, dataset_id: UUID) -> tuple:
    """The dedup predicate every ingestion lookup shares.

    Dedup is scoped to (dataset, owner, tenant): in a shared multi-writer
    dataset, two users adding the same bytes stay two rows — matching the
    old per-user identity semantics and keeping owner_id checks meaningful.
    ``identify``, ``identify_data`` and ``identify_many`` must agree on which
    row wins for a given hash, so they all build their filter here. The node
    set is the one scope SQL cannot compare (a JSON list), so callers that
    know it narrow the rows these predicates return with
    :func:`node_set_matches` — see ``identify_data_by_hash``.
    """
    tenant_filter = Data.tenant_id == user.tenant_id if user.tenant_id else Data.tenant_id.is_(None)
    return (
        Data.dataset_id == dataset_id,
        Data.content_hash == content_hash,
        Data.owner_id == user.id,
        tenant_filter,
    )


async def identify(
    data: IngestionData, user: User, dataset_id: UUID, node_set: Any = UNSCOPED
) -> UUID | None:
    """Resolve the existing ``Data`` row for this content in this dataset.

    Dedup is a lookup, not an identity: a hit returns the id of the row that
    already holds this content in this dataset for this owner/tenant; a miss
    returns ``None`` and the caller mints a fresh random id. The id itself
    carries no content information, so it stays stable when a document's
    content is updated. Rows are dataset-scoped (the startup migration
    backfills pre-refactor rows), so one scoped probe is sufficient. Pass
    ``node_set`` to match only the row stored under that scope (see
    ``identify_data_by_hash``).
    """
    row = await identify_data(data, user, dataset_id, node_set=node_set)
    return row.id if row is not None else None


async def identify_data(
    data: IngestionData,
    user: User,
    dataset_id: UUID,
    session: AsyncSession | None = None,
    node_set: Any = UNSCOPED,
) -> Data | None:
    """:func:`identify`, but return the whole row instead of its id.

    Callers that need the row's columns right after resolving it — the
    incremental pipeline wrapper reads ``pipeline_status`` — used to pay a
    second session (and, under ``NullPool``, a second TLS+SCRAM connection)
    to fetch by the id ``identify`` had just returned. One scoped probe
    returns the same row. Pass ``session`` to run inside a session the
    caller already holds (e.g. the one that goes on to update the row).
    """
    return await identify_data_by_hash(
        await data.aget_identifier(), user, dataset_id, session=session, node_set=node_set
    )


async def identify_data_by_hash(
    content_hash: str,
    user: User,
    dataset_id: UUID,
    session: AsyncSession | None = None,
    node_set: Any = UNSCOPED,
) -> Data | None:
    """:func:`identify_data` for callers that already hold the content hash.

    Ingestion computes the hash while the payload's bytes are in hand, so the
    dedup lookup does not need an ``IngestionData`` adapter around it — the
    hash IS the identity, scoped by the shared predicates.

    ``node_set`` narrows the match to the row stored under that scope (the
    same set of tags; ``None`` is "no node set"): the same content under
    another node set is a different data item (see
    :mod:`cognee.modules.ingestion.node_set_identity`). Left at the default
    the lookup is content-only and returns the first row, as before — for
    callers that do not know the node set.
    """
    predicates = content_hash_predicates(content_hash, user, dataset_id)

    async def _lookup(active_session: AsyncSession) -> Data | None:
        if node_set is UNSCOPED:
            return (
                await active_session.execute(select(Data).filter(*predicates).limit(1))
            ).scalar_one_or_none()
        rows = (await active_session.execute(select(Data).filter(*predicates))).scalars()
        return next((row for row in rows if node_set_matches(row.node_set, node_set)), None)

    if session is not None:
        return await _lookup(session)

    async with get_relational_engine().get_async_session() as own_session:
        return await _lookup(own_session)
