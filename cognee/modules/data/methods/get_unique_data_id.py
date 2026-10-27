from collections.abc import Iterable
from uuid import NAMESPACE_OID, UUID, uuid5

from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.methods.get_existing_data_ids import get_existing_data_ids
from cognee.modules.users.models import User


async def get_unique_data_id(
    data_identifier: str, user: User, dataset_id: UUID | None = None
) -> UUID:
    """Derive a deterministic data id from an identifier, scoped to its owners.

    The id is a UUID5 of (data_identifier, user id, tenant id) — and, when
    ``dataset_id`` is given, the dataset too. Rows are dataset-scoped, so any
    caller deriving ids for content that can live in several datasets (dlt is
    the one in-tree caller) MUST pass ``dataset_id``: without it the same
    identifier yields the same id in every dataset, which collides on the
    primary key and trips ingestion's foreign-pin guard. The dataset-free form
    exists for deriving the pre-scoping value of an id (e.g. dlt's adoption
    probe for rows ingested before ids were dataset-namespaced).

    Compatibility probe: data created before tenant-aware derivation carries a
    (identifier, user)-only UUID5 as its id; if such a row exists, that id is
    returned so existing rows keep resolving. (This "legacy" derivation
    predates — and is unrelated to — the ``Data.legacy_id`` fork-lineage
    column.)

    Args:
        data_identifier: A way to uniquely identify data (e.g. file hash, data name, etc.)
        user: User object adding the data
        dataset_id: Dataset the id is being derived for; namespaces the id so
            the same identifier in two datasets is two ids.

    Returns:
        UUID: Deterministic identifier for the data
    """
    return (await get_unique_data_ids([data_identifier], user, dataset_id))[0]


async def get_unique_data_ids(
    data_identifiers: Iterable[str],
    user: User,
    dataset_id: UUID | None = None,
    session: AsyncSession | None = None,
) -> list[UUID]:
    """``get_unique_data_id`` for many identifiers, in the same order and duplicates kept.

    The tenant-less probe runs once per batch of ids instead of once per identifier (a session
    each, 24,001 for a 12,000-row table). Pass ``session`` to probe inside one the caller already
    holds: under ``NullPool`` (the cloud pods) every session is a fresh TLS+SCRAM connection.
    """
    data_identifiers = list(data_identifiers)
    if dataset_id is not None:
        data_identifiers = [f"{dataset_id}:{identifier}" for identifier in data_identifiers]

    modern_ids = [
        uuid5(NAMESPACE_OID, f"{identifier}{user.id!s}{user.tenant_id!s}")
        for identifier in data_identifiers
    ]
    pre_tenant_ids = [
        uuid5(NAMESPACE_OID, f"{identifier}{user.id!s}") for identifier in data_identifiers
    ]

    if session is not None:
        existing_pre_tenant_ids = await get_existing_data_ids(session, pre_tenant_ids)
    else:
        async with get_relational_engine().get_async_session() as own_session:
            existing_pre_tenant_ids = await get_existing_data_ids(own_session, pre_tenant_ids)

    return [
        pre_tenant_id if pre_tenant_id in existing_pre_tenant_ids else modern_id
        for pre_tenant_id, modern_id in zip(pre_tenant_ids, modern_ids)
    ]
