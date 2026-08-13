from typing import Optional
from uuid import uuid5, NAMESPACE_OID, UUID

from sqlalchemy import select

from cognee.modules.data.models.Data import Data
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import User


async def get_unique_data_id(
    data_identifier: str, user: User, dataset_id: Optional[UUID] = None
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
    if dataset_id is not None:
        data_identifier = f"{dataset_id}:{data_identifier}"

    modern_data_id = uuid5(NAMESPACE_OID, f"{data_identifier}{str(user.id)}{str(user.tenant_id)}")
    pre_tenant_data_id = uuid5(NAMESPACE_OID, f"{data_identifier}{str(user.id)}")

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        pre_tenant_data_point = (
            await session.execute(select(Data).filter(Data.id == pre_tenant_data_id))
        ).scalar_one_or_none()

    return pre_tenant_data_id if pre_tenant_data_point else modern_data_id
