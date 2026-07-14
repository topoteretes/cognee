from uuid import uuid5, NAMESPACE_OID, UUID
from sqlalchemy import select

from cognee.modules.data.models.Data import Data
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import User


def _get_deprecated_unique_data_id(data_identifier: str, user: User) -> UUID:
    """Legacy id from data identifier + owner id (pre-tenant).

    Needed to support legacy data without tenant information.
    """
    return uuid5(NAMESPACE_OID, f"{data_identifier}{str(user.id)}")


def _get_modern_unique_data_id(data_identifier: str, user: User) -> UUID:
    """Modern id from data identifier + owner id + tenant id."""
    return uuid5(NAMESPACE_OID, f"{data_identifier}{str(user.id)}{str(user.tenant_id)}")


async def get_unique_data_id(data_identifier: str, user: User) -> UUID:
    """
    Function returns a unique UUID for data based on data identifier, user id and tenant id.
    If data with legacy ID exists, return that ID to maintain compatibility.

    Args:
        data_identifier: A way to uniquely identify data (e.g. file hash, data name, etc.)
        user: User object adding the data

    Returns:
        UUID: Unique identifier for the data
    """
    legacy_data_id = _get_deprecated_unique_data_id(data_identifier, user)

    # Check if data item with legacy_data_id exists, if so use that one, else use modern id
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        legacy_data_point = (
            await session.execute(select(Data).filter(Data.id == legacy_data_id))
        ).scalar_one_or_none()

    if legacy_data_point:
        return legacy_data_id
    return _get_modern_unique_data_id(data_identifier, user)


async def get_unique_data_ids(data_identifiers: list[str], user: User) -> list[UUID]:
    """
    Batch variant of get_unique_data_id: resolves all identifiers with a single
    query. Returns one id per identifier, preserving input order — the legacy
    id where a Data record with it exists, otherwise the modern id.
    """
    if not data_identifiers:
        return []

    legacy_ids = [
        _get_deprecated_unique_data_id(identifier, user) for identifier in data_identifiers
    ]

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        existing_legacy_ids = set(
            (await session.execute(select(Data.id).filter(Data.id.in_(legacy_ids)))).scalars().all()
        )

    return [
        legacy_id
        if legacy_id in existing_legacy_ids
        else _get_modern_unique_data_id(identifier, user)
        for identifier, legacy_id in zip(data_identifiers, legacy_ids)
    ]
