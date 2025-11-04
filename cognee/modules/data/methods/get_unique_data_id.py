from uuid import uuid5, NAMESPACE_OID, UUID
from sqlalchemy import select

from cognee.modules.data.models.Data import Data
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import User


async def get_unique_data_id(data_identifier: str, user: User) -> UUID:
    """
    Function returns a unique UUID for data based on data identifier, user id and tenant id.
    If data with legacy ID exists, return that ID to maintain compatibility.

    Args:
        data_identifier: A way to uniquely identify data (e.g. file hash, data name, etc.)
        user: User object adding the data
        tenant_id: UUID of the tenant for which data is being added

    Returns:
        UUID: Unique identifier for the data
    """

    def _get_deprecated_unique_data_id(data_identifier: str, user: User) -> UUID:
        """
        Deprecated function, returns a unique UUID for data based on data identifier and user id.
        Needed to support legacy data without tenant information.
        Args:
            data_identifier: A way to uniquely identify data (e.g. file hash, data name, etc.)
            user: User object adding the data

        Returns:
            UUID: Unique identifier for the data
        """
        # return UUID hash of file contents + owner id + tenant_id
        return uuid5(NAMESPACE_OID, f"{data_identifier}{str(user.id)}")

    def _get_modern_unique_data_id(data_identifier: str, user: User) -> UUID:
        """
        Function returns a unique UUID for data based on data identifier, user id and tenant id.
        Args:
            data_identifier: A way to uniquely identify data (e.g. file hash, data name, etc.)
            user: User object adding the data
            tenant_id: UUID of the tenant for which data is being added

        Returns:
            UUID: Unique identifier for the data
        """
        # return UUID hash of file contents + owner id + tenant_id
        return uuid5(NAMESPACE_OID, f"{data_identifier}{str(user.id)}{str(user.tenant_id)}")

    # Get all possible data_id values
    data_id = {
        "modern_data_id": _get_modern_unique_data_id(data_identifier=data_identifier, user=user),
        "legacy_data_id": _get_deprecated_unique_data_id(
            data_identifier=data_identifier, user=user
        ),
    }

    # Check if data item with legacy_data_id exists, if so use that one, else use modern_data_id
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        legacy_data_point = (
            await session.execute(select(Data).filter(Data.id == data_id["legacy_data_id"]))
        ).scalar_one_or_none()

        if not legacy_data_point:
            return data_id["modern_data_id"]
        return data_id["legacy_data_id"]
