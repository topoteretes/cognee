from uuid import UUID, uuid5, NAMESPACE_OID
from typing import Union
from sqlalchemy import select

from cognee.modules.data.models.Dataset import Dataset
from cognee.modules.users.models import User
from cognee.infrastructure.databases.relational import get_relational_engine


async def get_unique_dataset_id(dataset_name: Union[str, UUID], user: User) -> UUID:
    """
    Function returns a unique UUID for dataset based on dataset name, user id and tenant id.
    If dataset with legacy ID exists, return that ID to maintain compatibility.

    Args:
        dataset_name: string representing the dataset name
        user: User object adding the dataset
        tenant_id: UUID of the tenant for which dataset is being added

    Returns:
        UUID: Unique identifier for the dataset
    """

    def _get_legacy_unique_dataset_id(dataset_name: Union[str, UUID], user: User) -> UUID:
        """
        Legacy function, returns a unique UUID for dataset based on dataset name and user id.
        Needed to support legacy datasets without tenant information.
        Args:
            dataset_name: string representing the dataset name
            user: Current User object adding the dataset

        Returns:
            UUID: Unique identifier for the dataset
        """
        if isinstance(dataset_name, UUID):
            return dataset_name
        return uuid5(NAMESPACE_OID, f"{dataset_name}{str(user.id)}")

    def _get_modern_unique_dataset_id(dataset_name: Union[str, UUID], user: User) -> UUID:
        """
        Returns a unique UUID for dataset based on dataset name, user id and tenant_id.
        Args:
            dataset_name: string representing the dataset name
            user: Current User object adding the dataset
            tenant_id: UUID of the tenant for which dataset is being added

        Returns:
            UUID: Unique identifier for the dataset
        """
        if isinstance(dataset_name, UUID):
            return dataset_name
        return uuid5(NAMESPACE_OID, f"{dataset_name}{str(user.id)}{str(user.tenant_id)}")

    # Get all possible dataset_id values
    dataset_id = {
        "modern_dataset_id": _get_modern_unique_dataset_id(dataset_name=dataset_name, user=user),
        "legacy_dataset_id": _get_legacy_unique_dataset_id(dataset_name=dataset_name, user=user),
    }

    # Check if dataset with legacy_dataset_id exists, if so use that one, else use modern_dataset_id
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        legacy_dataset = (
            await session.execute(
                select(Dataset).filter(Dataset.id == dataset_id["legacy_dataset_id"])
            )
        ).scalar_one_or_none()

        if not legacy_dataset:
            return dataset_id["modern_dataset_id"]
        return dataset_id["legacy_dataset_id"]
