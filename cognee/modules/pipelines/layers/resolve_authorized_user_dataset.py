from uuid import UUID
from typing import Optional

from cognee.modules.users.models import User
from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
    resolve_authorized_user_datasets,
)


async def resolve_authorized_user_dataset(
    dataset_name: str, dataset_id: Optional[UUID] = None, user: Optional[User] = None
):
    """
    Function handles creation and dataset authorization if dataset already exist for Cognee.
    Verifies that provided user has necessary permission for provided Dataset.
    If Dataset does not exist creates the Dataset and gives permission for the user creating the dataset.

    Args:
        dataset_name: Name of the dataset.
        dataset_id: Id of the dataset.
        user: Cognee User request is being processed for, if None default user will be used.

    Returns:
        Tuple[User, Dataset]: A tuple containing the user and the authorized dataset.
    """

    user, authorized_datasets = await resolve_authorized_user_datasets(
        datasets=dataset_id if dataset_id else dataset_name, user=user
    )

    return user, authorized_datasets[0]
