from uuid import UUID

from cognee.api.v1.exceptions import DatasetNotFoundError
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.data.methods import (
    create_authorized_dataset,
    get_authorized_dataset,
    get_authorized_dataset_by_name,
)


async def resolve_authorized_user_dataset(dataset_id: UUID, dataset_name: str, user: User):
    """
    Function handles creation and dataset authorization if dataset already exist for Cognee.
    Verifies that provided user has necessary permission for provided Dataset.
    If Dataset does not exist creates the Dataset and gives permission for the user creating the dataset.

    Args:
        dataset_id: Id of the dataset.
        dataset_name: Name of the dataset.
        user: Cognee User request is being processed for, if None default user will be used.

    Returns:
        Tuple[User, Dataset]: A tuple containing the user and the authorized dataset.
    """
    if not user:
        user = await get_default_user()

    if dataset_id:
        authorized_dataset = await get_authorized_dataset(user, dataset_id, "write")
    elif dataset_name:
        authorized_dataset = await get_authorized_dataset_by_name(dataset_name, user, "write")

        if not authorized_dataset:
            authorized_dataset = await create_authorized_dataset(
                dataset_name=dataset_name, user=user
            )
    else:
        raise ValueError("Either dataset_id or dataset_name must be provided.")

    if not authorized_dataset:
        raise DatasetNotFoundError(
            message=f"Dataset ({str(dataset_id) or dataset_name}) not found."
        )

    return user, authorized_dataset
