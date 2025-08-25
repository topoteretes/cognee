from uuid import UUID
from typing import Union

from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.modules.data.exceptions import DatasetNotFoundError
from cognee.modules.data.methods import (
    get_authorized_existing_datasets,
    load_or_create_datasets,
    check_dataset_name,
)


async def authorized_user_datasets(user: User, datasets: Union[str, list[str], list[UUID]]):
    # If no user is provided use default user
    if user is None:
        user = await get_default_user()

    # Convert datasets to list
    if isinstance(datasets, str) or isinstance(datasets, UUID):
        datasets = [datasets]

    # Get datasets user wants write permissions for (verify user has permissions if datasets are provided as well)
    # NOTE: If a user wants to write to a dataset he does not own it must be provided through UUID
    existing_datasets = await get_authorized_existing_datasets(datasets, "write", user)

    if not datasets:
        # Get datasets from database if none sent.
        authorized_datasets = existing_datasets
    else:
        # If dataset matches an existing Dataset (by name or id), reuse it. Otherwise, create a new Dataset.
        authorized_datasets = await load_or_create_datasets(datasets, existing_datasets, user)

    if not authorized_datasets:
        raise DatasetNotFoundError("There are no datasets to work with.")

    for dataset in authorized_datasets:
        check_dataset_name(dataset.name)

    return user, authorized_datasets
