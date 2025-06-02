from typing import Union
from uuid import UUID

from cognee.modules.data.exceptions import DatasetTypeError
from cognee.modules.data.methods import get_datasets


async def get_dataset_ids(datasets: Union[list[str], list[UUID]], user):
    """
    Function returns dataset IDs necessary based on provided input.
    It transforms raw strings into real dataset_ids with keeping write permissions in mind.
    If a user wants to write to a dataset he is not the owner of it must be provided through UUID.
    Args:
        datasets:
        pipeline_name:
        user:

    Returns: a list of write access dataset_ids if they exist

    """
    if all(isinstance(dataset, UUID) for dataset in datasets):
        # Return list of dataset UUIDs
        dataset_ids = datasets
    else:
        # Convert list of dataset names to dataset UUID
        if all(isinstance(dataset, str) for dataset in datasets):
            # Get all user owned dataset objects (If a user wants to write to a dataset he is not the owner of it must be provided through UUID.)
            user_datasets = await get_datasets(user.id)
            # Filter out non name mentioned datasets
            dataset_ids = [dataset.id for dataset in user_datasets if dataset.name in datasets]
        else:
            raise DatasetTypeError(
                f"One or more of the provided dataset types is not handled: f{datasets}"
            )

    return dataset_ids
