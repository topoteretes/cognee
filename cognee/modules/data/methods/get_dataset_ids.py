from uuid import UUID

from cognee.modules.data.exceptions import DatasetNotFoundError, DatasetTypeError
from cognee.modules.data.methods import get_datasets


async def get_dataset_ids(datasets: list[str] | list[UUID], user, strict: bool = False):
    """
    Function returns dataset IDs necessary based on provided input.
    It transforms raw strings into real dataset_ids with keeping write permissions in mind.
    If a user wants to write to a dataset he is not the owner of it must be provided through UUID.
    Args:
        datasets:
        user:
        strict: When True, a name that resolves to no dataset the user owns in the
            current tenant raises DatasetNotFoundError naming the misses, instead of
            being dropped. Read paths use this; write paths keep the default so an
            unmatched name can still mean "create it".

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
            matched = [dataset for dataset in user_datasets if dataset.name in datasets]
            # Filter out non current tenant datasets
            matched = [dataset for dataset in matched if dataset.tenant_id == user.tenant_id]
            if strict:
                resolved_names = {dataset.name for dataset in matched}
                missing = [name for name in dict.fromkeys(datasets) if name not in resolved_names]
                if missing:
                    raise DatasetNotFoundError(
                        message=f"Dataset(s) not found: {', '.join(repr(name) for name in missing)}. "
                        "Dataset names resolve only among the datasets you own in the current "
                        "tenant; pass dataset_ids for datasets shared with you."
                    )
            dataset_ids = [dataset.id for dataset in matched]
        else:
            raise DatasetTypeError(
                f"One or more of the provided dataset types is not handled: {datasets}"
            )

    return dataset_ids
