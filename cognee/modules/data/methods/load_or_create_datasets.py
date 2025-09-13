from typing import List, Union
from uuid import UUID

from cognee.modules.data.models import Dataset
from cognee.modules.data.methods.create_authorized_dataset import create_authorized_dataset
from cognee.modules.data.exceptions import DatasetNotFoundError


async def load_or_create_datasets(
    dataset_names: List[Union[str, UUID]], existing_datasets: List[Dataset], user
) -> List[Dataset]:
    """
    Given a list of dataset identifiers (names or UUIDs), return Dataset instances:
      - If an identifier matches an existing Dataset (by name or id), reuse it.
      - Otherwise, create a new Dataset with a unique id.
    """
    result: List[Dataset] = []

    for identifier in dataset_names:
        # Try to find a matching dataset in the existing list
        # If no matching dataset is found return None
        match = next(
            (ds for ds in existing_datasets if ds.name == identifier or ds.id == identifier), None
        )

        if match:
            result.append(match)
            continue

        # If the identifier is a UUID but nothing matched, that's an error
        if isinstance(identifier, UUID):
            raise DatasetNotFoundError(f"Dataset with given UUID does not exist: {identifier}")

        new_dataset = await create_authorized_dataset(identifier, user)

        result.append(new_dataset)

    return result
