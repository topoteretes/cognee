from typing import Optional

from cognee.modules.users.models import User
from cognee.modules.data.methods.get_authorized_existing_datasets import (
    get_authorized_existing_datasets,
)

from ..models import Dataset


async def get_authorized_dataset_by_name(
    dataset_name: str, user: User, permission_type: str
) -> Optional[Dataset]:
    authorized_datasets = await get_authorized_existing_datasets([], permission_type, user)

    return next((dataset for dataset in authorized_datasets if dataset.name == dataset_name), None)
