from uuid import UUID
from typing import Optional

from cognee.modules.users.models import User
from cognee.modules.users.permissions.methods import get_principal_datasets

from ..models import Dataset


async def get_authorized_dataset(
    dataset_id: UUID, user: User, permission_type: str
) -> Optional[Dataset]:
    user_datasets = await get_principal_datasets(user, permission_type)

    return next((dataset for dataset in user_datasets if dataset.id == dataset_id), None)
