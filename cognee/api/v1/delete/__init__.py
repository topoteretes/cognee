from uuid import UUID
from typing import Optional

from deprecated import deprecated

from cognee.api.v1.datasets import datasets
from cognee.modules.users.models import User


@deprecated(
    reason="cognee.delete is deprecated. Use `datasets.delete_data` instead.", version="0.3.9"
)
async def delete(data_id: UUID, dataset_id: UUID, mode: str = "soft", user: Optional[User] = None):
    return await datasets.delete_data(data_id=data_id, dataset_id=dataset_id, user=user)
