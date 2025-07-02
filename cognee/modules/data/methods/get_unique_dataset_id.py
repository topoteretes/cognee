from uuid import UUID, uuid5, NAMESPACE_OID
from cognee.modules.users.models import User
from typing import Union


async def get_unique_dataset_id(dataset_name: Union[str, UUID], user: User) -> UUID:
    if isinstance(dataset_name, UUID):
        return dataset_name
    return uuid5(NAMESPACE_OID, f"{dataset_name}{str(user.id)}")
