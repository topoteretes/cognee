from uuid import UUID, uuid5, NAMESPACE_OID
from cognee.modules.users.models import User


async def get_unique_dataset_id(dataset_name: str, user: User) -> UUID:
    return uuid5(NAMESPACE_OID, f"{dataset_name}{str(user.id)}")
