from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import User
from cognee.modules.data.models import Dataset
from cognee.modules.users.permissions.methods import give_permission_on_dataset
from .create_dataset import create_dataset


async def create_authorized_dataset(dataset_name: str, user: User) -> Dataset:
    """
        Create a new dataset and give all permissions on this dataset to the given user.
    Args:
        dataset_name: Name of the dataset.
        user: The user object.

    Returns:
        Dataset: The new authorized dataset.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        new_dataset = await create_dataset(dataset_name, user, session)

    await give_permission_on_dataset(user, new_dataset.id, "read")
    await give_permission_on_dataset(user, new_dataset.id, "write")
    await give_permission_on_dataset(user, new_dataset.id, "delete")
    await give_permission_on_dataset(user, new_dataset.id, "share")

    return new_dataset
