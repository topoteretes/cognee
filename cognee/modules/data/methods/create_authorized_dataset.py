from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import User
from cognee.modules.data.models import Dataset
from cognee.modules.users.permissions.methods import give_permission_on_dataset
from cognee.modules.data.methods.get_unique_dataset_id import get_unique_dataset_id


async def create_authorized_dataset(dataset_name: str, user: User) -> Dataset:
    # Dataset id should be generated based on dataset_name and owner_id/user so multiple users can use the same dataset_name
    dataset_id = await get_unique_dataset_id(dataset_name=dataset_name, user=user)
    new_dataset = Dataset(id=dataset_id, name=dataset_name, data=[], owner_id=user.id)

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        session.add(new_dataset)

        await session.commit()

    await give_permission_on_dataset(user, new_dataset.id, "read")
    await give_permission_on_dataset(user, new_dataset.id, "write")
    await give_permission_on_dataset(user, new_dataset.id, "delete")
    await give_permission_on_dataset(user, new_dataset.id, "share")

    return new_dataset
