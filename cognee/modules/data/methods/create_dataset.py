from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload

from cognee.infrastructure.databases.relational import with_async_session

from cognee.modules.data.models import Dataset
from cognee.modules.data.methods.get_unique_dataset_id import get_unique_dataset_id

from cognee.modules.users.models import User


@with_async_session
async def create_dataset(dataset_name: str, user: User, session: AsyncSession) -> Dataset:
    owner_id = user.id

    dataset = (
        await session.scalars(
            select(Dataset)
            .options(joinedload(Dataset.data))
            .filter(Dataset.name == dataset_name)
            .filter(Dataset.owner_id == owner_id)
            .filter(Dataset.tenant_id == user.tenant_id)
        )
    ).first()

    if dataset is None:
        # Dataset id should be generated based on dataset_name and owner_id/user so multiple users can use the same dataset_name
        dataset_id = await get_unique_dataset_id(dataset_name=dataset_name, user=user)
        dataset = Dataset(
            id=dataset_id, name=dataset_name, data=[], owner_id=owner_id, tenant_id=user.tenant_id
        )

        session.add(dataset)

        try:
            await session.commit()
        except IntegrityError:
            # Concurrent calls race between the SELECT above and this INSERT
            # and, because the dataset id is deterministic, collide on the
            # primary key: another coroutine, worker, or process committed
            # this dataset first. Return its row.
            await session.rollback()
            dataset = (
                await session.scalars(
                    select(Dataset)
                    .options(joinedload(Dataset.data))
                    .filter(Dataset.id == dataset_id)
                )
            ).first()
            if dataset is None:
                # Winner's row vanished (concurrent delete) — surface the original error.
                raise

    return dataset
