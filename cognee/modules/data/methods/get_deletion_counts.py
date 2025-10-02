from uuid import UUID
from cognee.cli.exceptions import CliCommandException
from cognee.infrastructure.databases.exceptions.exceptions import EntityNotFoundError
from sqlalchemy import select
from sqlalchemy.sql import func
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Dataset, Data, DatasetData
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_user
from dataclasses import dataclass


@dataclass
class DeletionCountsPreview:
    datasets: int = 0
    data_entries: int = 0
    users: int = 0


async def get_deletion_counts(
    dataset_name: str = None, user_id: str = None, all_data: bool = False
) -> DeletionCountsPreview:
    """
    Calculates the number of items that will be deleted based on the provided arguments.
    """
    counts = DeletionCountsPreview()
    relational_engine = get_relational_engine()
    async with relational_engine.get_async_session() as session:
        if dataset_name:
            # Find the dataset by name
            dataset_result = await session.execute(
                select(Dataset).where(Dataset.name == dataset_name)
            )
            dataset = dataset_result.scalar_one_or_none()

            if dataset is None:
                raise CliCommandException(
                    f"No Dataset exists with the name {dataset_name}", error_code=1
                )

            # Count data entries linked to this dataset
            count_query = (
                select(func.count())
                .select_from(DatasetData)
                .where(DatasetData.dataset_id == dataset.id)
            )
            data_entry_count = (await session.execute(count_query)).scalar_one()
            counts.users = 1
            counts.datasets = 1
            counts.entries = data_entry_count
            return counts

        elif all_data:
            # Simplified logic: Get total counts directly from the tables.
            counts.datasets = (
                await session.execute(select(func.count()).select_from(Dataset))
            ).scalar_one()
            counts.entries = (
                await session.execute(select(func.count()).select_from(Data))
            ).scalar_one()
            counts.users = (
                await session.execute(select(func.count()).select_from(User))
            ).scalar_one()
            return counts

        # Placeholder for user_id logic
        elif user_id:
            user = None
            try:
                user_uuid = UUID(user_id)
                user = await get_user(user_uuid)
            except (ValueError, EntityNotFoundError):
                raise CliCommandException(f"No User exists with ID {user_id}", error_code=1)
            counts.users = 1
            # Find all datasets owned by this user
            datasets_query = select(Dataset).where(Dataset.owner_id == user.id)
            user_datasets = (await session.execute(datasets_query)).scalars().all()
            dataset_count = len(user_datasets)
            counts.datasets = dataset_count
            if dataset_count > 0:
                dataset_ids = [d.id for d in user_datasets]
                # Count all data entries across all of the user's datasets
                data_count_query = (
                    select(func.count())
                    .select_from(DatasetData)
                    .where(DatasetData.dataset_id.in_(dataset_ids))
                )
                data_entry_count = (await session.execute(data_count_query)).scalar_one()
                counts.entries = data_entry_count
            else:
                counts.entries = 0
            return counts
