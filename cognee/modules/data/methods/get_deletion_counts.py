from sqlalchemy import select
from sqlalchemy.sql import func
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Dataset, Data, DatasetData
from cognee.modules.users.models import User


async def get_deletion_counts(
    dataset_name: str = None, user_id: str = None, all_data: bool = False
) -> dict:
    """
    Calculates the number of items that will be deleted based on the provided arguments.
    """
    relational_engine = get_relational_engine()
    async with relational_engine.get_async_session() as session:
        if dataset_name:
            # Find the dataset by name
            dataset_result = await session.execute(
                select(Dataset).where(Dataset.name == dataset_name)
            )
            dataset = dataset_result.scalar_one_or_none()

            if dataset is None:
                return {"datasets": 0, "data_entries": 0}

            # Count data entries linked to this dataset
            count_query = (
                select(func.count())
                .select_from(DatasetData)
                .where(DatasetData.dataset_id == dataset.id)
            )
            data_entry_count = (await session.execute(count_query)).scalar_one()

            return {"datasets": 1, "data_entries": data_entry_count}

        if all_data:
            dataset_count = (
                await session.execute(select(func.count()).select_from(Dataset))
            ).scalar_one()
            data_entry_count = (
                await session.execute(select(func.count()).select_from(Data))
            ).scalar_one()
            user_count = (
                await session.execute(select(func.count()).select_from(User))
            ).scalar_one()
            return {
                "datasets": dataset_count,
                "data_entries": data_entry_count,
                "users": user_count,
            }

        # Placeholder for user_id logic
        if user_id:
            # TODO: Implement counting logic for a specific user
            return {"datasets": 0, "data_entries": 0, "users": 1}

        return {}
