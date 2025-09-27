from cognee.cli.exceptions import CliCommandException
from sqlalchemy import select
from sqlalchemy.sql import func
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Dataset, Data, DatasetData
from cognee.modules.users.models import User, DatasetDatabase
from cognee.modules.users.methods import get_user, get_default_user

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
                raise CliCommandException(f"No Dataset exists with the name {dataset_name}", error_code=1)

            # Count data entries linked to this dataset
            count_query = (
                select(func.count())
                .select_from(DatasetData)
                .where(DatasetData.dataset_id == dataset.id)
            )
            data_entry_count = (await session.execute(count_query)).scalar_one()

            return {"datasets": 1, "data_entries": data_entry_count}

        if all_data:
            users_query = select(User)
            users = (await session.execute(users_query)).scalars().all()
            users_ids = [u.id for u in users]
            user_count = len(users_ids)
            dataset_count = (
                await session.execute(select(func.count()).select_from(Dataset))
            ).scalar_one()
            data_entry_count = 0
            for user_id in users_ids:
                datasets_query = select(Dataset).where(Dataset.owner_id == user_id)
                user_datasets = (await session.execute(datasets_query)).scalars().all()
                if dataset_count > 0:
                    dataset_ids = [d.id for d in user_datasets]
                    # Count all data entries across all of the user's datasets
                    data_count_query = (
                        select(func.count())
                        .select_from(DatasetData)
                        .where(DatasetData.dataset_id.in_(dataset_ids))
                    )
                    data_entry_count += (await session.execute(data_count_query)).scalar_one()
            return {
                "datasets": dataset_count,
                "data_entries": data_entry_count,
                "users": user_count,
            }

        # Placeholder for user_id logic
        if user_id:
            counts = {}
            # TODO: How to query a user by its id?
            user = await get_default_user()
            if user:
                counts["users"] = 1
                # Find all datasets owned by this user
                datasets_query = select(Dataset).where(Dataset.owner_id == user.id)
                user_datasets = (await session.execute(datasets_query)).scalars().all()
                dataset_count = len(user_datasets)
                counts["datasets"] = dataset_count
                if dataset_count > 0:
                    dataset_ids = [d.id for d in user_datasets]
                    # Count all data entries across all of the user's datasets
                    data_count_query = (
                        select(func.count())
                        .select_from(DatasetData)
                        .where(DatasetData.dataset_id.in_(dataset_ids))
                    )
                    data_entry_count = (await session.execute(data_count_query)).scalar_one()
                    counts["data_entries"] = data_entry_count
                else:
                    counts["data_entries"] = 0
                return counts

        raise CliCommandException(f"No User exists with ID {user_id}", error_code=1)
