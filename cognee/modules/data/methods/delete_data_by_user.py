from uuid import UUID
from sqlalchemy import select, delete as sql_delete
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Dataset, DatasetData
from cognee.modules.users.methods import get_user
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def delete_data_by_user(user_id: UUID) -> dict[str, int]:
    """
    Delete all datasets and their associated data for a specific user.

    This function performs a comprehensive deletion of all data owned by a user,
    including datasets, data entries, and all related records in the database.

    Args:
        user_id: UUID of the user whose data should be deleted

    Returns:
        Dictionary containing deletion statistics:
        - datasets_deleted: Number of datasets deleted
        - data_entries_deleted: Number of data entries deleted

    Raises:
        ValueError: If user is not found
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        # Verify user exists
        user = await get_user(user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")

        # Get all datasets owned by this user
        datasets_query = select(Dataset).where(Dataset.owner_id == user_id)
        user_datasets = (await session.execute(datasets_query)).scalars().all()

        datasets_deleted = 0
        data_entries_deleted = 0

        # Delete each dataset and its data
        for dataset in user_datasets:
            # Get all data entries in this dataset
            data_query = select(DatasetData).where(DatasetData.dataset_id == dataset.id)
            dataset_data_links = (await session.execute(data_query)).scalars().all()

            # Delete dataset-data links
            for link in dataset_data_links:
                await session.execute(
                    sql_delete(DatasetData).where(DatasetData.id == link.id)
                )
                data_entries_deleted += 1

            # Delete the dataset itself
            await session.execute(
                sql_delete(Dataset).where(Dataset.id == dataset.id)
            )
            datasets_deleted += 1

        # Commit all changes
        await session.commit()

        logger.info(f"Deleted {datasets_deleted} datasets and {data_entries_deleted} data entries for user {user_id}")

        return {
            "datasets_deleted": datasets_deleted,
            "data_entries_deleted": data_entries_deleted,
        }