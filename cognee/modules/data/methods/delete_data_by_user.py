from uuid import UUID
from sqlalchemy import select
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Dataset
from cognee.modules.users.methods import get_user
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def delete_data_by_user(user_id: UUID):
    """
    Delete all datasets and their associated data for a specific user.

    This function performs a comprehensive deletion of all data owned by a user,
    including datasets, data entries, and all related records in the database.

    Args:
        user_id: UUID of the user whose data should be deleted

    Raises:
        EntityNotFoundError: If user is not found
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        # Verify user exists
        await get_user(user_id)
        # Get all datasets owned by this user
        datasets_query = select(Dataset.id).where(Dataset.owner_id == user_id)
        user_datasets_ids = (await session.execute(datasets_query)).scalars().all()
    if user_datasets_ids:
        await db_engine.delete_entities_by_id(Dataset.__table__.name, user_datasets_ids)
