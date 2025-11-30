from uuid import UUID
from sqlalchemy import select
from cognee.infrastructure.databases.relational import get_relational_engine
from ..models import Dataset


async def delete_dataset_by_name(dataset_name: str, user_id: UUID):
    """
    Delete a single dataset by name for a specific user.

    Args:
        dataset_name: The name of the dataset to delete (must be a single string).
        user_id: UUID of the dataset owner.

    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        dataset_id = (
            await session.scalars(
                select(Dataset.id)
                .filter(Dataset.owner_id == user_id)
                .filter(Dataset.name == dataset_name)
            )
        ).first()
    # Keeping this out of the first session, since delete_entities_by_id creates another session.
    if dataset_id:
        await db_engine.delete_entities_by_id(Dataset.__table__.name, dataset_id)
