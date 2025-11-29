from typing import Union
from uuid import UUID
from sqlalchemy import select
from cognee.infrastructure.databases.relational import get_relational_engine
from ..models import Dataset


async def delete_datasets_by_name(
    dataset_names: Union[str, list[str]], user_id: UUID
) -> dict[str, any]:
    """
    Delete datasets by name for a specific user.
    
    Args:
        dataset_names: Single dataset name or list of dataset names to delete
        user_id: UUID of the dataset owner
        
    Returns:
        Dictionary containing:
        - deleted_count: Number of datasets deleted
        - deleted_ids: List of deleted dataset IDs
        - not_found: List of dataset names that were not found
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        # Normalize input to list
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        # Retrieve datasets matching the names and user_id
        datasets = (
            await session.scalars(
                select(Dataset)
                .filter(Dataset.owner_id == user_id)
                .filter(Dataset.name.in_(dataset_names))
            )
        ).all()
        
        # Track results
        deleted_ids = []
        found_names = set()
        
        # Delete each dataset
        for dataset in datasets:
            await db_engine.delete_entity_by_id(dataset.__tablename__, dataset.id)
            deleted_ids.append(dataset.id)
            found_names.add(dataset.name)
        
        # Identify datasets that were not found
        not_found = [name for name in dataset_names if name not in found_names]
        
        return {
            "deleted_count": len(deleted_ids),
            "deleted_ids": deleted_ids,
            "not_found": not_found
        }