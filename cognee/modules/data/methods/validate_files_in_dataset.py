from typing import List
from uuid import UUID
from sqlalchemy import select, and_
from cognee.modules.data.models import Data
from cognee.infrastructure.databases.relational import get_relational_engine


async def validate_files_in_dataset(file_ids: List[UUID], dataset_id: UUID) -> List[UUID]:
    """Validate that files belong to a dataset and return valid file IDs."""
    if not file_ids:
        return []
    
    db_engine = get_relational_engine()
    
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(Data.id)
            .join(Data.datasets)
            .where(and_(
                Data.id.in_(file_ids),
                Data.datasets.any(id=dataset_id)
            ))
        )
        valid_file_ids = [row[0] for row in result.fetchall()]
        return valid_file_ids 