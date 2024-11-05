from uuid import UUID
from typing import Optional
from cognee.infrastructure.databases.relational import get_relational_engine
from ..models import Data

async def get_data(data_id: UUID) -> Optional[Data]:
    """Retrieve data by ID.

        Args:
            data_id (UUID): ID of the data to retrieve

        Returns:
            Optional[Data]: The requested data object if found, None otherwise
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        data = await session.get(Data, data_id)

        return data