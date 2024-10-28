from cognee.modules.data.models import Data
from cognee.infrastructure.databases.relational import get_relational_engine


async def delete_data(data: Data):
    db_engine = get_relational_engine()

    return await db_engine.delete_data_by_id(data.__tablename__, data.id)
