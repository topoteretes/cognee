from cognee.modules.data.exceptions.exceptions import InvalidTableAttributeError
from cognee.modules.data.models import Data
from cognee.infrastructure.databases.relational import get_relational_engine


async def delete_data(data: Data):
    """Delete a data record from the database.

    Args:
        data (Data): The data object to be deleted.

    Raises:
        ValueError: If the data object is invalid.
    """
    if not hasattr(data, "__tablename__"):
        raise InvalidTableAttributeError()

    db_engine = get_relational_engine()

    return await db_engine.delete_data_entity(data.id)
