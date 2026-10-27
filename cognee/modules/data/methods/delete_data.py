from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.exceptions.exceptions import InvalidTableAttributeError
from cognee.modules.data.models import Data


async def delete_data(data: Data, dataset_id: UUID):
    """Delete a data record from the database.

    Args:
        data (Data): The data object to be deleted.

    Raises:
        ValueError: If the data object is invalid.
    """
    if not hasattr(data, "__tablename__"):
        raise InvalidTableAttributeError()

    # The routing row's id derives from this Data row, so it goes first: once
    # the row is deleted nothing can address it again. Imported at call time,
    # the routing package imports this package.
    from cognee.modules.search.dataset_routing import remove_document_routes

    await remove_document_routes(dataset_id, [data.id])

    db_engine = get_relational_engine()

    return await db_engine.delete_data_entity(data.id, dataset_id)
