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

    # The routing rows' ids derive from this Data row, so they go first: once
    # the row is deleted nothing can address them again. The dataset's root
    # index summary names this document too, so it goes as well, from the
    # dataset's stores (callers run this in the dataset's database context,
    # as the graph delete before it needs). Imported at call time, both
    # packages import this one.
    from cognee.modules.search.dataset_routing import remove_dataset_routes
    from cognee.tasks.memify.global_context_index import remove_dataset_root

    await remove_dataset_routes(dataset_id, [data.id])
    await remove_dataset_root(dataset_id)

    db_engine = get_relational_engine()

    return await db_engine.delete_data_entity(data.id, dataset_id)
