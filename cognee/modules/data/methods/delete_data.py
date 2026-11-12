from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.exceptions.exceptions import InvalidTableAttributeError
from cognee.modules.data.models import Data


async def delete_data(data_items: list[Data], dataset_id: UUID, user) -> None:
    """Delete data records from the database, with the index and routing rows they own.

    Takes the rows as a list: a connector sweep removes thousands at once, and
    their index nodes and routing rows go in one round of store writes instead
    of one per row. ``user`` stamps the root written afterwards, as the index
    build stamps its nodes.

    Raises:
        InvalidTableAttributeError: If a data object is invalid.
    """
    if any(not hasattr(data, "__tablename__") for data in data_items):
        raise InvalidTableAttributeError()

    # The routing rows' and the index nodes' ids derive from these Data rows, so
    # they go first: once a row is deleted nothing can address them again.
    # The documents' buckets and the dataset's root, which names every
    # document, go from the dataset's stores (callers run this in the dataset's
    # database context, as the graph delete before it needs). Imported at call
    # time, both packages import this one.
    from cognee.modules.search.dataset_routing import remove_dataset_routes
    from cognee.tasks.memify.global_context_index import (
        rebuild_document_index_root,
        remove_document_index,
    )

    data_ids = [data.id for data in data_items]
    await remove_dataset_routes(dataset_id, data_ids)
    had_index = await remove_document_index(dataset_id, data_ids)

    db_engine = get_relational_engine()
    for data_id in data_ids:
        await db_engine.delete_data_entity(data_id, dataset_id)

    # The rows are gone, so the root computed now names the documents that
    # remain. Only a dataset that had an index gets one; it never fails the
    # deletion (see the function).
    if had_index:
        await rebuild_document_index_root(dataset_id, user)
