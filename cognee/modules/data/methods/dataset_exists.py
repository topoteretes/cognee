from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models import Dataset


async def dataset_exists(dataset_id: UUID) -> bool:
    """Whether a dataset with this id exists, regardless of who owns or may access it.

    ``get_dataset`` answers None for a dataset another user owns, so it cannot
    tell a deleted dataset from a foreign one. Use this where that difference
    matters and the caller is entitled to know, such as a session binding the
    caller's own write created.
    """
    async with get_relational_engine().get_async_session() as session:
        return await session.get(Dataset, dataset_id) is not None
