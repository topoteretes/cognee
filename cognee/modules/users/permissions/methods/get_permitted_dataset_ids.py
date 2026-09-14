from uuid import UUID

from cognee.modules.users.permissions.methods.get_readable_datasets import get_readable_datasets


async def get_permitted_dataset_ids(user_id: UUID) -> list[UUID]:
    """Return the UUIDs of datasets ``user_id`` has read permission for.

    Thin convenience over ``get_readable_datasets`` for callers that only
    need the IDs (e.g. to filter rows by ``dataset_id``), not the full
    dataset objects.
    """
    datasets = await get_readable_datasets(user_id)
    return [dataset.id for dataset in datasets]
