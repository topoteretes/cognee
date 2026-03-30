from uuid import UUID
from typing import Optional, Union

from cognee.shared.logging_utils import get_logger

logger = get_logger("forget")


async def forget(
    *,
    data_id: Optional[UUID] = None,
    dataset: Optional[Union[str, UUID]] = None,
    everything: bool = False,
    user=None,
) -> dict:
    """Remove data from the knowledge graph.

    Unified deletion command that replaces the separate prune/delete/
    empty_dataset APIs with a single mental model.

    Usage patterns::

        # Forget a specific data item from a dataset
        await cognee.forget(data_id=data_id, dataset=dataset_id)

        # Forget an entire dataset (all data + graph nodes + vector entries)
        await cognee.forget(dataset="scientists")

        # Forget everything the current user owns
        await cognee.forget(everything=True)

    Args:
        data_id: UUID of a specific data item to remove.
            Requires ``dataset`` to also be set.
        dataset: Dataset name or UUID. When set alone, deletes the
            entire dataset. When set with ``data_id``, deletes that
            item from this dataset.
        everything: If True, delete all datasets and data the user owns.
            Ignores ``data_id`` and ``dataset``.
        user: User context. Resolved to default user when None.

    Returns:
        Dict with deletion summary: items removed, datasets removed.
    """
    from cognee.modules.users.methods import get_default_user

    if user is None:
        user = await get_default_user()

    if everything:
        return await _forget_everything(user)

    if dataset is not None and data_id is not None:
        return await _forget_data_item(data_id, dataset, user)

    if dataset is not None:
        return await _forget_dataset(dataset, user)

    if data_id is not None:
        raise ValueError("data_id requires dataset to be specified.")

    raise ValueError("Specify dataset, data_id+dataset, or everything=True.")


async def _forget_everything(user) -> dict:
    """Delete all datasets and data owned by the user."""
    from cognee.api.v1.datasets.datasets import datasets

    user_datasets = await datasets.list_datasets(user=user)
    count = len(user_datasets)

    await datasets.delete_all(user=user)

    logger.info("forget: deleted all data for user=%s (%d datasets)", user.id, count)
    return {"datasets_removed": count, "status": "success"}


async def _forget_dataset(dataset_ref: Union[str, UUID], user) -> dict:
    """Delete an entire dataset by name or UUID."""
    from cognee.api.v1.datasets.datasets import datasets

    dataset_id = await _resolve_dataset_id(dataset_ref, user)

    await datasets.empty_dataset(dataset_id, user=user)

    logger.info("forget: deleted dataset=%s for user=%s", dataset_id, user.id)
    return {"dataset_id": str(dataset_id), "status": "success"}


async def _forget_data_item(data_id: UUID, dataset_ref: Union[str, UUID], user) -> dict:
    """Delete a single data item from a dataset."""
    from cognee.api.v1.datasets.datasets import datasets

    dataset_id = await _resolve_dataset_id(dataset_ref, user)

    await datasets.delete_data(
        dataset_id=dataset_id,
        data_id=data_id,
        user=user,
        delete_dataset_if_empty=False,
    )

    logger.info(
        "forget: deleted data_id=%s from dataset=%s for user=%s",
        data_id,
        dataset_id,
        user.id,
    )
    return {"data_id": str(data_id), "dataset_id": str(dataset_id), "status": "success"}


async def _resolve_dataset_id(dataset_ref: Union[str, UUID], user) -> UUID:
    """Resolve a dataset name or UUID to a UUID."""
    if isinstance(dataset_ref, UUID):
        return dataset_ref

    from cognee.modules.data.methods import get_authorized_dataset_by_name

    dataset = await get_authorized_dataset_by_name(dataset_ref, user, "delete")
    return dataset.id
