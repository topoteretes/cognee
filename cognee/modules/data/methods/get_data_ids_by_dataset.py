from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models import Data, Dataset


async def get_data_ids_by_dataset(
    dataset_ids: list[UUID] | None = None,
) -> dict[UUID, list[UUID]]:
    """Map each dataset to the ids of its data rows, in one query.

    ``dataset_ids=None`` covers every dataset in the registry. A dataset with no
    data rows is still a key, with an empty list, so a caller can derive the
    rows that exist for it without a second lookup.
    """
    if dataset_ids is not None and not dataset_ids:
        return {}

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        if dataset_ids is None:
            dataset_ids = list((await session.scalars(select(Dataset.id))).all())
        data_rows = select(Data.id, Data.dataset_id)
        if dataset_ids:
            data_rows = data_rows.where(Data.dataset_id.in_(dataset_ids))

        by_dataset: dict[UUID, list[UUID]] = {dataset_id: [] for dataset_id in dataset_ids}
        for data_id, dataset_id in (await session.execute(data_rows)).all():
            by_dataset.setdefault(dataset_id, []).append(data_id)
        return by_dataset
