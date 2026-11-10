from typing import Any
from uuid import UUID

from sqlalchemy import select

from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data


async def get_selected_dataset_data(dataset_id: UUID, data: Any) -> Any:
    """Return a pipeline's data selection with every Data row read again from the database.

    Call it under the dataset lock: a row may have been updated or deleted since the
    caller loaded it. Deleted rows are dropped, so a selection that empties runs over
    no data, never over the whole dataset. Items that are not Data rows pass through,
    and a selection without Data rows is returned unchanged.

    Raises CogneeValidationError when a selected row belongs to another dataset. A NULL
    dataset_id only exists before the startup backfill stamps legacy rows.
    """
    items = data if isinstance(data, (list, tuple)) else [data]
    selected_ids = [item.id for item in items if isinstance(item, Data)]
    if not selected_ids:
        return data

    async with get_relational_engine().get_async_session() as session:
        rows = await session.scalars(select(Data).where(Data.id.in_(selected_ids)))
        current = {row.id: row for row in rows}

    foreign_ids = [
        str(row.id) for row in current.values() if row.dataset_id not in (None, dataset_id)
    ]
    if foreign_ids:
        raise CogneeValidationError(
            f"Data rows {', '.join(foreign_ids)} do not belong to dataset {dataset_id}.",
            remediation=(
                "Pass only Data rows of the dataset the pipeline runs on, "
                "or run the pipeline on the dataset that owns them."
            ),
        )

    return [
        current[item.id] if isinstance(item, Data) else item
        for item in items
        if not isinstance(item, Data) or item.id in current
    ]
