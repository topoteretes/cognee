from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models import DatasetConfiguration


async def get_dataset_custom_prompt(dataset_id: UUID) -> str | None:
    """Return the extraction prompt stored for a dataset, or None when it has none.

    The prompt is stored with ``PUT /api/v1/datasets/{id}/schema``; a blank stored
    prompt counts as none, so storing ``""`` clears the dataset's default.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        custom_prompt = await session.scalar(
            select(DatasetConfiguration.custom_prompt).where(
                DatasetConfiguration.dataset_id == dataset_id
            )
        )

    return custom_prompt if custom_prompt and custom_prompt.strip() else None
