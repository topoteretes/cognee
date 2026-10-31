from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models import DatasetConfiguration


async def get_dataset_system_prompt(dataset_id: UUID) -> str | None:
    """Return the answer system prompt stored for a dataset, or None when it has none.

    The prompt is stored with ``PUT /api/v1/datasets/{id}/schema``; a blank stored
    prompt counts as none, so storing ``""`` clears the dataset's default.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        system_prompt = await session.scalar(
            select(DatasetConfiguration.system_prompt).where(
                DatasetConfiguration.dataset_id == dataset_id
            )
        )

    return system_prompt if system_prompt and system_prompt.strip() else None
