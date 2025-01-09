import json
from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models.Metadata import Metadata


async def get_metadata(metadata_id: UUID) -> Metadata:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        metadata = await session.get(Metadata, metadata_id)

        return metadata
