import warnings
from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models.Metadata import Metadata


async def delete_metadata(metadata_id: UUID):
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        metadata = await session.get(Metadata, metadata_id)
        if metadata is None:
            warnings.warn(f"metadata for metadata_id: {metadata_id} not found")

        session.delete(metadata)
        session.commit()
