from typing import Optional
from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models.Data import Data
from cognee.modules.users.models import User

from .data_types import IngestionData


async def identify(data: IngestionData, user: User, dataset_id: UUID) -> Optional[UUID]:
    """Resolve the existing ``Data`` row for this content in this dataset.

    Dedup is a lookup, not an identity: a hit returns the id of the row that
    already holds this content in this dataset for this owner/tenant; a miss
    returns ``None`` and the caller mints a fresh random id. The id itself
    carries no content information, so it stays stable when a document's
    content is updated. Rows are dataset-scoped (the startup migration
    backfills pre-refactor rows), so one scoped probe is sufficient.
    """
    content_hash: str = data.get_identifier()
    db_engine = get_relational_engine()

    # Dedup is scoped to (dataset, owner, tenant): in a shared multi-writer
    # dataset, two users adding the same bytes stay two rows — matching the
    # old per-user identity semantics and keeping owner_id checks meaningful.
    tenant_filter = Data.tenant_id == user.tenant_id if user.tenant_id else Data.tenant_id.is_(None)

    async with db_engine.get_async_session() as session:
        return (
            await session.execute(
                select(Data.id)
                .filter(
                    Data.dataset_id == dataset_id,
                    Data.content_hash == content_hash,
                    Data.owner_id == user.id,
                    tenant_filter,
                )
                .limit(1)
            )
        ).scalar_one_or_none()
