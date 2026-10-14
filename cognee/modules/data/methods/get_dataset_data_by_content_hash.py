from uuid import UUID

from sqlalchemy import or_, select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data


async def get_dataset_data_by_content_hash(
    dataset_id: UUID,
    content_hash: str,
    *,
    owner_id: UUID | None = None,
) -> list[Data]:
    """Return the dataset's ``Data`` rows whose content matches ``content_hash``.

    A row matches when either ``content_hash`` (the ingested payload, the
    value ``compute_content_hash`` produces for the text or bytes that were
    added) or ``raw_content_hash`` (the stored file the loader produced)
    equals the given hash. The ``content_hash`` branch is served by the
    ``data_dataset_content_lookup`` index that dedup already relies on.

    Pass ``owner_id`` to narrow a shared dataset to one writer's rows — the
    same scoping ingestion dedup applies. Results are ordered newest first,
    tiebroken on id, so a caller that expects one row can take the first.
    """
    predicates = [
        Data.dataset_id == dataset_id,
        or_(Data.content_hash == content_hash, Data.raw_content_hash == content_hash),
    ]
    if owner_id is not None:
        predicates.append(Data.owner_id == owner_id)

    query = select(Data).filter(*predicates).order_by(Data.created_at.desc().nullslast(), Data.id)

    async with get_relational_engine().get_async_session() as session:
        return list((await session.execute(query)).scalars().all())
