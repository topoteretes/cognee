from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
from cognee.modules.users.models import User


async def get_dataset_data_by_content_hash(
    dataset_id: UUID,
    content_hash: str,
    user: User,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> list[Data]:
    """Return ``user``'s ``Data`` rows in the dataset whose ``content_hash`` matches.

    The filter is the one ingestion dedup uses (``content_hash_predicates``):
    dataset, ``content_hash``, owner ``user.id`` and the user's tenant. That
    scoping is what makes the lookup answer "the row *I* added with this
    content" in a shared multi-writer dataset — two users adding the same
    bytes are two rows, and each finds only their own — and it is exactly the
    column order of the ``data_dataset_content_lookup`` index
    ``(dataset_id, owner_id, content_hash)``, so the query is an index probe,
    not a scan of the dataset.

    Only ``content_hash`` (the ingested payload) is matched. ``raw_content_hash``
    is the hash of the loader's *output* file — for a PDF that is its extracted
    text — so matching it would return documents the caller never added.

    Results are ordered newest first, tiebroken on id, so a caller that expects
    one row can take the first. ``limit``/``offset`` page the result like the
    dataset listing does; ``content_hash`` is expected to be normalized already
    (see ``normalize_content_hash``).
    """
    # Imported here: cognee.modules.ingestion pulls in the file-storage stack,
    # which must not load just because cognee.modules.data.methods is imported.
    from cognee.modules.ingestion.identify import content_hash_predicates

    query = (
        select(Data)
        .filter(*content_hash_predicates(content_hash, user, dataset_id))
        .order_by(Data.created_at.desc().nullslast(), Data.id)
        .offset(offset)
    )
    if limit is not None:
        query = query.limit(limit)

    async with get_relational_engine().get_async_session() as session:
        return list((await session.execute(query)).scalars().all())
