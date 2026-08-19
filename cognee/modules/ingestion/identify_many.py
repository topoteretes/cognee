from typing import Optional
from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models.Data import Data
from cognee.modules.users.models import User

# SQLite's default SQLITE_MAX_VARIABLE_NUMBER is 999.  Stay safely under it so
# a large unpinned batch never hits "too many SQL variables".  Postgres has no
# meaningful limit, so the chunk size is conservative but harmless there too.
_CHUNK_SIZE = 900


async def identify_many(
    hashes: list[str],
    user: User,
    dataset_id: UUID,
) -> dict[str, UUID]:
    """Batch version of :func:`identify`: resolve existing ``Data`` rows for
    multiple content hashes in one dataset.

    Returns a mapping ``{content_hash: data_id}`` for every hash that already
    has a row in ``(dataset_id, owner_id, tenant_id)`` scope.  Hashes with no
    existing row are absent from the result (not mapped to ``None``).

    The filter is identical to :func:`identify` — same four predicates — so
    both functions always agree on which row wins for a given hash.  Large
    inputs are chunked into batches of at most :data:`_CHUNK_SIZE` hashes to
    stay within SQLite's bind-parameter limit.
    """
    if not hashes:
        return {}

    # Dedup is scoped to (dataset, owner, tenant): in a shared multi-writer
    # dataset, two users adding the same bytes stay two rows — matching the
    # old per-user identity semantics and keeping owner_id checks meaningful.
    tenant_filter = (
        Data.tenant_id == user.tenant_id if user.tenant_id else Data.tenant_id.is_(None)
    )

    db_engine = get_relational_engine()
    result: dict[str, UUID] = {}

    async with db_engine.get_async_session() as session:
        for start in range(0, len(hashes), _CHUNK_SIZE):
            chunk = hashes[start : start + _CHUNK_SIZE]
            rows = await session.execute(
                select(Data.id, Data.content_hash).filter(
                    Data.dataset_id == dataset_id,
                    Data.content_hash.in_(chunk),
                    Data.owner_id == user.id,
                    tenant_filter,
                )
            )
            for data_id, content_hash in rows.fetchall():
                # Keep the first hit per hash (matches identify()'s .limit(1)
                # semantics: if duplicate rows exist, we take whichever the DB
                # returns first rather than silently picking the last one).
                result.setdefault(content_hash, data_id)

    return result
