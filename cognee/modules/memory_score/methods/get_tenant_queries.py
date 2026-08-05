"""Collect a tenant's REAL past questions, tenant-wide.

``Query`` carries no tenant column — only ``user_id`` (indexed) — so tenant
scope has to be expressed by joining ``queries.user_id -> users.id`` and
filtering on ``users.tenant_id``. The existing per-user
``cognee.modules.search.operations.get_queries`` is deliberately left untouched;
this is an additional, tenant-wide reader.

Real questions have NO golden answer, so they can only ever carry a
groundedness signal downstream — never a correctness score.
"""

from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.search.models.Query import Query
from cognee.modules.users.models import User


async def get_tenant_queries(tenant_id: UUID, limit: int) -> list[Query]:
    """Return the tenant's most recent questions, newest first.

    Args:
        tenant_id: the tenant to scope to. ``User.tenant_id`` is nullable, so
            ``None`` is accepted and means "the users with no tenant" — the
            OSS / single-tenant case, where every user row has a NULL tenant.
        limit: maximum number of ``Query`` rows to return.

    Returns:
        Up to ``limit`` ``Query`` rows ordered by ``queries.created_at`` desc.
        Empty when the tenant has no search history.
    """
    if limit <= 0:
        return []

    db_engine = get_relational_engine()

    tenant_filter = User.tenant_id.is_(None) if tenant_id is None else User.tenant_id == tenant_id

    async with db_engine.get_async_session() as session:
        queries = (
            await session.scalars(
                select(Query)
                .join(User, User.id == Query.user_id)
                .where(tenant_filter)
                .order_by(Query.created_at.desc())
                .limit(limit)
            )
        ).all()

        return list(queries)
