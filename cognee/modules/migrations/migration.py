"""Revision-chain primitives for graph/vector database migrations.

Mirrors Alembic's model: each migration declares a stable ``revision`` and the
``down_revision`` it builds on, forming a single linear chain. The database
stores the last-applied revision; the runner walks the chain forward to head.
Cognee versions are never compared for ordering — they are recorded only as
informational metadata on each migration and on the database row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional
from uuid import NAMESPACE_DNS, uuid5

# Stable namespace so revision ids are reproducible across machines and runs.
COGNEE_MIGRATION_NAMESPACE = uuid5(NAMESPACE_DNS, "migrations.cognee.ai")


def revision_id(slug: str) -> str:
    """Deterministically derive a migration revision id from its slug."""
    return str(uuid5(COGNEE_MIGRATION_NAMESPACE, slug))


@dataclass(frozen=True)
class Migration:
    """A single graph or vector database migration.

    Attributes:
        slug: Human-readable unique identifier; also seeds the revision id.
        cognee_version: Release this migration shipped in (informational only).
        up: Async callable receiving the resolved graph or vector engine.
        down_revision: Revision this builds on (``None`` for the first migration).
    """

    slug: str
    cognee_version: str
    up: Callable[[Any], Awaitable[None]]
    down_revision: Optional[str] = None

    @property
    def revision(self) -> str:
        return revision_id(self.slug)


def order_migrations(migrations: list[Migration]) -> list[Migration]:
    """Return migrations ordered from the root of the chain to its head.

    Raises ``ValueError`` if the chain branches (two migrations share a parent)
    or is disconnected (a parent is missing), enforcing Alembic's single-head,
    linear-history guarantee.
    """
    if not migrations:
        return []

    by_down: dict[Optional[str], Migration] = {}
    for migration in migrations:
        if migration.down_revision in by_down:
            raise ValueError(
                f"Migration chain branches at down_revision={migration.down_revision!r}: "
                f"{by_down[migration.down_revision].slug!r} and {migration.slug!r}"
            )
        by_down[migration.down_revision] = migration

    ordered: list[Migration] = []
    cursor: Optional[str] = None  # the chain root has down_revision == None
    while cursor in by_down:
        migration = by_down.pop(cursor)
        ordered.append(migration)
        cursor = migration.revision

    if by_down:
        raise ValueError(
            "Disconnected migrations (unreachable parents): "
            + ", ".join(sorted(m.slug for m in by_down.values()))
        )

    return ordered


def head_revision(migrations: list[Migration]) -> Optional[str]:
    """Return the revision at the head of the chain (``None`` if there are none)."""
    ordered = order_migrations(migrations)
    return ordered[-1].revision if ordered else None


def pending_migrations(
    migrations: list[Migration], stored_revision: Optional[str]
) -> list[Migration]:
    """Return the migrations needed to bring ``stored_revision`` up to head.

    - ``None`` stored revision -> the whole chain (a database with no recorded
      revision runs every migration).
    - stored revision at head -> empty list.
    - stored revision unknown to this chain (database ahead of, or diverged
      from, this code) -> empty list (no-op, like Alembic).
    """
    ordered = order_migrations(migrations)

    if stored_revision is None:
        return ordered

    for index, migration in enumerate(ordered):
        if migration.revision == stored_revision:
            return ordered[index + 1 :]

    return []
