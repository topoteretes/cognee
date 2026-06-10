"""Revision-chain primitives for graph/vector database migrations.

Mirrors Alembic's model: each migration declares a stable ``revision`` and the
``down_revision`` it builds on, forming a single linear chain. The database
stores the last-applied revision; the runner walks the chain forward to head.
Cognee versions are never compared for ordering — they are recorded only as
informational metadata on each migration and on the database row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional
from uuid import NAMESPACE_DNS, UUID, uuid5

logger = logging.getLogger(__name__)

# Stable namespace so revision ids are reproducible across machines and runs.
COGNEE_MIGRATION_NAMESPACE = uuid5(NAMESPACE_DNS, "migrations.cognee.ai")


@dataclass(frozen=True)
class MigrationContext:
    """Everything a migration needs to touch every store that holds a node id.

    A single Entity/EntityType id is a key in three places — the graph DB, the
    vector DB (point id), and the relational ledger (``nodes.slug`` /
    ``edges.source_node_id`` / ``edges.destination_node_id``). A migration that
    rewrites ids must update all three in lockstep, so it receives all the
    handles here rather than just one engine.

    Attributes:
        graph_engine: Resolved graph engine for this database.
        vector_engine: Resolved vector engine for this database.
        dataset_id: The dataset whose ledger rows this migration may touch
            (access control on: one database pair per dataset), or ``None`` in
            global mode (access control off: one global pair backs every
            dataset, so ledger updates apply unscoped).
    """

    graph_engine: Any
    vector_engine: Any
    dataset_id: Optional[UUID] = None


def revision_id(slug: str) -> str:
    """Deterministically derive a migration revision id from its slug."""
    return str(uuid5(COGNEE_MIGRATION_NAMESPACE, slug))


@dataclass(frozen=True)
class Migration:
    """A single graph or vector database migration.

    Attributes:
        slug: Human-readable unique identifier; also seeds the revision id.
        cognee_version: Release this migration shipped in (informational only).
        up: Async callable receiving a :class:`MigrationContext`.
        down_revision: Revision this builds on (``None`` for the first migration).
    """

    slug: str
    cognee_version: str
    up: Callable[["MigrationContext"], Awaitable[None]]
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
    - stored revision unknown to this chain -> empty list, with a WARNING.
      Unlike Alembic (which raises), this tolerates a database stamped by newer
      code (rollback); but the same state also arises from a renamed migration
      slug or a corrupted revision row — silently disabled migrations — so it
      must never pass without a trace in the logs.
    """
    ordered = order_migrations(migrations)

    if stored_revision is None:
        return ordered

    for index, migration in enumerate(ordered):
        if migration.revision == stored_revision:
            return ordered[index + 1 :]

    logger.warning(
        "Stored migration revision %r is unknown to this chain (head: %r) — no migrations "
        "will run for this database. Expected only after a rollback to older code; if no "
        "rollback happened, a migration slug was renamed or the revision row is corrupted, "
        "and this database will silently skip all future migrations until fixed.",
        stored_revision,
        ordered[-1].revision if ordered else None,
    )
    return []
