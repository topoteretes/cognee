"""Revision-chain primitives for graph/vector database migrations.

Mirrors Alembic's model: each migration declares a stable ``revision`` and the
``down_revision`` it builds on, forming a single linear chain. The database
stores the last-applied revision; the runner walks the chain forward to head.
Cognee versions are never compared for ordering — they are recorded only as
informational metadata on each migration and on the database row.

The revision IS the slug. It is stored verbatim in the database so an operator
reading a ``dataset_database`` row can see exactly which migration last ran —
no hashing, no lookup table. Slugs are therefore append-only and immutable:
renaming one orphans every stamped database (see ``pending_migrations``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

# Unknown stored revisions already warned about (once per process per value):
# the check runs on every startup AND on every cognify() pending-check, so an
# unrepaired legacy row must not flood the logs.
_warned_unknown_revisions: set = set()


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


@dataclass(frozen=True)
class Migration:
    """A single graph or vector database migration.

    Attributes:
        slug: Human-readable unique identifier; stored verbatim as the revision.
            IMMUTABLE once shipped — renaming orphans every stamped database.
        cognee_version: Release this migration shipped in (informational only).
        up: Async callable receiving a :class:`MigrationContext`.
        down_revision: Slug of the migration this builds on (``None`` for the
            first migration in a chain).
        down: Optional reverse transformation (same signature as ``up``).
            A chain can only be downgraded through migrations that define it.
    """

    slug: str
    cognee_version: str
    up: Callable[["MigrationContext"], Awaitable[None]]
    down_revision: Optional[str] = None
    down: Optional[Callable[["MigrationContext"], Awaitable[None]]] = None

    @property
    def revision(self) -> str:
        return self.slug


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
    migrations: list[Migration],
    stored_revision: Optional[str],
    target_revision: str = "head",
) -> list[Migration]:
    """Return the migrations needed to bring ``stored_revision`` up to ``target_revision``.

    - ``None`` stored revision -> everything up to the target (a database with
      no recorded revision runs every migration).
    - stored revision at/beyond the target -> empty list.
    - ``target_revision`` is ``"head"`` (default) or a slug; an unknown slug
      RAISES — an explicit target is an operator action, never best-effort.
    - stored revision unknown to this chain -> empty list, with a WARNING.
      Unlike Alembic (which raises), this tolerates a database stamped by newer
      code (rollback); but the same state also arises from a renamed migration
      slug or a corrupted revision row — silently disabled migrations — so it
      must never pass without a trace in the logs.
    """
    ordered = order_migrations(migrations)

    if target_revision == "head":
        end_index = len(ordered)
    else:
        revisions = [migration.revision for migration in ordered]
        if target_revision not in revisions:
            raise ValueError(
                f"Target revision {target_revision!r} is unknown to this chain; cannot upgrade."
            )
        end_index = revisions.index(target_revision) + 1

    if stored_revision is None:
        return ordered[:end_index]

    for index, migration in enumerate(ordered):
        if migration.revision == stored_revision:
            return ordered[index + 1 : end_index]

    if stored_revision not in _warned_unknown_revisions:
        _warned_unknown_revisions.add(stored_revision)
        logger.warning(
            "Stored migration revision %r is unknown to this chain (head: %r) — no migrations "
            "will run for this database. Expected only after a rollback to older code; if no "
            "rollback happened, a migration slug was renamed or the revision row is corrupted, "
            "and this database will silently skip all future migrations until repaired "
            "(`cognee-cli stamp base --dataset <id>` re-arms the chain). Logged once per process.",
            stored_revision,
            ordered[-1].revision if ordered else None,
        )
    return []


def migrations_to_downgrade(
    migrations: list[Migration],
    stored_revision: Optional[str],
    target_revision: Optional[str] = None,
) -> list[Migration]:
    """Return the migrations to revert (newest first) to reach ``target_revision``.

    - ``None`` stored revision -> nothing applied, nothing to revert.
    - ``None`` target -> revert every applied migration (back to pre-chain state).
    - Unlike :func:`pending_migrations`, an unknown stored or target revision
      RAISES: downgrading is always an explicit operator action against a state
      that must be fully understood, never a best-effort no-op.
    - Every migration in the returned span must define ``down`` — a chain
      cannot skip a step — otherwise this raises.
    """
    ordered = order_migrations(migrations)
    if stored_revision is None:
        return []

    revisions = [migration.revision for migration in ordered]
    if stored_revision not in revisions:
        raise ValueError(
            f"Stored revision {stored_revision!r} is unknown to this chain; cannot downgrade."
        )
    stored_index = revisions.index(stored_revision)

    if target_revision is None:
        target_index = -1
    else:
        if target_revision not in revisions:
            raise ValueError(
                f"Target revision {target_revision!r} is unknown to this chain; cannot downgrade."
            )
        target_index = revisions.index(target_revision)
        if target_index > stored_index:
            raise ValueError(
                f"Target revision {target_revision!r} is ahead of stored "
                f"{stored_revision!r}; nothing to downgrade."
            )

    to_revert = list(reversed(ordered[target_index + 1 : stored_index + 1]))
    irreversible = [migration.slug for migration in to_revert if migration.down is None]
    if irreversible:
        raise ValueError(
            "Cannot downgrade: migration(s) without a down() in the span: "
            + ", ".join(irreversible)
        )
    return to_revert
