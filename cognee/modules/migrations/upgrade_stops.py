"""Releases an old database has to be migrated by before this one.

A chain migration that writes through a store adapter runs with the adapter of the
release doing the upgrade. When that adapter no longer accepts the layout the
database is still in, the step fails half way down the chain and fails again on
every start. An upgrade stop names such a case and the release whose adapter
still accepts it. The runner checks the stops before it applies anything, so a
stopped database is left as it was, and the error says which release to upgrade
through.

Tested on cognee 1.3.0 databases (default stores and Postgres graph + pgvector,
with and without a document shared by two datasets): the only one this release
cannot migrate in one go is the Postgres graph with a shared document, and it
migrates through 1.6.10 and then this release.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sqlalchemy import select

from cognee.modules.migrations.migration import Migration, MigrationContext


@dataclass(frozen=True)
class UpgradeStop:
    """A database that has not passed ``before`` and ``applies`` must first be
    migrated by cognee ``release``."""

    before: str
    release: str
    reason: str
    applies: Callable[[MigrationContext], Awaitable[bool]]


class UpgradeThroughReleaseError(Exception):
    """This release cannot migrate the database; an older release has to first."""

    def __init__(self, stop: UpgradeStop, context: MigrationContext):
        database = (
            f"The databases of dataset {context.dataset_id}"
            if context.dataset_id is not None
            else "The global databases"
        )
        # The runner stores the first 500 characters of the error with the database
        # (``cognee-cli current``): keep the whole instruction inside them.
        super().__init__(
            f"{database} need cognee {stop.release} first, nothing was changed: "
            f"{stop.reason} Install cognee=={stop.release} and run `cognee-cli upgrade`, "
            "then install this release and run `cognee-cli upgrade` again."
        )
        self.stop = stop


async def _postgres_graph_with_forks_in_array_layout(context: MigrationContext) -> bool:
    """A Postgres graph still keeping source refs in array columns, with fork documents.

    ``rekey_fork_document_ids`` re-keys a fork document through the graph adapter,
    and this release's Postgres adapter refuses that layout, which only the later
    ``postgres_graph_source_ref_tables`` converts. Without fork documents the step
    has nothing to write, and the database migrates in one go.
    """
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data
    from cognee.modules.migrations.versions.postgres_graph_source_ref_tables import (
        stores_refs_in_array_columns,
    )

    if not await stores_refs_in_array_columns(context.graph_engine):
        return False
    forks = select(Data.id).where(Data.legacy_id.is_not(None)).limit(1)
    if context.dataset_id is not None:
        forks = forks.where(Data.dataset_id == context.dataset_id)
    async with get_relational_engine().get_async_session() as session:
        return (await session.execute(forks)).first() is not None


UPGRADE_STOPS = (
    UpgradeStop(
        before="rekey_fork_document_ids",
        release="1.6.10",
        reason=(
            "the Postgres graph is still in the source_ref_keys layout and has documents "
            "several datasets shared, which this release cannot re-key in that layout."
        ),
        applies=_postgres_graph_with_forks_in_array_layout,
    ),
)


async def check_upgrade_stops(context: MigrationContext, pending: list[Migration]) -> None:
    """Raise UpgradeThroughReleaseError when a stop applies to the database ``context``
    is for, about to run ``pending``."""
    pending_slugs = {migration.slug for migration in pending}
    for stop in UPGRADE_STOPS:
        if stop.before in pending_slugs and await stop.applies(context):
            raise UpgradeThroughReleaseError(stop, context)
