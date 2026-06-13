"""Startup migration orchestration.

Two stages, in order:

1. relational schema (Alembic) — must run first: it creates the revision
   columns / tables the next stage reads.
2. graph + vector script migrations (revision chains, ``runner.py``) — this
   includes the vector adapter's own storage-schema migration, which is a
   chain entry (``adapter_storage_migration``), so it is gated, locked and
   failure-isolated exactly like every other migration.

Triggered from the FastAPI lifespan on every server start, from the first
``remember()``/``cognify()`` call in an SDK process, and explicitly via
``cognee.run_startup_migrations()``. Set ``ENABLE_AUTO_MIGRATIONS=false`` to
disable ALL of these automatic runs (e.g. operating on deliberately
old-format data, or environments where migrations are operator-driven) —
``cognee-cli upgrade`` remains the explicit path and ignores the flag.

``run_startup_migrations`` is once-per-process: a cognee instance never needs
to run migrations twice (databases at head no-op anyway, but the relational
Alembic subprocess and the per-database row scan are not free). The guard
lives HERE, not in any caller — a concurrent second call waits on the lock,
and a FAILED run does not set the flag, so the next call retries.
"""

import asyncio
import logging
import os
import importlib.resources as pkg_resources

logger = logging.getLogger(__name__)

_startup_migrations_done = False
_startup_migrations_lock = None
_startup_migrations_lock_loop = None


def _get_startup_lock() -> asyncio.Lock:
    """The in-process migration lock, recreated when the event loop changes.

    SDK code commonly runs each call in its own ``asyncio.run()`` loop; an
    ``asyncio.Lock`` is bound to the loop it first awaited on and raises if
    reused on another, so a failed first attempt would otherwise make every
    retry from a fresh loop crash on the lock instead of retrying.
    """
    global _startup_migrations_lock, _startup_migrations_lock_loop
    loop = asyncio.get_running_loop()
    if _startup_migrations_lock is None or _startup_migrations_lock_loop is not loop:
        _startup_migrations_lock = asyncio.Lock()
        _startup_migrations_lock_loop = loop
    return _startup_migrations_lock


MIGRATIONS_PACKAGE = "cognee"
MIGRATIONS_DIR_NAME = "alembic"


class MigrationError(Exception):
    """Raised when migrations fail."""


async def abort_write_if_migration_blocked(failed: list[str], datasets, user) -> None:
    """Raise ``MigrationError`` if a database THIS write targets failed migration.

    ``failed`` is the result of :func:`run_startup_migrations` (the ids of the
    databases whose migration failed). Writing new-scheme data into a store still
    on the old scheme is the mixed-scheme corruption the migration exists to
    prevent — but the block is scoped, not global:

    * access control OFF — one global graph/vector pair backs EVERY dataset, so
      a failure is not dataset-scoped and any failure blocks all writes.
    * access control ON — each dataset has its own database pair, so block only
      when a dataset this call writes to is among the failed set. A brand-new
      dataset has no databases yet (so it cannot be in ``failed``);
      ``get_authorized_existing_datasets`` resolves existing datasets only and
      never creates one, so the check has no side effects.

    ``datasets`` is the caller's dataset selector (name(s)/UUID(s), or ``None``
    for "all of the user's datasets"); ``user`` may be ``None`` (default user).
    """
    if not failed:
        return

    from cognee.context_global_variables import backend_access_control_enabled

    if not backend_access_control_enabled():
        raise MigrationError(
            "Write aborted: database migration failed for the global database "
            f"({', '.join(failed)}). Writing now would mix id schemes. Inspect with "
            "`cognee-cli current`; it retries automatically on the next call."
        )

    from uuid import UUID

    from cognee.modules.data.methods import get_authorized_existing_datasets
    from cognee.modules.users.methods import get_default_user

    if isinstance(datasets, (str, UUID)):
        datasets = [datasets]
    if user is None:
        user = await get_default_user()

    failed_set = set(failed)
    targets = await get_authorized_existing_datasets(datasets, "write", user)
    blocked = [dataset for dataset in targets if str(dataset.id) in failed_set]
    if blocked:
        names = ", ".join(f"{dataset.name} ({dataset.id})" for dataset in blocked)
        raise MigrationError(
            f"Write aborted: database migration failed for dataset(s) {names}. "
            "Writing now would mix id schemes. Inspect with `cognee-cli current`; "
            "it retries automatically on the next call."
        )


def _auto_migrations_enabled() -> bool:
    """Read ENABLE_AUTO_MIGRATIONS dynamically — tests/embedders set it via
    os.environ after import, so it must not be frozen at module load."""
    return os.getenv("ENABLE_AUTO_MIGRATIONS", "true").lower() not in ("false", "0", "no")


async def run_relational_migrations():
    """Apply the Alembic relational-schema migrations, in-process.

    Runs ``alembic.command.upgrade`` in a worker thread (``env.py`` drives an
    async engine via ``asyncio.run``, which needs a thread with no running
    event loop). IN-PROCESS on purpose — not a subprocess: ``env.py`` resolves
    the database from ``get_relational_engine()``, so programmatically
    configured roots/credentials (``cognee.config.system_root_directory(...)``,
    as every test/example uses) are honored. A subprocess inherits only
    environment variables and silently migrates the DEFAULT-location database
    instead — the bug behind the library-test/example CI failures. The thread
    also keeps the caller's event loop unblocked for the duration.
    """
    # Locate the Alembic configuration within the installed package.
    package_root = str(pkg_resources.files(MIGRATIONS_PACKAGE))
    alembic_ini_path = os.path.join(package_root, "alembic.ini")
    script_location_path = os.path.join(package_root, MIGRATIONS_DIR_NAME)

    if not os.path.exists(alembic_ini_path):
        raise FileNotFoundError(
            f"Error: alembic.ini not found at expected locations for package '{MIGRATIONS_PACKAGE}'."
        )
    if not os.path.exists(script_location_path):
        raise FileNotFoundError(
            f"Error: Migrations directory not found at expected locations for package '{MIGRATIONS_PACKAGE}'."
        )

    def _upgrade_to_head():
        from alembic import command
        from alembic.config import Config

        alembic_config = Config(alembic_ini_path)
        alembic_config.set_main_option("script_location", script_location_path)
        # Tell env.py not to fileConfig() — that would reconfigure (and
        # disable) the host process's loggers from alembic.ini.
        alembic_config.attributes["configure_logger"] = False
        command.upgrade(alembic_config, "head")

    try:
        await asyncio.to_thread(_upgrade_to_head)
    except Exception as error:
        logger.error("Migration failed with unexpected error: %s", error)
        raise MigrationError("Relational DB Migrations failed.") from error

    logger.info("Migration completed successfully.")


async def run_startup_migrations() -> list[str]:
    """Run all startup migrations: relational schema first, then the graph +
    vector revision chains. Once per process (see module docstring); a failed
    run is retried on the next call.

    Returns the identifiers of the databases whose migration FAILED (empty list
    means everything is at head). Callers that are about to WRITE new-scheme
    data — ``cognify()`` / ``remember()`` — must treat a non-empty result as a
    hard stop: writing into an un-migrated store is exactly the mixed-scheme
    corruption the migration exists to prevent. Non-write callers (the API
    lifespan) may ignore it; per-request writes still block via those entry
    points, so the server can come up and migrations retry on the next call.
    """
    global _startup_migrations_done
    if not _auto_migrations_enabled():
        logger.info(
            "Automatic migrations are disabled (ENABLE_AUTO_MIGRATIONS=false); "
            "run `cognee-cli upgrade` to migrate explicitly."
        )
        return []
    if _startup_migrations_done:
        return []

    async with _get_startup_lock():
        if _startup_migrations_done:
            return []

        from cognee.modules.migrations.runner import run_database_migrations

        try:
            await run_relational_migrations()
        except MigrationError:
            # cognee's Alembic chain is not self-sufficient on an EMPTY
            # database — it patches a schema bootstrapped by
            # Base.metadata.create_all (e.g. the acls migration assumes its
            # table already exists). Bootstrap the schema and retry once: the
            # same first-boot recovery the API lifespan has always used, now
            # available to every caller (remember(), MCP, explicit SDK calls).
            from cognee.infrastructure.databases.relational import get_relational_engine

            logger.info(
                "Alembic failed on an unbootstrapped database; creating schema and retrying."
            )
            db_engine = get_relational_engine()
            await db_engine.create_database()
            await run_relational_migrations()

        summaries = await run_database_migrations()

        # Mark done only when every database succeeded: a failed dataset must
        # be retried by the next call in this process, exactly as the module
        # docstring promises. (An empty summary list — no datasets yet — is a
        # clean outcome.)
        failed = [
            summary.get("dataset_id") or summary.get("database", "?")
            for summary in summaries
            if summary.get("result") == "failed"
        ]
        if failed:
            logger.warning(
                "Migrations FAILED for %d database(s): %s. Writes into them are blocked "
                "(would duplicate entities until they migrate). Inspect with `cognee-cli "
                "current` (shows the recorded error); retried on the next call/startup.",
                len(failed),
                ", ".join(failed),
            )
        else:
            _startup_migrations_done = True

        return failed
