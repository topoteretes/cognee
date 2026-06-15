"""Startup migration orchestration.

Two stages, in order:

1. relational schema (Alembic) — must run first: it creates the revision
   columns / tables the next stage reads.
2. graph + vector script migrations (revision chain, ``runner.py``), followed
   by the vector adapter's own storage-schema sync. The chain runs once per
   database (gated by the stored revision slug); the adapter sync instead runs
   on every Cognee version change (gated by a library-vs-recorded
   ``cognee_version`` mismatch), after the chain — both under the same per-
   database lock and failure isolation.

Triggered from the FastAPI lifespan on every server start, from the first
``remember()``/``cognify()`` call in an SDK process, and explicitly via
``cognee.run_migrations()``. Set ``ENABLE_AUTO_MIGRATIONS=false`` to
disable ALL of these automatic runs (e.g. operating on deliberately
old-format data, or environments where migrations are operator-driven) —
``cognee-cli upgrade`` remains the explicit path and ignores the flag.

``run_migrations`` is once-per-process: a cognee instance never needs
to run migrations twice (databases at head no-op anyway, but the relational
Alembic subprocess and the per-database row scan are not free). The guard
lives HERE, not in any caller — a concurrent second call waits on the lock,
and a FAILED run does not set the flag, so the next call retries.
"""

import asyncio
import logging
import os
import importlib.resources as pkg_resources
from typing import Optional

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
    """Raise if a database this write targets failed migration — writing into a
    store still on the old scheme is the corruption the migration prevents.

    ``failed`` comes from :func:`run_migrations`. The block is scoped:
    access control OFF, one global pair backs every dataset, so any failure
    blocks all writes; access control ON, block only when a dataset this call
    targets is in ``failed`` (a brand-new dataset has no DB yet, and
    ``get_authorized_existing_datasets`` resolves existing datasets only — no
    side effects). ``datasets`` is the caller's selector (None = all of the
    user's); ``user`` may be None (default user).
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


def _build_alembic_config():
    """Alembic Config pointing at the package's alembic.ini and versions dir.

    ``configure_logger`` is off so env.py doesn't fileConfig()-reset the host
    process's loggers.
    """
    from alembic.config import Config

    package_root = str(pkg_resources.files(MIGRATIONS_PACKAGE))
    alembic_ini_path = os.path.join(package_root, "alembic.ini")
    script_location_path = os.path.join(package_root, MIGRATIONS_DIR_NAME)
    if not os.path.exists(alembic_ini_path):
        raise FileNotFoundError(f"alembic.ini not found for package '{MIGRATIONS_PACKAGE}'.")
    if not os.path.exists(script_location_path):
        raise FileNotFoundError(
            f"Alembic versions dir not found for package '{MIGRATIONS_PACKAGE}'."
        )

    config = Config(alembic_ini_path)
    config.set_main_option("script_location", script_location_path)
    config.attributes["configure_logger"] = False
    return config


async def run_relational_migrations(target: str = "head"):
    """Apply the Alembic relational-schema migrations up to ``target`` (default
    head), in-process.

    Run in a worker thread (env.py drives an async engine via asyncio.run, which
    needs a thread with no running loop), NOT a subprocess: env.py resolves the
    database from get_relational_engine(), so programmatic config
    (cognee.config.system_root_directory(...)) is honored. A subprocess would
    inherit only env vars and migrate the default-location database instead.
    """

    def _upgrade():
        from alembic import command

        command.upgrade(_build_alembic_config(), target)

    try:
        await asyncio.to_thread(_upgrade)
    except Exception as error:
        logger.error("Relational migration failed: %s", error)
        raise MigrationError("Relational DB Migrations failed.") from error

    logger.info("Relational migrations applied (target %s).", target)


async def run_relational_downgrade(target: str):
    """Revert the Alembic relational-schema migrations DOWN to ``target``
    (an Alembic revision, or ``"base"`` for the empty schema), in-process.

    Same in-thread, in-process rationale as ``run_relational_migrations``.
    EXPLICIT operator action only — the data chain must already be reverted past
    anything that relies on the schema being dropped (see ``revert_all_migrations``).
    """

    def _downgrade():
        from alembic import command

        command.downgrade(_build_alembic_config(), target)

    try:
        await asyncio.to_thread(_downgrade)
    except Exception as error:
        logger.error("Relational downgrade failed: %s", error)
        raise MigrationError("Relational DB downgrade failed.") from error

    logger.info("Relational schema downgraded (target %s).", target)


async def run_relational_stamp(revision: str = "head"):
    """Record an Alembic revision without running any migration.

    Used after ``create_database`` builds a fresh schema with
    ``Base.metadata.create_all``: that schema already IS head, so we stamp it
    instead of replaying every historical migration.
    """

    def _stamp():
        from alembic import command

        command.stamp(_build_alembic_config(), revision)

    await asyncio.to_thread(_stamp)
    logger.info("Stamped fresh relational schema at %s.", revision)


async def _relational_schema_exists() -> bool:
    """True if the relational database already holds cognee's schema.

    We look for EITHER the ``users`` table OR Alembic's ``alembic_version``,
    because neither alone is reliable across cognee's history:
      - ``users`` is a core table present in every initialized database,
        including LEGACY ones created before Alembic was wired in — those have
        no ``alembic_version`` at all, so checking only ``alembic_version``
        would mistake a populated legacy DB for an empty one and wrongly stamp
        it at head.
      - ``alembic_version`` is Alembic's own marker, so it still identifies a
        migration-managed DB even if the ``users`` table is renamed or
        restructured in the future.
    Only when NEITHER exists is the database genuinely empty — the one state
    where create_all + stamp head is safe.
    """
    from sqlalchemy import inspect

    from cognee.infrastructure.databases.relational import get_relational_engine

    try:
        async with get_relational_engine().engine.connect() as connection:
            tables = await connection.run_sync(
                lambda sync_conn: inspect(sync_conn).get_table_names()
            )
    except Exception:
        return False  # cannot inspect (e.g. brand-new SQLite file) -> treat as empty
    return "users" in tables or "alembic_version" in tables


# Alembic revisions that CREATE the cognee data-migration bookkeeping (the
# dataset_database tracking columns and the global_database_version table). The
# data-migration system reads/writes these, so the relational schema must not be
# downgraded below them while any data migration is still applied.
_DATA_BOOKKEEPING_ALEMBIC_REVISIONS = ("c1a2b3d4e5f9", "d8f4a1b2c3e9")


def _relational_downgrade_drops_bookkeeping(relational_target: str) -> bool:
    """True if downgrading the relational schema to ``relational_target`` would
    revert (drop) a migration that holds the data-migration bookkeeping.

    A revision survives a downgrade-to-``target`` iff it is ``target`` or one of
    its ancestors; anything above ``target`` is reverted. ``base`` reverts all.
    """
    if relational_target == "base":
        return True

    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(_build_alembic_config())
    # ``target`` + all its ancestors down to base = the revisions that stay applied.
    kept = {revision.revision for revision in script.iterate_revisions(relational_target, "base")}
    return any(revision not in kept for revision in _DATA_BOOKKEEPING_ALEMBIC_REVISIONS)


async def apply_all_migrations(
    data_target: str = "head", relational_target: str = "head"
) -> list[dict]:
    """UPGRADE every database under the ONE global migration lock: the relational
    schema FIRST (Alembic, to ``relational_target``), then the graph/vector data
    chain (to ``data_target``). Both default to head.

    Relational goes first because it owns the tables the data chain (and its own
    bookkeeping) live in. The single place the full upgrade sequence lives, so
    startup AND the CLI ``upgrade`` share identical behavior and one lock
    (run_database_migrations does not lock itself — see ``migration_lock``). Returns
    the runner's per-database summaries. Does NOT apply the ENABLE_AUTO_MIGRATIONS
    gate or the once-per-process guard — those belong to ``run_migrations``;
    the CLI must migrate even when automatic migrations are disabled.
    """
    from cognee.modules.migrations.runner import migration_lock, run_database_migrations
    from cognee.infrastructure.databases.relational import get_relational_engine

    async with migration_lock():
        if await _relational_schema_exists():
            # Existing database: apply pending migrations up to relational_target.
            await run_relational_migrations(relational_target)
        else:
            # Fresh database: create_all builds the schema at HEAD, so stamp head
            # rather than replaying history. A partial relational_target only makes
            # sense for an existing DB — a fresh one is head by construction.
            logger.info("Fresh database: creating schema and stamping at head.")
            await get_relational_engine().create_database()
            await run_relational_stamp("head")

        return await run_database_migrations(data_target)


async def revert_all_migrations(
    data_target: Optional[str] = None,
    relational_target: Optional[str] = None,
    dataset_ids: Optional[list] = None,
) -> list[dict]:
    """DOWNGRADE under the ONE global migration lock, in REVERSE order: the data
    chain FIRST, then the relational schema.

    Each target is INDEPENDENT and explicit — ``None`` means "leave this store
    alone", so you must name where to downgrade to:

      - ``data_target``:        ``None`` = don't touch the data chain; ``"base"`` =
                                revert EVERY data migration; a slug = revert down to it.
      - ``relational_target``:  ``None`` = don't touch the schema; ``"base"`` or an
                                Alembic revision = downgrade the schema to it.

    A call with BOTH targets ``None`` is a no-op and almost certainly a mistake, so
    it raises — a downgrade must say where to go. The relational schema cannot be
    dropped below the revisions that hold the data-migration bookkeeping unless the
    data chain is going to ``"base"`` in the same call.
    """
    from cognee.modules.migrations.runner import downgrade_database_migrations, migration_lock

    if data_target is None and relational_target is None:
        raise MigrationError(
            "Nothing to downgrade: specify a data target ('base' or a migration slug) "
            "and/or a relational target ('base' or an Alembic revision)."
        )

    # Coupling guard: dropping the relational tables that hold the data-migration
    # bookkeeping is only safe once the data chain is fully reverted (to 'base').
    if (
        relational_target is not None
        and data_target != "base"
        and _relational_downgrade_drops_bookkeeping(relational_target)
    ):
        raise MigrationError(
            f"Refusing to downgrade the relational schema to {relational_target!r}: it would "
            "drop the data-migration bookkeeping tables while data migrations remain applied "
            f"(data target {data_target!r}). Pass 'base' as the data target in the same call to "
            "revert the data chain fully first."
        )

    async with migration_lock():
        summaries: list[dict] = []
        if data_target is not None:
            # 'base' reverts everything -> the runner's None sentinel; else a slug.
            summaries = await downgrade_database_migrations(
                target_revision=None if data_target == "base" else data_target,
                dataset_ids=dataset_ids,
            )
        if relational_target is not None:
            await run_relational_downgrade(relational_target)
            summaries.append({"database": "relational", "downgraded_to": relational_target})
        return summaries


async def run_migrations() -> list[str]:
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

        # The full sequence (relational + graph/vector) under the ONE global
        # migration lock. _get_startup_lock above only serializes coroutines within
        # THIS process; the cross-process race (two cognify subprocesses both running
        # `alembic upgrade head`, the loser dying on the alembic_version create) is
        # serialized by the global lock inside apply_all_migrations.
        summaries = await apply_all_migrations("head")

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


async def run_migrations_and_block(datasets, user) -> None:
    """Run startup migrations, then block this write if a dataset it targets failed.

    The shared entry point for write paths (cognify/remember). The once-per-process
    guard lives in ``run_migrations``, so after the first run this is just a
    flag check plus the scoped block.
    """
    failed = await run_migrations()
    await abort_write_if_migration_blocked(failed, datasets, user)
