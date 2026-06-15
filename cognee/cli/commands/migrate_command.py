"""Alembic-style CLI commands for graph/vector database migrations.

    cognee upgrade [revision]     apply migrations up to a revision (default: head)
    cognee downgrade <revision>   revert migrations down to a revision ('base' = all)
    cognee history                show the migration chains (newest first)
    cognee current                show each database's stamped revision

Revisions are migration slugs in ONE chain covering graph + vector +
relational-ledger changes (migrations are cross-store transformations).
"""

import argparse
import asyncio

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException


def _validate_revision(revision: str, keywords: tuple) -> None:
    """Error like alembic's "Can't locate revision" for unknown slugs."""
    from cognee.modules.migrations.registry import MIGRATIONS

    if revision in keywords:
        return
    known = [migration.slug for migration in MIGRATIONS]
    if revision not in known:
        fmt.error(
            f"Can't locate revision identified by '{revision}'. Known revisions: "
            + ", ".join(known)
        )
        raise CliCommandException(f"Unknown revision: {revision}", error_code=1)


def _print_summaries(summaries: list, key: str, failure_hint: str) -> None:
    if not summaries:
        fmt.note("No databases found — nothing to do.")
        return
    for summary in summaries:
        target = summary.get("dataset_id") or summary.get("database", "?")
        if summary.get("result") == "failed":
            fmt.error(f"  {target}: FAILED ({failure_hint})")
            continue
        ran = summary.get(key) or []
        if ran:
            fmt.success(f"  {target}: {', '.join(ran)}")
        else:
            fmt.echo(f"  {target}: up to date")


def _raise_on_failures(summaries: list, action: str) -> None:
    if any(s.get("result") == "failed" for s in summaries):
        message = f"One or more databases failed to {action} — see the output above."
        fmt.error(message)
        raise CliCommandException(message, error_code=1)


def _bookkeeping_guard(error: Exception) -> None:
    """Translate missing-bookkeeping-schema errors into an actionable hint."""
    from sqlalchemy.exc import OperationalError, ProgrammingError

    if isinstance(error, (OperationalError, ProgrammingError)):
        fmt.error(
            "The migration bookkeeping schema is missing or outdated on this "
            "database — run `cognee-cli upgrade` first."
        )
        raise CliCommandException("Bookkeeping schema missing", error_code=1)
    raise error


class UpgradeCommand(SupportsCliCommand):
    command_string = "upgrade"
    help_string = "Upgrade databases to a later migration revision (like `alembic upgrade`)"
    docs_url = DEFAULT_DOCS_URL
    description = """
Upgrade databases to a later migration revision.

REVISION is 'head' (default) or a migration slug, which upgrades the chain up
to and including it (alembic-style partial upgrade). Runs the
relational (Alembic) schema migrations first. Safe to run anytime: databases
already at the target are skipped with an in-memory check.

Examples:
  cognee upgrade
  cognee upgrade head
  cognee upgrade namespace_entity_type_node_ids
"""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "revision",
            nargs="?",
            default="head",
            help="Data-migration target: 'head' (default) or a migration slug",
        )
        parser.add_argument(
            "--alembic",
            default="head",
            metavar="REV",
            help="Relational (Alembic) schema target: 'head' (default) or an Alembic revision",
        )

    def execute(self, args: argparse.Namespace) -> None:
        _validate_revision(args.revision, ("head",))

        async def run():
            from cognee.modules.migrations.startup import apply_all_migrations

            fmt.echo(
                f"Migrating all databases — relational schema to '{args.alembic}', "
                f"graph/vector to '{args.revision}'..."
            )
            # The SAME locked relational + graph/vector sequence startup runs — one
            # global migration lock, no duplicated bootstrap logic. Unlike
            # run_startup_migrations it is not gated by ENABLE_AUTO_MIGRATIONS, so an
            # explicit upgrade works even when automatic migrations are turned off.
            return await apply_all_migrations(
                data_target=args.revision, relational_target=args.alembic
            )

        try:
            summaries = asyncio.run(run())
        except Exception as error:  # noqa: BLE001 - translated to an actionable hint
            _bookkeeping_guard(error)
        _print_summaries(
            summaries,
            "migrations_applied",
            "see logs; it will be retried on the next startup/upgrade",
        )
        _raise_on_failures(summaries, "upgrade")
        fmt.success("Upgrade complete.")


class DowngradeCommand(SupportsCliCommand):
    command_string = "downgrade"
    help_string = "Revert databases to an earlier revision (like `alembic downgrade`)"
    docs_url = DEFAULT_DOCS_URL
    description = """
Revert applied graph/vector migrations down to a revision.

REVISION is required (alembic-style): 'base' reverts EVERY applied data
migration (revision back to NULL — the next upgrade re-applies everything), or a
migration slug, which downgrades down TO it (the slug itself stays applied).
Only spans where every migration defines a down() can be reverted. This REWRITES DATA — use it when rolling back releases.

The relational (Alembic) schema is left untouched UNLESS you pass --alembic with
a target; the data chain is reverted FIRST, then the schema. The schema cannot be
taken below the revisions that hold the data-migration bookkeeping unless the data
chain is going to 'base' in the same call.

Examples:
  cognee downgrade base
  cognee downgrade namespace_entity_type_node_ids
  cognee downgrade base --alembic base          # full rollback: data + schema
"""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "revision",
            help="Data-migration target: 'base' (revert everything) or a migration slug",
        )
        parser.add_argument(
            "--alembic",
            default=None,
            metavar="REV",
            help="Also downgrade the relational schema to this Alembic revision (or 'base'); "
            "omit to leave the schema untouched",
        )
        parser.add_argument(
            "--dataset",
            action="append",
            default=None,
            help="Restrict to a dataset UUID (repeatable; default: all datasets)",
        )
        parser.add_argument(
            "--force", "-f", action="store_true", help="Skip the confirmation prompt"
        )

    def execute(self, args: argparse.Namespace) -> None:
        from uuid import UUID

        _validate_revision(args.revision, ("base",))
        dataset_ids = [UUID(d) for d in args.dataset] if args.dataset else None

        if not args.force:
            scope = f"{len(dataset_ids)} dataset(s)" if dataset_ids else "ALL databases"
            schema_note = f" AND the relational schema to '{args.alembic}'" if args.alembic else ""
            if not fmt.confirm(
                f"Downgrade {scope} to '{args.revision}'{schema_note}? This rewrites data, and "
                "entities whose name collides across Entity/EntityType merge into one node on "
                "the old scheme (lossy — that was the old scheme's #2515 bug).",
                default=False,
            ):
                fmt.note("Aborted.")
                return

        async def run():
            from cognee.modules.migrations.startup import revert_all_migrations

            # One global lock, data chain first, then (opt-in) the relational schema.
            # `revision` is required, so a data target is always given here; --alembic
            # (None when omitted) leaves the schema untouched.
            return await revert_all_migrations(
                data_target=args.revision,
                relational_target=args.alembic,
                dataset_ids=dataset_ids,
            )

        try:
            summaries = asyncio.run(run())
        except Exception as error:  # noqa: BLE001 - translated to an actionable hint
            _bookkeeping_guard(error)
        _print_summaries(
            summaries,
            "migrations_reverted",
            "see logs; downgrades never run automatically — fix and re-run this command",
        )
        _raise_on_failures(summaries, "downgrade")
        fmt.success("Downgrade complete.")


class HistoryCommand(SupportsCliCommand):
    command_string = "history"
    help_string = "List migration revisions, newest first (like `alembic history`)"
    docs_url = DEFAULT_DOCS_URL
    description = """
List the migration chain, newest first, in alembic's
'down_revision -> revision' format. '(head)' marks each chain's latest
revision; '<base>' is the pre-chain state.
"""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> None:
        from cognee.modules.migrations.migration import order_migrations
        from cognee.modules.migrations.registry import MIGRATIONS

        ordered = order_migrations(MIGRATIONS)
        if not ordered:
            fmt.echo("(no migrations)")
        for index, migration in enumerate(reversed(ordered)):
            head = " (head)" if index == 0 else ""
            parent = migration.down_revision or "<base>"
            reversible = "reversible" if migration.down else "irreversible"
            fmt.echo(
                f"{parent} -> {migration.revision}{head}, "
                f"cognee {migration.cognee_version}, {reversible}"
            )


class CurrentCommand(SupportsCliCommand):
    command_string = "current"
    help_string = "Show each database's stamped revision (like `alembic current`)"
    docs_url = DEFAULT_DOCS_URL
    description = """
Show the currently stamped revision for every database — per
dataset with access control on, the global pair with it off. '<base>' means no
migration has been applied (everything pending).
"""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> None:
        async def run():
            from cognee.context_global_variables import backend_access_control_enabled
            from cognee.infrastructure.databases.relational import get_relational_engine
            from cognee.modules.data.methods.get_dataset_databases import get_dataset_databases
            from cognee.modules.migrations.migration import head_revision
            from cognee.modules.migrations.models import (
                GLOBAL_DATABASE_VERSION_ROW_ID,
                GlobalDatabaseVersion,
            )
            from cognee.modules.migrations.registry import MIGRATIONS

            head = head_revision(MIGRATIONS)

            def fmt_revision(revision):
                if revision is None:
                    return "<base>"
                return f"{revision} (head)" if revision == head else revision

            if backend_access_control_enabled():
                rows = await get_dataset_databases()
                if not rows:
                    fmt.note("No dataset databases found.")
                    return
                for row in rows:
                    fmt.echo(f"{row.dataset_id}  {fmt_revision(row.migration_revision)}")
                    if row.migration_last_error:
                        fmt.error(f"  last migration error: {row.migration_last_error}")
            else:
                db_engine = get_relational_engine()
                async with db_engine.get_async_session() as session:
                    record = await session.get(
                        GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID
                    )
                if record is None:
                    fmt.note("No global_database_version row yet — run `cognee upgrade`.")
                    return
                fmt.echo(f"global  {fmt_revision(record.global_migration_revision)}")
                if record.global_migration_last_error:
                    fmt.error(f"  last migration error: {record.global_migration_last_error}")

        try:
            asyncio.run(run())
        except Exception as error:  # noqa: BLE001 - translated to an actionable hint
            _bookkeeping_guard(error)


class StampCommand(SupportsCliCommand):
    command_string = "stamp"
    help_string = "Set the stored revision WITHOUT running migrations (like `alembic stamp`)"
    docs_url = DEFAULT_DOCS_URL
    description = """
Set the stored revision without running any migration.

For repairing bookkeeping that has drifted from reality — e.g. a restored
graph/vector backup sitting behind a head-stamped row (stamp 'base', then
`cognee upgrade` re-runs the idempotent chain against it), or data you have
verified by hand. REVISION is 'head', 'base', or a migration slug.
Never touches data.

Examples:
  cognee stamp base --dataset 7df514cd-...   # re-arm migrations for one dataset
  cognee stamp head                          # mark everything migrated (dangerous)
"""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "revision",
            help="Revision to stamp: 'head', 'base', or a migration slug",
        )
        parser.add_argument(
            "--dataset",
            action="append",
            default=None,
            help="Restrict to a dataset UUID (repeatable; default: all databases)",
        )
        parser.add_argument(
            "--force", "-f", action="store_true", help="Skip the confirmation prompt"
        )

    def execute(self, args: argparse.Namespace) -> None:
        from uuid import UUID

        _validate_revision(args.revision, ("head", "base"))
        dataset_ids = [UUID(d) for d in args.dataset] if args.dataset else None

        if not args.force:
            scope = f"{len(dataset_ids)} dataset(s)" if dataset_ids else "ALL databases"
            if not fmt.confirm(
                f"Stamp {scope} at '{args.revision}' WITHOUT running migrations? "
                "Stamping 'head' over unmigrated data permanently skips its migrations.",
                default=False,
            ):
                fmt.note("Aborted.")
                return

        async def run():
            from cognee.modules.migrations.runner import stamp_revisions

            return await stamp_revisions(target=args.revision, dataset_ids=dataset_ids)

        try:
            summaries = asyncio.run(run())
        except Exception as error:  # noqa: BLE001 - translated to an actionable hint
            _bookkeeping_guard(error)
        if not summaries:
            fmt.note("No databases found — nothing stamped.")
            return
        for summary in summaries:
            target = summary.get("dataset_id") or summary.get("database", "?")
            fmt.success(f"  {target}: {summary['revision'] or '<base>'}")
        fmt.success("Stamp complete.")
