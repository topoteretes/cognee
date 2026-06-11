"""Alembic-style CLI commands for graph/vector database migrations.

    cognee upgrade [revision]     apply migrations up to a revision (default: head)
    cognee downgrade <revision>   revert migrations down to a revision ('base' = all)
    cognee history                show the migration chains (newest first)
    cognee current                show each database's stamped revision

Revisions are migration slugs; a slug belongs to exactly one chain (graph or
vector), so targeting one upgrades/downgrades only that chain and leaves the
other untouched — like pointing alembic at one of two version locations.
"""

import argparse
import asyncio

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException


def _resolve_targets(revision: str, base_keyword: str, base_value):
    """Map a CLI revision argument onto (graph_target, vector_target).

    ``head``/``base`` apply to both chains; a slug applies to its own chain and
    KEEPs the other. Unknown revisions are an error, like alembic's
    "Can't locate revision".
    """
    from cognee.modules.migrations.graph_migrations import GRAPH_MIGRATIONS
    from cognee.modules.migrations.runner import KEEP
    from cognee.modules.migrations.vector_migrations import VECTOR_MIGRATIONS

    if revision == base_keyword:
        return base_value, base_value

    graph_slugs = {migration.slug for migration in GRAPH_MIGRATIONS}
    vector_slugs = {migration.slug for migration in VECTOR_MIGRATIONS}
    if revision in graph_slugs:
        return revision, KEEP
    if revision in vector_slugs:
        return KEEP, revision

    known = ", ".join(sorted(graph_slugs | vector_slugs))
    fmt.error(f"Can't locate revision identified by '{revision}'. Known revisions: {known}")
    raise CliCommandException(f"Unknown revision: {revision}", error_code=1)


def _print_summaries(summaries: list, keys: tuple, failure_hint: str) -> None:
    if not summaries:
        fmt.note("No databases found — nothing to do.")
        return
    for summary in summaries:
        target = summary.get("dataset_id") or summary.get("database", "?")
        if summary.get("result") == "failed":
            fmt.error(f"  {target}: FAILED ({failure_hint})")
            continue
        ran = [slug for key in keys for slug in (summary.get(key) or [])]
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

REVISION is 'head' (default — both chains to their latest) or a migration slug,
which upgrades only that slug's chain up to and including it. Runs the
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
            help="Target revision: 'head' (default) or a migration slug",
        )

    def execute(self, args: argparse.Namespace) -> None:
        graph_target, vector_target = _resolve_targets(args.revision, "head", "head")

        async def run():
            from cognee.modules.migrations.runner import run_database_migrations
            from cognee.modules.migrations.startup import run_relational_migrations

            fmt.echo("Running relational (Alembic) migrations...")
            await run_relational_migrations()
            fmt.echo(f"Upgrading graph/vector chains to '{args.revision}'...")
            return await run_database_migrations(
                graph_target=graph_target, vector_target=vector_target
            )

        summaries = asyncio.run(run())
        _print_summaries(
            summaries,
            ("graph_migrations_applied", "vector_migrations_applied"),
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

REVISION is required (alembic-style): 'base' reverts EVERY applied migration in
both chains (revisions back to NULL — the next upgrade re-applies everything),
or a migration slug, which downgrades only that slug's chain down TO it (the
slug itself stays applied). Only spans where every migration defines a down()
can be reverted. This REWRITES DATA — use it when rolling back releases.

The relational (Alembic) schema is NOT downgraded by this command.

Examples:
  cognee downgrade base
  cognee downgrade namespace_entity_type_node_ids
"""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "revision",
            help="Target revision: 'base' (revert everything) or a migration slug",
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

        graph_target, vector_target = _resolve_targets(args.revision, "base", None)
        dataset_ids = [UUID(d) for d in args.dataset] if args.dataset else None

        if not args.force:
            scope = f"{len(dataset_ids)} dataset(s)" if dataset_ids else "ALL databases"
            if not fmt.confirm(
                f"Downgrade {scope} to '{args.revision}'? This rewrites data, and entities "
                "whose name collides across Entity/EntityType merge into one node on the "
                "old scheme (lossy — that was the old scheme's #2515 bug).",
                default=False,
            ):
                fmt.note("Aborted.")
                return

        async def run():
            from cognee.modules.migrations.runner import downgrade_database_migrations

            return await downgrade_database_migrations(
                graph_target_revision=graph_target,
                vector_target_revision=vector_target,
                dataset_ids=dataset_ids,
            )

        try:
            summaries = asyncio.run(run())
        except Exception as error:  # noqa: BLE001 - translated to an actionable hint
            _bookkeeping_guard(error)
        _print_summaries(
            summaries,
            ("graph_migrations_reverted", "vector_migrations_reverted"),
            "see logs; downgrades never run automatically — fix and re-run this command",
        )
        _raise_on_failures(summaries, "downgrade")
        fmt.success("Downgrade complete.")


class HistoryCommand(SupportsCliCommand):
    command_string = "history"
    help_string = "List migration revisions, newest first (like `alembic history`)"
    docs_url = DEFAULT_DOCS_URL
    description = """
List the graph and vector migration chains, newest first, in alembic's
'down_revision -> revision' format. '(head)' marks each chain's latest
revision; '<base>' is the pre-chain state.
"""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> None:
        from cognee.modules.migrations.graph_migrations import GRAPH_MIGRATIONS
        from cognee.modules.migrations.migration import order_migrations
        from cognee.modules.migrations.vector_migrations import VECTOR_MIGRATIONS

        for title, migrations in (
            ("Graph chain:", GRAPH_MIGRATIONS),
            ("Vector chain:", VECTOR_MIGRATIONS),
        ):
            fmt.echo(fmt.bold(title))
            ordered = order_migrations(migrations)
            if not ordered:
                fmt.echo("  (no migrations)")
            for index, migration in enumerate(reversed(ordered)):
                head = " (head)" if index == 0 else ""
                parent = migration.down_revision or "<base>"
                reversible = "reversible" if migration.down else "irreversible"
                fmt.echo(
                    f"  {parent} -> {migration.revision}{head}, "
                    f"cognee {migration.cognee_version}, {reversible}"
                )
            fmt.echo()


class CurrentCommand(SupportsCliCommand):
    command_string = "current"
    help_string = "Show each database's stamped revision (like `alembic current`)"
    docs_url = DEFAULT_DOCS_URL
    description = """
Show the currently stamped graph/vector revision for every database — per
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
            from cognee.modules.migrations.graph_migrations import GRAPH_MIGRATIONS
            from cognee.modules.migrations.migration import head_revision
            from cognee.modules.migrations.models import (
                GLOBAL_DATABASE_VERSION_ROW_ID,
                GlobalDatabaseVersion,
            )
            from cognee.modules.migrations.vector_migrations import VECTOR_MIGRATIONS

            graph_head = head_revision(GRAPH_MIGRATIONS)
            vector_head = head_revision(VECTOR_MIGRATIONS)

            def fmt_revision(revision, head):
                if revision is None:
                    return "<base>"
                return f"{revision} (head)" if revision == head else revision

            if backend_access_control_enabled():
                rows = await get_dataset_databases()
                if not rows:
                    fmt.note("No dataset databases found.")
                    return
                for row in rows:
                    fmt.echo(
                        f"{row.dataset_id}  "
                        f"graph: {fmt_revision(row.graph_migration_revision, graph_head)}  "
                        f"vector: {fmt_revision(row.vector_migration_revision, vector_head)}"
                    )
            else:
                db_engine = get_relational_engine()
                async with db_engine.get_async_session() as session:
                    record = await session.get(
                        GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID
                    )
                if record is None:
                    fmt.note("No global_database_version row yet — run `cognee upgrade`.")
                    return
                fmt.echo(
                    f"global  "
                    f"graph: {fmt_revision(record.global_graph_migration_revision, graph_head)}  "
                    f"vector: {fmt_revision(record.global_vector_migration_revision, vector_head)}"
                )

        try:
            asyncio.run(run())
        except Exception as error:  # noqa: BLE001 - translated to an actionable hint
            _bookkeeping_guard(error)


class StampCommand(SupportsCliCommand):
    command_string = "stamp"
    help_string = "Set the stored revision WITHOUT running migrations (like `alembic stamp`)"
    docs_url = DEFAULT_DOCS_URL
    description = """
Set the stored graph/vector revisions without running any migration.

For repairing bookkeeping that has drifted from reality — e.g. a restored
graph/vector backup sitting behind a head-stamped row (stamp 'base', then
`cognee upgrade` re-runs the idempotent chain against it), or data you have
verified by hand. REVISION is 'head', 'base', or a migration slug; a slug
stamps only its own chain. Never touches data.

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

        from cognee.modules.migrations.runner import KEEP

        if args.revision in ("head", "base"):
            graph_target = vector_target = args.revision
        else:
            graph_target, vector_target = _resolve_targets(args.revision, "head", "head")
            # _resolve_targets maps the non-targeted chain to KEEP already; for
            # 'head' it returns ("head","head"), handled above.

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

            return await stamp_revisions(
                graph_target=graph_target,
                vector_target=vector_target,
                dataset_ids=dataset_ids,
            )

        try:
            summaries = asyncio.run(run())
        except Exception as error:  # noqa: BLE001 - translated to an actionable hint
            _bookkeeping_guard(error)
        if not summaries:
            fmt.note("No databases found — nothing stamped.")
            return
        for summary in summaries:
            target = summary.get("dataset_id") or summary.get("database", "?")
            fmt.success(
                f"  {target}: graph={summary['graph_revision']} vector={summary['vector_revision']}"
            )
        fmt.success("Stamp complete.")
