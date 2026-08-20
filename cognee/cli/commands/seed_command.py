import argparse
import asyncio
from pathlib import Path

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException
from cognee.cli.hints import hint_recall


class SeedCommand(SupportsCliCommand):
    command_string = "seed"
    help_string = "Ingest what already exists in this workspace so the first recall returns"
    docs_url = DEFAULT_DOCS_URL
    description = """
Day-one seeding: discover and ingest the knowledge that already exists around
this workspace — agent memory files (MEMORY.md, SOUL.md, AGENTS.md,
CLAUDE.md, ...), the README, recent coding-agent session logs, and the
codebase itself (when the workspace is a code project).

Seeding runs in stages ordered by size, memory files first, so a recall
issued moments later already has something to return. Re-running is safe:
it refuses to re-seed an existing seed dataset unless --force is given, and
content-hash dedup makes --force cheap on unchanged files.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--workspace",
            "-w",
            default=None,
            help="Workspace root to seed (default: auto-detected from the current directory)",
        )
        parser.add_argument(
            "--dataset-name",
            "-d",
            default=None,
            help="Dataset the seed lands in (default: workspace)",
        )
        parser.add_argument(
            "--no-code",
            action="store_true",
            help="Skip indexing the codebase",
        )
        parser.add_argument(
            "--no-session-logs",
            action="store_true",
            help="Skip ingesting recent agent session transcripts",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Only print what would be ingested, without ingesting anything",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-seed even when the seed dataset already exists",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            from cognee.modules.seeding import DEFAULT_SEED_DATASET, seed

            workspace = Path(args.workspace).expanduser() if args.workspace else None
            dataset_name = args.dataset_name or DEFAULT_SEED_DATASET

            async def run_seed():
                try:
                    return await seed(
                        workspace,
                        dataset_name=dataset_name,
                        include_codebase=not args.no_code,
                        include_session_logs=not args.no_session_logs,
                        force=args.force,
                        dry_run=args.dry_run,
                    )
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to seed: {str(e)}") from e

            if not args.dry_run:
                fmt.echo("Discovering day-one sources...")

            result = asyncio.run(run_seed())

            if args.dry_run:
                fmt.echo(result.plan.describe())
                return

            if result.plan.is_empty:
                fmt.warning(
                    "Nothing to seed: no memory files, README, session logs, or code "
                    f"project found around {result.plan.workspace}."
                )
                return

            if result.skipped_existing:
                fmt.note(result.summary())
                return

            fmt.echo(result.summary())
            if result.ingested_anything:
                fmt.success("Day-one seed complete.")
                hint_recall(dataset_name)
            else:
                raise CliCommandInnerException("Seeding ingested nothing; see errors above.")

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Failed to seed: {str(e)}", error_code=1) from e
