import argparse
import asyncio

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException


class UpdateCommand(SupportsCliCommand):
    command_string = "update"
    help_string = "Incrementally re-cognify changed sources and prune removed ones"
    docs_url = DEFAULT_DOCS_URL
    description = """
Incrementally sync source files/directories into a dataset.

Instead of rebuilding the whole graph, `cognee update` re-cognifies only the inputs
that actually changed, skips unchanged ones, and (by default) prunes graph content for
sources that were removed from disk. Entities that other still-present sources also
contribute are preserved.

Examples:
    cognee update ./docs                       # sync a directory into main_dataset
    cognee update ./docs -d my_project         # into a named dataset
    cognee update ./docs --no-prune            # only add/update, never delete

Pair it with `cognee hook install` to refresh the graph automatically on every commit.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "paths",
            nargs="+",
            help="Files or directories to sync (directories are scanned recursively)",
        )
        parser.add_argument(
            "--dataset-name",
            "-d",
            default="main_dataset",
            help="Dataset to update (default: main_dataset)",
        )
        parser.add_argument(
            "--no-prune",
            action="store_true",
            help="Do not delete graph content for sources removed from disk",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            import cognee

            async def run_update():
                try:
                    from cognee.cli.user_resolution import resolve_cli_user

                    user = await resolve_cli_user(getattr(args, "user_id", None))

                    fmt.echo(
                        f"Updating dataset '{args.dataset_name}' from "
                        f"{len(args.paths)} path(s)..."
                    )
                    result = await cognee.incremental_update(
                        data=args.paths,
                        dataset_name=args.dataset_name,
                        user=user,
                        prune_removed=not args.no_prune,
                    )
                    fmt.success(
                        f"Updated '{result['dataset_name']}': "
                        f"{result['processed']} source(s) processed, "
                        f"{result['removed']} removed."
                    )
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to update: {str(e)}") from e

            asyncio.run(run_update())

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Failed to update: {str(e)}", error_code=1) from e
