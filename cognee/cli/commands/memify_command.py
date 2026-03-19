import argparse
import asyncio

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException


class MemifyCommand(SupportsCliCommand):
    command_string = "memify"
    help_string = "Run the memory enrichment pipeline on a dataset"
    docs_url = DEFAULT_DOCS_URL
    description = """
Run the Cognee memify (memory enrichment) pipeline.

Enriches an existing knowledge graph with additional context, triplet
embeddings, and coding rules. Requires data to have been added and
cognified first.

Examples:
  cognee-cli memify -d my_dataset
  cognee-cli memify --dataset-id 550e8400-e29b-41d4-a716-446655440000
  cognee-cli memify -d my_dataset --node-name "Python" "FastAPI"
  cognee-cli memify -d my_dataset --background
"""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "-d",
            "--dataset-name",
            default=None,
            help="Dataset name to memify",
        )
        group.add_argument(
            "--dataset-id",
            default=None,
            help="Dataset UUID to memify",
        )
        parser.add_argument(
            "--node-name",
            nargs="*",
            default=None,
            help="Filter to specific named entities in the graph",
        )
        parser.add_argument(
            "--data",
            default=None,
            help="Optional text data to feed into the pipeline instead of using the existing graph",
        )
        parser.add_argument(
            "-b",
            "--background",
            action="store_true",
            help="Run in background (non-blocking)",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:

            async def run():
                import cognee
                from uuid import UUID
                from cognee.cli.user_resolution import resolve_cli_user

                user = await resolve_cli_user(getattr(args, "user_id", None))

                dataset = args.dataset_name
                if args.dataset_id:
                    dataset = UUID(args.dataset_id)

                label = args.dataset_id or args.dataset_name
                fmt.echo(f"Running memify on '{label}'...")

                result = await cognee.memify(
                    dataset=dataset,
                    user=user,
                    data=args.data,
                    node_name=args.node_name,
                    run_in_background=args.background,
                )

                if args.background:
                    fmt.success("Memify started in background.")
                else:
                    fmt.success("Memify completed.")

                if result:
                    fmt.echo(str(result))

            asyncio.run(run())

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Memify failed: {str(e)}", error_code=1) from e
