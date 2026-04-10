import argparse
import asyncio
from uuid import UUID

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException


class ForgetCommand(SupportsCliCommand):
    command_string = "forget"
    help_string = "Remove data from the knowledge graph"
    docs_url = DEFAULT_DOCS_URL
    description = """
Remove data from the knowledge graph.

Use --everything to delete all user data, --dataset to delete a dataset,
or --dataset with --data-id to delete a single item.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--dataset",
            help="Dataset name or UUID to delete",
        )
        parser.add_argument(
            "--data-id",
            help="UUID of a specific data item to delete (requires --dataset)",
        )
        parser.add_argument(
            "--everything",
            action="store_true",
            default=False,
            help="Delete all datasets and data",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            import cognee

            data_id = UUID(args.data_id) if args.data_id else None

            # Try parsing dataset as UUID, fall back to string name
            dataset = args.dataset
            if dataset:
                try:
                    dataset = UUID(dataset)
                except ValueError:
                    pass

            if not args.everything and not dataset and not data_id:
                fmt.error("Specify --dataset, --data-id with --dataset, or --everything.")
                return

            async def run_forget():
                try:
                    return await cognee.forget(
                        data_id=data_id,
                        dataset=dataset,
                        everything=args.everything,
                    )
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to forget: {str(e)}") from e

            result = asyncio.run(run_forget())
            fmt.success(f"Done: {result}")

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Error during forget: {str(e)}", error_code=1) from e
