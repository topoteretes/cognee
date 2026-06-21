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

Use --everything (alias --all) to delete all user data, --dataset/--dataset-id
to delete a dataset, or dataset + --data-id to delete a single item.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--dataset", help="Dataset name to delete")
        parser.add_argument(
            "--dataset-id",
            help="Dataset UUID to delete",
        )
        parser.add_argument(
            "--data-id",
            help="UUID of a specific data item to delete (requires dataset or dataset-id)",
        )
        parser.add_argument(
            "--everything",
            "--all",
            action="store_true",
            default=False,
            help="Delete all datasets and data",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            import cognee

            data_id = UUID(args.data_id) if args.data_id else None
            dataset_id = UUID(args.dataset_id) if args.dataset_id else None
            dataset = args.dataset

            if dataset and dataset_id:
                fmt.error("Provide either --dataset or --dataset-id, not both.")
                return

            if not args.everything and not dataset and not dataset_id and not data_id:
                fmt.error(
                    "Specify --dataset or --dataset-id, --data-id with dataset, or --everything/--all."
                )
                return

            async def run_forget():
                try:
                    return await cognee.forget(
                        data_id=data_id,
                        dataset=dataset,
                        dataset_id=dataset_id,
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
