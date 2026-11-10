import argparse
import asyncio
from uuid import UUID

import cognee.cli.echo as fmt
from cognee.cli import DEFAULT_DOCS_URL
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException
from cognee.cli.hints import hint_remember
from cognee.cli.reference import SupportsCliCommand


class ForgetCommand(SupportsCliCommand):
    command_string = "forget"
    help_string = "Remove data from the knowledge graph"
    docs_url = DEFAULT_DOCS_URL
    description = """
Remove data from the knowledge graph.

Use --everything (alias --all) to delete all user data, --dataset/--dataset-id
to delete a dataset, or dataset + --data-id to delete a single item. Add
--memory-only to clear graph/vector memory while keeping raw files and data
records, so the dataset (or item) can be re-cognified later.
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
        parser.add_argument(
            "--memory-only",
            action="store_true",
            default=False,
            help=(
                "Delete only graph/vector memory (requires --dataset or --dataset-id); "
                "raw files and data records are preserved so the dataset can be "
                "re-cognified with different settings"
            ),
        )
        parser.add_argument(
            "--clean-history",
            action="store_true",
            default=False,
            help=(
                "Also delete your own search and recall history, in every dataset, "
                "before the data is deleted; alone, delete only the history"
            ),
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

            if (
                not args.everything
                and not dataset
                and not dataset_id
                and not data_id
                and not args.clean_history
            ):
                fmt.error(
                    "Specify --dataset or --dataset-id, --data-id with dataset, "
                    "--everything/--all, or --clean-history."
                )
                return

            if data_id and not dataset and not dataset_id:
                fmt.error("Specify --dataset or --dataset-id when using --data-id.")
                return

            if args.everything and args.memory_only:
                fmt.error(
                    "--memory-only has no effect with --everything: everything deletes all "
                    "datasets and data outright. Specify --dataset or --dataset-id with "
                    "--memory-only instead."
                )
                return

            async def run_forget():
                try:
                    return await cognee.forget(
                        data_id=data_id,
                        dataset=dataset,
                        dataset_id=dataset_id,
                        everything=args.everything,
                        memory_only=args.memory_only,
                        clean_history=args.clean_history,
                    )
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to forget: {e!s}") from e

            result = asyncio.run(run_forget())
            if isinstance(result, dict) and result.get("status") == "not_found":
                # A single-item forget whose id owned nothing: wrong id, or already
                # deleted. Not a success line — nothing was removed.
                fmt.warning(
                    f"Nothing to delete: data item {data_id} was not found in the dataset "
                    f"(wrong id, or already deleted). {result}"
                )
                return
            fmt.success(f"Done: {result}")

            # After a successful forget the natural next step is to seed the
            # graph again with remember; a placeholder is used when the user
            # wiped everything so no specific dataset name is meaningful.
            hint_dataset = dataset or "<dataset-name>"
            hint_remember(hint_dataset)

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Error during forget: {e!s}", error_code=1) from e
