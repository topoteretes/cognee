import argparse
import asyncio

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException
from cognee.api.v1.datasets.datasets import datasets as cognee_datasets
from cognee.modules.data.methods import get_datasets_by_name
from cognee.modules.data.methods.get_deletion_counts import get_deletion_counts


class DeleteCommand(SupportsCliCommand):
    command_string = "delete"
    help_string = "Delete data from cognee knowledge base"
    docs_url = DEFAULT_DOCS_URL
    description = """
The `cognee delete` command removes data from your knowledge base.

You can delete:
- Specific datasets by name
- All data (with confirmation)

Be careful with deletion operations as they are irreversible.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--dataset-name", "-d", help="Specific dataset to delete")
        parser.add_argument(
            "--all", action="store_true", help="Delete all data (requires confirmation)"
        )
        parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompts")

    def execute(self, args: argparse.Namespace) -> None:
        try:
            if not any(
                [
                    getattr(args, "dataset_name", None),
                    getattr(args, "all", False),
                ]
            ):
                fmt.error("Please specify what to delete: --dataset-name or --all")
                return

            # If --force is used, skip the preview and go straight to deletion
            if not getattr(args, "force", False):
                fmt.echo("Gathering data for preview...")
                try:
                    preview_data = asyncio.run(
                        get_deletion_counts(
                            dataset_name=getattr(args, "dataset_name", None),
                            user_id=getattr(args, "user_id", None),
                            all_data=getattr(args, "all", False),
                        )
                    )
                except CliCommandException as e:
                    fmt.error(f"Error occurred when fetching preview data: {str(e)}")
                    return

                if not preview_data:
                    fmt.success("No data found to delete.")
                    return

                fmt.echo("You are about to delete:")
                fmt.echo(
                    f"Datasets: {preview_data.datasets}\n"
                    f"Entries: {preview_data.data_entries}\n"
                    f"Users: {preview_data.users}"
                )
                fmt.echo("-" * 20)

            if getattr(args, "all", False):
                confirm_msg = "Delete ALL data from cognee?"
                operation = "all data"
            else:
                confirm_msg = f"Delete dataset '{args.dataset_name}'?"
                operation = f"dataset '{args.dataset_name}'"

            if not getattr(args, "force", False):
                fmt.warning("This operation is irreversible!")
                if not fmt.confirm(confirm_msg):
                    fmt.echo("Deletion cancelled.")
                    return

            fmt.echo(f"Deleting {operation}...")

            async def run_delete():
                try:
                    from cognee.cli.user_resolution import resolve_cli_user

                    user = await resolve_cli_user(getattr(args, "user_id", None))

                    if getattr(args, "all", False):
                        await cognee_datasets.delete_all(user=user)
                    elif getattr(args, "dataset_name", None):
                        datasets = await get_datasets_by_name(args.dataset_name, user_id=user.id)
                        if not datasets:
                            raise CliCommandException(
                                f"No dataset found for name '{args.dataset_name}'."
                            )
                        await cognee_datasets.empty_dataset(dataset_id=datasets[0].id, user=user)
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to delete: {str(e)}") from e

            asyncio.run(run_delete())
            fmt.success(f"Successfully deleted {operation}")

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Error deleting data: {str(e)}", error_code=1) from e
