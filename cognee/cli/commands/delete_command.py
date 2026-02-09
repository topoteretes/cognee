import argparse
import asyncio

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException
from cognee.api.v1.datasets.datasets import datasets as cognee_datasets
from cognee.modules.data.methods import get_datasets_by_name
from cognee.modules.data.methods.get_deletion_counts import get_deletion_counts
from cognee.modules.users.methods import get_default_user, get_user


class DeleteCommand(SupportsCliCommand):
    command_string = "delete"
    help_string = "Delete data from cognee knowledge base"
    docs_url = DEFAULT_DOCS_URL
    description = """
The `cognee delete` command removes data from your knowledge base.

You can delete:
- Specific datasets by name
- All data (with confirmation)
- Data for specific users

Be careful with deletion operations as they are irreversible.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--dataset-name", "-d", help="Specific dataset to delete")
        parser.add_argument("--user-id", "-u", help="User ID to delete data for")
        parser.add_argument(
            "--all", action="store_true", help="Delete all data (requires confirmation)"
        )
        parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompts")

    def execute(self, args: argparse.Namespace) -> None:
        try:
            # Validate arguments
            if not any(
                [
                    getattr(args, "dataset_name", None),
                    getattr(args, "dataset_id", None),
                    getattr(args, "data_id", None),
                    getattr(args, "all", False) and getattr(args, "user_id", None),
                ]
            ):
                fmt.error(
                    "Please specify what to delete: --dataset-name, --dataset-id, --data-id, --user-id, or --all"
                )
                return

            # If --force is used, skip the preview and go straight to deletion
            if not getattr(args, "force", False):
                # --- START PREVIEW LOGIC ---
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
                    fmt.error(f"Error occured when fetching preview data: {str(e)}")
                    return

                if not preview_data:
                    fmt.success("No data found to delete.")
                    return

                fmt.echo("You are about to delete:")
                fmt.echo(
                    f"Datasets: {preview_data.datasets}\nEntries: {preview_data.data_entries}\nUsers: {preview_data.users}"
                )
                fmt.echo("-" * 20)
                # --- END PREVIEW LOGIC ---

            # Build operation message for success/failure logging
            if getattr(args, "all", False):
                confirm_msg = "Delete ALL data from cognee?"
                operation = "all data"
            elif hasattr(args, "dataset_name"):
                confirm_msg = f"Delete dataset '{args.dataset_name}'?"
                operation = f"dataset '{args.dataset_name}'"
            elif hasattr(args, "user_id"):
                confirm_msg = f"Delete all data for user '{args.user_id}'?"
                operation = f"data for user '{args.user_id}'"
            else:
                operation = "data"

            if not getattr(args, "force", False):
                fmt.warning("This operation is irreversible!")
                if not fmt.confirm(confirm_msg):
                    fmt.echo("Deletion cancelled.")
                    return

            fmt.echo(f"Deleting {operation}...")

            # Run the async delete function
            async def run_delete():
                try:
                    if getattr(args, "all", False):
                        if not hasattr(args, "user_id"):
                            raise CliCommandException(
                                "No user ID provided for '--all' deletion. Please specify using --user-id param."
                            )
                        await cognee_datasets.delete_all(user_id=args.user_id)
                    elif hasattr(args, "dataset_name") or hasattr(args, "dataset_id"):
                        dataset_id = getattr(args, "dataset_id", None)

                        if hasattr(args, "dataset_name") and not hasattr(args, "dataset_id"):
                            datasets = await get_datasets_by_name(
                                args.dataset_name, user_id=args.user_id
                            )

                            if not datasets:
                                raise CliCommandException(
                                    f"No dataset found for name '{args.dataset_name}'."
                                )

                            dataset = datasets[0]
                            dataset_id = dataset.id

                        if not hasattr(args, "user_id"):
                            raise CliCommandException(
                                "No user ID provided for deletion. Please specify using --user-id param."
                            )

                        if not args.user_id:
                            user = await get_default_user()
                        else:
                            user = await get_user(args.user_id)

                        await cognee_datasets.empty_dataset(dataset_id=dataset_id, user=user)
                    elif hasattr(args, "dataset_id") and hasattr(args, "data_id"):
                        await cognee_datasets.delete_data(args.dataset_id, args.data_id)
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to delete: {str(e)}") from e

            asyncio.run(run_delete())
            # This success message may be inaccurate due to the underlying bug, but we leave it for now.
            fmt.success(f"Successfully deleted {operation}")

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Error deleting data: {str(e)}", error_code=1) from e
