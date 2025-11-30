import argparse
import asyncio
from uuid import UUID
from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException
from cognee.modules.data.methods.get_deletion_counts import get_deletion_counts
from cognee.modules.data.methods.delete_dataset_by_name import delete_datasets_by_name
from cognee.modules.data.methods.delete_data_by_user import delete_data_by_user
from cognee.modules.users.methods import get_default_user


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
            # Import cognee here to avoid circular imports
            import cognee

            # Validate arguments
            if not any([args.dataset_name, args.user_id, args.all]):
                fmt.error("Please specify what to delete: --dataset-name, --user-id, or --all")
                return

            # If --force is used, skip the preview and go straight to deletion
            if not args.force:
                # --- START PREVIEW LOGIC ---
                fmt.echo("Gathering data for preview...")
                try:
                    preview_data = asyncio.run(
                        get_deletion_counts(
                            dataset_name=args.dataset_name,
                            user_id=args.user_id,
                            all_data=args.all,
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
                    f"Datasets: {preview_data.datasets}\nEntries: {preview_data.entries}\nUsers: {preview_data.users}"
                )
                fmt.echo("-" * 20)
                # --- END PREVIEW LOGIC ---

            # Build operation message for success/failure logging
            if args.all:
                confirm_msg = "Delete ALL data from cognee?"
                operation = "all data"
            elif args.dataset_name:
                confirm_msg = f"Delete dataset '{args.dataset_name}'?"
                operation = f"dataset '{args.dataset_name}'"
            elif args.user_id:
                confirm_msg = f"Delete all data for user '{args.user_id}'?"
                operation = f"data for user '{args.user_id}'"
            else:
                operation = "data"

            if not args.force:
                fmt.warning("This operation is irreversible!")
                if not fmt.confirm(confirm_msg):
                    fmt.echo("Deletion cancelled.")
                    return

            fmt.echo(f"Deleting {operation}...")

            # Run the async delete function
            async def run_delete():
                try:
                    if args.dataset_name:
                        # Use delete_datasets_by_name for dataset deletion
                        user = await get_default_user()
                        result = await delete_datasets_by_name(args.dataset_name, user.id)
                        
                        if result["not_found"]:
                            fmt.warning(f"Dataset '{args.dataset_name}' not found")
                            return False
                        
                        fmt.success(f"Successfully deleted {result['deleted_count']} dataset(s)")
                        return True
                    else:
                        # For user_id deletion, use the original cognee.delete
                        result = await delete_data_by_user(UUID(args.user_id))
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to delete: {str(e)}") from e
                return True

            success = asyncio.run(run_delete())
            if success and not args.dataset_name:
                fmt.success(f"Successfully deleted {operation}")

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Error deleting data: {str(e)}", error_code=1) from e
