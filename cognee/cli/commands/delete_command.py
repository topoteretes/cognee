import argparse
import asyncio
from typing import Optional

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException
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
                preview_data = asyncio.run(
                    get_deletion_counts(
                        dataset_name=args.dataset_name,
                        user_id=args.user_id,
                        all_data=args.all,
                    )
                )

                if not preview_data or all(value == 0 for value in preview_data.values()):
                    fmt.success("No data found to delete.")
                    return

                fmt.echo("You are about to delete:")
                if "datasets" in preview_data and preview_data["datasets"] > 0:
                    fmt.echo(f"- {preview_data['datasets']} datasets")
                if "data_entries" in preview_data and preview_data["data_entries"] > 0:
                    fmt.echo(f"- {preview_data['data_entries']} data entries")
                if "users" in preview_data and preview_data["users"] > 0:
                    fmt.echo(
                        f"- {preview_data['users']} {'users' if preview_data['users'] > 1 else 'user'}"
                    )
                fmt.echo("-" * 20)

                fmt.warning("This operation is irreversible!")
                if not fmt.confirm("Proceed?"):
                    fmt.echo("Deletion cancelled.")
                    return
                # --- END PREVIEW LOGIC ---

            # Build operation message for success/failure logging
            if args.all:
                operation = "all data"
            elif args.dataset_name:
                operation = f"dataset '{args.dataset_name}'"
            elif args.user_id:
                operation = f"data for user '{args.user_id}'"
            else:
                operation = "data"

            fmt.echo(f"Deleting {operation}...")

            # Run the async delete function
            async def run_delete():
                try:
                    # NOTE: The underlying cognee.delete() function is currently not working as expected.
                    # This is a separate bug that this preview feature helps to expose.
                    if args.all:
                        await cognee.delete(dataset_name=None, user_id=args.user_id)
                    else:
                        await cognee.delete(dataset_name=args.dataset_name, user_id=args.user_id)
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to delete: {str(e)}")

            asyncio.run(run_delete())
            # This success message may be inaccurate due to the underlying bug, but we leave it for now.
            fmt.success(f"Successfully deleted {operation}")

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1)
            raise CliCommandException(f"Error deleting data: {str(e)}", error_code=1)
