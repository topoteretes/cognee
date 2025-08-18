import argparse
import asyncio
from typing import Optional

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException


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

            # Build confirmation message
            if args.all:
                confirm_msg = "Delete ALL data from cognee?"
                operation = "all data"
            elif args.dataset_name:
                confirm_msg = f"Delete dataset '{args.dataset_name}'?"
                operation = f"dataset '{args.dataset_name}'"
            elif args.user_id:
                confirm_msg = f"Delete all data for user '{args.user_id}'?"
                operation = f"data for user '{args.user_id}'"

            # Confirm deletion unless forced
            if not args.force:
                fmt.warning("This operation is irreversible!")
                if not fmt.confirm(confirm_msg):
                    fmt.echo("Deletion cancelled.")
                    return

            fmt.echo(f"Deleting {operation}...")

            # Run the async delete function
            async def run_delete():
                try:
                    if args.all:
                        await cognee.delete(dataset_name=None, user_id=args.user_id)
                    else:
                        await cognee.delete(dataset_name=args.dataset_name, user_id=args.user_id)
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to delete: {str(e)}")

            asyncio.run(run_delete())
            fmt.success(f"Successfully deleted {operation}")

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1)
            raise CliCommandException(f"Error deleting data: {str(e)}", error_code=1)
