import argparse
import asyncio
from typing import Optional

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException


class DatasetsCommand(SupportsCliCommand):
    command_string = "datasets"
    help_string = "List and inspect datasets"
    docs_url = DEFAULT_DOCS_URL
    description = """
List available datasets and their contents.

Use `cognee datasets list` to discover what data has been ingested.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest="datasets_action", help="Dataset actions")
        subparsers.add_parser("list", help="List all datasets")

    def execute(self, args: argparse.Namespace) -> Optional[dict]:
        action = getattr(args, "datasets_action", None)
        if action is None or action == "list":
            return self._list_datasets()
        else:
            fmt.error(f"Unknown action: {action}")
            return None

    def _list_datasets(self) -> Optional[dict]:
        try:
            import cognee
            from cognee.modules.users.methods import get_default_user

            async def _fetch():
                user = await get_default_user()
                ds_list = await cognee.datasets.list_datasets(user=user)
                results = []
                for ds in ds_list or []:
                    has_data = await cognee.datasets.has_data(str(ds.id), user=user)
                    results.append(
                        {
                            "id": str(ds.id),
                            "name": ds.name,
                            "has_data": has_data,
                            "created_at": ds.created_at.isoformat()
                            if hasattr(ds, "created_at") and ds.created_at
                            else None,
                        }
                    )
                return results

            datasets = asyncio.run(_fetch())

            if not datasets:
                fmt.echo("No datasets found. Use 'cognee-cli add' to ingest data.")
                return {"datasets": [], "count": 0}

            fmt.echo(f"Found {len(datasets)} dataset(s):\n")
            for ds in datasets:
                data_marker = "has data" if ds["has_data"] else "empty"
                fmt.echo(f"  {ds['name']}  ({data_marker})  id={ds['id']}")

            return {"datasets": datasets, "count": len(datasets)}

        except Exception as e:
            raise CliCommandException(f"Failed to list datasets: {e}", error_code=1) from e
