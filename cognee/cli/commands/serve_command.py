import argparse
import asyncio

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt


class ServeCommand(SupportsCliCommand):
    command_string = "serve"
    help_string = "Connect to Cognee Cloud (remote mode)"
    docs_url = DEFAULT_DOCS_URL
    description = """
Connect the local Cognee SDK to a remote Cognee Cloud instance.

Authenticates via browser-based device code flow, discovers your tenant,
and redirects all subsequent V2 operations (remember, recall, improve,
forget) to the cloud.

Use `cognee serve --logout` to disconnect and clear saved credentials.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--management-url",
            help="Override the Management API URL (default: COGNEE_CLOUD_URL env var)",
        )
        parser.add_argument(
            "--logout",
            action="store_true",
            help="Disconnect and clear saved credentials",
        )

    def execute(self, args: argparse.Namespace) -> None:
        if args.logout:
            asyncio.run(_logout())
        else:
            asyncio.run(_serve(args.management_url))


async def _serve(management_url=None):
    import cognee

    fmt.note("Connecting to Cognee Cloud...")
    try:
        client = await cognee.serve(management_url=management_url)
        fmt.success(f"Connected to {client.service_url}")
    except KeyboardInterrupt:
        fmt.warning("Authentication cancelled.")
    except Exception as e:
        fmt.error(f"Failed to connect: {e}")


async def _logout():
    import cognee

    await cognee.disconnect(clear_saved=True)
