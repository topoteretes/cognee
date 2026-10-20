import argparse
import asyncio

import cognee.cli.echo as fmt
from cognee.cli import DEFAULT_DOCS_URL
from cognee.cli.reference import SupportsCliCommand
from cognee.shared.logging_utils import get_logger

logger = get_logger()


class ServeCommand(SupportsCliCommand):
    command_string = "serve"
    help_string = "Connect to a Cognee instance (cloud or local)"
    docs_url = DEFAULT_DOCS_URL
    description = """
Connect the local Cognee SDK to a Cognee instance.

Pass the instance URL and its API key (or set COGNEE_SERVICE_URL and
COGNEE_API_KEY). A successful connection is saved, so a later bare
`cognee serve` reconnects to the same instance when it was connected with
an API key; a local instance without a key needs --url again.

Examples:
  cognee serve --url http://localhost:8000  # Local instance
  cognee serve --url https://my.cognee.ai --api-key ck_...
  cognee serve                              # Reconnect to the saved keyed instance
  cognee serve --logout                     # Disconnect + clear credentials
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--url",
            help="URL of a Cognee instance (or set COGNEE_SERVICE_URL)",
        )
        parser.add_argument(
            "--api-key",
            help="API key for the instance (optional for local, required for cloud instances)",
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
            asyncio.run(_serve(url=args.url, api_key=args.api_key))


async def _serve(url=None, api_key=None):
    import cognee

    fmt.note("Connecting to Cognee...")
    try:
        client = await cognee.serve(url=url, api_key=api_key or "")
        fmt.success(f"Connected to {client.service_url}")
    except KeyboardInterrupt:
        fmt.warning("Connection cancelled.")
    except Exception as e:
        logger.debug("Ignoring exception in _serve", exc_info=True)
        fmt.error(f"Failed to connect: {e}")


async def _logout():
    import cognee

    await cognee.disconnect(clear_saved=True)
