import argparse
import asyncio

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt


class ServeCommand(SupportsCliCommand):
    command_string = "serve"
    help_string = "Connect to a Cognee instance (cloud or local)"
    docs_url = DEFAULT_DOCS_URL
    description = """
Connect the local Cognee SDK to a Cognee instance.

Cloud mode (default): authenticates via browser-based device code flow,
discovers your tenant, and connects automatically.

Local mode: connect directly to a running Cognee backend.

Examples:
  cognee serve                              # Cloud (Auth0 device flow)
  cognee serve --url http://localhost:8000  # Local instance
  cognee serve --url https://my.cognee.ai --api-key ck_...
  cognee serve --logout                     # Disconnect + clear credentials
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--url",
            help="Direct URL of a Cognee instance (skips Auth0 + tenant discovery)",
        )
        parser.add_argument(
            "--api-key",
            help="API key for the instance (optional for local, required for cloud instances)",
        )
        parser.add_argument(
            "--management-url",
            help="Override the Management API URL (cloud mode only)",
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
            asyncio.run(
                _serve(url=args.url, api_key=args.api_key, management_url=args.management_url)
            )


async def _serve(url=None, api_key=None, management_url=None):
    import cognee

    mode = "local" if url else "cloud"
    fmt.note(f"Connecting to Cognee ({mode})...")
    try:
        client = await cognee.serve(url=url, api_key=api_key or "", management_url=management_url)
        fmt.success(f"Connected to {client.service_url}")
    except KeyboardInterrupt:
        fmt.warning("Authentication cancelled.")
    except Exception as e:
        fmt.error(f"Failed to connect: {e}")


async def _logout():
    import cognee

    await cognee.disconnect(clear_saved=True)
