import argparse
import asyncio

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException

# Tiny constants-only import; the cognee package is already loaded by the CLI.
from cognee.modules.migration.sources.base import IMPORT_MODES


class PushCommand(SupportsCliCommand):
    command_string = "push"
    help_string = "Upload a local dataset's knowledge graph to Cognee Cloud"
    docs_url = DEFAULT_DOCS_URL
    description = """
Upload a local dataset's knowledge graph to a Cognee Cloud instance.

The dataset's graph is exported as a COGX archive and imported on the remote
instance, preserving locally extracted entities and relationships instead of
re-deriving them from raw files.

Authentication reuses the serve credentials: run `cognee serve` once to log
in, then push any time. Alternatively pass --url/--api-key or set
COGNEE_SERVICE_URL and COGNEE_API_KEY.

Import modes:
  preserve   (default) map exported entities/facts directly — zero LLM calls
  hybrid     preserve the graph and also cognify the raw content
  re-derive  ignore the exported graph, rebuild from raw content remotely

Examples:
  cognee push                                  # push main_dataset
  cognee push my_dataset
  cognee push my_dataset --target-dataset prod_dataset
  cognee push my_dataset --mode hybrid
  cognee push --url https://my.cognee.ai --api-key ck_...
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "dataset",
            nargs="?",
            default="main_dataset",
            help="Local dataset name to push (default: main_dataset)",
        )
        parser.add_argument(
            "--target-dataset",
            help="Dataset name on the remote instance (default: same as local)",
        )
        parser.add_argument(
            "--mode",
            choices=list(IMPORT_MODES),
            default="preserve",
            help="Remote import mode (default: preserve)",
        )
        parser.add_argument(
            "--url",
            help=(
                "Remote instance URL (default: active serve connection, "
                "COGNEE_SERVICE_URL, or saved serve credentials)"
            ),
        )
        parser.add_argument(
            "--api-key",
            help="API key for the remote instance",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            import cognee

            fmt.echo(f"Pushing dataset '{args.dataset}' to Cognee Cloud...")

            async def run_push():
                try:
                    from cognee.api.v1.push.push import _resolve_client
                    from cognee.cli.user_resolution import resolve_cli_user

                    # Resolve once up front so the target host is visible
                    # before the upload starts.
                    client, created = _resolve_client(args.url, args.api_key)
                    fmt.echo(f"Remote instance: {client.service_url}")
                    if created:
                        await client.close()

                    user = await resolve_cli_user(getattr(args, "user_id", None))

                    return await cognee.push(
                        args.dataset,
                        target_dataset=args.target_dataset,
                        mode=args.mode,
                        url=args.url,
                        api_key=args.api_key,
                        user=user,
                    )
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to push: {str(e)}") from e

            result = asyncio.run(run_push())

            fmt.success(
                f"Pushed {result.num_nodes} nodes and {result.num_edges} edges "
                f"to remote dataset '{result.target_dataset}'."
            )
        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Error during push: {str(e)}", error_code=1) from e
