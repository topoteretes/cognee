import argparse

import cognee.cli.echo as fmt
from cognee.cli import DEFAULT_DOCS_URL
from cognee.cli.reference import SupportsCliCommand

# Shared with the --api-url dispatch, which behaves the same.
MEMIFY_CLI_DEPRECATION_MESSAGE = (
    "cognee-cli memify is deprecated and will be removed in the next release. "
    "It has nothing left to run: its default tasks built triplet embeddings, which were "
    "removed. Use `cognee-cli improve` to enrich a dataset."
)


class MemifyCommand(SupportsCliCommand):
    command_string = "memify"
    help_string = "Deprecated: does nothing (use improve)"
    docs_url = DEFAULT_DOCS_URL
    description = """
Deprecated: removed in the next release. Use `cognee-cli improve` to enrich
a dataset.

Triplet embeddings were removed, and with them the tasks this command ran, so
it no longer runs anything: it prints this notice and exits. Its options are
still accepted so existing scripts keep working.
"""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "-d",
            "--dataset-name",
            default=None,
            help="Ignored (deprecated)",
        )
        group.add_argument(
            "--dataset-id",
            default=None,
            help="Ignored (deprecated)",
        )
        parser.add_argument(
            "--node-name",
            nargs="*",
            default=None,
            help="Ignored (deprecated)",
        )
        parser.add_argument(
            "--data",
            default=None,
            help="Ignored (deprecated)",
        )
        parser.add_argument(
            "-b",
            "--background",
            action="store_true",
            help="Ignored (deprecated)",
        )

    def execute(self, args: argparse.Namespace) -> None:
        fmt.warning(MEMIFY_CLI_DEPRECATION_MESSAGE)
