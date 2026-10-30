"""Inspect a local COGX archive before importing it."""

import argparse

import cognee.cli.echo as fmt
from cognee.cli import DEFAULT_DOCS_URL
from cognee.cli.reference import SupportsCliCommand


class InspectArchiveCommand(SupportsCliCommand):
    command_string = "inspect-archive"
    help_string = "Inspect a local COGX archive's records and integrity without importing it"
    docs_url = DEFAULT_DOCS_URL
    description = """
Inspect a COGX archive directory locally, without importing data or invoking models.

Checks record schemas, declared counts, format compatibility, and available
checksums. Reports the presence of permissions.json without reading its contents.
Legacy records without checksums remain supported and are reported as unverified.
Packed .cogx.tar.gz files must be unpacked before inspection.

Examples:
  cognee-cli inspect-archive backup_cogx
  cognee-cli inspect-archive backup_cogx --json

Exit status is 0 when no errors are found and 1 when inspection reports errors.
Warnings do not fail inspection. This command always runs locally, including
when the global --api-url option is supplied.
"""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("directory", help="Path to the unpacked COGX archive directory")
        parser.add_argument("--json", action="store_true", help="Print the structured JSON report")

    def execute(self, args: argparse.Namespace) -> None:
        from cognee.migration import inspect_cogx_archive

        report = inspect_cogx_archive(args.directory)
        if args.json:
            fmt.echo(report.model_dump_json(indent=2))
        else:
            fmt.echo(f"Archive: {report.directory}")
            fmt.echo(f"COGX version: {report.cogx_version or 'not declared'}")
            fmt.echo(f"Source system: {report.source_system or 'not declared'}")
            fmt.echo(f"Records: {report.total_records}")
            for kind, count in sorted(report.record_counts.items()):
                fmt.echo(f"  {kind}: {count}")
            fmt.echo(
                "Checksums: "
                + ", ".join(f"{key}={value}" for key, value in report.checksum_counts.items())
            )
            if report.contains_permissions:
                fmt.warning(
                    "permissions.json is present; the archive may contain account credentials."
                )
            for diagnostic in report.diagnostics:
                location = diagnostic.file
                if diagnostic.line is not None:
                    location += f":{diagnostic.line}"
                fmt.echo(
                    f"{diagnostic.severity.upper()} {location} "
                    f"[{diagnostic.code}]: {diagnostic.message}"
                )
            fmt.echo("Inspection passed." if report.valid else "Inspection failed.")

        if not report.valid:
            # Avoid the generic command exception handler adding non-JSON help
            # text to stdout after a structured report.
            raise SystemExit(1)
