import argparse
import asyncio

from cognee.cli import DEFAULT_DOCS_URL
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException
from cognee.cli.reference import SupportsCliCommand
import cognee.cli.echo as fmt


class ReportCommand(SupportsCliCommand):
    """Generate a Graph Insight Report (hub nodes, surprising links, confidence tags, questions)."""

    command_string = "report"
    help_string = (
        "Generate a Graph Insight Report — hub nodes, surprising cross-set connections, "
        "confidence tags, and LLM-suggested questions"
    )
    docs_url = DEFAULT_DOCS_URL

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--datasets",
            "-d",
            nargs="*",
            default=["main_dataset"],
            help="Dataset name(s) to analyse (default: main_dataset)",
        )
        parser.add_argument(
            "--output",
            "-o",
            default="graph_report.md",
            help="Output file path for the Markdown report (default: graph_report.md)",
        )
        parser.add_argument(
            "--top-n",
            "-n",
            type=int,
            default=10,
            help="Number of hub nodes and surprising connections to surface (default: 10)",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            async def _run() -> str:
                try:
                    from cognee.api.v1.report import report
                    from cognee.cli.user_resolution import resolve_cli_user

                    user = await resolve_cli_user(getattr(args, "user_id", None))
                    datasets = args.datasets or ["main_dataset"]
                    return await report(
                        datasets=datasets,
                        output_path=args.output,
                        top_n=args.top_n,
                        user=user,
                    )
                except Exception as exc:
                    raise CliCommandInnerException(
                        f"Failed to generate report: {exc}"
                    ) from exc

            report_md = asyncio.run(_run())

            fmt.echo("\nGraph Insight Report generated successfully!")
            fmt.echo(f"Report written to: {args.output}")
            fmt.echo("\nPreview (first 500 chars):")
            fmt.echo("=" * 60)
            fmt.echo(report_md[:500] + ("…" if len(report_md) > 500 else ""))

        except CliCommandInnerException as exc:
            raise CliCommandException(str(exc), error_code=1) from exc
        except Exception as exc:
            raise CliCommandException(f"Error generating report: {exc}", error_code=1) from exc
