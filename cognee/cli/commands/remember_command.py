import argparse
import asyncio
from importlib import resources

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
from cognee.cli.config import CHUNKER_CHOICES
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException
from cognee.cli.hints import hint_recall
from cognee.modules.data.constants import DEFAULT_DATASET_NAME


_SAMPLE_FIXTURE = "quickstart.txt"


def _resolve_sample_path() -> str:
    """Return the absolute path of the bundled quickstart fixture.

    ``importlib.resources.files`` gives a path that survives both editable
    and wheel installs. The fixture is copied into the package on install,
    so no environment variable or ``DATA_ROOT_DIRECTORY`` override is
    required to run the demo.
    """
    return str(resources.files("cognee.cli.samples").joinpath(_SAMPLE_FIXTURE))


class RememberCommand(SupportsCliCommand):
    command_string = "remember"
    help_string = "Add data and build the knowledge graph in one step"
    docs_url = DEFAULT_DOCS_URL
    description = """
Add data and build the knowledge graph in one step.

This combines the `add` and `cognify` commands: data is ingested first,
then automatically processed into a structured knowledge graph.

After completion, use `cognee recall` (or `cognee search`) to query the graph.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "data",
            nargs="*",
            help="Data to add: text content, file paths, file URLs, or S3 paths",
        )
        parser.add_argument(
            "--sample-data",
            action="store_true",
            help=(
                "Ingest the bundled quickstart fixture instead of user data. "
                "Handy for a first-run smoke test; still requires LLM_API_KEY."
            ),
        )
        parser.add_argument(
            "--dataset-name",
            "-d",
            default=DEFAULT_DATASET_NAME,
            help="Dataset name (default: main_dataset)",
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            help="Maximum tokens per chunk (auto-calculated if not specified)",
        )
        parser.add_argument(
            "--chunker",
            choices=CHUNKER_CHOICES,
            default="TextChunker",
            help="Text chunking strategy (default: TextChunker)",
        )
        parser.add_argument(
            "--background",
            "-b",
            action="store_true",
            help="Run cognify step in background (add always completes first)",
        )
        parser.add_argument(
            "--chunks-per-batch",
            type=int,
            help="Number of chunks to process per task batch",
        )
        parser.add_argument(
            "--dry-run",
            nargs="?",
            const=True,
            default=False,
            metavar="presort",
            help=(
                "Estimate LLM token usage and cost without ingesting data or making LLM "
                "calls. With the value 'presort', scan a folder instead: report junk, "
                "duplicates, version candidates, potential personal data, already-in-cognee "
                "status, and proposed dataset groupings — then apply with --from-report."
            ),
        )

        presort = parser.add_argument_group("presort (--dry-run presort / --from-report)")
        presort.add_argument(
            "--from-report",
            metavar="REPORT_JSON",
            help="Apply a saved presort report: ingest its proposed groups as datasets",
        )
        presort.add_argument(
            "--report-output",
            "-o",
            metavar="PATH",
            help="Also save the presort report to this JSON file",
        )
        presort.add_argument(
            "--use-llm",
            action="store_true",
            help="Presort: enable LLM content classification, deeper PII detection, "
            "and semantic grouping",
        )
        presort.add_argument(
            "--no-pii", action="store_true", help="Presort: skip personal-data detection"
        )
        presort.add_argument(
            "--no-subdirectories",
            action="store_true",
            help="Presort: scan only the top level of the folder",
        )
        presort.add_argument(
            "--no-check-existing",
            action="store_true",
            help="Presort: skip checking which files cognee already knows/cognified",
        )
        presort.add_argument(
            "--dataset-prefix",
            default="",
            help="Presort: prefix for proposed dataset names",
        )
        presort.add_argument(
            "--spec",
            metavar="SPEC_JSON",
            help="Presort: JSON file with a custom relationship spec (graph-model DSL)",
        )
        presort.add_argument(
            "--allow-root",
            action="store_true",
            help="Allow scanning folders outside the default allowed roots by extending "
            "COGNEE_ALLOWED_LOCAL_FILE_ROOTS for this process",
        )
        presort.add_argument(
            "--apply",
            action="store_true",
            help="Presort: auto-apply the report as soon as it is produced — scan and "
            "ingest the proposed groups in one run (the report JSON is still saved first)",
        )
        presort.add_argument(
            "--apply-group",
            action="append",
            metavar="NAME",
            help="Apply only this group from the report (repeatable; default: all groups)",
        )
        presort.add_argument(
            "--exclude-pii",
            action="store_true",
            help="Apply: skip files with potential personal data",
        )
        presort.add_argument(
            "--apply-graph",
            action="store_true",
            help="Apply: also write the report's relationship graph (files, groups, "
            "duplicates, PII tags per the relationship spec) into its own dataset",
        )
        presort.add_argument(
            "--graph-dataset",
            metavar="NAME",
            help="Dataset for --apply-graph (default: <folder>_presort_graph)",
        )
        presort.add_argument(
            "--keep-duplicates",
            action="store_true",
            help="Apply: ingest duplicate copies too (default keeps one per cluster)",
        )

    def _extend_allowed_roots(self, paths) -> None:
        import os
        import tempfile
        from pathlib import Path

        from cognee.infrastructure.files.utils.local_path_safety import (
            ALLOWED_LOCAL_FILE_ROOTS_ENV,
        )

        existing = os.environ.get(ALLOWED_LOCAL_FILE_ROOTS_ENV)
        # When the env var was unset the defaults were cwd+tempdir — keep them,
        # and always APPEND (never replace) so no previously-allowed root is lost.
        roots = existing.split(os.pathsep) if existing else [str(Path.cwd()), tempfile.gettempdir()]
        roots.extend(str(Path(path).expanduser()) for path in paths)
        os.environ[ALLOWED_LOCAL_FILE_ROOTS_ENV] = os.pathsep.join(dict.fromkeys(roots))

    def _execute_presort(self, args: argparse.Namespace) -> None:
        import json

        if not args.data or len(args.data) != 1:
            raise CliCommandInnerException("--dry-run presort expects exactly one folder path.")
        folder = args.data[0]
        if args.allow_root:
            self._extend_allowed_roots([folder])

        relationship_spec = None
        if args.spec:
            with open(args.spec, encoding="utf-8") as spec_file:
                relationship_spec = json.load(spec_file)

        from cognee.modules.presort.llm_availability import llm_is_configured

        if not llm_is_configured():
            fmt.warning(
                "LLM API key not configured — presort runs deterministically; any apply "
                "step will stage files (add) without building graphs (cognify skipped)."
            )

        fmt.echo(f"Presorting '{folder}'...")

        async def run_presort():
            import cognee

            return await cognee.remember(
                data=folder,
                dry_run="presort",
                use_llm=args.use_llm,
                detect_pii=not args.no_pii,
                check_existing=not args.no_check_existing,
                include_subdirectories=not args.no_subdirectories,
                dataset_prefix=args.dataset_prefix,
                relationship_spec=relationship_spec,
                auto_apply=args.apply or args.apply_graph,
                apply_groups=args.apply_group if args.apply else [],
                apply_graph=args.apply_graph,
                graph_dataset=args.graph_dataset,
                exclude_pii=args.exclude_pii or None,
                skip_duplicates=False if args.keep_duplicates else None,
                run_in_background=args.background,
            )

        report = asyncio.run(run_presort())

        if args.report_output:
            report.save(args.report_output)

        self._print_presort_report(report)

    def _print_presort_report(self, report) -> None:
        summary = report.summary()
        status = summary["cognee_status"]
        fmt.success(f"Scanned {summary['files']} file(s), skipped {summary['junk']} junk file(s)")
        fmt.echo(
            f"  Already in cognee: {status['cognified']} cognified, {status['staged']} staged; "
            f"{status['new']} new, {status['unknown']} unknown"
        )
        fmt.echo(
            f"  Duplicates: {summary['duplicate_clusters']} cluster(s), "
            f"{summary['wasted_bytes'] / 1_000_000:.1f} MB wasted"
        )
        fmt.echo(f"  Version candidates: {summary['version_candidates']}")
        fmt.echo(
            f"  Potential personal data: {summary['pii_findings']} finding(s) in "
            f"{summary['files_with_pii']} file(s)"
        )
        fmt.echo(f"  Proposed groups ({summary['groups']}):")
        for group in report.groups:
            fmt.echo(
                f"    - {group.name} -> dataset '{group.dataset_name}' "
                f"({len(group.file_paths)} file(s), {group.kind})"
            )
        for warning in report.warnings:
            fmt.warning(warning)
        if report.report_path:
            fmt.echo(f"  Report saved to: {report.report_path}")

        if report.apply_results is not None:
            fmt.success(f"Auto-applied: ingested {len(report.apply_results)} dataset(s)")
            for dataset_name in report.apply_results:
                fmt.echo(f"  - {dataset_name}")
                hint_recall(dataset_name)
        elif report.report_path:
            fmt.echo(f"  Apply with: cognee-cli remember --from-report {report.report_path}")

    def _execute_apply(self, args: argparse.Namespace) -> None:
        from cognee.tasks.presort.models import PresortReport

        if args.data:
            raise CliCommandInnerException("--from-report cannot be combined with data arguments.")
        report = PresortReport.from_json(args.from_report)
        if args.allow_root and report.root_path:
            self._extend_allowed_roots([report.root_path])

        from cognee.modules.presort.llm_availability import llm_is_configured

        if not llm_is_configured():
            fmt.warning(
                "LLM API key not configured — files will be staged (add) without building "
                "graphs; run cognify on the datasets once a key is set."
            )

        fmt.echo(f"Applying presort report for '{report.root_path}'...")

        async def run_apply():
            import cognee

            return await cognee.remember(
                data=report,
                apply_groups=args.apply_group,
                apply_graph=args.apply_graph,
                graph_dataset=args.graph_dataset,
                exclude_pii=args.exclude_pii or None,
                skip_duplicates=False if args.keep_duplicates else None,
                run_in_background=args.background,
            )

        results = asyncio.run(run_apply())
        fmt.success(f"Ingested {len(results)} dataset(s) from the presort report")
        for dataset_name in results:
            fmt.echo(f"  - {dataset_name}")
            hint_recall(dataset_name)

    def execute(self, args: argparse.Namespace) -> None:
        try:
            import cognee

            if args.from_report:
                self._execute_apply(args)
                return

            if args.dry_run == "presort":
                self._execute_presort(args)
                return
            if args.dry_run not in (True, False):
                raise CliCommandInnerException(
                    f"Unsupported --dry-run value {args.dry_run!r}; use --dry-run or "
                    "--dry-run presort."
                )

            if args.allow_root and args.data:
                # Plain folder inputs auto-presort; the scan needs the folder
                # inside the allowed local file roots.
                self._extend_allowed_roots(args.data)

            if args.sample_data:
                if args.data:
                    raise CliCommandInnerException(
                        "--sample-data cannot be combined with explicit data arguments."
                    )
                sample_path = _resolve_sample_path()
                fmt.note(f"Using bundled sample fixture: {sample_path}")
                args.data = [sample_path]
            elif not args.data:
                raise CliCommandInnerException(
                    "No data supplied. Pass file paths, text, or --sample-data."
                )

            dry_run = getattr(args, "dry_run", False)
            action = "Estimating" if dry_run else "Remembering"
            fmt.echo(f"{action} {len(args.data)} item(s) in dataset '{args.dataset_name}'...")

            async def run_remember():
                try:
                    from cognee.modules.chunking.TextChunker import TextChunker

                    chunker_class = TextChunker
                    if args.chunker == "LangchainChunker":
                        try:
                            from cognee.modules.chunking.LangchainChunker import LangchainChunker

                            chunker_class = LangchainChunker
                        except ImportError:
                            fmt.warning("LangchainChunker not available, using TextChunker")
                    elif args.chunker == "CsvChunker":
                        try:
                            from cognee.modules.chunking.CsvChunker import CsvChunker

                            chunker_class = CsvChunker
                        except ImportError:
                            fmt.warning("CsvChunker not available, using TextChunker")

                    data_to_add = args.data[0] if len(args.data) == 1 else args.data

                    result = await cognee.remember(
                        data=data_to_add,
                        dataset_name=args.dataset_name,
                        chunker=chunker_class,
                        chunk_size=args.chunk_size,
                        chunks_per_batch=args.chunks_per_batch,
                        run_in_background=args.background,
                        dry_run=dry_run,
                    )
                    return result
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to remember: {str(e)}") from e

            result = asyncio.run(run_remember())

            from cognee.tasks.presort.models import PresortReport

            if isinstance(result, PresortReport):
                # A plain folder input was presorted automatically.
                self._print_presort_report(result)
                return

            if dry_run:
                fmt.echo(str(result))
                return

            if args.background:
                fmt.success("Data ingested and cognification started in background!")
            else:
                fmt.success("Data ingested and knowledge graph built successfully!")

            if result:
                if result.dataset_id:
                    fmt.echo(f"  Dataset ID: {result.dataset_id}")
                if result.items_processed:
                    fmt.echo(f"  Items processed: {result.items_processed}")
                if result.content_hash:
                    fmt.echo(f"  Content hash: {result.content_hash}")
                if result.elapsed_seconds is not None:
                    fmt.echo(f"  Elapsed: {result.elapsed_seconds:.1f}s")

            hint_recall(args.dataset_name)

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Failed to remember: {str(e)}", error_code=1) from e
