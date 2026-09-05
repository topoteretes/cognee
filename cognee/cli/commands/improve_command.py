import argparse
import asyncio
from typing import Any, Optional

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException
from cognee.modules.data.constants import DEFAULT_DATASET_NAME


def _field(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from an ``ImproveResult``/``StageResult`` or its JSON dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def format_stage_line(stage: Any) -> str:
    """One CLI line for one stage: name, status, then the reason / counts / error.

    Accepts a ``StageResult`` (in-process) or its serialized dict (``--api-url``).
    """
    name = str(_field(stage, "stage", "?"))
    status = str(_field(stage, "status", "?"))
    details = []
    reason = _field(stage, "reason")
    if reason:
        details.append(str(reason))
    counts = _field(stage, "counts") or {}
    if isinstance(counts, dict) and counts:
        details.append(", ".join(f"{key}={value}" for key, value in counts.items()))
    error = _field(stage, "error")
    if error and status == "errored":
        details.append(str(error))
    duration_ms = _field(stage, "duration_ms") or 0
    if duration_ms:
        details.append(f"{int(duration_ms)} ms")
    detail_text = f"  {'; '.join(details)}" if details else ""
    return f"  {name:<24} {status:<18}{detail_text}"


def print_improve_result(result: Any, background: bool = False) -> None:
    """Print the outcome of an improve run: a headline plus one line per stage."""
    if background:
        fmt.success("Improvement started in background!")
        return

    status = _field(result, "status")
    stages = _field(result, "stages") or []
    lock_held = bool(stages) and all(
        _field(stage, "status") == "skipped" and _field(stage, "reason") == "lock_held"
        for stage in stages
    )

    if status == "errored":
        fmt.warning("Knowledge graph improvement finished with errors.")
    elif status == "skipped":
        if lock_held:
            fmt.warning(
                "Another improve run already holds this dataset/session; nothing ran (lock_held)."
            )
        else:
            fmt.warning("Nothing to improve: every stage was skipped.")
    else:
        fmt.success("Knowledge graph improved successfully!")

    for stage in stages:
        fmt.echo(format_stage_line(stage))

    error = _field(result, "error")
    if error and status == "errored" and not any(_field(s, "error") == error for s in stages):
        fmt.echo(f"  error: {error}")


class ImproveCommand(SupportsCliCommand):
    command_string = "improve"
    help_string = "Enrich and improve the knowledge graph"
    docs_url = DEFAULT_DOCS_URL
    description = """
Enrich and improve the knowledge graph.

Runs the self-improvement loop over a dataset: nine stages in a fixed order,
each of which first declines work it cannot do (no LLM calls) and then runs.
Stages 1-7 (feedback weights, session Q&A / trace persistence, agent context,
distillation, user preferences, truth subspace) need --session-ids; triplet
enrichment always runs when the graph changed; the truth subspace and the
global context index are opt-in flags. The result prints one line per stage.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--dataset-name",
            "-d",
            default=DEFAULT_DATASET_NAME,
            help="Dataset name (default: main_dataset)",
        )
        parser.add_argument(
            "--dataset-id",
            help="Dataset UUID (alternative to --dataset-name)",
        )
        parser.add_argument(
            "--node-name",
            nargs="*",
            help="Filter to specific named entities",
        )
        parser.add_argument(
            "--session-ids",
            "-s",
            nargs="+",
            help="Session IDs whose feedback and Q&A content should be bridged into the permanent graph",
        )
        parser.add_argument(
            "--feedback-alpha",
            type=float,
            default=None,
            help=(
                "Learning rate in (0, 1] for feedback weight updates "
                "(default: IMPROVE_FEEDBACK_ALPHA, 0.1)"
            ),
        )
        parser.add_argument(
            "--build-global-context-index",
            action="store_true",
            help="Build the global context index after enrichment (opt-in stage)",
        )
        parser.add_argument(
            "--build-truth-subspace",
            action="store_true",
            help=(
                "Build the truth subspace from the sessions' distilled learnings "
                "(opt-in stage; needs --session-ids and a backend with truth state)"
            ),
        )
        parser.add_argument(
            "--background",
            "-b",
            action="store_true",
            help="Run processing in background",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            import cognee

            dataset = args.dataset_id if args.dataset_id else args.dataset_name
            fmt.echo(f"Improving knowledge graph for dataset '{dataset}'...")

            feedback_alpha: Optional[float] = getattr(args, "feedback_alpha", None)
            build_global_context_index = bool(getattr(args, "build_global_context_index", False))
            build_truth_subspace = bool(getattr(args, "build_truth_subspace", False))

            async def run_improve():
                try:
                    from uuid import UUID

                    dataset_arg = UUID(args.dataset_id) if args.dataset_id else args.dataset_name

                    improve_kwargs = {}
                    if feedback_alpha is not None:
                        improve_kwargs["feedback_alpha"] = feedback_alpha

                    result = await cognee.improve(
                        dataset=dataset_arg,
                        node_name=args.node_name,
                        session_ids=args.session_ids,
                        build_global_context_index=build_global_context_index,
                        build_truth_subspace=build_truth_subspace,
                        run_in_background=args.background,
                        **improve_kwargs,
                    )
                    return result
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to improve: {str(e)}") from e

            result = asyncio.run(run_improve())

            print_improve_result(result, background=bool(args.background))

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Error improving: {str(e)}", error_code=1) from e
