import argparse
import asyncio

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException


class EvalCommand(SupportsCliCommand):
    command_string = "eval"
    help_string = "Run a memory-quality benchmark end to end (corpus -> answer -> evaluate)"
    docs_url = DEFAULT_DOCS_URL
    description = """
Run a reproducible memory-quality benchmark in one command.

This chains the evaluation harness pipeline - corpus building, question
answering, evaluation, and an optional dashboard - for a single deterministic
config. No Modal or Docker is required for local runs.

Examples:
  cognee eval --benchmark HotPotQA --engine direct_llm --limit 5
  cognee eval --benchmark Dummy --no-dashboard --output-dir eval_results

The cognee[eval] extra provides the HTML dashboard (on by default), the DeepEval
engine, and some benchmark dataset downloads; a --engine direct_llm
--no-dashboard run works without it. Install it with:  pip install "cognee[eval]"

Once a LongMemEval adapter is registered in the benchmark registry, run it with:
  cognee eval --benchmark LongMemEval --engine direct_llm
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        # Import here to keep CLI startup light and avoid importing the eval
        # harness (and its optional deps) unless the eval command is used.
        from cognee.eval_framework.runner import add_eval_arguments

        add_eval_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        try:
            from cognee.eval_framework.runner import (
                config_from_namespace,
                run_eval,
                summarize_result,
            )

            config = config_from_namespace(args)

            fmt.echo(
                f"Running eval: benchmark={config.benchmark}, "
                f"engine={config.evaluation_engine}, "
                f"samples={config.number_of_samples_in_corpus}, seed={config.seed}"
            )

            async def _run():
                try:
                    return await run_eval(config)
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to run eval: {str(e)}") from e

            result = asyncio.run(_run())

            fmt.success("Evaluation completed successfully!")
            for line in summarize_result(result):
                fmt.echo(line)

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1) from e
            raise CliCommandException(f"Error during evaluation: {str(e)}", error_code=1) from e
