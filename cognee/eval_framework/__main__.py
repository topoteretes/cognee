"""Module entry point: ``python -m cognee.eval_framework [options]``.

Thin wrapper around :func:`cognee.eval_framework.runner.run_eval` that shares its
argument surface with the ``cognee eval`` CLI command.
"""

import argparse
import asyncio

from cognee.eval_framework.runner import (
    add_eval_arguments,
    config_from_namespace,
    run_eval,
    summarize_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m cognee.eval_framework",
        description="Run a cognee memory-quality benchmark end to end in one command.",
    )
    add_eval_arguments(parser)
    args = parser.parse_args()

    config = config_from_namespace(args)
    result = asyncio.run(run_eval(config))

    print("\nEvaluation complete.")
    for line in summarize_result(result):
        print(line)


if __name__ == "__main__":
    main()
