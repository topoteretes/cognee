"""Legacy script entry point for the eval harness.

Kept for backward compatibility. New code should prefer the one-command runner:

    from cognee.eval_framework.runner import run_eval
    result = await run_eval(EvalConfig())

or the CLI:  ``cognee eval ...`` / ``python -m cognee.eval_framework``.
"""

import asyncio

from cognee.shared.logging_utils import get_logger
from cognee.eval_framework.eval_config import EvalConfig
from cognee.eval_framework.runner import run_eval, summarize_result

logger = get_logger()


async def main():
    result = await run_eval(EvalConfig())
    for line in summarize_result(result):
        logger.info(line)
    return result


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        print("Done")
