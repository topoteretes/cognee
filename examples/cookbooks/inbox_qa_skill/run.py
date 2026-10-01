"""Orchestrator: check setup, then run the three steps in order.

    uv run python examples/cookbooks/inbox_qa_skill/run.py --check      # setup only, no work
    uv run python examples/cookbooks/inbox_qa_skill/run.py              # all steps
    uv run python examples/cookbooks/inbox_qa_skill/run.py --no-email --file my.txt \
        --question "What did we decide?"

Exit codes: 0 done, 2 setup missing (the message says what to fix).
"""

import argparse
import asyncio
import sys
from pathlib import Path

import answer
import ingest_email
import ingest_file
from common import SAMPLE_FILE, missing_setup

from cognee.shared.logging_utils import ERROR, setup_logging


async def main(args: argparse.Namespace) -> None:
    if not args.no_email:
        await ingest_email.run()
    await ingest_file.run(args.file)
    await answer.run(args.question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report what is missing")
    parser.add_argument("--no-email", action="store_true", help="skip the Gmail step")
    parser.add_argument("--file", type=Path, default=SAMPLE_FILE, help="text file to remember")
    parser.add_argument("--question", default=answer.DEFAULT_QUESTION)
    args = parser.parse_args()
    setup_logging(log_level=ERROR)

    missing = missing_setup(need_gmail=not args.no_email)
    for line in missing:
        print(f"[setup] MISSING: {line}")
    if missing:
        sys.exit(2)
    if args.check:
        print("[setup] OK: ready to run.")
        sys.exit(0)

    asyncio.run(main(args))
