"""Check setup, then run the scripts in order: ingest email, ingest Granola, answer.

    uv run python examples/cookbooks/inbox_qa/inbox_qa.py --check      # setup only, no work
    uv run python examples/cookbooks/inbox_qa/inbox_qa.py              # everything
    uv run python examples/cookbooks/inbox_qa/inbox_qa.py --no-email --days 7 \
        --question "What did we decide?"

Each script in scripts/ also runs alone. Exit codes: 0 done, 2 setup missing, 1 a script
failed (its message says why).
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from scripts.answer import DEFAULT_QUESTION, answer
from scripts.ingest_email import ingest_email
from scripts.ingest_granola import ingest_granola

from cognee.shared.logging_utils import ERROR, setup_logging

SKILL_DIR = Path(__file__).parent


def missing_setup(need_gmail: bool, need_granola: bool) -> list[str]:
    """What still has to be set up, one line each. Empty when everything is ready."""
    missing = []
    if not os.environ.get("LLM_API_KEY"):
        missing.append("LLM_API_KEY is not set (put it in .env).")
    if need_gmail and not (SKILL_DIR / "credentials.json").exists():
        missing.append("Gmail OAuth client not found at credentials.json in the skill folder.")
    if need_granola and not os.environ.get("GRANOLA_API_KEY"):
        missing.append("GRANOLA_API_KEY is not set (put it in .env).")
    return missing


async def run(args: argparse.Namespace) -> None:
    if not args.no_email:
        await ingest_email(args.emails)
    if not args.no_granola:
        await ingest_granola(args.days)
    await answer(args.question or DEFAULT_QUESTION, args.sender)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report what is missing")
    parser.add_argument("--no-email", action="store_true", help="skip the Gmail step")
    parser.add_argument("--no-granola", action="store_true", help="skip the Granola step")
    parser.add_argument("--emails", type=int, default=20, help="inbox emails to remember")
    parser.add_argument("--days", type=int, default=30, help="Granola meetings to remember")
    parser.add_argument("--question", help="what to ask (default: draft a reply)")
    parser.add_argument("--sender", help="answer the newest email from this name or address")
    args = parser.parse_args()

    missing = missing_setup(need_gmail=not args.no_email, need_granola=not args.no_granola)
    for line in missing:
        print(f"[setup] MISSING: {line}")
    if missing:
        sys.exit(2)
    if args.check:
        print("[setup] OK: ready to run.")
        sys.exit(0)

    setup_logging(log_level=ERROR)
    asyncio.run(run(args))
