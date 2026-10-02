"""Personalized email: check setup, then remember Granola and Gmail and draft a reply.

    uv run python examples/cookbooks/personalized_email/personalized_email.py --check
    uv run python examples/cookbooks/personalized_email/personalized_email.py
    uv run python examples/cookbooks/personalized_email/personalized_email.py --no-granola

With neither Gmail nor Granola set up it runs on the sample mailbox from setup.py, so it
works with only LLM_API_KEY. Each script in scripts/ also runs alone. Exit codes: 0 done, 2 setup missing, 1 a script
failed (its message says why).
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from scripts.draft import draft
from scripts.ingest_email import ingest_email
from scripts.ingest_granola import ingest_granola
from setup import write_sample

from cognee.shared.logging_utils import ERROR, setup_logging

COOKBOOK_DIR = Path(__file__).parent


def no_sources(args: argparse.Namespace) -> bool:
    """True when neither Gmail nor Granola is set up, so the sample stands in."""
    return not (COOKBOOK_DIR / "credentials.json").exists() and not os.environ.get(
        "GRANOLA_API_KEY"
    )


def missing_setup(args: argparse.Namespace) -> list[str]:
    """What still has to be set up, one line each. Empty when everything is ready."""
    missing = []
    if not os.environ.get("LLM_API_KEY"):
        missing.append("LLM_API_KEY is not set (put it in .env).")
    if args.sample:  # the sample replaces Gmail and Granola
        return missing
    if not (COOKBOOK_DIR / "credentials.json").exists():
        missing.append("Gmail OAuth client not found at credentials.json in the cookbook folder.")
    if not args.no_granola and not os.environ.get("GRANOLA_API_KEY"):
        missing.append("GRANOLA_API_KEY is not set (put it in .env), or pass --no-granola.")
    return missing


async def run(args: argparse.Namespace) -> None:
    if not args.no_granola:
        await ingest_granola(args.days, args.sample)
    await ingest_email(args.emails, args.sample)
    await draft(args.sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report what is missing")
    parser.add_argument("--sample", action="store_true", help="use the sample from setup.py")
    parser.add_argument("--no-granola", action="store_true", help="skip the Granola step")
    parser.add_argument("--days", type=int, default=30, help="Granola meetings to remember")
    parser.add_argument("--emails", type=int, default=50, help="emails to remember per label")
    args = parser.parse_args()

    import cognee  # loads .env, so keys set there are seen by the check

    if not args.sample and no_sources(args):
        print("[setup] Neither Gmail nor Granola is set up, so this runs on the sample.")
        args.sample = True

    missing = missing_setup(args)
    for line in missing:
        print(f"[setup] MISSING: {line}")
    if missing:
        sys.exit(2)
    if args.check:
        print("[setup] OK: ready to run.")
        sys.exit(0)

    setup_logging(log_level=ERROR)
    if args.sample:
        write_sample()
        print("[setup] Wrote the sample from setup.py.")
    asyncio.run(run(args))
