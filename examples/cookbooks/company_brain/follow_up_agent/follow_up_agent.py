"""Follow-up agent: check setup, remember calls, issues and email, then post next steps.

    uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py --check
    uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py
    uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py --no-email --ui

Granola is required. Linear, Gmail and Slack are used when they are set up and skipped
otherwise. Each script in scripts/ also runs alone. Exit codes: 0 done, 2 setup missing,
1 a script failed (its message says why).
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from scripts.follow_up import follow_up
from scripts.ingest_email import ingest_email
from scripts.ingest_granola import ingest_granola
from scripts.ingest_linear import ingest_linear
from scripts.ui import open_ui

from cognee.shared.logging_utils import ERROR, setup_logging

COOKBOOK_DIR = Path(__file__).parent


def missing_setup() -> list[str]:
    """What still has to be set up, one line each. Empty when everything is ready."""
    missing = []
    if not os.environ.get("LLM_API_KEY"):
        missing.append("LLM_API_KEY is not set (put it in .env).")
    if not os.environ.get("GRANOLA_API_KEY"):
        missing.append("GRANOLA_API_KEY is not set (put it in .env).")
    return missing


def optional_sources(args: argparse.Namespace) -> dict[str, bool]:
    """Which optional sources this run uses. Each one is used only when it is set up."""
    return {
        "linear": not args.no_linear and bool(os.environ.get("LINEAR_API_KEY")),
        "email": not args.no_email and (COOKBOOK_DIR / "credentials.json").exists(),
        "slack": bool(os.environ.get("SLACK_BOT_TOKEN") and os.environ.get("SLACK_CHANNEL")),
    }


async def run(args: argparse.Namespace, sources: dict[str, bool]) -> None:
    await ingest_granola(args.days)
    if sources["linear"]:
        await ingest_linear(args.days)
    if sources["email"]:
        await ingest_email(args.emails)
    await follow_up(args.days)
    if args.ui:
        await open_ui()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report what is missing")
    parser.add_argument("--no-linear", action="store_true", help="skip the Linear step")
    parser.add_argument("--no-email", action="store_true", help="skip the Gmail step")
    parser.add_argument("--days", type=int, default=30, help="calls and issues to remember")
    parser.add_argument("--emails", type=int, default=50, help="inbox emails to remember")
    parser.add_argument("--ui", action="store_true", help="browse the graph afterwards")
    args = parser.parse_args()

    import cognee  # loads .env, so keys set there are seen by the check

    missing = missing_setup()
    for line in missing:
        print(f"[setup] MISSING: {line}")
    if missing:
        sys.exit(2)
    sources = optional_sources(args)
    skipped = {
        "linear": "Linear (LINEAR_API_KEY not set, or --no-linear): issues are not remembered.",
        "email": "Gmail (no credentials.json, or --no-email): emails are not remembered.",
        "slack": "Slack (SLACK_BOT_TOKEN / SLACK_CHANNEL not set): steps are printed.",
    }
    for source, used in sources.items():
        if not used:
            print(f"[setup] SKIPPED: {skipped[source]}")
    if args.check:
        print("[setup] OK: ready to run.")
        sys.exit(0)

    setup_logging(log_level=ERROR)
    asyncio.run(run(args, sources))
