"""Follow-up agent: check setup, remember calls, issues and email, then post next steps.

    uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py --check
    uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py
    uv run python examples/cookbooks/company_brain/follow_up_agent/follow_up_agent.py --no-email --ui

Granola is required. Linear, Gmail and Slack are used when they are set up and skipped
otherwise. With none of Granola, Linear or Gmail set up it runs on the sample from setup.py, so it works
with only LLM_API_KEY; a sample run never posts to Slack. Each script in scripts/ also runs alone. Exit codes: 0 done, 2 setup missing,
1 a script failed (its message says why).
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

from scripts.clear import clear
from scripts.follow_up import follow_up
from scripts.ingest_email import ingest_email
from scripts.ingest_granola import ingest_granola
from scripts.ingest_linear import ingest_linear
from scripts.ui import open_ui
from setup import write_sample

COOKBOOK_DIR = Path(__file__).parent


def no_sources(args: argparse.Namespace) -> bool:
    """True when none of Granola, Linear and Gmail is set up, so the sample stands in."""
    return not (
        os.environ.get("GRANOLA_API_KEY")
        or os.environ.get("LINEAR_API_KEY")
        or (COOKBOOK_DIR / "credentials.json").exists()
    )


def missing_setup(args: argparse.Namespace) -> list[str]:
    """What still has to be set up, one line each. Empty when everything is ready."""
    missing = []
    if not os.environ.get("LLM_API_KEY"):
        missing.append("LLM_API_KEY is not set (put it in .env).")
    if not args.sample and not os.environ.get("GRANOLA_API_KEY"):
        missing.append("GRANOLA_API_KEY is not set (put it in .env).")
    return missing


def optional_sources(args: argparse.Namespace) -> dict[str, bool]:
    """Which optional sources this run uses. Each one is used only when it is set up."""
    if args.sample:  # sample issues and email; never Slack
        return {"linear": True, "email": True, "slack": False}
    return {
        "linear": not args.no_linear and bool(os.environ.get("LINEAR_API_KEY")),
        "email": not args.no_email and (COOKBOOK_DIR / "credentials.json").exists(),
        "slack": bool(os.environ.get("SLACK_BOT_TOKEN") and os.environ.get("SLACK_CHANNEL")),
    }


async def run(args: argparse.Namespace, sources: dict[str, bool]) -> None:
    if args.clear:
        await clear()
    await ingest_granola(args.days, args.sample)
    if sources["linear"]:
        await ingest_linear(args.days, args.sample)
    if sources["email"]:
        await ingest_email(args.emails, args.sample)
    await follow_up(args.days, args.sample)
    if args.ui:
        await open_ui()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report what is missing")
    parser.add_argument("--sample", action="store_true", help="use the sample from setup.py")
    parser.add_argument(
        "--clear",
        action=argparse.BooleanOptionalAction,
        help="forget the dataset before remembering (default: on for the sample, off otherwise)",
    )
    parser.add_argument("--no-linear", action="store_true", help="skip the Linear step")
    parser.add_argument("--no-email", action="store_true", help="skip the Gmail step")
    parser.add_argument("--days", type=int, default=30, help="calls and issues to remember")
    parser.add_argument("--emails", type=int, default=50, help="inbox emails to remember")
    parser.add_argument("--ui", action="store_true", help="browse the graph afterwards")
    args = parser.parse_args()

    import cognee  # loads .env, so keys set there are seen by the check

    if not args.sample and no_sources(args):
        print("[setup] None of Granola, Linear or Gmail is set up, so this runs on the sample.")
        args.sample = True

    if args.clear is None:  # a sample run starts from an empty dataset unless --no-clear
        args.clear = args.sample
    if args.clear:
        print("[setup] CLEAR: the cookbook's dataset is forgotten before this run.")

    missing = missing_setup(args)
    for line in missing:
        print(f"[setup] MISSING: {line}")
    if missing:
        sys.exit(2)
    sources = optional_sources(args)
    skipped = {
        "linear": "Linear (LINEAR_API_KEY not set, or --no-linear): issues are not remembered.",
        "email": "Gmail (no credentials.json, or --no-email): emails are not remembered.",
        "slack": "Slack (not set up, or a sample run): steps are printed.",
    }
    for source, used in sources.items():
        if not used:
            print(f"[setup] SKIPPED: {skipped[source]}")
    if args.check:
        print("[setup] OK: ready to run.")
        sys.exit(0)

    if args.sample:
        write_sample()
        print("[setup] Wrote the sample from setup.py.")
    asyncio.run(run(args, sources))
