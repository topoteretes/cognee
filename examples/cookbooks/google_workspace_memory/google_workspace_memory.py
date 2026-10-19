"""Google Workspace memory: check setup, remember Calendar, Gmail and Drive, then ask across them.

    uv run python examples/cookbooks/google_workspace_memory/google_workspace_memory.py --check
    uv run python examples/cookbooks/google_workspace_memory/google_workspace_memory.py
    uv run python examples/cookbooks/google_workspace_memory/google_workspace_memory.py --no-drive

Without credentials.json it runs on the sample workspace from setup.py, so it works with
only LLM_API_KEY. Each script in scripts/ also runs alone. Exit codes: 0 done, 2 setup
missing, 1 a script failed (its message says why).
"""

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

from scripts.ask import ask
from scripts.clear import clear
from scripts.ingest_calendar import ingest_calendar
from scripts.ingest_drive import ingest_drive
from scripts.ingest_gmail import email_count, ingest_gmail
from setup import write_sample

COOKBOOK_DIR = Path(__file__).parent
# Installed by any of the extras gmail, google-drive and google-calendar.
GOOGLE_LIBRARIES = ("googleapiclient", "google_auth_oauthlib")


def missing_setup(args: argparse.Namespace) -> list[str]:
    """What still has to be set up, one line each. Empty when everything is ready."""
    missing = []
    if not os.environ.get("LLM_API_KEY"):
        missing.append("LLM_API_KEY is not set (put it in .env).")
    if args.no_calendar and args.no_gmail and args.no_drive:
        missing.append("Every app is skipped; drop one of --no-calendar, --no-gmail, --no-drive.")
    if args.sample:  # the sample replaces all three apps
        return missing
    if not all(importlib.util.find_spec(name) for name in GOOGLE_LIBRARIES):
        missing.append(
            "Google client libraries are not installed "
            "(uv sync --extra gmail --extra google-drive --extra google-calendar)."
        )
    if not (COOKBOOK_DIR / "credentials.json").exists():
        missing.append("Google OAuth client not found at credentials.json in the cookbook folder.")
    if not args.no_drive and not os.environ.get("GOOGLE_DRIVE_FOLDER_ID"):
        missing.append("GOOGLE_DRIVE_FOLDER_ID is not set (put it in .env), or pass --no-drive.")
    return missing


async def run(args: argparse.Namespace) -> None:
    if args.clear:
        await clear(args.sample)
    if not args.no_calendar:
        await ingest_calendar(args.sample)
    if not args.no_gmail:
        await ingest_gmail(args.emails, args.sample)
    if not args.no_drive:
        await ingest_drive(args.sample)
    await ask(args.ask, args.sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report what is missing")
    parser.add_argument("--sample", action="store_true", help="use the sample from setup.py")
    parser.add_argument(
        "--clear",
        action=argparse.BooleanOptionalAction,
        help="forget the dataset before remembering (default: on for the sample, off otherwise)",
    )
    parser.add_argument("--no-calendar", action="store_true", help="skip the Calendar step")
    parser.add_argument("--no-gmail", action="store_true", help="skip the Gmail step")
    parser.add_argument("--no-drive", action="store_true", help="skip the Drive step")
    parser.add_argument("--emails", type=email_count, default=50, help="inbox emails to remember")
    parser.add_argument("--ask", help="your own question instead of the meeting prep")
    args = parser.parse_args()

    import cognee  # loads .env, so keys set there are seen by the check

    if not args.sample and not (COOKBOOK_DIR / "credentials.json").exists():
        print("[setup] No Google OAuth client is set up, so this runs on the sample.")
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
    if args.check:
        print("[setup] OK: ready to run.")
        sys.exit(0)

    if args.sample:
        write_sample()
        print("[setup] Wrote the sample from setup.py.")
    asyncio.run(run(args))
