"""Google Workspace memory: check setup, remember Calendar, Gmail and Drive, then ask across them.

    uv run python examples/cookbooks/google_workspace_memory/google_workspace_memory.py --check
    uv run python examples/cookbooks/google_workspace_memory/google_workspace_memory.py
    uv run python examples/cookbooks/google_workspace_memory/google_workspace_memory.py --no-drive
    uv run python examples/cookbooks/google_workspace_memory/google_workspace_memory.py --api

Without credentials.json it runs on the sample workspace from setup.py, so it works with
only LLM_API_KEY. With --api a running cognee server syncs your connected Google account
and answers instead (scripts/sync_api.py). Each script in scripts/ also runs alone.

Exit codes: 0 done, 2 setup missing (each [setup] MISSING line names one fix), 1 a step
failed and stopped the run: its [<step>] line says why, or, for an unexpected error, the
traceback does.
"""

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

from scripts.ask import ask, ask_api
from scripts.clear import clear
from scripts.ingest_calendar import ingest_calendar
from scripts.ingest_drive import ingest_drive
from scripts.ingest_gmail import email_count, ingest_gmail
from scripts.sync_api import missing_connections, sync_api
from setup import write_sample

COOKBOOK_DIR = Path(__file__).parent
# Installed by any of the extras gmail, google-drive and google-calendar.
GOOGLE_LIBRARIES = ("googleapiclient", "google_auth_oauthlib")


def apps(args: argparse.Namespace) -> list[str]:
    """The apps this run reads, in order."""
    skipped = {"calendar": args.no_calendar, "gmail": args.no_gmail, "drive": args.no_drive}
    return [app for app, skip in skipped.items() if not skip]


def missing_setup(args: argparse.Namespace) -> list[str]:
    """What still has to be set up, one line each. Empty when everything is ready."""
    missing = []
    if not apps(args):
        missing.append("Every app is skipped; drop one of --no-calendar, --no-gmail, --no-drive.")
    if args.api:  # the server holds the LLM key and the Google clients
        if args.sample or args.clear:
            missing.append("--sample and --clear work on local memory only; drop them with --api.")
        if not os.environ.get("COGNEE_API_TOKEN"):
            missing.append("COGNEE_API_TOKEN is not set: your bearer token for the cognee server.")
        elif apps(args):
            missing.extend(asyncio.run(missing_connections(apps(args))))
        if not args.no_drive and not os.environ.get("GOOGLE_DRIVE_FOLDER_ID"):
            missing.append(
                "GOOGLE_DRIVE_FOLDER_ID is not set (put it in .env), or pass --no-drive."
            )
        return missing
    if not os.environ.get("LLM_API_KEY"):
        missing.append("LLM_API_KEY is not set (put it in .env).")
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
    if args.api:
        await ask_api(args.ask, await sync_api(apps(args)))
        return
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
        "--api", action="store_true", help="sync and ask through a running cognee server"
    )
    parser.add_argument(
        "--clear",
        action=argparse.BooleanOptionalAction,
        help="forget the dataset before remembering (default: on for the sample, off otherwise)",
    )
    parser.add_argument("--no-calendar", action="store_true", help="skip the Calendar step")
    parser.add_argument("--no-gmail", action="store_true", help="skip the Gmail step")
    parser.add_argument("--no-drive", action="store_true", help="skip the Drive step")
    parser.add_argument(
        "--emails",
        type=email_count,
        default=50,
        help="inbox emails to remember (not with --api: the server reads the label)",
    )
    parser.add_argument("--ask", help="your own question instead of the meeting prep")
    args = parser.parse_args()

    import cognee  # loads .env, so keys set there are seen by the check

    if not args.api and not args.sample and not (COOKBOOK_DIR / "credentials.json").exists():
        print("[setup] No Google OAuth client is set up, so this runs on the sample.")
        args.sample = True

    if args.clear is None:  # a sample run starts from an empty dataset unless --no-clear
        args.clear = args.sample

    missing = missing_setup(args)
    for line in missing:
        print(f"[setup] MISSING: {line}")
    if missing:
        sys.exit(2)
    if args.clear:
        print("[setup] CLEAR: the cookbook's dataset is forgotten before this run.")
    if args.check:
        print("[setup] OK: ready to run.")
        sys.exit(0)

    if args.sample:
        write_sample()
        print("[setup] Wrote the sample from setup.py.")
    asyncio.run(run(args))
