"""Remember your Granola calls from the last 30 days (node set `calls`).

Needs GRANOLA_API_KEY (create one in Granola's settings) in .env at the repo root.

Run alone: uv run python examples/cookbooks/company_brain/follow_up_agent/scripts/ingest_granola.py [--days N]
"""

import argparse
import asyncio
import os
from datetime import datetime, timedelta, timezone

import httpx

import cognee  # also loads .env, so a GRANOLA_API_KEY set there is seen
from cognee.shared.logging_utils import ERROR, setup_logging

DATASET = "company_brain"  # the same in every script


def granola_calls(days: int) -> list[str]:
    """Your Granola calls from the last ``days`` days, each as text."""
    if not os.environ.get("GRANOLA_API_KEY"):
        raise SystemExit("[ingest_granola] MISSING: GRANOLA_API_KEY is not set (put it in .env).")
    # The API rejects microseconds and "+00:00"; it wants a plain UTC "...Z" timestamp.
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    headers = {"Authorization": f"Bearer {os.environ['GRANOLA_API_KEY']}"}
    api = httpx.Client(base_url="https://public-api.granola.ai/v1/", headers=headers, timeout=60)
    calls, params = [], {"created_after": since, "page_size": 30}
    while True:
        page = api.get("notes", params=params).raise_for_status().json()
        for listed in page["notes"]:
            note = api.get(f"notes/{listed['id']}", params={"include": "transcript"}).json()
            attendees = ", ".join(a.get("name") or a["email"] for a in note.get("attendees") or [])
            transcript = "\n".join(
                f"{(turn['speaker'] or {}).get('name') or 'Speaker'}: {turn['text']}"
                for turn in note.get("transcript") or []
            )
            calls.append(
                f"Call: {note.get('title')}\nDate: {note.get('created_at')}\n"
                f"Attendees: {attendees}\n\n{note.get('summary_text') or ''}\n\n{transcript}"
            )
        if not page.get("hasMore"):
            return calls
        params["cursor"] = page["cursor"]


async def ingest_granola(days: int = 30) -> None:
    calls = granola_calls(days)
    if not calls:
        raise SystemExit(f"[ingest_granola] No Granola calls in the last {days} days.")
    await cognee.remember(calls, dataset_name=DATASET, node_set=["calls"], self_improvement=False)
    print(f"[ingest_granola] Remembered {len(calls)} Granola calls from the last {days} days")


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=30, help="how far back to read calls")
    asyncio.run(ingest_granola(parser.parse_args().days))
