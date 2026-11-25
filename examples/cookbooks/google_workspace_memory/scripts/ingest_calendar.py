"""Remember every event of your primary Google Calendar (node set `calendar`).

Each event becomes one document with its time, organizer, attendees and description, so the
same client links across meetings. Calendar comes in through cognee's Google Calendar
connector, `google_calendar_source`. Needs credentials.json (a Google OAuth Desktop client)
in the cookbook folder; calendar_token.json is written there on the first run. The first run
reads the whole calendar, past and future; later runs fetch only what changed. With
--sample, it reads the events setup.py wrote into the dataset `google_workspace_memory_sample`
instead.

Run alone: uv run python examples/cookbooks/google_workspace_memory/scripts/ingest_calendar.py [--sample]
"""

import argparse
import asyncio
import os
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

import cognee
from cognee.tasks.ingestion.connectors import google_calendar_source
from cognee.tasks.ingestion.connectors.google_calendar import GoogleCalendarSettings

DATASET = "google_workspace_memory"  # the same in every script
SAMPLE_DATASET = "google_workspace_memory_sample"  # the sample never mixes with your data
COOKBOOK_DIR = Path(__file__).parent.parent
CREDENTIALS, TOKEN = COOKBOOK_DIR / "credentials.json", COOKBOOK_DIR / "calendar_token.json"
SAMPLE = COOKBOOK_DIR / "sample" / "calendar"


async def ingest_calendar(sample: bool = False) -> None:
    if sample:
        events = [path.read_text() for path in sorted(SAMPLE.glob("*.txt"))]
        await cognee.remember(
            events, dataset_name=SAMPLE_DATASET, node_set=["calendar"], self_improvement=False
        )
        print(f"[ingest_calendar] Remembered {len(events)} sample calendar events")
        return
    if not CREDENTIALS.exists():
        raise SystemExit(f"[ingest_calendar] MISSING: Google OAuth client at {CREDENTIALS}")
    print(
        "[ingest_calendar] Remembering your primary calendar. The first run reads every "
        "event and can take ten minutes or more; later runs fetch only changes.",
        flush=True,  # shows before the long remember(), even when output is piped
    )
    # The empty tokens keep a GOOGLE_CALENDAR_REFRESH_TOKEN or _ACCESS_TOKEN in .env from
    # taking precedence over the cookbook's credentials.json.
    settings = GoogleCalendarSettings(
        credentials_path=str(CREDENTIALS), token_path=str(TOKEN), refresh_token="", access_token=""
    )
    source = google_calendar_source(["primary"], settings=settings)
    await cognee.remember(
        source,
        dataset_name=DATASET,
        node_set=["calendar"],
        write_disposition="merge",  # the connector's rows are merged by event id
        primary_key="id",
        self_improvement=False,
    )
    stats = source.cognee_sync_stats
    print(
        f"[ingest_calendar] Remembered your primary calendar: {stats.get('scanned', 0)} new or "
        f"changed events, {stats.get('deleted', 0)} deleted"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", action="store_true", help="use the sample from setup.py")
    asyncio.run(ingest_calendar(parser.parse_args().sample))
