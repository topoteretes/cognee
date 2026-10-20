"""Ingest a Google Calendar into cognee memory, incrementally.

Demonstrates the Google Calendar DLT connector: each event becomes one text
document with its title, time, location, organizer, attendees and their
responses, and description, so cognify links the same people across meetings.
A recurring series is one document with its repeat rule. Re-running this script
fetches only what changed through Google's sync token, and deleted events are
forgotten automatically.

The guide syncs your primary calendar only. The first sync reads every event
in it, past and future, and cognifies each one with your LLM, so a calendar
with years of history takes several minutes. Later syncs are fast.

Install:
    pip install "cognee[google-calendar]"     # or: uv sync --extra google-calendar

Auth setup (OAuth user credentials):
  - In the Google Cloud Console, enable the "Google Calendar API", add your
    account as a test user under Google Auth Platform → Audience, and add the
    ``calendar.readonly`` scope under Data Access.
  - Create an OAuth client of type "Desktop app" under Clients and download its JSON.
  - Set:
      GOOGLE_CALENDAR_CREDENTIALS_PATH=/path/to/oauth-client-secret.json
      GOOGLE_CALENDAR_TOKEN_PATH=/path/to/cache-the-user-token.json
  - The first run opens a browser for one-time consent; later runs reuse the
    cached, auto-refreshed token.
  - A GOOGLE_CALENDAR_ACCESS_TOKEN or GOOGLE_CALENDAR_REFRESH_TOKEN in the
    environment takes precedence over the two paths; unset them to use this flow.

Also set the usual cognee LLM_API_KEY (see .env.template) — this example
calls cognee.recall(), which needs an LLM for the final completion.

Run it:

    uv run python examples/guides/google_calendar.py
"""

import asyncio

import cognee
from cognee.tasks.ingestion.connectors import google_calendar_source

DATASET_NAME = "google_calendar_demo"
CALENDARS = ["primary"]

# "merge" is required: it makes re-runs incremental and propagates deleted
# events through orphan cleanup.
REMEMBER_KWARGS = {
    "dataset_name": DATASET_NAME,
    "primary_key": "id",
    "write_disposition": "merge",
}


async def main():
    print("=== Initial sync ===")
    # Your own calendar. Pass other ids, like "team@group.calendar.google.com",
    # to sync more; GOOGLE_CALENDAR_* env vars supply the credentials.
    source = google_calendar_source(CALENDARS)
    await cognee.remember(source, **REMEMBER_KWARGS)
    print("Sync stats:", source.cognee_sync_stats)

    answer = await cognee.recall(
        "Who do I meet with most often, and what are those meetings about?",
        datasets=[DATASET_NAME],
    )
    print("Recall:", answer[0].text if answer else "nothing found")

    print("\n=== Incremental re-sync (only new, changed or deleted events) ===")
    source = google_calendar_source(CALENDARS)
    await cognee.remember(source, **REMEMBER_KWARGS)
    print("Sync stats:", source.cognee_sync_stats)


if __name__ == "__main__":
    asyncio.run(main())
