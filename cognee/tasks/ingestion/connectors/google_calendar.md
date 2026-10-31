# Google Calendar SDK connector

A Google Calendar data-source connector for [cognee](https://github.com/topoteretes/cognee):
sync your calendars into memory, incrementally, and forget events when they are deleted.

It exposes a `dlt` source you hand to `cognee.remember(...)`. Each event becomes a **normal
document** in cognee (it flows through chunking and LLM entity extraction), via cognee's
document-mode marker. The calendar name, organizer and attendees with their responses are
written into the content, so the same person is linked across meetings.

## Requirements

Bundled with the Cognee SDK, including table-scoped document cleanup. No separate connector
package is required. You need a Google account.

## Install

```bash
uv pip install "cognee[google-calendar]"
# or, from this monorepo:
uv sync --extra google-calendar
```

## Setup

1. In the [Google Cloud Console](https://console.cloud.google.com), create a project (free)
   and enable the **Google Calendar API**.
2. Under **Google Auth Platform**, set up the consent screen as **External**, keep it in
   **Testing**, and add your own Google account as a **test user**. Add the scope
   `https://www.googleapis.com/auth/calendar.readonly` only.
3. Under **Clients**, create an OAuth client of type **Desktop app** and download its JSON.
4. Point the connector at it and sign in once (a browser opens for consent):

   ```bash
   export GOOGLE_CALENDAR_CREDENTIALS_PATH=~/client_secret.json
   export GOOGLE_CALENDAR_TOKEN_PATH=~/.cognee_google_calendar_token.json
   ```

   ```python
   from cognee.tasks.ingestion.connectors.google_calendar import (
       build_calendar_service,
       list_calendars,
   )

   # Caches the token in the second file, written private.
   service = build_calendar_service("client_secret.json", "token.json")
   print(list_calendars(service))  # the calendar ids you can sync
   ```

   Apps left in **Testing** get refresh tokens that expire after 7 days; publish the app,
   or sign in again, for longer use.
5. Set your `LLM_API_KEY` like any other cognee run.

**Headless or hosted:** instead of the browser flow, set `GOOGLE_CALENDAR_REFRESH_TOKEN`
with `GOOGLE_CALENDAR_CLIENT_ID` and `GOOGLE_CALENDAR_CLIENT_SECRET`, or pass a ready-made
`access_token` (never refreshed).

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import google_calendar_source

await cognee.remember(
    google_calendar_source(calendar_ids=["primary"]),  # GOOGLE_CALENDAR_* settings for auth
    dataset_name="my_calendar",
    primary_key="id",
    write_disposition="merge",  # required: incremental upserts and deletes
    max_rows_per_table=0,  # no row cap; a calendar exceeds the default 50
)

answer = await cognee.recall("Who is in my Thursday meetings?", datasets=["my_calendar"])
```

Run the same `remember` call again whenever you want to sync.

| Argument | Setting (env var) | Default |
|---|---|---|
| `calendar_ids` | `ids` (`GOOGLE_CALENDAR_IDS`, comma-separated) | `primary` |
| `settings` | `client_id`, `client_secret`, `refresh_token`, `credentials_path`, `token_path` (`GOOGLE_CALENDAR_*`) | read from the environment |
| `access_token` | `access_token` (`GOOGLE_CALENDAR_ACCESS_TOKEN`) | for hosts and quick tries only |
| `resource_name` | none | `google_calendar_events`; give each account its own name when several sync into one dataset |

`primary` and the real id of the primary calendar are the same calendar: listing both gives
one set of documents.

After a run, `source.cognee_sync_stats` holds counts only: `scanned`, `skipped`, `deleted`,
`calendars` and `failed`, plus `failed_auth` (HTTP 401), `failed_rate_limit` (HTTP 429, or 403 with a rate-limit reason) or
`failed_calendar_sync` (anything else) for the reason a calendar failed. Never titles or content.

## How sync and forget-on-delete work

- **Sync tokens:** the first run lists every event of each calendar; later runs pass that
  calendar's `nextSyncToken` and get only what changed. The token only arrives on the last
  page, so it is saved in dlt state once the whole round has finished. Sync-token requests
  keep the same parameters on every page and use no date window, as the API requires.
- **Recurring events** are listed with `singleEvents=False`: a series is one document with its
  `RRULE`, and a changed single occurrence is its own document.
- **All-day events** show their last day. Google's `end.date` is the day after the event, so a
  one-day event reads "all day, 2026-11-08" and a three-day one "2026-11-08 to 2026-11-10".
- **Deleted events** come back with `status: cancelled` and are forgotten, if they ever reached
  memory. An event created and deleted between two syncs is neither stored nor counted.
  - A **cancelled single occurrence** of a series only forgets that occurrence, never the
    series.
  - **Deleting a whole series** also forgets its changed occurrences.
- Deleted items are emitted as `{"id": ..., "_deleted": True}` with a `hard_delete` column.
  dlt removes them on `merge`, and cognee's orphan cleanup removes them from the graph and
  vector stores.
- **Errors never delete.**
  - `410 Gone` means the sync token expired: that calendar is listed again and reconciled
    like a snapshot, so events deleted meanwhile are still forgotten.
  - A calendar is forgotten only when it was synced before and has left the account's calendar
    list. A 404 for a calendar that is still listed, or that was never synced (most likely a
    typo in the config), is a failure.
  - A calendar that fails keeps its old state and is retried on the next run. The other
    calendars finish and save theirs. The run raises only when no calendar finished, because
    dlt rolls back the state of a failed run. A 401, a 429 and a 403 with a rate-limit reason
    end the run early, since the remaining calendars would fail the same way; any other 403
    (lost access) only fails its own calendar.
  - This holds for HTTP responses only. A transport error, or an auth error raised by the
    Google client library itself (google-auth's `RefreshError` is raised before any HTTP call),
    ends the whole run: the other calendars are not tried and the stats do not count it.
- Removing a calendar from `calendar_ids` forgets the events of that calendar.
- `check_active` (for hosts) runs before every API request, not only between rows.

## Privacy

This connector reads personal data. Be clear about what it does:

**What is read**

- From the selected calendars only: title, calendar name, start and end with time zone,
  location, organizer, attendees with their name, email and response, the recurrence rule,
  status, event type, the Google Meet link, the description and the event link.
- Not read: attachments, conference details beyond the Meet link, reminders, colors, and
  other calendars. `workingLocation` events are skipped.
- The only scope is `calendar.readonly`. The connector never creates, changes or deletes
  anything in Google Calendar.

**Where it goes**

- The text above becomes cognee documents. During cognify it is sent to your configured LLM
  and embedding providers, like any document you add to cognee. Use local models if that
  matters to you.
- dlt state stores each calendar's id, its sync token and its event ids. It does not store
  titles or descriptions. Document ids contain the calendar id.
- Logs and `cognee_sync_stats` contain counts and HTTP statuses only. Calendar ids, usually
  email addresses, are not logged, and the connector's errors leave out the request URL that
  Google's client puts in its own error text.
- The token file holds a refresh token. It is written private (0600); never commit it.
  Revoke access any time at <https://myaccount.google.com/permissions> and delete the file.

**How deletion is honoured**

- Deleted events, deleted series and cancelled occurrences are forgotten on the **next sync**.
  Run syncs in the foreground: forgetting happens as part of `remember`.
- Removing a calendar from `calendar_ids` forgets its events.
- To remove everything at once, forget the whole dataset with `cognee.forget`.

## Limitations

- The full history of each calendar is read on the first run; the sync-token API allows no
  date window.
- `workingLocation` events are skipped.
- Attachments are not ingested.

## Testing

```bash
uv run pytest cognee/tests/unit/tasks/connectors/ cognee/tests/integration/tasks/connectors/
```

No Google account or API key is needed. `cognee/tests/unit/tasks/connectors/fake_calendar.py`
is an in-memory Calendar API (`service.events().list(...).execute()`). It keeps a change log
per calendar and replays it through sync tokens, returning deleted events as `cancelled` stubs
and all-day events with an exclusive end date, like Google does.
`test_google_calendar_forget.py` runs the real cognee pipeline with the LLM and embeddings
mocked.
