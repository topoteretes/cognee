# cognee-connector-google-calendar

Google Calendar connector for [cognee](https://github.com/topoteretes/cognee): sync your
calendars into memory, incrementally, and forget events when they are deleted.

It exposes a `dlt` source you hand to `cognee.remember(...)`. Each event becomes a **normal
document** in cognee (it flows through chunking and LLM entity extraction), via cognee's
document-mode marker. The calendar name, organizer and attendees with their responses are
written into the content, so the same person is linked across meetings. The sync is
incremental (one `nextSyncToken` per calendar) and forgets deleted events, deleted series and
cancelled occurrences. The package also carries the provider's connection contract
(`CONNECTOR`: the Google OAuth flow, token refresh and revocation, the calendar picker and the
sync scope) for a cognee server or Cognee Cloud, and a `cognee_host/` adapter that plugs it
into a cognee server's `/api/v1/integrations/google_calendar/*` routes.

## Install

```bash
pip install cognee-connector-google-calendar             # the connector and its DLT source
pip install "cognee-connector-google-calendar[cognee]"   # plus the adapter for a cognee server (installs cognee)
```

The source needs a cognee whose ingestion scopes DLT document cleanup by table
(`DOCUMENT_SYNC_VERSION >= 1`, cognee 1.6.5 or later); an older host is refused with a clear
message when the source is built. `cognee-connector-google`, the OAuth client and contract the
three Google connectors share, is installed with it. You need a Google account.

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
   from cognee_connector_google_calendar import build_calendar_service, list_calendars

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

## Configuration

Two settings classes read the `GOOGLE_CALENDAR_` prefix, exactly as in cognee.
`cognee_connector_google_calendar.settings.GoogleCalendarOAuthSettings` is the OAuth
application block a cognee server uses; `GoogleCalendarSettings` in
`cognee_connector_google_calendar.source` adds what a local, headless or hosted SDK sync needs
(refresh token, credential files, calendar ids) and reads the same block. Hosted syncs never
use those fields. Names are unchanged from cognee.

```bash
# Google Calendar: its own OAuth client and consent screen, same variables as
# Drive. Calendars are opt-in: nothing syncs until the user selects calendars in
# the integrations page. The browser comes back with ?google_calendar=<outcome>.
# The SDK connector reads the same GOOGLE_CALENDAR_* prefix for local use
# (GOOGLE_CALENDAR_IDS, GOOGLE_CALENDAR_REFRESH_TOKEN and friends); hosted syncs
# never use those.
# Note: calendar.readonly is a sensitive scope, so a public launch needs
# Google's OAuth verification first. Test users work without it.
#GOOGLE_CALENDAR_CLIENT_ID=""
#GOOGLE_CALENDAR_CLIENT_SECRET=""
#GOOGLE_CALENDAR_STATE_SECRET=""
#GOOGLE_CALENDAR_REDIRECT_URI="http://localhost:8000/api/v1/integrations/google_calendar/callback"
#GOOGLE_CALENDAR_FRONTEND_BASE_URL="http://localhost:3000"
```

| Variable | Read by | Purpose |
| --- | --- | --- |
| `GOOGLE_CALENDAR_CLIENT_ID` | both | The OAuth client id (the server's web client, or the SDK's desktop client with a refresh token) |
| `GOOGLE_CALENDAR_CLIENT_SECRET` | both | Its client secret |
| `GOOGLE_CALENDAR_REDIRECT_URI` | settings | The callback URL, listed verbatim on the client (cognee host) |
| `GOOGLE_CALENDAR_FRONTEND_BASE_URL` | settings | Where the browser lands after the OAuth round-trip (cognee host) |
| `GOOGLE_CALENDAR_STATE_SECRET` | settings | Signs the OAuth `state` parameter (cognee host) |
| `GOOGLE_CALENDAR_REFRESH_TOKEN` | `google_calendar_source` | Headless sign-in: google-auth refreshes the access token itself |
| `GOOGLE_CALENDAR_CREDENTIALS_PATH` | `google_calendar_source` | The desktop client's JSON, for the browser flow |
| `GOOGLE_CALENDAR_TOKEN_PATH` | `google_calendar_source` | Where the browser flow caches the token (default `~/.cognee_google_calendar_token.json`) |
| `GOOGLE_CALENDAR_ACCESS_TOKEN` | `google_calendar_source` | A ready-made token from a host, never refreshed (quick tries) |
| `GOOGLE_CALENDAR_IDS` | `google_calendar_source` | Comma-separated calendar ids when `calendar_ids` is not passed; empty means the primary calendar |

## Usage

```python
import cognee
from cognee_connector_google_calendar import google_calendar_source

await cognee.remember(
    google_calendar_source(calendar_ids=["primary"]),  # GOOGLE_CALENDAR_* settings for auth
    dataset_name="my_calendar",
    primary_key="id",
    write_disposition="merge",  # required: incremental upserts and deletes
    max_rows_per_table=0,  # no row cap; a calendar exceeds the default 50
)

answer = await cognee.recall("Who is in my Thursday meetings?", datasets=["my_calendar"])
```

Run the same `remember` call again whenever you want to sync. See
`examples/google_calendar.py` for the full two-sync demo.

| Argument | Setting (env var) | Default |
|---|---|---|
| `calendar_ids` | `ids` (`GOOGLE_CALENDAR_IDS`, comma-separated) | `primary` |
| `settings` | `client_id`, `client_secret`, `refresh_token`, `credentials_path`, `token_path` (`GOOGLE_CALENDAR_*`) | read from the environment |
| `access_token` | `access_token` (`GOOGLE_CALENDAR_ACCESS_TOKEN`) | for hosts and quick tries only |
| `resource_name` | none | `google_calendar_events`; give each account its own name when several sync into one dataset |

`primary` and the real id of the primary calendar are the same calendar: listing both gives
one set of documents. An empty `calendar_ids` is refused, because it would forget every
synced event.

After a run, `source.cognee_sync_stats` holds counts only: `scanned`, `skipped`, `deleted`,
`calendars` and `failed`, plus `failed_auth` (HTTP 401), `failed_rate_limit` (HTTP 429, or 403
with a rate-limit reason) or `failed_calendar_sync` (anything else) for the reason a calendar
failed. Never titles or content.

## Running on a cognee server

```python
from cognee_connector_google_calendar.cognee_host import register

register()  # before the app starts; the provider appears in GET /api/v1/integrations/status
```

`routers()` in the same module is empty: Calendar mounts no routes of its own. The OAuth flow
runs on the generic `/api/v1/integrations/google_calendar/...` routes, and the server
receives no webhooks from Google Calendar; a connection syncs on the user's Refresh action.

On a cognee server the integration does three things:

- **Connects a Google account.** The consent asks for `openid email` and `calendar.readonly`
  only, with `access_type=offline` and `prompt=consent` so Google issues a refresh token, and
  without `include_granted_scopes`, so this grant is never folded together with a Drive or
  Gmail one. The credential is keyed on the account's stable Google subject; its email, hosted
  domain and account type (`workspace` or `personal`) are kept as cleartext metadata.
- **Keeps calendars opt-in.** A new connection starts with an empty selection
  (`selected_calendar_ids: []`) and syncs nothing until the user picks calendars with
  `GET`/`PUT /api/v1/integrations/google_calendar/resources` (hidden calendars are offered
  too). Unlike Drive and Gmail, a null selection is refused rather than read as "all": the
  source reads `calendar_ids=None` as the primary calendar or the server's
  `GOOGLE_CALENDAR_IDS`, which is no answer for a user. Each sync hands the exact selection to
  `google_calendar_source` under one DLT resource per account, into the dataset
  `google_calendar_<account email>_<hash>`, named as cognee always named it; the source keeps
  a sync token per calendar and forgets the events of a deselected one.
- **Rotates and revokes tokens.** The access token is refreshed five minutes before it
  expires; Google issues no new refresh token, so the stored one is carried forward. An
  `invalid_grant` answer means the account revoked access on Google's side and revokes the
  local credential, since no webhook would ever say so. Disconnecting revokes the grant at
  Google, best-effort.

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

## The connection contract

`cognee_connector_google_calendar.CONNECTOR` is Google Calendar on the
`cognee-connectors-core` contract, for a host that drives connectors generically (Cognee
Cloud, a future cognee release): the shared Google descriptor
(`cognee_connector_google.google_descriptor`) with Calendar's read-only scope, on
`GoogleConnector` for the userinfo lookup that keys the credential on the Google subject
(tier 2), plus the calendar picker (`list_resources`) and the sync scope (`sync_scope`). With
no resources named, the scope is every calendar the account can see, listed and named one by
one (never the source's own null default); with resources, the calendars named. A host that
keeps calendars opt-in narrows before calling, as the contract has it. The connector refreshes
through the refresh grant, keeps the stored refresh token Google does not rotate, revokes the
grant on disconnect without ever raising, and polls hourly for a host that schedules syncs.
Its dataset is `google_calendar_<google subject>`, as Cognee Cloud names it; the cognee server
keeps its own email-based name (see above).

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

## Development

```bash
export COGNEE_ENV_FILE=<an empty file> TELEMETRY_DISABLED=1 LITELLM_MODE=PRODUCTION LLM_API_KEY=sk-test-dummy
PYTHONPATH=core:connectors/google:connectors/google-calendar python -m pytest connectors/google-calendar/tests -q
PYTHONPATH=core:connectors/google:connectors/google-calendar python scripts/check_cognee_free.py cognee_connector_google_calendar
```

No Google account or API key is needed. `tests/unit/fake_calendar.py` is an in-memory
Calendar API (`service.events().list(...).execute()`). It keeps a change log per calendar and
replays it through sync tokens, returning deleted events as `cancelled` stubs and all-day
events with an exclusive end date, like Google does. `tests/unit` also covers the contract
object with Google's token and userinfo endpoints faked, and the cognee adapter and sync;
`tests/integration` runs the example and the forget-on-delete path through the real cognee
pipeline with the LLM and embeddings mocked, which needs cognee installed.
