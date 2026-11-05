# Adapted from topoteretes/cognee-community (Apache-2.0), PR #370, commit 5e78950.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""Google Calendar connector for cognee - a ``dlt`` source that turns calendars into memory.

Sync one or more Google calendars into cognee, incrementally and with
forget-on-deletion.  The resource produced here is handed directly to
:func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import google_calendar_source

    await cognee.remember(
        google_calendar_source(calendar_ids=["primary"]),
        dataset_name="my_calendar",
        primary_key="id",
        write_disposition="merge",   # incremental upsert by event id
        max_rows_per_table=0,        # 0 = no row cap (calendars exceed the default 50)
    )

Design
------
* **Auth** - read-only ``calendar.readonly`` scope.  Settings come from
  ``GoogleCalendarSettings`` (``GOOGLE_CALENDAR_*``).  Three ways in: a refresh
  token plus client id and secret (headless), the installed-app browser flow
  with a cached 0600 token file (local), or a ready-made access token (hosts).
* **Incremental cursor** - one ``nextSyncToken`` per calendar, kept in dlt
  resource state.  It only arrives on the last page, so it is saved once the
  whole round has finished.  ``410 Gone`` means the token expired: that
  calendar gets a full listing, reconciled like a snapshot.
* **Recurring events** - listed with ``singleEvents=False``, so a series is
  one document with its recurrence rule, and a changed single occurrence is
  its own document.
* **Forget-on-delete** - deleted events come back with ``status: cancelled``.
  A cancelled event is forgotten; a cancelled single occurrence of a series
  only forgets that occurrence, never the series.  Deleting a whole series also
  forgets its changed occurrences.  Deletes are emitted with the ``_deleted``
  hard-delete marker and removed by cognee's ``orphan_cleanup``.
* **Errors never delete** - a calendar that fails keeps its previous state and is
  retried on the next run.  Only a calendar that was synced before and has left
  the account's calendar list is forgotten; a 404 for one that is still listed,
  or for one that was never synced (most likely a typo), is a failure.
* **One bad calendar does not block the others** - when Google answers with an
  HTTP error, the other calendars finish and save their state.  The run raises
  only when no calendar finished: ``dlt`` rolls back the state of a failed run,
  so raising after a partial success would throw the finished calendars away.
  Failures are counted in ``cognee_sync_stats`` (``failed``, ``failed_auth``,
  ``failed_rate_limit``, ``failed_calendar_sync``).  A 401, a 429 and a 403 with
  a rate-limit reason end the run, since the remaining calendars would fail the
  same way; any other 403 only fails its own calendar.  This holds for HTTP
  responses only.  A transport error, or an auth error raised by the Google
  client library itself (google-auth's ``RefreshError`` is raised before any
  HTTP call), ends the whole run: the other calendars are not tried and the
  stats do not count it.
* **Document path** - each event becomes a text document with the calendar
  name, organizer and attendees with their responses written into the content,
  so cognify links the same person across meetings.
* **Privacy** - logs and errors carry counts and HTTP statuses, never calendar
  ids (usually email addresses) or the request URLs googleapiclient puts in its
  own error text.

Limitations
-----------
* Attachments and conference details beyond the meeting link are not ingested.
* ``workingLocation`` events are skipped: they repeat every day and carry no
  meeting content.
"""

import html
import os
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from pydantic_settings import SettingsConfigDict

from cognee.modules.integrations.google import GoogleAuthError, GoogleSettings
from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

logger = get_logger("google_calendar_connector")

CALENDAR_READONLY_SCOPE = "https://www.googleapis.com/auth/calendar.readonly"
TOKEN_URI = "https://oauth2.googleapis.com/token"
DEFAULT_TOKEN_PATH = "~/.cognee_google_calendar_token.json"
DEFAULT_CALENDARS = ("primary",)
# Event types that carry no meeting content and would only add noise.
SKIPPED_EVENT_TYPES = {"workingLocation"}
# 250 is the largest maxResults Google accepts for events.list.
PAGE_SIZE = 250
# googleapiclient retries 429 and 5xx with exponential backoff this many times.
NUM_RETRIES = 5
ID_PREFIX = "gcal:"
# Google's reasons for a 403 that is a quota rejection. Same list as gmail.py.
_RATE_LIMIT_REASONS = ("rateLimitExceeded", "userRateLimitExceeded")
# Tags Google Calendar descriptions can carry. Anything else between angle brackets,
# like "<5 lakh>", is the user's text and stays.
_HTML_TAG = re.compile(
    r"</?(?:a|abbr|b|blockquote|br|code|div|em|font|h[1-6]|hr|i|img|li|ol|p|pre|s|span|strike"
    r"|strong|sub|sup|table|tbody|td|th|thead|tr|u|ul)\b[^>]*>",
    re.IGNORECASE,
)


class GoogleCalendarSettings(GoogleSettings):
    """cognee's Google OAuth settings, read from ``GOOGLE_CALENDAR_*``.

    Inherits ``client_id`` and ``client_secret`` from cognee's
    ``GoogleSettings`` and adds what a local, headless or hosted sync needs.
    """

    model_config = SettingsConfigDict(env_prefix="GOOGLE_CALENDAR_", extra="ignore")

    refresh_token: str = ""
    credentials_path: str = ""
    token_path: str = ""
    # A ready-made token from a host. It is never refreshed.
    access_token: str = ""
    # Comma-separated calendar ids (GOOGLE_CALENDAR_IDS); empty means the primary calendar.
    ids: str = ""


class GoogleCalendarError(RuntimeError):
    """A Google Calendar request or round failed.

    The message holds the HTTP status at most, never the calendar id or the
    request URL that googleapiclient puts in its own error text.
    """

    def __init__(self, message: str, status: int | None = None, rate_limited: bool = False):
        super().__init__(message)
        self.status = status
        # Set where the HttpError is still in hand: its text must not be kept, so the
        # reason Google gave for a 403 would otherwise be lost.
        self.rate_limited = rate_limited


@dataclass(frozen=True)
class _Round:
    """What one calendar's sync round produced."""

    state: dict | None  # None: the calendar is gone and its events were forgotten.
    rows: list[dict]
    deletes: list[str]
    skipped: int


# ---------------------------------------------------------------------------
# Auth / service construction
# ---------------------------------------------------------------------------
def _google_libraries() -> tuple[Any, Any, Any, Any]:
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            'The Google Calendar connector requires the "google-calendar" extra. '
            'Install it with: pip install "cognee[google-calendar]"'
        ) from exc
    return Request, Credentials, InstalledAppFlow, build


def build_calendar_service(credentials_path: str, token_path: str) -> Any:
    """Build a Calendar client through the OAuth installed-app flow.

    The first run opens a browser for consent and caches the token at
    ``token_path`` (written 0600); later runs reuse and refresh it.  Mirrors
    ``build_gmail_service``, with only the calendar.readonly scope.
    """
    request, credentials_cls, installed_app_flow, build = _google_libraries()
    scopes = [CALENDAR_READONLY_SCOPE]
    creds = None
    if os.path.exists(token_path):
        creds = credentials_cls.from_authorized_user_file(token_path, scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"Google OAuth client secrets not found at '{credentials_path}'. "
                    "Download an OAuth client (Desktop app) from the Google Cloud Console "
                    "and set GOOGLE_CALENDAR_CREDENTIALS_PATH."
                )
            flow = installed_app_flow.from_client_secrets_file(credentials_path, scopes)
            creds = flow.run_local_server(port=0)
        # The token file holds the long-lived refresh token: keep it private.
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())
        os.chmod(token_path, 0o600)  # tighten a pre-existing token file too

    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def build_calendar_service_from_access_token(access_token: str) -> Any:
    """Build a Calendar client from a short-lived token supplied by a host.

    Cloud hosts keep refresh tokens in their control plane and pass only an
    access token to the ingestion worker. This helper deliberately does not
    persist the token or attempt an interactive OAuth flow.
    """
    _, credentials_cls, _, build = _google_libraries()
    credentials = credentials_cls(token=access_token, scopes=[CALENDAR_READONLY_SCOPE])
    return build("calendar", "v3", credentials=credentials, cache_discovery=False)


def _build_service_from_refresh_token(settings: GoogleCalendarSettings) -> Any:
    """Headless: google-auth refreshes the access token itself when needed."""
    if not (settings.client_id and settings.client_secret):
        raise GoogleAuthError("calendar sign-in", "missing_client_credentials")
    _, credentials_cls, _, build = _google_libraries()
    credentials = credentials_cls(
        token=None,
        refresh_token=settings.refresh_token,
        token_uri=TOKEN_URI,
        client_id=settings.client_id,
        client_secret=settings.client_secret,
        scopes=[CALENDAR_READONLY_SCOPE],
    )
    return build("calendar", "v3", credentials=credentials, cache_discovery=False)


# ---------------------------------------------------------------------------
# Google API calls
# ---------------------------------------------------------------------------
def _http_error() -> Any:
    # Evaluated only when a request has failed, so a source built on a pre-made
    # service (tests, hosts) never needs the Google libraries.
    from googleapiclient.errors import HttpError

    return HttpError


def _execute(request: Any, check_active: Callable[[], None] | None) -> Any:
    """Run one API request after the host's authorization checkpoint.

    The checkpoint sits before the call and outside the ``try``, so a revoked
    connection stops the sync between API calls and is never turned into a
    calendar failure.  Google's HTTP errors become ``GoogleCalendarError``:
    their own text contains the request URL, and with it the calendar id.
    """
    if check_active is not None:
        check_active()
    try:
        return request.execute(num_retries=NUM_RETRIES)
    except _http_error() as error:
        status = error.resp.status
        raise GoogleCalendarError(
            f"Google Calendar request failed: HTTP {status}", status, _is_rate_limited(error)
        ) from None


def _is_rate_limited(error: Any) -> bool:
    """Whether Google rejected the call for quota: a 429, or a 403 with a rate-limit reason.

    Calendar mostly reports quota as 403 ``rateLimitExceeded``, so the status alone
    cannot tell it from lost access. A local copy of ``gmail._is_rate_limited``:
    importing it would pull Gmail's ``limits`` dependency into this extra. Like
    Gmail it falls back to the message for older googleapiclient versions, where
    the reason strings cannot appear in a request URL.
    """
    status = error.resp.status
    if status == 429:
        return True
    if status != 403:
        return False
    details = getattr(error, "error_details", None) or []
    reasons = {d.get("reason") for d in details if isinstance(d, dict)}
    if reasons.intersection(_RATE_LIMIT_REASONS):
        return True
    return any(reason in str(error) for reason in _RATE_LIMIT_REASONS)


def list_calendars(service: Any, check_active: Callable[[], None] | None = None) -> list[dict]:
    """Every calendar the account can see: id, name and whether it is primary."""
    calendars: list[dict] = []
    page_token = None
    while True:
        # showHidden: a hidden calendar still has to be listed, or "primary" would stop
        # resolving to its real id and a named calendar would lose its name, which changes
        # every row id or content hash. showDeleted stays off its default (on) on purpose:
        # a deleted calendar must drop out of the names so the 404 branch can forget it.
        request = service.calendarList().list(
            pageToken=page_token, maxResults=PAGE_SIZE, showHidden=True
        )
        response = _execute(request, check_active)
        for entry in response.get("items", []):
            calendars.append(
                {
                    "id": entry["id"],
                    "name": entry.get("summaryOverride") or entry.get("summary") or entry["id"],
                    "primary": bool(entry.get("primary")),
                }
            )
        page_token = response.get("nextPageToken")
        if not page_token:
            return calendars


def _list_events(
    service: Any,
    calendar_id: str,
    sync_token: str | None,
    check_active: Callable[[], None] | None,
) -> tuple[list[dict], str]:
    """Every page of one round. Returns (events, nextSyncToken).

    A sync-token request must keep the same parameters on every page and
    cannot use a time window, so both kinds of round use the same plain
    parameters.
    """
    events: list[dict] = []
    page_token = None
    while True:
        # No showDeleted or other filter: forgetting a deleted event relies on the
        # sync-token round returning it as status "cancelled", which Google's sync guide
        # documents. FakeCalendar always does, so the tests cannot prove it; confirm it
        # once on a live account.
        params: dict[str, Any] = {
            "calendarId": calendar_id,
            "maxResults": PAGE_SIZE,
            "singleEvents": False,
        }
        if sync_token:
            params["syncToken"] = sync_token
        if page_token:
            params["pageToken"] = page_token
        response = _execute(service.events().list(**params), check_active)
        events.extend(response.get("items", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            next_sync_token = response.get("nextSyncToken")
            if not next_sync_token:
                # Without it the next run could not follow this calendar's changes.
                raise GoogleCalendarError("Google Calendar returned no nextSyncToken.")
            return events, next_sync_token


# ---------------------------------------------------------------------------
# Sync state machine (pure given a service + state dict - unit-testable)
# ---------------------------------------------------------------------------
def _iter_rows(
    service: Any,
    calendar_ids: tuple[str, ...],
    state: dict,
    stats: dict[str, int],
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict]:
    """Yield one row per new or changed event, plus hard-delete tombstones.

    ``state`` is a plain dict standing in for dlt's resource state, so this is
    directly unit-testable.  Every API call happens before the first yield and
    the state is written after the last one.  A calendar that fails keeps its
    old state; the run raises only when none finished (see the module docstring).
    """
    stats.clear()
    stats.update(scanned=0, skipped=0, failed=0, deleted=0, calendars=0)

    try:
        listed = list_calendars(service, check_active)
    except GoogleCalendarError as error:
        _count_failure(stats, error)
        raise
    names = {calendar["id"]: calendar["name"] for calendar in listed}
    # "primary" and the real id name one calendar: one state entry, one set of rows.
    primary = next((c["id"] for c in listed if c["primary"]), "primary")
    wanted = tuple(dict.fromkeys(primary if c == "primary" else c for c in calendar_ids))

    old_calendars: dict[str, dict] = dict(state.get("calendars", {}))
    # Calendars dropped from the config: their events are out of scope now.
    deletes = [
        row_id
        for calendar_id, calendar_state in old_calendars.items()
        if calendar_id not in wanted
        for row_id in _row_ids(calendar_id, calendar_state.get("known", []))
    ]
    # A failed or skipped calendar keeps its old state and is retried next run.
    new_calendars = {c: s for c, s in old_calendars.items() if c in wanted}

    rounds: list[_Round] = []
    last_error: GoogleCalendarError | None = None
    for position, calendar_id in enumerate(wanted, start=1):
        try:
            result = _sync_calendar(
                service, calendar_id, names, old_calendars.get(calendar_id, {}), check_active
            )
        except GoogleCalendarError as error:
            last_error = error
            _count_failure(stats, error)
            logger.warning(
                "Google Calendar: calendar %d of %d failed (%s); its state is kept and it is "
                "retried on the next run.",
                position,
                len(wanted),
                error,
            )
            if _failure_reason(error) != "calendar_sync":
                break
            continue
        rounds.append(result)
        if result.state is None:
            new_calendars.pop(calendar_id, None)
        else:
            new_calendars[calendar_id] = result.state
            stats["calendars"] += 1

    if last_error is not None and not rounds:
        # Nothing finished, so there is no state to save and nothing is lost by raising.
        raise GoogleCalendarError(
            f"Google Calendar sync failed for all {len(wanted)} configured calendar(s): "
            f"{last_error}",
            last_error.status,
            last_error.rate_limited,
        ) from last_error

    for result in rounds:
        deletes.extend(result.deletes)
        stats["skipped"] += result.skipped
        stats["scanned"] += len(result.rows)
    stats["deleted"] = len(deletes)

    for row_id in deletes:
        yield {"id": row_id, "_deleted": True}
    for result in rounds:
        yield from result.rows

    state["calendars"] = new_calendars
    logger.info("Google Calendar: sync stats %s", stats)


def _failure_reason(error: GoogleCalendarError) -> str:
    """Why a calendar failed. Auth and rate limit end the run: the others would fail alike."""
    if error.status == 401:
        return "auth"
    if error.rate_limited:
        return "rate_limit"
    return "calendar_sync"


def _count_failure(stats: dict[str, int], error: GoogleCalendarError) -> None:
    reason = _failure_reason(error)
    stats["failed"] += 1
    stats[f"failed_{reason}"] = stats.get(f"failed_{reason}", 0) + 1


def _sync_calendar(
    service: Any,
    calendar_id: str,
    names: dict[str, str],
    calendar_state: dict,
    check_active: Callable[[], None] | None,
) -> _Round:
    """One sync round for one calendar.

    ``names`` is the account's calendar list (id to name); ``calendar_state``
    is empty for a calendar that was never synced.
    """
    known: set[str] = set(calendar_state.get("known", []))
    token = calendar_state.get("sync_token")

    items = None
    if token:
        try:
            items, token = _list_events(service, calendar_id, token, check_active)
        except GoogleCalendarError as error:
            # 410 is an expired token. A 404 may mean the calendar is gone,
            # which the full listing below confirms before anything is forgotten.
            if error.status not in (404, 410):
                raise
            logger.info("Google Calendar: a sync token is unusable; listing the calendar again.")

    snapshot = items is None
    if snapshot:
        try:
            items, token = _list_events(service, calendar_id, None, check_active)
        except GoogleCalendarError as error:
            # A 404 only means "deleted or no longer shared" for a calendar we synced
            # before and that has left the account's list. Still listed, or never
            # synced (a typo in the config), it is a failure, and failures forget nothing.
            if error.status != 404 or calendar_id in names or not calendar_state:
                raise
            logger.warning(
                "Google Calendar: a synced calendar left the account's calendar list; "
                "forgetting its %d event(s).",
                len(known),
            )
            return _Round(None, [], _row_ids(calendar_id, known), 0)

    present: set[str] = set()
    changed: dict[str, dict] = {}
    gone: set[str] = set()
    skipped = 0
    for event in items:
        event_id = event["id"]
        if event.get("status") == "cancelled":
            if not event.get("recurringEventId"):
                # A deleted event or a deleted whole series: forget its changed
                # single occurrences too, whose ids start with the series id.
                gone.add(event_id)
                gone |= {k for k in known if k.startswith(event_id + "_")}
            else:
                # One cancelled occurrence of a series that we stored because it
                # had been changed. The series itself stays.
                gone.add(event_id)
            continue
        if event.get("eventType") in SKIPPED_EVENT_TYPES:
            skipped += 1
            continue
        present.add(event_id)
        changed[event_id] = event

    if snapshot:
        # A full listing has every live event, so anything known but missing
        # was deleted while the token was not usable.
        gone |= known - present
    # An event that was created and deleted between two syncs never reached
    # memory, so there is nothing to forget.
    gone &= known
    gone -= present

    known -= gone
    known |= changed.keys()
    name = names.get(calendar_id, calendar_id)
    rows = [_event_row(calendar_id, name, event) for event in changed.values()]
    new_state = {"sync_token": token, "known": sorted(known)}
    return _Round(new_state, rows, _row_ids(calendar_id, gone), skipped)


def _row_ids(calendar_id: str, event_ids: Iterable[str]) -> list[str]:
    return [_row_id(calendar_id, event_id) for event_id in sorted(event_ids)]


def _row_id(calendar_id: str, event_id: str) -> str:
    # The same invitation can sit in several calendars under one event id.
    return f"{ID_PREFIX}{calendar_id}:{event_id}"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _event_row(calendar_id: str, calendar_name: str, event: dict) -> dict:
    title = event.get("summary") or "(no title)"
    lines = [f"Calendar event: {title}", f"Calendar: {calendar_name}"]
    start, end = event.get("start") or {}, event.get("end") or {}
    if start.get("date"):
        # A confirmed event always has an end, so a missing one is a bug to see, not hide.
        lines.append(f"When: {_all_day(start['date'], end['date'])}")
    else:
        lines.append(f"When: {_when(start)} to {_when(end)}")
    if event.get("location"):
        lines.append(f"Location: {event['location']}")
    if event.get("organizer"):
        lines.append(f"Organizer: {_person(event['organizer'])}")
    # Google's order is not stable between fetches, and the content hash decides
    # whether the event is re-cognified.
    attendees = sorted(
        event.get("attendees") or [], key=lambda a: a.get("email") or a.get("displayName") or ""
    )
    if attendees:
        line = "Attendees: " + ", ".join(
            f"{_person(a)} ({a.get('responseStatus') or 'needsAction'})" for a in attendees
        )
        if event.get("attendeesOmitted"):
            line += ", and more not listed by Google"
        lines.append(line)
    if event.get("recurrence"):
        lines.append("Repeats: " + "; ".join(event["recurrence"]))
    if event.get("recurringEventId"):
        original = _when(event.get("originalStartTime") or {})
        lines.append(f"Changed occurrence of series {event['recurringEventId']}, was {original}")
    if event.get("status") == "tentative":
        lines.append("Status: tentative")
    if event.get("eventType") and event["eventType"] != "default":
        lines.append(f"Type: {event['eventType']}")
    if event.get("hangoutLink"):
        lines.append(f"Meeting link: {event['hangoutLink']}")
    description = _plain_text(event.get("description") or "")
    content = "\n".join(lines) + (f"\n\n{description}" if description else "")
    return {
        "id": _row_id(calendar_id, event["id"]),
        "title": title,
        "content": content,
        "url": event.get("htmlLink"),
        "_deleted": False,
    }


def _person(person: dict) -> str:
    name, email = person.get("displayName"), person.get("email")
    if name and email and name != email:
        return f"{name} <{email}>"
    return email or name or "unknown"


def _when(value: dict) -> str:
    moment = value.get("dateTime") or value.get("date") or "unknown"
    zone = value.get("timeZone")
    return f"{moment} ({zone})" if zone else moment


def _all_day(first_day: str, exclusive_end: str) -> str:
    """The span of an all-day event, with its last day inclusive.

    Google's ``end.date`` is the day after the event: a one-day event on
    2026-11-08 ends on 2026-11-09, so a multi-day one reads "2026-11-08 to
    2026-11-10" for three days, and a one-day one needs no "to".
    """
    start = date.fromisoformat(first_day)
    last = date.fromisoformat(exclusive_end) - timedelta(days=1)
    if last == start:
        return f"all day, {start}"
    return f"all day, {start} to {last}"


def _plain_text(description: str) -> str:
    """Google allows simple HTML in descriptions; keep the text only."""
    text = re.sub(r"<br\s*/?>|</p>|</div>|</li>", "\n", description, flags=re.IGNORECASE)
    return html.unescape(_HTML_TAG.sub("", text)).strip()


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def google_calendar_source(
    calendar_ids: list[str] | str | None = None,
    *,
    resource_name: str = "google_calendar_events",
    check_active: Callable[[], None] | None = None,
    access_token: str | None = None,
    settings: GoogleCalendarSettings | None = None,
    service: Any = None,
):
    """Return a ``dlt`` resource yielding one row per new or changed calendar event.

    Hand the result to ``cognee.remember(...)`` with ``write_disposition="merge"``
    and ``primary_key="id"``.

    Args:
        calendar_ids: Calendars to sync, like ``["primary", "team@group.calendar.google.com"]``.
            Defaults to the primary calendar; ``GOOGLE_CALENDAR_IDS`` takes a
            comma-separated list.  ``list_calendars`` shows the ids.  An empty list
            is refused, because it would forget every synced event.
        resource_name: Stable dlt resource name. Give each account its own name
            when several sync into one dataset, so their state stays separate.
        check_active: Optional host authorization checkpoint during extraction.
        access_token: A ready-made token from a host. It is never refreshed.
        settings: Google OAuth settings; read from ``GOOGLE_CALENDAR_*`` when
            omitted.  A refresh token with client id and secret is used first,
            then the installed-app flow with ``credentials_path`` and ``token_path``.
        service: A pre-built Calendar client, mainly for tests.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError(
            "The Google Calendar connector requires the google-calendar extra: "
            'pip install "cognee[google-calendar]".'
        ) from exc

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 1:
        raise RuntimeError(
            "Google Calendar sync requires a Cognee build with table-scoped DLT document "
            "cleanup. Upgrade Cognee before syncing to avoid deleting another source's data."
        )

    settings = settings or GoogleCalendarSettings()
    wanted_ids = _normalize_calendar_ids(calendar_ids, settings)
    access_token = access_token or settings.access_token
    if service is None and not (
        access_token or settings.refresh_token or settings.credentials_path
    ):
        # Fail when the source is built, not halfway through a pipeline run.
        raise ValueError(
            "Google Calendar credentials are missing. Set GOOGLE_CALENDAR_REFRESH_TOKEN with "
            "GOOGLE_CALENDAR_CLIENT_ID and GOOGLE_CALENDAR_CLIENT_SECRET, or "
            "GOOGLE_CALENDAR_CREDENTIALS_PATH and GOOGLE_CALENDAR_TOKEN_PATH, or "
            "GOOGLE_CALENDAR_ACCESS_TOKEN for a quick try."
        )

    def make_service() -> Any:
        if service is not None:
            return service
        if access_token:
            return build_calendar_service_from_access_token(access_token)
        if settings.refresh_token:
            return _build_service_from_refresh_token(settings)
        return build_calendar_service(
            os.path.expanduser(settings.credentials_path),
            os.path.expanduser(settings.token_path or DEFAULT_TOKEN_PATH),
        )

    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        write_disposition="merge",
        primary_key="id",
        # `_deleted` is a boolean hard-delete marker: rows where it is True are
        # removed from the dlt destination on merge, which propagates the
        # deletion through cognee's orphan_cleanup.
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def google_calendar_events():
        rows = _iter_rows(
            make_service(), wanted_ids, dlt.current.resource_state(), stats, check_active
        )
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = google_calendar_events()
    # Opt into the document ingestion path: each event becomes a text document
    # that flows through normal cognify (LLM graph extraction).
    setattr(resource, DOCUMENT_SOURCE_ATTR, "google_calendar")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    # Host-readable diagnostics contain counts only, never titles or content.
    resource.cognee_sync_stats = stats
    return resource


def _normalize_calendar_ids(
    calendar_ids: list[str] | str | None, settings: GoogleCalendarSettings
) -> tuple[str, ...]:
    if calendar_ids is None:
        candidates = settings.ids.split(",") if settings.ids.strip() else DEFAULT_CALENDARS
    elif isinstance(calendar_ids, str):
        candidates = [calendar_ids]
    else:
        candidates = calendar_ids
    cleaned = tuple(dict.fromkeys(c.strip() for c in candidates if c.strip()))
    if not cleaned:
        # An empty list would forget every synced event, so refuse it.
        raise ValueError('calendar_ids is empty. Pass calendars like ["primary"].')
    return cleaned
