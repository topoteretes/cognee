# Adapted from topoteretes/cognee-community (Apache-2.0), PR #370, commit 5e78950.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""An in-memory Google Calendar API for tests.

Mimics the googleapiclient surface the connector uses, in the same style as
cognee's own Gmail tests: ``service.events().list(...).execute()`` and
``service.calendarList().list(...).execute()``.  Every change made through the
helpers is appended to a per-calendar change log, and sync tokens replay it,
collapsed to the latest state per event like Google does.  Deleted events
come back as ``status: cancelled`` stubs.
"""

import itertools
import sys
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace


class HttpError(Exception):
    """Mimics googleapiclient.errors.HttpError: the status sits on ``resp``.

    Like the real one, its text embeds the request URI, which for an events
    request contains the calendar id.
    """

    def __init__(self, status: int, uri: str = "", reason: str | None = None):
        super().__init__(f'<HttpError {status} when requesting {uri} returned "error">')
        self.resp = type("Resp", (), {"status": status})()
        # Like googleapiclient 2.x: a list of dicts, each with the error's reason.
        self.error_details = [{"reason": reason}] if reason else []


def install_google_errors(monkeypatch) -> None:
    """Make ``from googleapiclient.errors import HttpError`` yield the fake class.

    The connector imports ``HttpError`` when it makes a request, so the tests
    run without the Google libraries and with the same class the fake raises.
    """
    monkeypatch.setitem(sys.modules, "googleapiclient.errors", SimpleNamespace(HttpError=HttpError))


class _Request:
    def __init__(self, fn):
        self._fn = fn

    def execute(self, num_retries=0):
        return self._fn()


class _Events:
    def __init__(self, fake):
        self.fake = fake

    def list(self, **params):
        return _Request(lambda: self.fake._list_events(params))


class _CalendarList:
    def __init__(self, fake):
        self.fake = fake

    def list(self, pageToken=None, maxResults=None, showHidden=None):
        self.fake.calendar_list_params.append(
            {"pageToken": pageToken, "maxResults": maxResults, "showHidden": showHidden}
        )
        return _Request(lambda: self.fake._list_calendars())


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S+05:30")


class FakeCalendar:
    def __init__(self, page_size: int = 2):
        self.page_size = page_size
        self.calendars: dict[str, dict] = {}
        self.logs: dict[str, list[dict]] = {}
        self.expired_tokens: set[str] = set()
        self.fail_next: list[int] = []
        # Calendar id -> status that every events.list for it answers with.
        self.event_errors: dict[str, int] = {}
        # Calendar id -> reason Google gives with that status (a 403 is told apart by it).
        self.event_error_reasons: dict[str, str] = {}
        self.calendar_list_params: list[dict] = []
        self.list_calls: list[dict] = []
        # One entry per API request, in order: "calendarList" or "events".
        self.api_calls: list[str] = []
        self._pages: dict[str, tuple[list[dict], str]] = {}
        self._ids = itertools.count(1)
        self.add_calendar("lay@gmail.com", "Lay Patel", primary=True)

    # ------------------------------------------------------------------
    # Helpers that change the fake calendars (and record the change)
    # ------------------------------------------------------------------
    def add_calendar(self, calendar_id: str, name: str, primary: bool = False) -> str:
        self.calendars[calendar_id] = {"summary": name, "primary": primary, "events": {}}
        self.logs[calendar_id] = []
        return calendar_id

    def delete_calendar(self, calendar_id: str) -> None:
        del self.calendars[calendar_id]

    def put_event(self, calendar: str, summary: str, when: datetime, **fields) -> str:
        calendar = self._resolve(calendar)
        event_id = fields.pop("id", None) or f"evt{next(self._ids)}"
        event = {
            "id": event_id,
            "status": "confirmed",
            "summary": summary,
            "start": {"dateTime": _stamp(when), "timeZone": "Asia/Kolkata"},
            "end": {"dateTime": _stamp(when + timedelta(hours=1)), "timeZone": "Asia/Kolkata"},
            "organizer": {"email": "meera@example.com", "displayName": "Meera Joshi"},
            "htmlLink": f"https://calendar.google.com/event?eid={event_id}",
            "eventType": "default",
            **fields,
        }
        return self._save(calendar, event)

    def put_all_day(self, calendar: str, summary: str, day: str, days: int = 1, **fields) -> str:
        # Google's end.date is exclusive: a one-day event ends on the next day.
        end = (date.fromisoformat(day) + timedelta(days=days)).isoformat()
        return self.put_event(
            calendar,
            summary,
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            start={"date": day},
            end={"date": end},
            **fields,
        )

    def put_series(self, calendar: str, summary: str, start: datetime, rule: str, **fields):
        return self.put_event(calendar, summary, start, recurrence=[rule], **fields)

    def edit_event(self, calendar: str, event_id: str, **fields) -> None:
        calendar = self._resolve(calendar)
        event = {**self.calendars[calendar]["events"][event_id], **fields}
        self._save(calendar, event)

    def edit_occurrence(self, calendar: str, series_id: str, original: datetime, **fields):
        """A changed single occurrence: its own event with recurringEventId."""
        calendar = self._resolve(calendar)
        series = self.calendars[calendar]["events"][series_id]
        event = {
            **{k: v for k, v in series.items() if k != "recurrence"},
            "id": f"{series_id}_{original.strftime('%Y%m%dT%H%M%SZ')}",
            "recurringEventId": series_id,
            "originalStartTime": {"dateTime": _stamp(original)},
            **fields,
        }
        return self._save(calendar, event)

    def cancel_occurrence(self, calendar: str, series_id: str, original: datetime) -> str:
        """One occurrence of a series cancelled: a stub with recurringEventId."""
        calendar = self._resolve(calendar)
        occurrence_id = f"{series_id}_{original.strftime('%Y%m%dT%H%M%SZ')}"
        self.calendars[calendar]["events"].pop(occurrence_id, None)
        self.logs[calendar].append(
            {
                "id": occurrence_id,
                "status": "cancelled",
                "recurringEventId": series_id,
                "originalStartTime": {"dateTime": _stamp(original)},
            }
        )
        return occurrence_id

    def delete_event(self, calendar: str, event_id: str) -> None:
        calendar = self._resolve(calendar)
        del self.calendars[calendar]["events"][event_id]
        self.logs[calendar].append({"id": event_id, "status": "cancelled"})

    # ------------------------------------------------------------------
    # googleapiclient surface
    # ------------------------------------------------------------------
    def events(self):
        return _Events(self)

    def calendarList(self):
        return _CalendarList(self)

    def _list_calendars(self) -> dict:
        self.api_calls.append("calendarList")
        self._maybe_fail()
        return {
            "items": [
                {"id": cid, "summary": c["summary"], "primary": c["primary"]}
                for cid, c in self.calendars.items()
            ]
        }

    def _list_events(self, params: dict) -> dict:
        self.list_calls.append(dict(params))
        self.api_calls.append("events")
        self._maybe_fail()
        assert params["singleEvents"] is False, "series must stay one event"
        if "pageToken" in params:
            items, final = self._pages.pop(params["pageToken"])
            return self._page(items, final)
        calendar = self._resolve(params["calendarId"])
        uri = f"https://www.googleapis.com/calendar/v3/calendars/{params['calendarId']}/events"
        if calendar in self.event_errors:
            raise HttpError(
                self.event_errors[calendar], uri, self.event_error_reasons.get(calendar)
            )
        if calendar not in self.calendars:
            raise HttpError(404, uri)
        token = params.get("syncToken")
        if token is None:
            items = list(self.calendars[calendar]["events"].values())
        else:
            if token in self.expired_tokens:
                raise HttpError(410, uri)
            position = int(token.split("~")[1])
            latest: dict[str, dict] = {}
            for event in self.logs[calendar][position:]:
                latest[event["id"]] = event
            items = list(latest.values())
        return self._page(items, f"{calendar}~{len(self.logs[calendar])}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _save(self, calendar: str, event: dict) -> str:
        self.calendars[calendar]["events"][event["id"]] = event
        self.logs[calendar].append(dict(event))
        return event["id"]

    def _resolve(self, calendar: str) -> str:
        if calendar == "primary":
            return next(cid for cid, c in self.calendars.items() if c["primary"])
        return calendar

    def _maybe_fail(self) -> None:
        if self.fail_next:
            raise HttpError(self.fail_next.pop(0))

    def _page(self, items: list[dict], final_token: str) -> dict:
        page, rest = items[: self.page_size], items[self.page_size :]
        if rest:
            token = f"page{len(self._pages)}-{len(self.list_calls)}"
            self._pages[token] = (rest, final_token)
            return {"items": page, "nextPageToken": token}
        return {"items": page, "nextSyncToken": final_token}
