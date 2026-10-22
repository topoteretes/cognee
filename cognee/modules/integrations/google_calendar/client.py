"""Minimal Google Calendar REST client used by the core integration."""

from typing import Any

from cognee.modules.integrations.google.client import _authorized_get

CALENDAR_LIST_URL = "https://www.googleapis.com/calendar/v3/users/me/calendarList"
# 250 is the largest maxResults Google accepts for calendarList.list.
_PAGE_SIZE = "250"


async def list_calendars(access_token: str) -> list[dict[str, Any]]:
    """Every calendar the account can see, across all pages."""
    calendars: list[dict[str, Any]] = []
    params = {
        "maxResults": _PAGE_SIZE,
        # The source lists hidden calendars too, so the picker has to offer them.
        "showHidden": "true",
    }
    while True:
        page = await _authorized_get(
            "Calendar listing", CALENDAR_LIST_URL, access_token, dict(params)
        )
        calendars.extend(page.get("items", []) or [])
        page_token = page.get("nextPageToken")
        if not page_token:
            return calendars
        params["pageToken"] = page_token
