"""What a Google Calendar connection can enumerate, given an access token.

Split from cognee.modules.integrations.google_calendar.adapter (cognee dev, 2026-10-26):
:func:`list_resources` is the body of ``GoogleCalendarIntegration.list_resources`` there,
taken out of the host adapter so the contract object (``connector.py``) can offer the same
picker without a credential row. The entries keep the shape cognee's generic picker reads
(``id``, ``name``, ``description``, ``attributes``); the contract object wraps them in
``ConnectorResource``. Pure: nothing here imports cognee.
"""

from typing import Any

from cognee_connector_google_calendar import client as calendar_client


async def calendar_ids(access_token: str) -> list[str]:
    """Every calendar the account can see, by id, in listing order."""
    ids: list[str] = []
    for calendar in await calendar_client.list_calendars(access_token):
        calendar_id = str(calendar.get("id") or "")
        if calendar_id and calendar_id not in ids:
            ids.append(calendar_id)
    return ids


async def list_resources(access_token: str) -> list[dict[str, Any]]:
    """Every calendar the account can see, hidden ones included, for a picker."""
    calendars = await calendar_client.list_calendars(access_token)
    return [
        {
            "id": str(calendar["id"]),
            "name": str(
                calendar.get("summaryOverride") or calendar.get("summary") or calendar["id"]
            ),
            "description": calendar.get("description"),
            "attributes": {
                "primary": bool(calendar.get("primary")),
                "access_role": calendar.get("accessRole"),
            },
        }
        for calendar in calendars
        if calendar.get("id")
    ]
