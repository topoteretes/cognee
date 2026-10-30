"""Calendly connector for cognee, a ``dlt`` source that turns scheduled events into memory.

Sync a set of explicit Calendly resources into cognee incrementally.

    import cognee
    from cognee.tasks.ingestion.connectors import calendly_source

    await cognee.remember(
        calendly_source(),
        dataset_name="my_calendly_data",
        primary_key="id",
        write_disposition="merge",
    )
"""

import logging
import os
from collections.abc import Iterator

import httpx

from cognee.tasks.ingestion import dlt_utils

logger = logging.getLogger(__name__)

CALENDLY_API_URL = "https://api.calendly.com"


def _get_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def calendly_source(token: str | None = None, resource_name: str = "calendly_events"):
    """Create a dlt source yielding Calendly scheduled events.

    Args:
        token: Calendly Personal Access Token. Falls back to CALENDLY_API_KEY.
        resource_name: The dlt resource / table name to yield rows into.
    """
    try:
        import dlt
    except ImportError as e:
        raise ImportError("The Calendly connector requires dlt: pip install dlt") from e

    token = token or os.environ.get("CALENDLY_API_KEY")
    if not token:
        raise ValueError(
            "Calendly API key must be provided or set in CALENDLY_API_KEY environment variable"
        )

    @dlt.resource(name=resource_name, write_disposition="merge", primary_key="id")
    def calendly_events(
        last_updated_time: dlt.sources.incremental = dlt.sources.incremental("updated_at"),  # noqa: B008
    ) -> Iterator[dict]:
        headers = _get_headers(token)

        # Get the current user to fetch their events
        user_resp = httpx.get(f"{CALENDLY_API_URL}/users/me", headers=headers, timeout=30)
        user_resp.raise_for_status()
        user_uri = user_resp.json()["resource"]["uri"]

        url = f"{CALENDLY_API_URL}/scheduled_events"
        params = {"user": user_uri}

        while url:
            resp = httpx.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()

            data = resp.json()
            events = data.get("collection", [])

            for event in events:
                event_id = event.get("uri").split("/")[-1]
                name = event.get("name")
                status = event.get("status")
                start_time = event.get("start_time")
                end_time = event.get("end_time")
                updated_at = event.get("updated_at")

                content_lines = [
                    f"Calendly Event: {name}",
                    f"Status: {status}",
                    f"Start: {start_time}",
                    f"End: {end_time}",
                ]

                yield {
                    "id": event_id,
                    "url": event.get("uri"),
                    "title": f"Calendly Event: {name}",
                    "content": "\n".join(content_lines),
                    "updated_at": updated_at,
                    "status": status,
                    "start_time": start_time,
                    "end_time": end_time,
                }

            # Pagination
            pagination = data.get("pagination", {})
            url = pagination.get("next_page")
            params = {}  # The next_page URL already contains the required parameters

    @dlt.source(name="calendly")
    def _calendly():
        return calendly_events

    source = _calendly()
    setattr(source, dlt_utils.DOCUMENT_SOURCE_ATTR, "calendly")
    return source
