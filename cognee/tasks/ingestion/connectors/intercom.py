"""Intercom connector for cognee, a ``dlt`` source that turns contacts into memory.

Sync a set of explicit Intercom resources into cognee incrementally.

    import cognee
    from cognee.tasks.ingestion.connectors import intercom_source

    await cognee.remember(
        intercom_source(),
        dataset_name="my_intercom_data",
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

INTERCOM_API_URL = "https://api.intercom.io"
INTERCOM_VERSION = "2.11"


def _get_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Intercom-Version": INTERCOM_VERSION,
    }


def intercom_source(token: str | None = None, resource_name: str = "intercom_contacts"):
    """Create a dlt source yielding Intercom contacts.

    Args:
        token: Intercom Access Token. Falls back to INTERCOM_ACCESS_TOKEN.
        resource_name: The dlt resource / table name to yield rows into.
    """
    try:
        import dlt
    except ImportError as e:
        raise ImportError("The Intercom connector requires dlt: pip install dlt") from e

    token = token or os.environ.get("INTERCOM_ACCESS_TOKEN")
    if not token:
        raise ValueError(
            "Intercom access token must be provided or set in INTERCOM_ACCESS_TOKEN environment variable"
        )

    @dlt.resource(name=resource_name, write_disposition="merge", primary_key="id")
    def intercom_contacts(
        last_updated_time: dlt.sources.incremental = dlt.sources.incremental("updated_at"),  # noqa: B008
    ) -> Iterator[dict]:
        headers = _get_headers(token)
        url = f"{INTERCOM_API_URL}/contacts"

        while url:
            resp = httpx.get(url, headers=headers, timeout=30)
            resp.raise_for_status()

            data = resp.json()
            contacts = data.get("data", [])

            for contact in contacts:
                contact_id = contact.get("id")
                name = contact.get("name")
                email = contact.get("email")
                role = contact.get("role")
                updated_at = contact.get("updated_at")

                content_lines = [
                    f"Intercom Contact: {name or 'Unknown'}",
                    f"Email: {email}",
                    f"Role: {role}",
                ]

                yield {
                    "id": contact_id,
                    "title": f"Intercom Contact: {name or email or contact_id}",
                    "content": "\n".join(content_lines),
                    "updated_at": updated_at,
                    "email": email,
                    "role": role,
                }

            pages = data.get("pages", {})
            url = pages.get("next")

    @dlt.source(name="intercom")
    def _intercom():
        return intercom_contacts

    source = _intercom()
    setattr(source, dlt_utils.DOCUMENT_SOURCE_ATTR, "intercom")
    return source
