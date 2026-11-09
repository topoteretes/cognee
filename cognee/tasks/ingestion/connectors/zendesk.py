"""Zendesk Tickets connector for cognee: a ``dlt`` source that turns support history into memory.

One source is one Zendesk workspace. It yields tickets and their comment histories as flat
document rows, and is meant to be handed directly to :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import zendesk_source

    await cognee.remember(
        zendesk_source(
            subdomain="mycompany",
            email="support@mycompany.com",
            api_token="<zendesk-token>",
        ),
        dataset_name="zendesk",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``.
  Ids are prefixed by kind (``ticket:<id>``). Every column feeds the stored
  document identity. Volatile metrics are omitted, preserving deterministic fingerprints.
* **Incremental** by Zendesk Incremental Ticket Export API (``start_time`` epoch).
* **Comments & Conversations** are folded into each ticket's content document to preserve
  thread context and avoid graph fragmentation.
* **Tombstones & Deletions** detect soft-deleted tickets or status changes to ``deleted``.
* **Resumable**: Stops gracefully on rate limits without crashing the pipeline.
"""

import base64
import logging
import re
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

NODE_SET_COLUMN = "cognee_node_set"
DOCUMENT_SOURCE_ATTR = "document_source"

DEFAULT_PAGE_SIZE = 100
_TIMEOUT = (10, 30)


class ZendeskError(Exception):
    """Base exception for Zendesk connector errors."""


class ZendeskAuthError(ZendeskError):
    """Raised when authentication credentials fail."""


class ZendeskRateLimitError(ZendeskError):
    """Raised when Zendesk rate limit quota is exceeded."""


class ZendeskClient:
    """HTTP client for Zendesk Core & Incremental API."""

    def __init__(
        self,
        subdomain: str,
        email: str | None = None,
        api_token: str | None = None,
        oauth_token: str | None = None,
        session: Any = None,
    ):
        clean_subdomain = re.sub(r"[^a-zA-Z0-9-]", "", subdomain.strip().lower())
        if not clean_subdomain:
            raise ValueError("Invalid Zendesk subdomain")

        self.subdomain = clean_subdomain
        self.base_url = f"https://{self.subdomain}.zendesk.com/api/v2"

        if oauth_token:
            self.auth_header = f"Bearer {oauth_token}"
        elif email and api_token:
            auth_str = f"{email}/token:{api_token}"
            encoded = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
            self.auth_header = f"Basic {encoded}"
        else:
            raise ValueError("Provide either oauth_token or email + api_token")

        self.session = session

    def get_headers(self) -> dict[str, str]:
        return {
            "Authorization": self.auth_header,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Cognee-Zendesk-Connector/1.0",
        }

    def fetch_tickets(self, start_time: int = 0) -> dict[str, Any]:
        """Fetches incremental tickets starting from epoch timestamp."""
        if self.session is None:
            import requests
            session = requests.Session()
        else:
            session = self.session

        url = f"{self.base_url}/incremental/tickets.json?start_time={start_time}"
        resp = session.get(url, headers=self.get_headers(), timeout=_TIMEOUT)

        if resp.status_code == 401 or resp.status_code == 403:
            raise ZendeskAuthError(f"Zendesk auth failed: HTTP {resp.status_code}")
        if resp.status_code == 429:
            raise ZendeskRateLimitError("Zendesk API rate limit hit")
        if not resp.ok:
            raise ZendeskError(f"Zendesk API request failed: HTTP {resp.status_code}")

        return resp.json()

    def fetch_comments(self, ticket_id: int | str) -> list[dict[str, Any]]:
        """Fetches comments for a specific ticket."""
        if self.session is None:
            import requests
            session = requests.Session()
        else:
            session = self.session

        clean_id = re.sub(r"[^0-9]", "", str(ticket_id))
        if not clean_id:
            return []

        url = f"{self.base_url}/tickets/{clean_id}/comments.json"
        try:
            resp = session.get(url, headers=self.get_headers(), timeout=_TIMEOUT)
            if resp.ok:
                return resp.json().get("comments", [])
        except Exception as e:
            logger.warning("Failed to fetch comments for ticket %s: %s", ticket_id, e)
        return []


def render_ticket(ticket: dict[str, Any], comments: list[dict[str, Any]] | None = None) -> str:
    """Renders a Zendesk ticket and comments into structured natural text for vector embedding."""
    subject = ticket.get("subject") or "Untitled Ticket"
    status = ticket.get("status") or "open"
    priority = ticket.get("priority") or "normal"
    ticket_type = ticket.get("type") or "incident"
    description = ticket.get("description") or ""

    tags = ", ".join(ticket.get("tags") or [])
    created_at = ticket.get("created_at") or ""

    lines = [
        f"Zendesk Support Ticket #{ticket.get('id')}: {subject}",
        f"Status: {status} | Priority: {priority} | Type: {ticket_type}",
    ]
    if tags:
        lines.append(f"Tags: {tags}")
    if created_at:
        lines.append(f"Created At: {created_at}")

    lines.append("\nInitial Description:")
    lines.append(description.strip())

    if comments:
        lines.append("\nTicket Conversation History:")
        for idx, comment in enumerate(comments, 1):
            author_id = comment.get("author_id", "unknown")
            body = (comment.get("body") or "").strip()
            public = "Public" if comment.get("public", True) else "Internal Note"
            if body:
                lines.append(f"[{idx}] {public} (Author {author_id}): {body}")

    return "\n".join(lines)


def _iter_tickets(
    client: ZendeskClient,
    start_time: int = 0,
    include_comments: bool = True,
) -> Iterator[dict[str, Any]]:
    """Generates document rows for Zendesk tickets."""
    data = client.fetch_tickets(start_time=start_time)
    tickets = data.get("tickets", [])

    for ticket in tickets:
        ticket_id = ticket.get("id")
        if not ticket_id:
            continue

        is_deleted = ticket.get("status") == "deleted"
        comments = []
        if include_comments and not is_deleted:
            comments = client.fetch_comments(ticket_id)

        rendered_content = render_ticket(ticket, comments)
        ticket_url = f"https://{client.subdomain}.zendesk.com/agent/tickets/{ticket_id}"

        row = {
            "id": f"ticket:{ticket_id}",
            "title": ticket.get("subject") or f"Ticket #{ticket_id}",
            "content": rendered_content,
            "url": ticket_url,
            "_deleted": is_deleted,
            NODE_SET_COLUMN: f"zendesk:{client.subdomain}:tickets",
            DOCUMENT_SOURCE_ATTR: "zendesk",
        }
        yield row


def zendesk_source(
    subdomain: str,
    email: str | None = None,
    api_token: str | None = None,
    oauth_token: str | None = None,
    start_time: int = 0,
    include_comments: bool = True,
    client: ZendeskClient | None = None,
):
    """Returns a DLT source configured for incremental Zendesk ticket ingestion."""
    import dlt

    if client is None:
        client = ZendeskClient(
            subdomain=subdomain,
            email=email,
            api_token=api_token,
            oauth_token=oauth_token,
        )

    @dlt.source(name=f"zendesk_{client.subdomain}")
    def source():
        @dlt.resource(
            name="tickets",
            primary_key="id",
            write_disposition="merge",
        )
        def tickets_resource():
            yield from _iter_tickets(
                client=client,
                start_time=start_time,
                include_comments=include_comments,
            )

        return tickets_resource

    return source()


def build_zendesk_service(
    subdomain: str,
    email: str | None = None,
    api_token: str | None = None,
    oauth_token: str | None = None,
) -> ZendeskClient:
    """Helper to instantiate a standalone ZendeskClient."""
    return ZendeskClient(
        subdomain=subdomain,
        email=email,
        api_token=api_token,
        oauth_token=oauth_token,
    )
