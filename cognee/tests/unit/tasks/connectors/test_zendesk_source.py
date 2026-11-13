"""Unit tests for the Zendesk Support connector (zendesk_source)."""

from __future__ import annotations

from typing import Any
import pytest

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.connectors import build_zendesk_service, zendesk_source
from cognee.tasks.ingestion.connectors.zendesk import (
    ZendeskAPIError,
    ZendeskAuthError,
    ZendeskClient,
    ZendeskRateLimitedError,
    _normalize_subdomain,
    _render_ticket_row,
    _iter_rows,
)
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
)


class FakeZendeskClient(ZendeskClient):
    """In-memory fake Zendesk client for deterministic hermetic tests."""

    def __init__(self, subdomain: str = "acme"):
        super().__init__(subdomain=subdomain, access_token="test-token")
        self.tickets: list[dict[str, Any]] = []
        self.comments_by_ticket: dict[str, list[dict[str, Any]]] = {}
        self.queries: list[str] = []
        self.raise_on_call: int | None = None
        self.exception_to_raise: Exception | None = None
        self._call_count = 0

    def search_tickets(
        self,
        query: str,
        *,
        page_after: str | None = None,
        page_size: int = 100,
    ) -> dict[str, Any]:
        self._call_count += 1
        self.queries.append(query)
        if self.raise_on_call is not None and self._call_count >= self.raise_on_call:
            assert self.exception_to_raise is not None
            raise self.exception_to_raise

        return {
            "results": self.tickets,
            "meta": {"has_more": False, "after_cursor": None},
        }

    def list_tickets(
        self,
        *,
        start_time: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        self._call_count += 1
        if self.raise_on_call is not None and self._call_count >= self.raise_on_call:
            assert self.exception_to_raise is not None
            raise self.exception_to_raise

        return {
            "tickets": self.tickets,
            "after_cursor": "cur_next_999",
            "end_of_stream": True,
        }

    def get_ticket_comments(self, ticket_id: int | str) -> list[dict[str, Any]]:
        self._call_count += 1
        if self.raise_on_call is not None and self._call_count >= self.raise_on_call:
            assert self.exception_to_raise is not None
            raise self.exception_to_raise
        return self.comments_by_ticket.get(str(ticket_id), [])


def _sample_ticket(
    ticket_id: str = "101",
    subject: str = "Cannot connect database",
    description: str = "Database connection timed out on prod cluster.",
) -> dict[str, Any]:
    return {
        "id": ticket_id,
        "subject": subject,
        "description": description,
        "status": "open",
        "priority": "urgent",
        "type": "incident",
        "tags": ["db", "production"],
    }


def _empty_stats() -> dict[str, int]:
    return {
        "synced": 0,
        "deleted": 0,
        "failed": 0,
        "failed_rate_limit": 0,
        "failed_budget": 0,
        "failed_auth": 0,
    }


def test_normalize_subdomain():
    sub, url = _normalize_subdomain("acme")
    assert sub == "acme"
    assert url == "https://acme.zendesk.com"

    sub, url = _normalize_subdomain("https://my-company.zendesk.com/")
    assert sub == "my-company"
    assert url == "https://my-company.zendesk.com"

    with pytest.raises(ValueError, match="subdomain must not be empty"):
        _normalize_subdomain("")

    with pytest.raises(ValueError, match="Invalid Zendesk subdomain"):
        _normalize_subdomain("bad@domain!")


def test_render_ticket_row():
    ticket = _sample_ticket()
    comments = [
        {"author_id": 999, "body": "Looking into it now.", "public": True},
        {"author_id": 888, "body": "Internal note: check VPC peering.", "public": False},
    ]
    row = _render_ticket_row("https://acme.zendesk.com", "acme", ticket, comments)
    assert row["id"] == "zendesk:ticket:101"
    assert row["title"] == "Zendesk #101: Cannot connect database"
    assert row["url"] == "https://acme.zendesk.com/agent/tickets/101"
    assert row["_deleted"] is False
    assert row[NODE_SET_COLUMN] == ["zendesk:acme"]
    assert "Database connection timed out" in row["content"]
    assert "Looking into it now." in row["content"]
    assert "Internal note: check VPC peering." in row["content"]
    assert "Tags: db, production" in row["content"]


def test_iter_rows_search():
    client = FakeZendeskClient(subdomain="acme")
    client.tickets = [_sample_ticket("101"), _sample_ticket("102", subject="SSO issue")]
    client.comments_by_ticket["101"] = [{"author_id": 1, "body": "Fixed."}]

    state: dict[str, Any] = {}
    stats = _empty_stats()

    rows = list(_iter_rows(client, query="status:open", state=state, stats=stats))
    assert len(rows) == 2
    assert stats["synced"] == 2
    assert stats["failed"] == 0
    assert rows[0]["id"] == "zendesk:ticket:101"
    assert rows[1]["id"] == "zendesk:ticket:102"
    assert state["known_ids"] == ["101", "102"]


def test_iter_rows_incremental_list():
    client = FakeZendeskClient(subdomain="acme")
    client.tickets = [_sample_ticket("201")]

    state: dict[str, Any] = {}
    stats = _empty_stats()

    rows = list(_iter_rows(client, query=None, state=state, stats=stats))
    assert len(rows) == 1
    assert stats["synced"] == 1
    assert state["cursor"] == "cur_next_999"
    assert state["known_ids"] == ["201"]


def test_rate_limit_handling():
    client = FakeZendeskClient(subdomain="acme")
    client.raise_on_call = 1
    client.exception_to_raise = ZendeskRateLimitedError(retry_after=60)

    state: dict[str, Any] = {}
    stats = _empty_stats()

    rows = list(_iter_rows(client, query=None, state=state, stats=stats))
    assert len(rows) == 0
    assert stats["failed"] == 1
    assert stats["failed_rate_limit"] == 1


def test_auth_error_aborts_when_no_prior_progress():
    client = FakeZendeskClient(subdomain="acme")
    client.raise_on_call = 1
    client.exception_to_raise = ZendeskAuthError("401 Unauthorized")

    state: dict[str, Any] = {}
    stats = _empty_stats()

    with pytest.raises(ZendeskAuthError):
        list(_iter_rows(client, query=None, state=state, stats=stats))
