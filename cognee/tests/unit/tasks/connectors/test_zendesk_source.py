"""Unit tests for Cognee Zendesk Tickets connector."""

import json
from unittest.mock import MagicMock

import pytest

try:
    from cognee.tasks.ingestion.connectors.zendesk import (
        ZendeskAuthError,
        ZendeskClient,
        ZendeskError,
        ZendeskRateLimitError,
        _iter_tickets,
        build_zendesk_service,
        render_ticket,
    )
except (ImportError, ModuleNotFoundError):
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../tasks/ingestion/connectors")))
    from zendesk import (
        ZendeskAuthError,
        ZendeskClient,
        ZendeskError,
        ZendeskRateLimitError,
        _iter_tickets,
        build_zendesk_service,
        render_ticket,
    )


class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._json_data


def test_zendesk_client_auth_basic():
    client = build_zendesk_service(
        subdomain="testcompany",
        email="dev@testcompany.com",
        api_token="secrettoken123",
    )
    headers = client.get_headers()
    assert "Authorization" in headers
    assert headers["Authorization"].startswith("Basic ")
    assert client.base_url == "https://testcompany.zendesk.com/api/v2"


def test_zendesk_client_auth_oauth():
    client = build_zendesk_service(
        subdomain="testcompany",
        oauth_token="oauth_secret_abc_456",
    )
    headers = client.get_headers()
    assert headers["Authorization"] == "Bearer oauth_secret_abc_456"


def test_zendesk_client_missing_auth():
    with pytest.raises(ValueError, match="Provide either oauth_token or email"):
        ZendeskClient(subdomain="testcompany")


def test_render_ticket_with_comments():
    ticket = {
        "id": 101,
        "subject": "Login authentication loop on Safari",
        "status": "open",
        "priority": "high",
        "type": "incident",
        "tags": ["auth", "safari", "frontend"],
        "created_at": "2026-11-09T08:00:00Z",
        "description": "User experiences endless redirect loop after entering OTP.",
    }
    comments = [
        {"author_id": 991, "body": "Checked session cookie; SameSite attribute is missing.", "public": False},
        {"author_id": 992, "body": "Fix deployed to staging, awaiting verification.", "public": True},
    ]

    rendered = render_ticket(ticket, comments)
    assert "Zendesk Support Ticket #101: Login authentication loop on Safari" in rendered
    assert "Status: open | Priority: high | Type: incident" in rendered
    assert "Tags: auth, safari, frontend" in rendered
    assert "Internal Note (Author 991): Checked session cookie" in rendered
    assert "Public (Author 992): Fix deployed to staging" in rendered


def test_iter_tickets_emission():
    mock_session = MagicMock()

    # Mock tickets response
    tickets_payload = {
        "tickets": [
            {
                "id": 101,
                "subject": "Payment webhook failure",
                "status": "solved",
                "priority": "urgent",
                "description": "Stripe webhook dropped event",
                "tags": ["billing"],
            },
            {
                "id": 102,
                "subject": "Spam ticket deleted",
                "status": "deleted",
                "description": "Spam entry",
            },
        ],
        "count": 2,
    }

    # Mock comments response
    comments_payload = {
        "comments": [
            {"author_id": 888, "body": "Webhook retried and confirmed.", "public": True}
        ]
    }

    def fake_get(url, **kwargs):
        if "incremental/tickets.json" in url:
            return FakeResponse(tickets_payload, 200)
        if "tickets/101/comments.json" in url:
            return FakeResponse(comments_payload, 200)
        return FakeResponse({}, 404)

    mock_session.get.side_effect = fake_get

    client = ZendeskClient(
        subdomain="apexenterprise",
        oauth_token="test_token",
        session=mock_session,
    )

    rows = list(_iter_tickets(client=client, start_time=0, include_comments=True))

    assert len(rows) == 2

    # Verify first active ticket
    ticket_1 = rows[0]
    assert ticket_1["id"] == "ticket:101"
    assert ticket_1["title"] == "Payment webhook failure"
    assert ticket_1["_deleted"] is False
    assert "Webhook retried and confirmed" in ticket_1["content"]
    assert ticket_1["cognee_node_set"] == "zendesk:apexenterprise:tickets"
    assert ticket_1["document_source"] == "zendesk"

    # Verify second tombstone ticket
    ticket_2 = rows[1]
    assert ticket_2["id"] == "ticket:102"
    assert ticket_2["_deleted"] is True


def test_zendesk_auth_error_handling():
    mock_session = MagicMock()
    mock_session.get.return_value = FakeResponse({}, status_code=401)

    client = ZendeskClient(
        subdomain="apexenterprise",
        oauth_token="invalid_token",
        session=mock_session,
    )

    with pytest.raises(ZendeskAuthError, match="Zendesk auth failed: HTTP 401"):
        client.fetch_tickets()


def test_zendesk_rate_limit_error_handling():
    mock_session = MagicMock()
    mock_session.get.return_value = FakeResponse({}, status_code=429)

    client = ZendeskClient(
        subdomain="apexenterprise",
        oauth_token="valid_token",
        session=mock_session,
    )

    with pytest.raises(ZendeskRateLimitError, match="Zendesk API rate limit hit"):
        client.fetch_tickets()


if __name__ == "__main__":
    print("Running Zendesk connector unit tests...")
    test_zendesk_client_auth_basic()
    test_zendesk_client_auth_oauth()
    test_zendesk_client_missing_auth()
    test_render_ticket_with_comments()
    test_iter_tickets_emission()
    test_zendesk_auth_error_handling()
    test_zendesk_rate_limit_error_handling()
    print("✅ All 7 unit tests passed successfully!")
