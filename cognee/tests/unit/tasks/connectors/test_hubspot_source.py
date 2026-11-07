"""Unit tests for the HubSpot CRM connector (hubspot_source)."""

from __future__ import annotations

from typing import Any
import pytest

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.connectors import build_hubspot_service, hubspot_source
from cognee.tasks.ingestion.connectors.hubspot import (
    HubSpotAPIError,
    HubSpotAuthError,
    HubSpotClient,
    HubSpotRateLimitedError,
    _render_company_row,
    _render_contact_row,
    _render_deal_row,
    _render_note_row,
    _iter_object_rows,
)
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
)


class FakeHubSpotClient(HubSpotClient):
    """Hermetic in-memory fake client for deterministic HubSpot tests."""

    def __init__(self, portal_id: str = "98765432"):
        super().__init__(access_token="test-pat-token", portal_id=portal_id)
        self.objects: dict[str, list[dict[str, Any]]] = {
            "contacts": [],
            "companies": [],
            "deals": [],
            "notes": [],
        }
        self.call_count = 0
        self.raise_on_call: int | None = None
        self.exception_to_raise: Exception | None = None

    def search_objects(
        self,
        object_type: str,
        *,
        properties: list[str] | None = None,
        after: str | None = None,
        limit: int = 100,
        filter_groups: list[dict[str, Any]] | None = None,
        sorts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.call_count += 1
        if self.raise_on_call is not None and self.call_count >= self.raise_on_call:
            assert self.exception_to_raise is not None
            raise self.exception_to_raise

        items = self.objects.get(object_type, [])
        return {"results": items, "paging": None}


def test_contact_row_rendering():
    raw_contact = {
        "id": "101",
        "properties": {
            "firstname": "John",
            "lastname": "Doe",
            "email": "john.doe@example.com",
            "jobtitle": "VP Engineering",
            "company": "Acme Corp",
            "lifecyclestage": "customer",
        },
        "archived": False,
    }
    row = _render_contact_row(raw_contact, "98765432")
    assert row["id"] == "contact:101"
    assert row["title"] == "John Doe"
    assert "Title: VP Engineering" in row["content"]
    assert "Company: Acme Corp" in row["content"]
    assert "Lifecycle Stage: customer" in row["content"]
    assert row["url"] == "https://app.hubspot.com/contacts/98765432/record/0-1/101"
    assert row["_deleted"] is False
    assert row[NODE_SET_COLUMN] == ["hubspot:98765432:contacts", "hubspot:98765432"]


def test_company_row_rendering():
    raw_company = {
        "id": "202",
        "properties": {
            "name": "Acme Corp",
            "domain": "acme.com",
            "industry": "Software",
            "city": "San Francisco",
            "country": "USA",
        },
        "archived": False,
    }
    row = _render_company_row(raw_company, "98765432")
    assert row["id"] == "company:202"
    assert row["title"] == "Acme Corp"
    assert "Domain: acme.com" in row["content"]
    assert "Industry: Software" in row["content"]
    assert "Location: San Francisco, USA" in row["content"]
    assert row["_deleted"] is False
    assert row[NODE_SET_COLUMN] == ["hubspot:98765432:companies", "hubspot:98765432"]


def test_deal_row_rendering():
    raw_deal = {
        "id": "303",
        "properties": {
            "dealname": "Enterprise Expansion 2026",
            "dealstage": "presentationscheduled",
            "amount": "120000",
            "closedate": "2026-12-31",
        },
        "archived": False,
    }
    row = _render_deal_row(raw_deal, "98765432")
    assert row["id"] == "deal:303"
    assert row["title"] == "Enterprise Expansion 2026"
    assert "Stage: presentationscheduled" in row["content"]
    assert "Amount: 120000" in row["content"]
    assert row[NODE_SET_COLUMN] == ["hubspot:98765432:deals", "hubspot:98765432"]


def test_note_row_rendering_html_sanitized():
    raw_note = {
        "id": "404",
        "properties": {
            "hs_note_body": "<p>Met with CTO regarding <strong>security compliance</strong> roadmap.</p>",
        },
        "archived": False,
    }
    row = _render_note_row(raw_note, "98765432")
    assert row["id"] == "note:404"
    assert "<p>" not in row["content"]
    assert "Met with CTO regarding security compliance roadmap." in row["content"]
    assert row[NODE_SET_COLUMN] == ["hubspot:98765432:notes", "hubspot:98765432"]


def test_validation_and_construction():
    with pytest.raises(ValueError, match="access_token"):
        hubspot_source(access_token="")

    with pytest.raises(ValueError, match="resource_name"):
        hubspot_source(access_token="valid_token", resource_name="   ")
