"""Unit tests for the Square POS data-source connector.

Tests pagination, incremental cursors, catalog variations, category caching,
order line-items, customer PII masking, tombstones, and token hygiene.
All tests use in-memory mock clients with zero external network requests.
"""

from typing import Any

import httpx
import pytest

from cognee.tasks.ingestion.connectors.square import (
    SquareAPIError,
    SquareAuthError,
    SquareClient,
    SquareRateLimitError,
    SquareSettings,
    format_currency,
    iter_square_records,
    mask_contact_info,
    mask_token,
    render_catalog_item,
    render_customer,
    render_order,
    square_source,
)
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR


class MockSquareTransport(httpx.BaseTransport):
    """Mock HTTP transport returning controlled Square API responses."""

    def __init__(self, handler):
        self.handler = handler

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self.handler(request)


def test_token_masking():
    """Verify that bearer tokens are masked across string representations."""
    assert mask_token(None) == "<not-set>"
    assert mask_token("short") == "***"
    assert mask_token("sq0at-secret-token-123456789") == "sq0a...6789"

    client = SquareClient(access_token="sq0at-secret-token-123456789", environment="sandbox")
    repr_str = repr(client)
    assert "sq0at-secret-token-123456789" not in repr_str
    assert "sq0a...6789" in repr_str
    assert "sandbox" in repr_str


def test_format_currency():
    """Verify formatting of standard 2-decimal and zero-decimal currency amounts."""
    assert format_currency(None, "USD") == "0.00 USD"
    assert format_currency(450, "USD") == "$4.50 USD"
    assert format_currency(1200, "EUR") == "12.00 EUR"
    assert format_currency(500, "JPY") == "500 JPY"
    assert format_currency(15000, "KRW") == "15000 KRW"


def test_mask_contact_info():
    """Verify PII email and phone redaction for customer privacy."""
    assert mask_contact_info("") == ""
    assert mask_contact_info("alice@example.com") == "a***@example.com"
    assert mask_contact_info("+15551234567") == "***-4567"


def test_render_catalog_item_with_variations():
    """Verify rendering of a catalog item with multiple variations, SKU, and category resolution."""
    category_map = {"CAT_1": "Specialty Coffee"}
    raw_item = {
        "id": "ITEM_ESPRESSO",
        "type": "ITEM",
        "updated_at": "2026-10-18T10:00:00Z",
        "item_data": {
            "name": "Artisan Espresso",
            "description": "Double shot extraction with notes of dark chocolate.",
            "category_id": "CAT_1",
            "variations": [
                {
                    "id": "VAR_SINGLE",
                    "item_variation_data": {
                        "name": "Single Shot",
                        "sku": "ESP-SGL",
                        "price_money": {"amount": 350, "currency": "USD"},
                    },
                },
                {
                    "id": "VAR_DOUBLE",
                    "item_variation_data": {
                        "name": "Double Shot",
                        "sku": "ESP-DBL",
                        "price_money": {"amount": 475, "currency": "USD"},
                    },
                },
            ],
        },
    }

    doc = render_catalog_item(
        raw_item, category_map=category_map, base_url="https://connect.squareupsandbox.com"
    )
    assert doc["id"] == "catalog:ITEM_ESPRESSO"
    assert doc["title"] == "Product: Artisan Espresso"
    assert doc["category"] == "Specialty Coffee"
    assert doc["updated_at"] == "2026-10-18T10:00:00Z"
    assert "Specialty Coffee" in doc["content"]
    assert "Artisan Espresso" in doc["content"]
    assert "Double shot extraction" in doc["content"]
    assert "Single Shot" in doc["content"]
    assert "$3.50 USD" in doc["content"]
    assert "Double Shot" in doc["content"]
    assert "$4.75 USD" in doc["content"]
    assert "[SKU: ESP-DBL]" in doc["content"]


def test_render_catalog_item_tombstone():
    """Verify deleted catalog objects produce tombstone records."""
    raw_deleted_item = {
        "id": "ITEM_DELETED_1",
        "type": "ITEM",
        "is_deleted": True,
        "updated_at": "2026-10-18T11:00:00Z",
    }
    tombstone = render_catalog_item(raw_deleted_item)
    assert tombstone == {"id": "catalog:ITEM_DELETED_1", "_deleted": True}


def test_render_customer_pii_safety():
    """Verify customer rendering preserves notes while safeguarding PII contact info."""
    raw_customer = {
        "id": "CUST_99",
        "given_name": "Eleanor",
        "family_name": "Vance",
        "company_name": "Hill House Hospitality",
        "note": "Allergic to hazelnut syrup. Prefers oat milk flat white.",
        "email_address": "eleanor@hillhouse.org",
        "phone_number": "+15559876543",
        "group_ids": ["VIP_MEMBERS", "WEEKLY_REGULARS"],
        "updated_at": "2026-10-18T09:30:00Z",
    }

    doc = render_customer(raw_customer)
    assert doc["id"] == "customer:CUST_99"
    assert doc["title"] == "Customer: Eleanor Vance"
    assert doc["company"] == "Hill House Hospitality"
    assert "Allergic to hazelnut syrup" in doc["content"]
    assert "VIP_MEMBERS" in doc["content"]
    assert "eleanor@hillhouse.org" not in doc["content"]
    assert "e***@hillhouse.org" in doc["content"]
    assert "+15559876543" not in doc["content"]
    assert "***-6543" in doc["content"]


def test_render_order_line_items_and_totals():
    """Verify order rendering formats line items, totals, and links customer."""
    raw_order = {
        "id": "ORD_54321_ABCD",
        "state": "COMPLETED",
        "customer_id": "CUST_99",
        "created_at": "2026-10-18T12:00:00Z",
        "updated_at": "2026-10-18T12:05:00Z",
        "line_items": [
            {
                "name": "Artisan Espresso",
                "variation_name": "Double Shot",
                "quantity": "2",
                "gross_sales_money": {"amount": 950, "currency": "USD"},
            },
            {
                "name": "Almond Croissant",
                "quantity": "1",
                "gross_sales_money": {"amount": 500, "currency": "USD"},
            },
        ],
        "total_money": {"amount": 1625, "currency": "USD"},
        "total_tax_money": {"amount": 125, "currency": "USD"},
        "total_tip_money": {"amount": 200, "currency": "USD"},
        "total_discount_money": {"amount": 150, "currency": "USD"},
    }

    doc = render_order(raw_order)
    assert doc["id"] == "order:ORD_54321_ABCD"
    assert doc["title"] == "Order #ORD_5432"
    assert doc["customer_id"] == "customer:CUST_99"
    assert doc["state"] == "COMPLETED"
    assert "customer:CUST_99" in doc["content"]
    assert "2x **Artisan Espresso** (Double Shot) — $9.50 USD" in doc["content"]
    assert "1x **Almond Croissant** — $5.00 USD" in doc["content"]
    assert "Total Amount**: $16.25 USD" in doc["content"]
    assert "Tip: $2.00 USD" in doc["content"]
    assert "Tax: $1.25 USD" in doc["content"]


def test_iter_square_records_catalog_and_category_caching():
    """Verify incremental catalog sync, category name caching, and cursor update."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/catalog/search-objects"
        return httpx.Response(
            200,
            json={
                "objects": [
                    {
                        "id": "CAT_PASTRY",
                        "type": "CATEGORY",
                        "updated_at": "2026-10-18T08:00:00Z",
                        "category_data": {"name": "Fresh Pastries"},
                    },
                    {
                        "id": "ITEM_CROISSANT",
                        "type": "ITEM",
                        "updated_at": "2026-10-18T08:15:00Z",
                        "item_data": {
                            "name": "Butter Croissant",
                            "category_id": "CAT_PASTRY",
                            "variations": [
                                {
                                    "id": "VAR_CR",
                                    "item_variation_data": {
                                        "name": "Standard",
                                        "price_money": {"amount": 425, "currency": "USD"},
                                    },
                                }
                            ],
                        },
                    },
                    {
                        "id": "ITEM_OBSOLETE",
                        "type": "ITEM",
                        "is_deleted": True,
                        "updated_at": "2026-10-18T08:20:00Z",
                    },
                ],
                "cursor": None,
            },
        )

    mock_client = httpx.Client(transport=MockSquareTransport(handler))
    client = SquareClient(access_token="test_token", http_client=mock_client)

    state: dict[str, Any] = {}
    stats: dict[str, int] = {}
    records = list(iter_square_records(client, resources=["catalog"], state=state, stats=stats))

    assert len(records) == 2  # 1 item + 1 tombstone (category is cached, not yielded as document)
    assert records[0]["id"] == "catalog:ITEM_CROISSANT"
    assert records[0]["category"] == "Fresh Pastries"
    assert records[1] == {"id": "catalog:ITEM_OBSOLETE", "_deleted": True}
    assert state["catalog_cursor"] == "2026-10-18T08:20:00Z"
    assert state["category_map"]["CAT_PASTRY"] == "Fresh Pastries"
    assert stats["catalog_count"] == 2


def test_iter_square_records_customers_and_orders():
    """Verify customer and order streaming with pagination."""
    customer_page = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal customer_page
        if request.url.path == "/v2/customers/search":
            customer_page += 1
            if customer_page == 1:
                return httpx.Response(
                    200,
                    json={
                        "customers": [
                            {
                                "id": "CUST_1",
                                "given_name": "Bob",
                                "updated_at": "2026-10-18T07:00:00Z",
                            }
                        ],
                        "cursor": "cursor_page_2",
                    },
                )
            return httpx.Response(
                200,
                json={
                    "customers": [
                        {
                            "id": "CUST_2",
                            "given_name": "Charlie",
                            "updated_at": "2026-10-18T07:30:00Z",
                        }
                    ],
                    "cursor": None,
                },
            )
        if request.url.path == "/v2/orders/search":
            return httpx.Response(
                200,
                json={
                    "orders": [
                        {
                            "id": "ORD_1",
                            "state": "COMPLETED",
                            "customer_id": "CUST_1",
                            "updated_at": "2026-10-18T08:00:00Z",
                            "line_items": [{"name": "Coffee", "quantity": "1"}],
                        }
                    ],
                    "cursor": None,
                },
            )
        return httpx.Response(404, json={})

    mock_client = httpx.Client(transport=MockSquareTransport(handler))
    client = SquareClient(access_token="test_token", http_client=mock_client)

    state: dict[str, Any] = {}
    stats: dict[str, int] = {}
    records = list(
        iter_square_records(client, resources=["customers", "orders"], state=state, stats=stats)
    )

    assert len(records) == 3
    assert records[0]["id"] == "customer:CUST_1"
    assert records[1]["id"] == "customer:CUST_2"
    assert records[2]["id"] == "order:ORD_1"
    assert records[2]["customer_id"] == "customer:CUST_1"
    assert state["customers_cursor"] == "2026-10-18T07:30:00Z"
    assert state["orders_cursor"] == "2026-10-18T08:00:00Z"
    assert stats["customer_count"] == 2
    assert stats["order_count"] == 1


def test_rate_limit_graceful_abort():
    """Verify 429 rate limit triggers graceful abort without raising or discarding state."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, json={"errors": [{"category": "RATE_LIMIT_ERROR", "code": "RATE_LIMITED"}]}
        )

    mock_client = httpx.Client(transport=MockSquareTransport(handler))
    client = SquareClient(access_token="test_token", http_client=mock_client)

    state: dict[str, Any] = {}
    stats: dict[str, int] = {}
    records = list(iter_square_records(client, resources=["catalog"], state=state, stats=stats))

    assert records == []
    assert stats["rate_limited"] == 1


def test_square_source_contract():
    """Verify square_source builder attaches correct metadata and attributes."""
    client = SquareClient(access_token="fake_token", environment="sandbox")
    resource = square_source(
        service=client,
        resource_name="square_test_pos",
        resources=["catalog"],
    )

    assert getattr(resource, DOCUMENT_SOURCE_ATTR) == "square"
    assert getattr(resource, PIPELINE_SCOPE_ATTR) == "square_test_pos"
    assert hasattr(resource, "cognee_sync_stats")


def test_square_client_auth_error():
    """Verify 401 raises SquareAuthError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401, json={"errors": [{"code": "UNAUTHORIZED", "detail": "Invalid access token"}]}
        )

    mock_client = httpx.Client(transport=MockSquareTransport(handler))
    client = SquareClient(access_token="invalid_token", http_client=mock_client)

    with pytest.raises(SquareAuthError) as exc_info:
        client.search_catalog_objects()
    assert "authentication failed" in str(exc_info.value).lower()


def test_square_client_general_api_error():
    """Verify general API error parsing and category extraction."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "errors": [{"category": "INVALID_REQUEST_ERROR", "detail": "Invalid limit param"}]
            },
        )

    mock_client = httpx.Client(transport=MockSquareTransport(handler))
    client = SquareClient(access_token="test_token", http_client=mock_client)

    with pytest.raises(SquareAPIError) as exc_info:
        client.search_catalog_objects()
    assert "Invalid limit param" in str(exc_info.value)
    assert exc_info.value.status_code == 400
    assert exc_info.value.category == "INVALID_REQUEST_ERROR"
