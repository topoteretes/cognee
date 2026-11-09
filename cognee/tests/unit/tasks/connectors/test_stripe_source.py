"""Unit tests for Cognee Stripe billing & invoice connector."""

from unittest.mock import MagicMock

import pytest

try:
    from cognee.tasks.ingestion.connectors.stripe import (
        StripeAuthError,
        StripeClient,
        StripeError,
        StripeRateLimitError,
        _iter_stripe_resources,
        build_stripe_service,
        render_charge,
        render_invoice,
    )
except (ImportError, ModuleNotFoundError):
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../tasks/ingestion/connectors")))
    from stripe import (
        StripeAuthError,
        StripeClient,
        StripeError,
        StripeRateLimitError,
        _iter_stripe_resources,
        build_stripe_service,
        render_charge,
        render_invoice,
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


def test_stripe_client_auth_valid():
    client = build_stripe_service(api_key="sk_live_test_valid_key_123")
    assert client.headers["Authorization"] == "Bearer sk_live_test_valid_key_123"
    assert client.headers["Accept"] == "application/json"


def test_stripe_client_auth_invalid():
    with pytest.raises(ValueError, match="Invalid Stripe API key format"):
        StripeClient(api_key="invalid_plain_key")


def test_render_invoice():
    invoice = {
        "id": "in_1N4XYZ",
        "number": "INV-2026-0042",
        "currency": "usd",
        "amount_due": 150000,
        "amount_paid": 150000,
        "status": "paid",
        "customer": "cus_999",
        "customer_name": "Acme Corp",
        "created": 1794211200,
        "lines": {
            "data": [
                {"description": "Enterprise Cloud Seats", "quantity": 10, "amount": 100000},
                {"description": "Dedicated Vector Host", "quantity": 1, "amount": 50000},
            ]
        },
    }

    rendered = render_invoice(invoice)
    assert "Stripe Commercial Invoice #INV-2026-0042 (in_1N4XYZ)" in rendered
    assert "Status: paid | Currency: USD | Amount Due: $1,500.00 | Amount Paid: $1,500.00" in rendered
    assert "Customer: Acme Corp (ID: cus_999)" in rendered
    assert "- Enterprise Cloud Seats (Qty: 10): $1,000.00" in rendered
    assert "- Dedicated Vector Host (Qty: 1): $500.00" in rendered


def test_render_charge():
    charge = {
        "id": "ch_3M0ABC",
        "amount": 25000,
        "currency": "usd",
        "status": "succeeded",
        "paid": True,
        "refunded": False,
        "description": "Monthly Pro Subscription",
        "created": 1794211200,
        "payment_method_details": {
            "card": {"brand": "visa", "last4": "4242"}
        },
    }

    rendered = render_charge(charge)
    assert "Stripe Transaction Charge #ch_3M0ABC" in rendered
    assert "Monthly Pro Subscription" in rendered
    assert "Amount: $250.00 USD | Status: succeeded | Paid: True | Refunded: False" in rendered
    assert "Payment Rail: Visa ending in 4242" in rendered


def test_iter_stripe_resources_emission():
    mock_session = MagicMock()

    invoices_payload = {
        "data": [
            {
                "id": "in_001",
                "number": "INV-001",
                "currency": "usd",
                "amount_due": 5000,
                "amount_paid": 5000,
                "status": "paid",
            },
            {
                "id": "in_002",
                "number": "INV-002",
                "currency": "usd",
                "amount_due": 3000,
                "amount_paid": 0,
                "status": "void",
            },
        ],
        "has_more": False,
    }

    charges_payload = {
        "data": [
            {
                "id": "ch_001",
                "amount": 5000,
                "currency": "usd",
                "status": "succeeded",
                "paid": True,
                "refunded": False,
                "payment_method_details": {"card": {"brand": "mastercard", "last4": "8888"}},
            }
        ],
        "has_more": False,
    }

    def fake_get(url, **kwargs):
        if "/invoices" in url:
            return FakeResponse(invoices_payload, 200)
        if "/charges" in url:
            return FakeResponse(charges_payload, 200)
        return FakeResponse({}, 404)

    mock_session.get.side_effect = fake_get

    client = StripeClient(api_key="sk_live_test_123", session=mock_session)
    rows = list(_iter_stripe_resources(client=client, include_invoices=True, include_charges=True))

    assert len(rows) == 3

    # Invoice 1 (active)
    inv_1 = rows[0]
    assert inv_1["id"] == "stripe:invoice:in_001"
    assert inv_1["_deleted"] is False
    assert inv_1["node_set"] if "node_set" in inv_1 else inv_1["cognee_node_set"] == "stripe:invoices"
    assert inv_1["document_source"] == "stripe"

    # Invoice 2 (void tombstone)
    inv_2 = rows[1]
    assert inv_2["id"] == "stripe:invoice:in_002"
    assert inv_2["_deleted"] is True

    # Charge 1 (active)
    ch_1 = rows[2]
    assert ch_1["id"] == "stripe:charge:ch_001"
    assert ch_1["_deleted"] is False
    assert "Mastercard ending in 8888" in ch_1["content"]


def test_stripe_error_handling():
    mock_session = MagicMock()
    mock_session.get.return_value = FakeResponse({}, status_code=401)
    client = StripeClient(api_key="sk_live_unauthorized", session=mock_session)

    with pytest.raises(StripeAuthError, match="Stripe API key invalid"):
        client.fetch_invoices()

    mock_session.get.return_value = FakeResponse({}, status_code=429)
    with pytest.raises(StripeRateLimitError, match="Stripe API rate limit exceeded"):
        client.fetch_charges()


if __name__ == "__main__":
    print("Running Stripe connector unit tests...")
    test_stripe_client_auth_valid()
    test_stripe_client_auth_invalid()
    test_render_invoice()
    test_render_charge()
    test_iter_stripe_resources_emission()
    test_stripe_error_handling()
    print("✅ All 6 Stripe unit tests passed successfully!")
