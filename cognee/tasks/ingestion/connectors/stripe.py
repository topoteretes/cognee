"""Stripe connector for cognee: turn enterprise billing, invoices, and charges into memory.

One source handles a Stripe account or live billing stream. It yields invoices, customer billing profiles,
and charge summaries as flat document rows for :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import stripe_source

    await cognee.remember(
        stripe_source(
            api_key="<stripe-restricted-api-key>",
            include_invoices=True,
            include_charges=True,
        ),
        dataset_name="enterprise_billing",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``.
  Ids are prefixed by resource kind (``stripe:invoice:<id>``, ``stripe:charge:<id>``).
* **Incremental**: Uses Stripe's cursor pagination (``starting_after``) to ingest streams chronologically.
* **Line Items**: Invoices fold line item descriptions, unit amounts, and quantity breakdowns into
  the document text for unified semantic recall.
* **Privacy & Security**: Raw card numbers, CVVs, and banking secrets are never rendered or fetched;
  only sanitized last4, brand, and transaction reference metadata are kept.
* **Tombstones**: Voided or uncollectible invoices emit ``_deleted: True`` to ensure graphs stay synchronized.
"""

import logging
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

BASE_URL = "https://api.stripe.com/v1"
_TIMEOUT = (10, 30)
NODE_SET_COLUMN = "cognee_node_set"
DOCUMENT_SOURCE_ATTR = "document_source"


class StripeError(Exception):
    """Base exception for Stripe connector errors."""


class StripeAuthError(StripeError):
    """Raised when Stripe API authentication fails."""


class StripeRateLimitError(StripeError):
    """Raised when Stripe API rate limit quota is exceeded."""


class StripeClient:
    """HTTP client for Stripe Core REST API."""

    def __init__(self, api_key: str, session: Any = None):
        clean_key = api_key.strip()
        if not clean_key or not (clean_key.startswith("sk_") or clean_key.startswith("rk_")):
            raise ValueError("Invalid Stripe API key format (expected sk_live_... or rk_live_...)")

        self.api_key = clean_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "Cognee-Stripe-Connector/1.0",
        }
        self.session = session

    def fetch_invoices(self, limit: int = 100, starting_after: str | None = None) -> dict[str, Any]:
        """Fetches paginated list of Stripe invoices."""
        if self.session is None:
            import requests
            session = requests.Session()
        else:
            session = self.session

        url = f"{BASE_URL}/invoices?limit={min(limit, 100)}"
        if starting_after:
            url += f"&starting_after={starting_after}"

        resp = session.get(url, headers=self.headers, timeout=_TIMEOUT)

        if resp.status_code == 401:
            raise StripeAuthError("Stripe API key invalid or unauthorized")
        if resp.status_code == 429:
            raise StripeRateLimitError("Stripe API rate limit exceeded")
        if not resp.ok:
            raise StripeError(f"Stripe request failed: HTTP {resp.status_code}")

        return resp.json()

    def fetch_charges(self, limit: int = 100, starting_after: str | None = None) -> dict[str, Any]:
        """Fetches paginated list of Stripe charges."""
        if self.session is None:
            import requests
            session = requests.Session()
        else:
            session = self.session

        url = f"{BASE_URL}/charges?limit={min(limit, 100)}"
        if starting_after:
            url += f"&starting_after={starting_after}"

        resp = session.get(url, headers=self.headers, timeout=_TIMEOUT)

        if resp.status_code == 401:
            raise StripeAuthError("Stripe API key invalid or unauthorized")
        if resp.status_code == 429:
            raise StripeRateLimitError("Stripe API rate limit exceeded")
        if not resp.ok:
            raise StripeError(f"Stripe request failed: HTTP {resp.status_code}")

        return resp.json()


def render_invoice(invoice: dict[str, Any]) -> str:
    """Renders structured invoice metadata and line items into text for memory vectorization."""
    inv_id = invoice.get("id", "unknown")
    number = invoice.get("number") or inv_id
    currency = (invoice.get("currency") or "usd").upper()
    amount_due = (invoice.get("amount_due") or 0) / 100.0
    amount_paid = (invoice.get("amount_paid") or 0) / 100.0
    status = invoice.get("status") or "draft"

    customer_id = invoice.get("customer") or "guest"
    customer_name = invoice.get("customer_name") or ""
    customer_email = invoice.get("customer_email") or ""

    created_epoch = invoice.get("created")
    created_str = (
        datetime.fromtimestamp(created_epoch, tz=timezone.utc).isoformat()
        if created_epoch
        else "Unknown"
    )

    lines = [
        f"Stripe Commercial Invoice #{number} ({inv_id})",
        f"Status: {status} | Currency: {currency} | Amount Due: ${amount_due:,.2f} | Amount Paid: ${amount_paid:,.2f}",
        f"Billing Date: {created_str}",
    ]

    if customer_name or customer_id:
        lines.append(f"Customer: {customer_name} (ID: {customer_id})")

    # Fold line items
    line_items = invoice.get("lines", {}).get("data", [])
    if line_items:
        lines.append("\nInvoice Line Items:")
        for item in line_items:
            desc = item.get("description") or "Item"
            qty = item.get("quantity") or 1
            amt = (item.get("amount") or 0) / 100.0
            lines.append(f"- {desc} (Qty: {qty}): ${amt:,.2f}")

    return "\n".join(lines)


def render_charge(charge: dict[str, Any]) -> str:
    """Renders structured charge and settlement metadata into text."""
    ch_id = charge.get("id", "unknown")
    amount = (charge.get("amount") or 0) / 100.0
    currency = (charge.get("currency") or "usd").upper()
    status = charge.get("status") or "succeeded"
    paid = charge.get("paid", False)
    refunded = charge.get("refunded", False)
    description = charge.get("description") or "Charge"

    created_epoch = charge.get("created")
    created_str = (
        datetime.fromtimestamp(created_epoch, tz=timezone.utc).isoformat()
        if created_epoch
        else "Unknown"
    )

    # Extract payment method sanitized details
    pm_details = charge.get("payment_method_details", {})
    card = pm_details.get("card", {})
    card_brand = card.get("brand", "Card").capitalize()
    last4 = card.get("last4", "XXXX")

    lines = [
        f"Stripe Transaction Charge #{ch_id}",
        f"Description: {description}",
        f"Amount: ${amount:,.2f} {currency} | Status: {status} | Paid: {paid} | Refunded: {refunded}",
        f"Payment Rail: {card_brand} ending in {last4}",
        f"Processed At: {created_str}",
    ]

    return "\n".join(lines)


def _iter_stripe_resources(
    client: StripeClient,
    include_invoices: bool = True,
    include_charges: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yields flat documents representing Stripe invoices and charges."""
    if include_invoices:
        has_more = True
        starting_after = None
        while has_more:
            try:
                data = client.fetch_invoices(limit=100, starting_after=starting_after)
            except Exception as e:
                logger.error("Error fetching Stripe invoices: %s", e)
                break

            invoices = data.get("data", [])
            for inv in invoices:
                inv_id = inv.get("id")
                status = inv.get("status")
                is_deleted = status in ("void", "uncollectible")

                yield {
                    "id": f"stripe:invoice:{inv_id}",
                    "title": f"Stripe Invoice {inv.get('number') or inv_id}",
                    "content": render_invoice(inv),
                    "url": f"https://dashboard.stripe.com/invoices/{inv_id}",
                    "_deleted": is_deleted,
                    NODE_SET_COLUMN: "stripe:invoices",
                    DOCUMENT_SOURCE_ATTR: "stripe",
                }

            has_more = data.get("has_more", False)
            if invoices and has_more:
                starting_after = invoices[-1].get("id")
            else:
                has_more = False

    if include_charges:
        has_more = True
        starting_after = None
        while has_more:
            try:
                data = client.fetch_charges(limit=100, starting_after=starting_after)
            except Exception as e:
                logger.error("Error fetching Stripe charges: %s", e)
                break

            charges = data.get("data", [])
            for ch in charges:
                ch_id = ch.get("id")
                is_deleted = ch.get("refunded", False) or ch.get("status") == "failed"

                yield {
                    "id": f"stripe:charge:{ch_id}",
                    "title": f"Stripe Charge {ch_id} (${(ch.get('amount', 0)/100):,.2f})",
                    "content": render_charge(ch),
                    "url": f"https://dashboard.stripe.com/payments/{ch_id}",
                    "_deleted": is_deleted,
                    NODE_SET_COLUMN: "stripe:charges",
                    DOCUMENT_SOURCE_ATTR: "stripe",
                }

            has_more = data.get("has_more", False)
            if charges and has_more:
                starting_after = charges[-1].get("id")
            else:
                has_more = False


def stripe_source(
    api_key: str,
    include_invoices: bool = True,
    include_charges: bool = True,
    client: StripeClient | None = None,
):
    """Returns a DLT source for Stripe billing and transaction ingestion."""
    import dlt

    if client is None:
        client = StripeClient(api_key=api_key)

    @dlt.source(name="stripe_billing")
    def source():
        @dlt.resource(
            name="billing_events",
            primary_key="id",
            write_disposition="merge",
        )
        def billing_resource():
            yield from _iter_stripe_resources(
                client=client,
                include_invoices=include_invoices,
                include_charges=include_charges,
            )

        return billing_resource

    return source()


def build_stripe_service(api_key: str) -> StripeClient:
    """Helper to instantiate a standalone StripeClient."""
    return StripeClient(api_key=api_key)
