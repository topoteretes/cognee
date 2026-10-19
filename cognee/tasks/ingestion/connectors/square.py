"""Square point-of-sale connector for cognee.

A ``dlt`` source that turns physical commerce knowledge from Square into AI memory.
Extracts catalog items (with variations, categories, and prices), customer profiles
(with preferences, notes, and segments), and orders (with line items, modifiers,
totals, and customer links):

    import cognee
    from cognee.tasks.ingestion.connectors import square_source

    await cognee.remember(
        square_source(
            access_token="<square access token>",
            environment="sandbox",  # or "production"
            resources=["catalog", "customers", "orders"],
        ),
        dataset_name="pos_memory",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

Design
------
* **Scope**: Ingests Square physical point-of-sale data:
  - **Catalog**: Products, item variations (size, flavor, color), categories, and pricing.
  - **Customers**: Names, company, notes, preferences, and group associations (with PII protection).
  - **Orders**: Point-of-sale orders, line items, itemized subtotals, tips, taxes, state, and customer links.
* **Incremental**:
  - Catalog searches use ``begin_time`` filtering to only fetch items created or modified
    since the previous run cursor.
  - Customers and orders search endpoints filter and sort by ``updated_at``.
  - Cursors are tracked per-resource in the dlt state dictionary.
* **Forget-on-Delete (Tombstones)**:
  - Square's ``SearchCatalogObjects`` returns deleted catalog items and variations with
    ``is_deleted: true``. These are emitted as ``_deleted=True`` tombstone rows under
    ``write_disposition="merge"``, enabling cognee's orphan cleanup to remove stale entities.
* **Security & Privacy**:
  - Bearer tokens are masked in ``__repr__`` and stripped from exception messages to prevent
    leakage into application logs.
  - Customer contact details (email addresses and phone numbers) are masked/redacted to safeguard
    customer PII in AI memory.
"""

import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Optional

import httpx
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR

logger = get_logger("square_connector")

SQUARE_PRODUCTION_URL = "https://connect.squareup.com"
SQUARE_SANDBOX_URL = "https://connect.squareupsandbox.com"
SQUARE_DEFAULT_API_VERSION = "2024-10-16"
DEFAULT_PAGE_SIZE = 100

ZERO_DECIMAL_CURRENCIES = {
    "BIF",
    "CLP",
    "DJF",
    "GNF",
    "JPY",
    "KMF",
    "KRW",
    "MGA",
    "PYG",
    "RWF",
    "UGX",
    "VND",
    "VUV",
    "XAF",
    "XOF",
    "XPF",
}


class SquareAPIError(Exception):
    """Raised when the Square REST API returns an error response."""

    def __init__(self, message: str, status_code: int | None = None, category: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.category = category


class SquareAuthError(SquareAPIError):
    """Raised on authentication failures (HTTP 401 or invalid token)."""


class SquareRateLimitError(SquareAPIError):
    """Raised when Square returns HTTP 429 rate limit exceeded."""


class SquareSettings(BaseSettings):
    """Configuration settings for Square API connection."""

    square_access_token: str | None = Field(
        default=None, description="Square OAuth or Personal Access Token"
    )
    square_environment: str = Field(
        default="sandbox", description="Square environment: 'sandbox' or 'production'"
    )
    square_api_version: str = Field(
        default=SQUARE_DEFAULT_API_VERSION, description="Square API version header"
    )

    model_config = SettingsConfigDict(
        env_prefix="SQUARE_",
        env_file=".env",
        extra="ignore",
    )


def format_currency(amount: int | None, currency: str = "USD") -> str:
    """Format Square's smallest-unit integer money amounts into human-readable strings."""
    if amount is None:
        return f"0.00 {currency}"
    curr = (currency or "USD").upper()
    if curr in ZERO_DECIMAL_CURRENCIES:
        return f"{amount} {curr}"
    formatted = f"{amount / 100.0:.2f}"
    if curr == "USD":
        return f"${formatted} USD"
    return f"{formatted} {curr}"


def mask_token(token: str | None) -> str:
    """Mask a secret token, preserving only a short prefix and suffix."""
    if not token:
        return "<not-set>"
    if len(token) <= 8:
        return "***"
    return f"{token[:4]}...{token[-4:]}"


def mask_contact_info(value: str | None) -> str:
    """Mask sensitive PII such as email addresses or phone numbers."""
    if not value:
        return ""
    if "@" in value:
        parts = value.split("@", 1)
        name = parts[0]
        domain = parts[1]
        masked_name = name[0] + "***" if name else "***"
        return f"{masked_name}@{domain}"
    if len(value) >= 4:
        return f"***-{value[-4:]}"
    return "***"


class SquareClient:
    """HTTP client for Square POS REST API."""

    def __init__(
        self,
        access_token: str,
        environment: str = "sandbox",
        base_url: str | None = None,
        api_version: str = SQUARE_DEFAULT_API_VERSION,
        http_client: httpx.Client | None = None,
    ):
        if not access_token:
            raise ValueError("SquareClient requires an access_token.")
        self.access_token = access_token
        self.environment = environment.lower() if environment else "sandbox"
        if base_url:
            self.base_url = base_url.rstrip("/")
        elif self.environment == "production":
            self.base_url = SQUARE_PRODUCTION_URL
        else:
            self.base_url = SQUARE_SANDBOX_URL
        self.api_version = api_version
        self._http_client = http_client
        self._owns_http_client = http_client is None

    def __repr__(self) -> str:
        return f"SquareClient(environment='{self.environment}', token='{mask_token(self.access_token)}')"

    def _get_client(self) -> httpx.Client:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.Client(
                timeout=30.0,
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Square-Version": self.api_version,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            self._owns_http_client = True
        return self._http_client

    def close(self) -> None:
        """Close the underlying HTTP client session if owned."""
        if (
            self._owns_http_client
            and self._http_client is not None
            and not self._http_client.is_closed
        ):
            self._http_client.close()

    def _post(self, path: str, json_data: dict[str, Any]) -> dict[str, Any]:
        """Make an authenticated POST request to Square API."""
        client = self._get_client()
        url = f"{self.base_url}{path}"
        try:
            response = client.post(url, json=json_data)
        except httpx.RequestError as exc:
            raise SquareAPIError(f"Square API connection error: {exc}") from exc

        if response.status_code == 401:
            raise SquareAuthError(
                "Square API authentication failed. Verify SQUARE_ACCESS_TOKEN.", status_code=401
            )
        if response.status_code == 429:
            raise SquareRateLimitError("Square API rate limit exceeded.", status_code=429)
        if response.is_error:
            error_data = {}
            try:
                error_data = response.json()
            except ValueError:
                error_data = {}
            errors = error_data.get("errors", [])
            msg = errors[0].get("detail") if errors else response.text
            cat = errors[0].get("category") if errors else None
            raise SquareAPIError(
                f"Square API error ({response.status_code}): {msg}",
                status_code=response.status_code,
                category=cat,
            )

        return response.json()

    def search_catalog_objects(
        self,
        object_types: Sequence[str] = ("ITEM", "CATEGORY"),
        begin_time: str | None = None,
        include_deleted_objects: bool = True,
        cursor: str | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> dict[str, Any]:
        """Search catalog objects (items, categories) incrementally."""
        payload: dict[str, Any] = {
            "object_types": list(object_types),
            "include_deleted_objects": include_deleted_objects,
            "limit": limit,
        }
        if begin_time:
            payload["begin_time"] = begin_time
        if cursor:
            payload["cursor"] = cursor
        return self._post("/v2/catalog/search-objects", payload)

    def search_customers(
        self,
        begin_time: str | None = None,
        cursor: str | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> dict[str, Any]:
        """Search customers ordered by update timestamp."""
        payload: dict[str, Any] = {
            "limit": limit,
            "query": {
                "sort": {
                    "field": "UPDATED_AT",
                    "order": "ASC",
                }
            },
        }
        if begin_time:
            payload["query"]["filter"] = {
                "updated_at": {
                    "start_at": begin_time,
                }
            }
        if cursor:
            payload["cursor"] = cursor
        return self._post("/v2/customers/search", payload)

    def search_orders(
        self,
        location_ids: Sequence[str] | None = None,
        begin_time: str | None = None,
        cursor: str | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> dict[str, Any]:
        """Search point-of-sale orders with line items and states."""
        payload: dict[str, Any] = {
            "limit": limit,
            "query": {
                "sort": {
                    "sort_field": "UPDATED_AT",
                    "sort_order": "ASC",
                }
            },
        }
        if location_ids:
            payload["location_ids"] = list(location_ids)
        if begin_time:
            payload["query"]["filter"] = {
                "date_time_filter": {
                    "updated_at": {
                        "start_at": begin_time,
                    }
                }
            }
        if cursor:
            payload["cursor"] = cursor
        return self._post("/v2/orders/search", payload)


def render_catalog_item(
    obj: dict[str, Any],
    category_map: dict[str, str] | None = None,
    base_url: str = "",
) -> dict[str, Any]:
    """Render a Square catalog item into a structured document or tombstone."""
    obj_id = obj.get("id", "")
    doc_id = f"catalog:{obj_id}"

    # Handle deleted catalog object tombstones
    if obj.get("is_deleted") is True:
        return {"id": doc_id, "_deleted": True}

    item_data = obj.get("item_data", {})
    name = item_data.get("name") or "Unnamed Catalog Item"
    description = item_data.get("description", "").strip()
    category_id = item_data.get("category_id")
    category_name = (
        (category_map or {}).get(category_id, "Uncategorized") if category_id else "Uncategorized"
    )

    variations_list = []
    for var in item_data.get("variations", []):
        var_data = var.get("item_variation_data", {})
        var_name = var_data.get("name") or "Regular"
        sku = var_data.get("sku")
        price_money = var_data.get("price_money") or {}
        price_str = format_currency(price_money.get("amount"), price_money.get("currency", "USD"))
        sku_str = f" [SKU: {sku}]" if sku else ""
        variations_list.append(f"- **{var_name}**{sku_str}: {price_str}")

    variations_md = "\n".join(variations_list) if variations_list else "- *No variations listed.*"

    content = (
        f"# Catalog Item: {name}\n\n"
        f"**Category**: {category_name}\n"
        f"**Square ID**: `{obj_id}`\n\n"
        f"## Description\n{description if description else 'No product description provided.'}\n\n"
        f"## Variations & Pricing\n{variations_md}"
    )

    return {
        "id": doc_id,
        "title": f"Product: {name}",
        "content": content,
        "category": category_name,
        "updated_at": obj.get("updated_at"),
        "url": f"{base_url}/dashboard/items/library/{obj_id}" if base_url else "",
    }


def render_customer(
    customer: dict[str, Any],
    base_url: str = "",
) -> dict[str, Any]:
    """Render a Square customer profile into a structured document with PII protection."""
    cust_id = customer.get("id", "")
    doc_id = f"customer:{cust_id}"

    given_name = customer.get("given_name", "")
    family_name = customer.get("family_name", "")
    display_name = (
        f"{given_name} {family_name}".strip() or customer.get("company_name") or "Valued Customer"
    )
    company = customer.get("company_name", "")
    note = customer.get("note", "").strip()
    groups = customer.get("group_ids") or customer.get("groups") or []
    group_str = ", ".join(str(g) for g in groups) if groups else "None"

    # Redact sensitive contact info to keep PII safe in AI memory
    email_masked = mask_contact_info(customer.get("email_address"))
    phone_masked = mask_contact_info(customer.get("phone_number"))

    contact_parts = []
    if email_masked:
        contact_parts.append(f"Email: {email_masked}")
    if phone_masked:
        contact_parts.append(f"Phone: {phone_masked}")
    contact_info = (
        " | ".join(contact_parts) if contact_parts else "No direct contact details on file"
    )

    content = (
        f"# Customer: {display_name}\n\n"
        f"**Company**: {company if company else 'N/A'}\n"
        f"**Customer ID**: `{cust_id}`\n"
        f"**Groups/Segments**: {group_str}\n"
        f"**Contact**: {contact_info}\n\n"
        f"## Staff Notes & Preferences\n{note if note else 'No customer notes recorded.'}"
    )

    return {
        "id": doc_id,
        "title": f"Customer: {display_name}",
        "content": content,
        "company": company,
        "updated_at": customer.get("updated_at"),
        "url": f"{base_url}/dashboard/customers/directory/{cust_id}" if base_url else "",
    }


def render_order(
    order: dict[str, Any],
    base_url: str = "",
) -> dict[str, Any]:
    """Render a Square point-of-sale order into a structured document with customer links."""
    order_id = order.get("id", "")
    doc_id = f"order:{order_id}"
    state = order.get("state", "UNKNOWN")
    customer_id = order.get("customer_id")
    created_at = order.get("created_at", "")

    # Format line items
    items_list = []
    for item in order.get("line_items", []):
        item_name = item.get("name", "Unknown Item")
        variation_name = item.get("variation_name")
        qty = item.get("quantity", "1")
        gross_money = item.get("gross_sales_money") or item.get("total_money") or {}
        price_str = format_currency(gross_money.get("amount"), gross_money.get("currency", "USD"))
        var_str = f" ({variation_name})" if variation_name else ""
        items_list.append(f"- {qty}x **{item_name}**{var_str} — {price_str}")

    items_md = "\n".join(items_list) if items_list else "- *No line items recorded.*"

    total_money = order.get("total_money") or {}
    total_str = format_currency(total_money.get("amount"), total_money.get("currency", "USD"))
    tax_money = order.get("total_tax_money") or {}
    tax_str = format_currency(tax_money.get("amount"), tax_money.get("currency", "USD"))
    tip_money = order.get("total_tip_money") or {}
    tip_str = format_currency(tip_money.get("amount"), tip_money.get("currency", "USD"))
    discount_money = order.get("total_discount_money") or {}
    discount_str = format_currency(
        discount_money.get("amount"), discount_money.get("currency", "USD")
    )

    customer_ref = f"`customer:{customer_id}`" if customer_id else "Guest (Walk-in)"

    content = (
        f"# Order #{order_id[:8]}\n\n"
        f"**Order ID**: `{order_id}`\n"
        f"**State**: {state}\n"
        f"**Date**: {created_at}\n"
        f"**Customer**: {customer_ref}\n\n"
        f"## Purchased Items\n{items_md}\n\n"
        f"## Financial Summary\n"
        f"- **Total Amount**: {total_str}\n"
        f"- Tax: {tax_str}\n"
        f"- Tip: {tip_str}\n"
        f"- Discounts Applied: {discount_str}"
    )

    return {
        "id": doc_id,
        "title": f"Order #{order_id[:8]}",
        "content": content,
        "customer_id": f"customer:{customer_id}" if customer_id else None,
        "state": state,
        "updated_at": order.get("updated_at"),
        "url": f"{base_url}/dashboard/orders/overview/{order_id}" if base_url else "",
    }


def iter_square_records(
    client: SquareClient,
    resources: Sequence[str] = ("catalog", "customers", "orders"),
    location_ids: Sequence[str] | None = None,
    state: dict[str, Any] | None = None,
    stats: dict[str, int] | None = None,
    check_active: Callable[[], None] | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> Iterator[dict[str, Any]]:
    """Yield changed documents and tombstones across Square resources.

    Pure Python generator decoupled from DLT runtime for fast testing.
    """
    if state is None:
        state = {}
    if stats is None:
        stats = {}

    category_map: dict[str, str] = state.setdefault("category_map", {})

    for resource in resources:
        if check_active:
            check_active()

        cursor_key = f"{resource}_cursor"
        last_sync_time = state.get(cursor_key)
        latest_time = last_sync_time
        pagination_cursor: str | None = None

        if resource == "catalog":
            while True:
                if check_active:
                    check_active()
                try:
                    res = client.search_catalog_objects(
                        object_types=("ITEM", "CATEGORY"),
                        begin_time=last_sync_time,
                        include_deleted_objects=True,
                        cursor=pagination_cursor,
                        limit=page_size,
                    )
                except SquareRateLimitError:
                    stats["rate_limited"] = stats.get("rate_limited", 0) + 1
                    logger.warning(
                        "Square API rate limit reached during catalog sync; resuming next run."
                    )
                    break
                except SquareAPIError as exc:
                    stats["errors"] = stats.get("errors", 0) + 1
                    logger.error("Square catalog error: %s", exc)
                    raise

                objects = res.get("objects", [])
                for obj in objects:
                    obj_type = obj.get("type")
                    updated_at = obj.get("updated_at")
                    if updated_at and (latest_time is None or updated_at > latest_time):
                        latest_time = updated_at

                    # Cache category names so items reflect renamed or assigned categories
                    if obj_type == "CATEGORY":
                        cat_data = obj.get("category_data", {})
                        cat_name = cat_data.get("name")
                        if cat_name:
                            category_map[obj.get("id")] = cat_name

                    # Render items or deletion tombstones
                    if obj_type == "ITEM" or obj.get("is_deleted") is True:
                        row = render_catalog_item(
                            obj, category_map=category_map, base_url=client.base_url
                        )
                        stats["catalog_count"] = stats.get("catalog_count", 0) + 1
                        yield row

                pagination_cursor = res.get("cursor")
                if not pagination_cursor or not objects:
                    break

        elif resource == "customers":
            while True:
                if check_active:
                    check_active()
                try:
                    res = client.search_customers(
                        begin_time=last_sync_time,
                        cursor=pagination_cursor,
                        limit=page_size,
                    )
                except SquareRateLimitError:
                    stats["rate_limited"] = stats.get("rate_limited", 0) + 1
                    logger.warning(
                        "Square API rate limit reached during customer sync; resuming next run."
                    )
                    break
                except SquareAPIError as exc:
                    stats["errors"] = stats.get("errors", 0) + 1
                    logger.error("Square customers error: %s", exc)
                    raise

                customers = res.get("customers", [])
                for cust in customers:
                    updated_at = cust.get("updated_at")
                    if updated_at and (latest_time is None or updated_at > latest_time):
                        latest_time = updated_at
                    row = render_customer(cust, base_url=client.base_url)
                    stats["customer_count"] = stats.get("customer_count", 0) + 1
                    yield row

                pagination_cursor = res.get("cursor")
                if not pagination_cursor or not customers:
                    break

        elif resource == "orders":
            while True:
                if check_active:
                    check_active()
                try:
                    res = client.search_orders(
                        location_ids=location_ids,
                        begin_time=last_sync_time,
                        cursor=pagination_cursor,
                        limit=page_size,
                    )
                except SquareRateLimitError:
                    stats["rate_limited"] = stats.get("rate_limited", 0) + 1
                    logger.warning(
                        "Square API rate limit reached during order sync; resuming next run."
                    )
                    break
                except SquareAPIError as exc:
                    stats["errors"] = stats.get("errors", 0) + 1
                    logger.error("Square orders error: %s", exc)
                    raise

                orders = res.get("orders", [])
                for order in orders:
                    updated_at = order.get("updated_at")
                    if updated_at and (latest_time is None or updated_at > latest_time):
                        latest_time = updated_at
                    row = render_order(order, base_url=client.base_url)
                    stats["order_count"] = stats.get("order_count", 0) + 1
                    yield row

                pagination_cursor = res.get("cursor")
                if not pagination_cursor or not orders:
                    break

        if latest_time:
            state[cursor_key] = latest_time


def square_source(
    *,
    access_token: str | None = None,
    environment: str = "sandbox",
    resources: Sequence[str] = ("catalog", "customers", "orders"),
    location_ids: Sequence[str] | None = None,
    resource_name: str | None = None,
    service: SquareClient | None = None,
    check_active: Callable[[], None] | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
):
    """Return a ``dlt`` resource yielding Square POS documents and tombstones for ``remember``.

    Args:
        access_token: Square OAuth or Personal Access Token. Defaults to SQUARE_ACCESS_TOKEN env var.
        environment: Square environment ('sandbox' or 'production'). Defaults to 'sandbox'.
        resources: Sequence of Square resources to sync ('catalog', 'customers', 'orders').
        location_ids: Optional list of Square location IDs for order filtering.
        resource_name: Stable connection identifier for cognee dataset staging.
        service: Optional pre-configured SquareClient (useful for testing and dependency injection).
        check_active: Optional callback to check credential freshness before each batch.
        page_size: Maximum records per page returned by Square REST endpoints.

    Returns:
        A ``dlt`` resource configured with primary_key="id", write_disposition="merge",
        and tombstone support (_deleted=True).
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError("The Square connector requires dlt (a cognee core dependency).") from exc

    resolved_token = access_token or SquareSettings().square_access_token
    if not service and not resolved_token:
        raise ValueError("square_source requires access_token or SQUARE_ACCESS_TOKEN env var.")

    client = service or SquareClient(
        access_token=resolved_token,  # type: ignore
        environment=environment,
    )

    if resource_name is None:
        resource_name = f"square_{client.environment}"

    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
    )
    def square_pos_documents():
        # Retrieve dlt resource state to preserve cursors across sync runs
        dlt_state = dlt.current.resource_state()
        yield from dlt_utils.guarded_rows(
            iter_square_records(
                client=client,
                resources=resources,
                location_ids=location_ids,
                state=dlt_state,
                stats=stats,
                check_active=check_active,
                page_size=page_size,
            ),
            check_active=check_active,
        )

    resource = square_pos_documents()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "square")
    setattr(resource, PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
