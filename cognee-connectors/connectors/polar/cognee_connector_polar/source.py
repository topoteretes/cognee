"""Polar connector for cognee, a ``dlt`` source that turns Polar customers,
subscriptions, and benefit grants into searchable memory.

Sync Polar customer state, active/past subscriptions, and benefit grants
into cognee incrementally.

Design
------
* **Transport**: plain ``httpx`` client with authorization header using Polar Personal Access Token (or Organization Access Token).
* **API version**: Polar REST API (`https://api.polar.sh/v1`).
* **Entities**:
  - `customers`: Customer profiles & organization context.
  - `subscriptions`: Customer subscription history and current status.
  - `benefit_grants`: Granted product benefit entitlements for customers.
"""

import logging
import os
from collections.abc import Callable
from typing import Any

from cognee_connectors_core import dlt_conventions as dlt_utils
from cognee_connectors_core.dlt_conventions import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    STRUCTURE_COLUMN,
)

logger = logging.getLogger(__name__)

POLAR_API_BASE_URL = "https://api.polar.sh/v1"
_MAX_RETRIES = 5
_PAGE_SIZE = 100

_EXTRA_HINT = "Set token= or the POLAR_ORGANIZATION_TOKEN / POLAR_ACCESS_TOKEN environment variable."


class PolarAPIError(Exception):
    """A Polar API error."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class _RetryableStatusError(PolarAPIError):
    """A 429 or 5xx answer, retried by the client."""

    def __init__(self, status_code: int, message: str, headers: Any = None):
        super().__init__(status_code, message)
        self.headers = headers


class _PolarClient:
    """One Polar API session."""

    def __init__(
        self, http_client: Any, token: str, check_active: Callable[[], None] | None = None
    ):
        self._http = http_client
        self._check_active = check_active
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def request(
        self, method: str, path: str, *, params: dict | None = None, json_body: dict | None = None
    ) -> dict:
        import httpx

        try:
            return dlt_utils.with_retries(
                lambda: self._attempt(method, path, params, json_body),
                retry_on=(httpx.TransportError, _RetryableStatusError),
                delay=lambda error, number: _retry_delay(path, error, number),
                attempts=_MAX_RETRIES,
                check_active=self._check_active,
            )
        except _RetryableStatusError as error:
            raise PolarAPIError(error.status_code, str(error)) from None

    def _attempt(self, method: str, path: str, params: dict | None, json_body: dict | None) -> dict:
        url = f"{POLAR_API_BASE_URL}/{path.lstrip('/')}"
        response = self._http.request(
            method,
            url,
            params=params,
            json=json_body,
            headers=self._headers,
        )
        status = response.status_code
        if status == 429 or status >= 500:
            raise _RetryableStatusError(status, f"Polar API error {status} on {path}: {response.text}", response.headers)
        if status not in (200, 201):
            raise PolarAPIError(status, f"Polar API error {status} on {path}: {response.text}")
        return response.json()

    def list_customers(self, page: int = 1, limit: int = _PAGE_SIZE) -> dict:
        return self.request("GET", "customers", params={"page": page, "limit": limit})

    def list_subscriptions(self, page: int = 1, limit: int = _PAGE_SIZE) -> dict:
        return self.request("GET", "subscriptions", params={"page": page, "limit": limit})

    def list_benefit_grants(self, page: int = 1, limit: int = _PAGE_SIZE) -> dict:
        return self.request("GET", "customer-portal/benefit-grants", params={"page": page, "limit": limit})


def _retry_delay(path: str, error: BaseException, number: int) -> float:
    delay = float(2**number)
    logger.warning("Polar: retry attempt %d for %s due to %s", number + 1, path, error)
    return delay


def _paginate_endpoint(fetch_fn: Callable[[int, int], dict]) -> list[dict]:
    items = []
    page = 1
    while True:
        res = fetch_fn(page, _PAGE_SIZE)
        page_items = res.get("items", [])
        items.extend(page_items)
        pagination = res.get("pagination", {})
        max_page = pagination.get("max_page") or 1
        if page >= max_page or not page_items:
            break
        page += 1
    return items


def _render_customer_document(customer: dict, organization_id: str) -> dict:
    cid = customer.get("id")
    email = customer.get("email") or ""
    name = customer.get("name") or email or cid
    
    content_lines = [
        f"Customer Name: {name}",
        f"Customer Email: {email}",
        f"Customer ID: {cid}",
    ]
    if customer.get("created_at"):
        content_lines.append(f"Created At: {customer.get('created_at')}")

    return {
        "id": f"polar_customer_{cid}",
        "title": f"Polar Customer: {name}",
        "content": "\n".join(content_lines),
        "_deleted": False,
        NODE_SET_COLUMN: [f"polar:{organization_id}:customers"],
        STRUCTURE_COLUMN: {"ancestors": [{"kind": "organization", "id": organization_id}]},
    }


def _render_subscription_document(sub: dict, organization_id: str) -> dict:
    sub_id = sub.get("id")
    status = sub.get("status") or "unknown"
    customer_id = sub.get("customer_id") or (sub.get("customer") or {}).get("id") or "unknown"
    product_name = (sub.get("product") or {}).get("name") or sub.get("product_id") or "Subscription"

    content_lines = [
        f"Subscription ID: {sub_id}",
        f"Status: {status}",
        f"Customer ID: {customer_id}",
        f"Product: {product_name}",
    ]
    if sub.get("current_period_start"):
        content_lines.append(f"Current Period Start: {sub.get('current_period_start')}")
    if sub.get("current_period_end"):
        content_lines.append(f"Current Period End: {sub.get('current_period_end')}")

    return {
        "id": f"polar_subscription_{sub_id}",
        "title": f"Polar Subscription: {product_name} ({status})",
        "content": "\n".join(content_lines),
        "_deleted": False,
        NODE_SET_COLUMN: [f"polar:{organization_id}:subscriptions"],
        STRUCTURE_COLUMN: {"ancestors": [{"kind": "customer", "id": customer_id}]},
    }


def _render_grant_document(grant: dict, organization_id: str) -> dict:
    grant_id = grant.get("id")
    benefit_id = grant.get("benefit_id") or (grant.get("benefit") or {}).get("id") or "unknown"
    benefit_type = (grant.get("benefit") or {}).get("type") or "benefit"
    customer_id = grant.get("customer_id") or (grant.get("customer") or {}).get("id") or "unknown"
    is_granted = grant.get("is_granted", True)

    content_lines = [
        f"Benefit Grant ID: {grant_id}",
        f"Benefit ID: {benefit_id}",
        f"Type: {benefit_type}",
        f"Customer ID: {customer_id}",
        f"Granted: {'Yes' if is_granted else 'No'}",
    ]

    return {
        "id": f"polar_grant_{grant_id}",
        "title": f"Polar Benefit Grant: {benefit_type}",
        "content": "\n".join(content_lines),
        "_deleted": False,
        NODE_SET_COLUMN: [f"polar:{organization_id}:benefit_grants"],
        STRUCTURE_COLUMN: {"ancestors": [{"kind": "customer", "id": customer_id}]},
    }


def polar_source(
    token: str | None = None,
    organization_id: str = "default",
    resource_name: str = "polar_data",
    check_active: Callable[[], None] | None = None,
    http_client: Any = None,
):
    """Return a ``dlt`` resource yielding customer state, subscriptions, and benefit grants from Polar."""
    import dlt
    import httpx

    dlt_utils.require_document_sync_version(
        3, provider="Polar", feature="reads per-row node sets and structure"
    )

    resolved_token = token or os.getenv("POLAR_ORGANIZATION_TOKEN") or os.getenv("POLAR_ACCESS_TOKEN")
    if not resolved_token:
        raise ValueError(f"Polar access token required: {_EXTRA_HINT}")

    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def polar_data():
        owned_client = None if http_client is not None else httpx.Client(timeout=30.0)
        try:
            client = _PolarClient(http_client or owned_client, resolved_token, check_active)

            # 1. Fetch customers
            try:
                customers = _paginate_endpoint(client.list_customers)
                for cust in customers:
                    yield _render_customer_document(cust, organization_id)
            except PolarAPIError as err:
                logger.warning("Failed to fetch Polar customers: %s", err)

            # 2. Fetch subscriptions
            try:
                subscriptions = _paginate_endpoint(client.list_subscriptions)
                for sub in subscriptions:
                    yield _render_subscription_document(sub, organization_id)
            except PolarAPIError as err:
                logger.warning("Failed to fetch Polar subscriptions: %s", err)

            # 3. Fetch benefit grants
            try:
                grants = _paginate_endpoint(client.list_benefit_grants)
                for grant in grants:
                    yield _render_grant_document(grant, organization_id)
            except PolarAPIError as err:
                logger.warning("Failed to fetch Polar benefit grants: %s", err)

        finally:
            if owned_client is not None:
                owned_client.close()

    resource = polar_data()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "polar")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
