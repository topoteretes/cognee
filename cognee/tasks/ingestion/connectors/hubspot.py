"""HubSpot CRM connector for cognee: a ``dlt`` source turning CRM objects into memory.

One source connects to a HubSpot portal (via Private App Access Token) and yields
contacts, companies, deals, and engagement notes as flat document rows, meant to
be handed directly to :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import hubspot_source

    await cognee.remember(
        hubspot_source(
            access_token="pat-na1-xxxxxxxx",
            portal_id="12345678",
            include_deals=True,
            include_notes=True,
        ),
        dataset_name="hubspot",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``.
  Ids are prefixed by kind (``contact:<id>``, ``company:<id>``, ``deal:<id>``, ``note:<id>``).
  Rendering is deterministic and frozen: volatile timestamps live only in resource state.
* **Node set** ``hubspot:<portal_id>:<kind>`` scopes recall to HubSpot, to one portal,
  or to a specific entity type (``contacts``, ``companies``, ``deals``, ``notes``).
* **Incremental** by ``updatedAt`` timestamp stored in ``dlt.current.resource_state()``.
  On subsequent runs, only records modified since the previous cursor are fetched.
* **Forget-on-delete** via archiving checks or tombstone emissions.
* **Bounded and resumable.** On HTTP 429 rate limit or exhausted request budget,
  the source stops cleanly and records diagnostic metrics in ``cognee_sync_stats``.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import quote, urlparse

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
)

logger = logging.getLogger(__name__)

HUBSPOT_API_BASE = "https://api.hubapi.com"
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 100
DEFAULT_MAX_REQUESTS = 500


class HubSpotSourceError(RuntimeError):
    """Base error for HubSpot connector issues."""


class HubSpotAuthError(HubSpotSourceError):
    """Raised when authentication credentials (Private App Token) fail."""


class HubSpotRateLimitedError(HubSpotSourceError):
    """Raised when API rate limits (HTTP 429) are encountered."""


class HubSpotAPIError(HubSpotSourceError):
    """Raised for unexpected upstream API errors."""


class HubSpotClient:
    """HTTP client interface for HubSpot CRM API v3."""

    def __init__(
        self,
        access_token: str,
        *,
        portal_id: str | None = None,
        base_url: str = HUBSPOT_API_BASE,
        timeout: float = 30.0,
    ):
        if not access_token or not isinstance(access_token, str) or not access_token.strip():
            raise ValueError("HubSpot access_token must be a non-empty string.")
        self.access_token = access_token.strip()
        self.portal_id = str(portal_id).strip() if portal_id else None
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "cognee-hubspot-connector/1.0",
        }

    def fetch_portal_info(self) -> dict[str, Any]:
        """Fetch account/portal info to verify token and identify portal ID."""
        import urllib.request
        import json

        url = f"{self.base_url}/account-info/v3/details"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if not self.portal_id and "portalId" in data:
                    self.portal_id = str(data["portalId"])
                return data
        except Exception as exc:
            status = getattr(exc, "code", None)
            if status in (401, 403):
                raise HubSpotAuthError("Invalid or unauthorized HubSpot access token.") from exc
            if status == 429:
                raise HubSpotRateLimitedError("HubSpot rate limit exceeded (HTTP 429).") from exc
            logger.warning("Could not fetch portal details from account-info API: %s", exc)
            return {"portalId": self.portal_id or "default"}

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
        """Query HubSpot CRM v3 objects via search endpoint."""
        import urllib.request
        import urllib.error
        import json

        url = f"{self.base_url}/crm/v3/objects/{object_type}/search"
        payload: dict[str, Any] = {
            "limit": max(1, min(limit, MAX_PAGE_SIZE)),
            "filterGroups": filter_groups or [],
        }
        if properties:
            payload["properties"] = properties
        if after:
            payload["after"] = after
        if sorts:
            payload["sorts"] = sorts

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise HubSpotAuthError(f"HubSpot authentication failed for {object_type}.") from exc
            if exc.code == 429:
                raise HubSpotRateLimitedError(f"HubSpot rate limit hit on {object_type}.") from exc
            raise HubSpotAPIError(f"HubSpot API error {exc.code} for {object_type}: {exc.reason}") from exc
        except Exception as exc:
            raise HubSpotAPIError(f"Network error querying HubSpot {object_type}: {exc}") from exc


def build_hubspot_service(
    access_token: str,
    portal_id: str | None = None,
    base_url: str = HUBSPOT_API_BASE,
) -> HubSpotClient:
    """Factory helper to construct a configured HubSpotClient."""
    return HubSpotClient(access_token=access_token, portal_id=portal_id, base_url=base_url)


def _render_contact_row(record: dict[str, Any], portal_id: str) -> dict[str, Any]:
    contact_id = str(record.get("id"))
    props = record.get("properties") or {}
    first_name = props.get("firstname") or ""
    last_name = props.get("lastname") or ""
    name = f"{first_name} {last_name}".strip() or props.get("email") or f"Contact {contact_id}"
    job_title = props.get("jobtitle") or ""
    company_name = props.get("company") or ""
    lifecycle_stage = props.get("lifecyclestage") or ""

    lines = [f"Contact: {name}"]
    if job_title:
        lines.append(f"Title: {job_title}")
    if company_name:
        lines.append(f"Company: {company_name}")
    if lifecycle_stage:
        lines.append(f"Lifecycle Stage: {lifecycle_stage}")

    content = "\n".join(lines)
    url = f"https://app.hubspot.com/contacts/{portal_id}/record/0-1/{contact_id}" if portal_id else f"https://app.hubspot.com/contacts/0-1/{contact_id}"
    node_set = [f"hubspot:{portal_id}:contacts", f"hubspot:{portal_id}"]

    return {
        "id": f"contact:{contact_id}",
        "title": name,
        "content": content,
        "url": url,
        "_deleted": bool(record.get("archived", False)),
        NODE_SET_COLUMN: node_set,
    }


def _render_company_row(record: dict[str, Any], portal_id: str) -> dict[str, Any]:
    company_id = str(record.get("id"))
    props = record.get("properties") or {}
    name = props.get("name") or props.get("domain") or f"Company {company_id}"
    domain = props.get("domain") or ""
    industry = props.get("industry") or ""
    city = props.get("city") or ""
    country = props.get("country") or ""

    lines = [f"Company: {name}"]
    if domain:
        lines.append(f"Domain: {domain}")
    if industry:
        lines.append(f"Industry: {industry}")
    loc = ", ".join(filter(None, [city, country]))
    if loc:
        lines.append(f"Location: {loc}")

    content = "\n".join(lines)
    url = f"https://app.hubspot.com/contacts/{portal_id}/record/0-2/{company_id}" if portal_id else f"https://app.hubspot.com/contacts/0-2/{company_id}"
    node_set = [f"hubspot:{portal_id}:companies", f"hubspot:{portal_id}"]

    return {
        "id": f"company:{company_id}",
        "title": name,
        "content": content,
        "url": url,
        "_deleted": bool(record.get("archived", False)),
        NODE_SET_COLUMN: node_set,
    }


def _render_deal_row(record: dict[str, Any], portal_id: str) -> dict[str, Any]:
    deal_id = str(record.get("id"))
    props = record.get("properties") or {}
    name = props.get("dealname") or f"Deal {deal_id}"
    stage = props.get("dealstage") or ""
    amount = props.get("amount") or ""
    close_date = props.get("closedate") or ""

    lines = [f"Deal: {name}"]
    if stage:
        lines.append(f"Stage: {stage}")
    if amount:
        lines.append(f"Amount: {amount}")
    if close_date:
        lines.append(f"Target Close: {close_date}")

    content = "\n".join(lines)
    url = f"https://app.hubspot.com/contacts/{portal_id}/record/0-3/{deal_id}" if portal_id else f"https://app.hubspot.com/contacts/0-3/{deal_id}"
    node_set = [f"hubspot:{portal_id}:deals", f"hubspot:{portal_id}"]

    return {
        "id": f"deal:{deal_id}",
        "title": name,
        "content": content,
        "url": url,
        "_deleted": bool(record.get("archived", False)),
        NODE_SET_COLUMN: node_set,
    }


def _render_note_row(record: dict[str, Any], portal_id: str) -> dict[str, Any]:
    note_id = str(record.get("id"))
    props = record.get("properties") or {}
    body = props.get("hs_note_body") or ""
    # Strip HTML tags from note body if present
    clean_body = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)).strip()
    preview = clean_body[:80].strip() or f"Engagement Note {note_id}"

    lines = [f"Engagement Note: {preview}"]
    if clean_body:
        lines.append(f"Body: {clean_body}")

    content = "\n".join(lines)
    url = f"https://app.hubspot.com/contacts/{portal_id}/record/0-4/{note_id}" if portal_id else f"https://app.hubspot.com/contacts/0-4/{note_id}"
    node_set = [f"hubspot:{portal_id}:notes", f"hubspot:{portal_id}"]

    return {
        "id": f"note:{note_id}",
        "title": f"Note: {preview[:40]}",
        "content": content,
        "url": url,
        "_deleted": bool(record.get("archived", False)),
        NODE_SET_COLUMN: node_set,
    }


def _iter_object_rows(
    client: HubSpotClient,
    object_type: str,
    renderer: Callable[[dict[str, Any], str], dict[str, Any]],
    state: dict[str, Any],
    stats: dict[str, int],
    properties: list[str],
    portal_id: str,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_requests: int = DEFAULT_MAX_REQUESTS,
) -> Iterator[dict[str, Any]]:
    state_key = f"last_updated_{object_type}"
    last_cursor = state.get(state_key)
    filter_groups: list[dict[str, Any]] = []

    if last_cursor:
        filter_groups.append({
            "filters": [
                {
                    "propertyName": "hs_lastmodifieddate",
                    "operator": "GT",
                    "value": str(last_cursor),
                }
            ]
        })

    sorts = [{"propertyName": "hs_lastmodifieddate", "direction": "ASCENDING"}]
    after: str | None = None
    req_count = 0
    max_updated_seen = last_cursor

    while True:
        if req_count >= max_requests:
            logger.info("Reached maximum request budget (%d) for HubSpot %s.", max_requests, object_type)
            stats["failed_budget"] = stats.get("failed_budget", 0) + 1
            break

        try:
            req_count += 1
            res = client.search_objects(
                object_type=object_type,
                properties=properties + ["hs_lastmodifieddate"],
                after=after,
                limit=page_size,
                filter_groups=filter_groups,
                sorts=sorts,
            )
        except HubSpotRateLimitedError:
            stats["failed_rate_limit"] = stats.get("failed_rate_limit", 0) + 1
            break
        except HubSpotAuthError:
            stats["failed_auth"] = stats.get("failed_auth", 0) + 1
            break
        except Exception:
            stats["failed"] = stats.get("failed", 0) + 1
            break

        results = res.get("results") or []
        for item in results:
            updated_str = (item.get("properties") or {}).get("hs_lastmodifieddate")
            if updated_str and (max_updated_seen is None or updated_str > max_updated_seen):
                max_updated_seen = updated_str

            row = renderer(item, portal_id)
            if row.get("_deleted"):
                stats["deleted"] = stats.get("deleted", 0) + 1
            else:
                stats["synced"] = stats.get("synced", 0) + 1
            yield row

        paging = res.get("paging") or {}
        after = (paging.get("next") or {}).get("after")
        if not after or not results:
            break

    if max_updated_seen:
        state[state_key] = max_updated_seen


def hubspot_source(
    access_token: str | None = None,
    *,
    portal_id: str | None = None,
    service: HubSpotClient | None = None,
    resource_name: str | None = None,
    include_contacts: bool = True,
    include_companies: bool = True,
    include_deals: bool = True,
    include_notes: bool = True,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    check_active: Callable[[], bool] | None = None,
) -> Any:
    """Produce a ``dlt`` resource for HubSpot CRM entities.

    Configured with ``primary_key="id"``, ``write_disposition="merge"``, and
    deletion tombstones supported via ``_deleted`` boolean column.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError("The HubSpot connector requires dlt (a cognee core dependency).") from exc

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 1:
        raise RuntimeError(
            "HubSpot sync requires a Cognee build with table-scoped DLT document cleanup."
        )

    if service is None and not access_token:
        raise ValueError("hubspot_source requires either a service instance or an access_token.")

    client = service if service is not None else HubSpotClient(access_token=access_token or "", portal_id=portal_id)
    resolved_portal = client.portal_id or "default"

    if resource_name is None:
        digest = hashlib.sha256(f"{resolved_portal}".encode("utf-8")).hexdigest()[:8]
        resource_name = f"hubspot_{digest}"
    elif not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("resource_name must be a non-empty string.")

    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={
            "_deleted": {"data_type": "bool", "hard_delete": True},
            NODE_SET_COLUMN: {"data_type": "json", "nullable": True},
        },
    )
    def hubspot_crm_data():
        stats.update(
            synced=0,
            deleted=0,
            failed=0,
            failed_rate_limit=0,
            failed_budget=0,
            failed_auth=0,
        )
        state = dlt.current.resource_state()

        generators = []
        if include_contacts:
            generators.append(
                _iter_object_rows(
                    client,
                    "contacts",
                    _render_contact_row,
                    state,
                    stats,
                    ["firstname", "lastname", "email", "jobtitle", "company", "lifecyclestage"],
                    resolved_portal,
                    page_size=page_size,
                    max_requests=max_requests,
                )
            )
        if include_companies:
            generators.append(
                _iter_object_rows(
                    client,
                    "companies",
                    _render_company_row,
                    state,
                    stats,
                    ["name", "domain", "industry", "city", "country"],
                    resolved_portal,
                    page_size=page_size,
                    max_requests=max_requests,
                )
            )
        if include_deals:
            generators.append(
                _iter_object_rows(
                    client,
                    "deals",
                    _render_deal_row,
                    state,
                    stats,
                    ["dealname", "dealstage", "amount", "closedate"],
                    resolved_portal,
                    page_size=page_size,
                    max_requests=max_requests,
                )
            )
        if include_notes:
            generators.append(
                _iter_object_rows(
                    client,
                    "notes",
                    _render_note_row,
                    state,
                    stats,
                    ["hs_note_body"],
                    resolved_portal,
                    page_size=page_size,
                    max_requests=max_requests,
                )
            )

        def _combined():
            for gen in generators:
                yield from gen

        yield from dlt_utils.guarded_rows(_combined(), check_active)

    resource = hubspot_crm_data()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "hubspot")
    setattr(resource, PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
