"""Teable no-code database connector for cognee.

A ``dlt`` source that turns structured records, relational link fields, and no-code tables
from Teable (open-source Airtable alternative built on Postgres) into AI memory.
Supports teable.ai cloud and self-hosted instances with content-hash change detection,
cross-table relational links, and automatic deletion tombstones:

    import cognee
    from cognee.tasks.ingestion.connectors import teable_source

    await cognee.remember(
        teable_source(
            api_key="<personal access token>",
            base_url="https://app.teable.ai/api",
            table_ids=["tblXyz123"],
        ),
        dataset_name="crm_memory",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

Design
------
* **Scope**: Ingests structured tables and records from Teable:
  - **Tables & Records**: Field-name keyed records converted into structured markdown documents.
  - **Cross-Table Links**: Link fields (e.g., Contacts -> Companies) rendered as relational references
    so Cognee can construct cross-record graph edges.
* **Content-Hash Change Detection**:
  - Rather than relying solely on ``lastModifiedTime`` (which does not move when linked records are
    renamed, fields are renamed, or computed formulas update), each record is content-hashed via SHA-256.
  - Only new or modified records are re-yielded, saving downstream LLM entity extraction costs.
* **Forget-on-Delete & Tombstones**:
  - Unseen records from complete table listings and records returned by the Teable trash API
    (``GET /trash/items``) are emitted as ``_deleted=True`` tombstone rows under
    ``write_disposition="merge"``, enabling Cognee's orphan cleanup to remove purged entities.
* **Security & Privacy**:
  - Bearer tokens are masked in ``__repr__`` and stripped from exception messages.
  - User details (names, emails in ``userMap`` metadata) from trash/audit payloads are omitted
    to protect team PII in AI memory.
"""

import hashlib
import json
import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import httpx
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR

logger = get_logger("teable_connector")

DEFAULT_TEABLE_BASE_URL = "https://app.teable.ai/api"
DEFAULT_PAGE_SIZE = 100


class TeableAPIError(Exception):
    """Raised when the Teable REST API returns an error response."""

    def __init__(self, message: str, status_code: int | None = None, error_code: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class TeableAuthError(TeableAPIError):
    """Raised on authentication failures (HTTP 401 or invalid token)."""


class TeableRateLimitError(TeableAPIError):
    """Raised when Teable returns HTTP 429 rate limit exceeded."""


class TeableSettings(BaseSettings):
    """Configuration settings for Teable API connection."""

    teable_api_key: str | None = Field(default=None, description="Teable Personal Access Token")
    teable_access_token: str | None = Field(
        default=None, description="Teable Personal Access Token (alias)"
    )
    teable_base_url: str = Field(default=DEFAULT_TEABLE_BASE_URL, description="Teable API base URL")

    model_config = SettingsConfigDict(
        env_prefix="TEABLE_",
        env_file=".env",
        extra="ignore",
    )


def mask_token(token: str | None) -> str:
    """Mask a secret token, preserving only a short prefix and suffix."""
    if not token:
        return "<not-set>"
    if len(token) <= 8:
        return "***"
    return f"{token[:4]}...{token[-4:]}"


def compute_content_hash(content: str) -> str:
    """Compute a deterministic SHA-256 hash of rendered record content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class TeableClient:
    """HTTP client for Teable REST API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_TEABLE_BASE_URL,
        http_client: httpx.Client | None = None,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("TeableClient requires an api_key.")
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self._http_client = http_client
        self._owns_http_client = http_client is None

    def __repr__(self) -> str:
        return f"TeableClient(base_url='{self.base_url}', token='{mask_token(self.api_key)}')"

    def _get_client(self) -> httpx.Client:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.Client(
                timeout=30.0,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
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

    def _request(
        self, method: str, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make an authenticated HTTP request to Teable API with rate-limit detection."""
        client = self._get_client()
        url = f"{self.base_url}{path}"
        try:
            response = client.request(method, url, params=params)
        except httpx.RequestError as exc:
            raise TeableAPIError(f"Teable API connection error: {exc}") from exc

        if response.status_code == 401:
            raise TeableAuthError(
                "Teable authentication failed. Check TEABLE_API_KEY.", status_code=401
            )
        if response.status_code == 429:
            raise TeableRateLimitError("Teable API rate limit exceeded.", status_code=429)
        if response.is_error:
            error_data = {}
            try:
                error_data = response.json()
            except ValueError:
                error_data = {}
            msg = error_data.get("message") or response.text
            code = error_data.get("code")
            raise TeableAPIError(
                f"Teable API error ({response.status_code}): {msg}",
                status_code=response.status_code,
                error_code=code,
            )

        try:
            return response.json()
        except ValueError as exc:
            raise TeableAPIError(f"Teable API returned invalid JSON: {exc}") from exc

    def get_table_records(
        self,
        table_id: str,
        field_key_type: str = "name",
        take: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Fetch a paginated page of records from a Teable table."""
        params: dict[str, Any] = {
            "fieldKeyType": field_key_type,
            "take": take,
        }
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", f"/table/{table_id}/record", params=params)

    def get_table_fields(self, table_id: str) -> list[dict[str, Any]]:
        """Fetch field definitions for a Teable table."""
        res = self._request("GET", f"/table/{table_id}/field")
        if isinstance(res, list):
            return res
        return res.get("fields", [])

    def get_trash_items(self, table_id: str) -> list[dict[str, Any]]:
        """Fetch items currently in trash for a given table to reconcile deletions."""
        params = {
            "resourceId": table_id,
            "resourceType": "table",
        }
        try:
            res = self._request("GET", "/trash/items", params=params)
            if isinstance(res, list):
                return res
            return res.get("trashItems", [])
        except TeableAPIError:
            # Trash endpoint is optional; ignore if unsupported on self-hosted
            return []


def format_cell_value(val: Any) -> tuple[str, list[dict[str, str]]]:
    """Format a cell value and extract linked records."""
    links: list[dict[str, str]] = []

    if val is None:
        return "N/A", links

    # Single link field: {"id": "rec...", "title": "..."}
    if isinstance(val, dict) and "id" in val and ("title" in val or "name" in val):
        rec_id = val["id"]
        title = val.get("title") or val.get("name") or rec_id
        links.append({"id": rec_id, "title": str(title)})
        return str(title), links

    # Multiple link field: [{"id": "rec...", "title": "..."}, ...]
    if isinstance(val, list):
        formatted_items = []
        is_link_list = False
        for item in val:
            if isinstance(item, dict) and "id" in item and ("title" in item or "name" in item):
                is_link_list = True
                rec_id = item["id"]
                title = item.get("title") or item.get("name") or rec_id
                links.append({"id": rec_id, "title": str(title)})
                formatted_items.append(str(title))
            else:
                formatted_items.append(str(item))
        if is_link_list:
            return ", ".join(formatted_items), links
        return ", ".join(formatted_items), links

    if isinstance(val, bool):
        return "Yes" if val else "No", links

    if isinstance(val, (int, float)):
        return str(val), links

    return str(val).strip(), links


def render_teable_record(
    record: dict[str, Any],
    table_id: str,
    table_name: str | None = None,
    base_url: str = "",
) -> dict[str, Any]:
    """Render a Teable record into a structured Cognee document with cross-table relations."""
    record_id = record.get("id", "")
    doc_id = f"teable:{table_id}:{record_id}"

    # Handle deletion flag if record is marked deleted
    if record.get("is_deleted") is True:
        return {"id": doc_id, "_deleted": True}

    fields = record.get("fields", {})
    t_name = table_name or table_id
    primary_name = (
        record.get("name") or fields.get("Name") or fields.get("Title") or f"Record {record_id[:8]}"
    )

    properties_lines = []
    all_links = []

    for field_name, field_val in sorted(fields.items()):
        val_str, links = format_cell_value(field_val)
        properties_lines.append(f"- **{field_name}**: {val_str}")
        if links:
            all_links.extend(links)

    properties_md = "\n".join(properties_lines) if properties_lines else "- *No fields recorded.*"

    relations_lines = []
    for link in all_links:
        relations_lines.append(f"- [{link['title']}](teable:{link['id']})")
    relations_md = (
        "\n".join(relations_lines) if relations_lines else "- *No cross-table relations.*"
    )

    content = (
        f"# {t_name}: {primary_name}\n\n"
        f"**Table**: {t_name}\n"
        f"**Record ID**: `{record_id}`\n\n"
        f"## Properties\n{properties_md}\n\n"
        f"## Linked Relations\n{relations_md}"
    )

    content_hash = compute_content_hash(content)

    return {
        "id": doc_id,
        "title": f"{t_name}: {primary_name}",
        "content": content,
        "table_id": table_id,
        "table_name": t_name,
        "content_hash": content_hash,
        "created_time": record.get("createdTime"),
        "last_modified_time": record.get("lastModifiedTime"),
        "url": f"{base_url}/table/{table_id}/{record_id}" if base_url else "",
    }


def iter_teable_records(
    client: TeableClient,
    table_ids: Sequence[str],
    table_names: dict[str, str] | None = None,
    state: dict[str, Any] | None = None,
    stats: dict[str, int] | None = None,
    check_active: Callable[[], None] | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> Iterator[dict[str, Any]]:
    """Yield changed records and tombstones for Teable tables using content hashing.

    Pure Python generator decoupled from DLT runtime for fast testing.
    """
    if state is None:
        state = {}
    if stats is None:
        stats = {}
    if table_names is None:
        table_names = {}

    committed_hashes: dict[str, str] = state.setdefault("content_hashes", {})
    known_records: dict[str, list[str]] = state.setdefault("known_records", {})

    for table_id in table_ids:
        if check_active:
            check_active()

        table_name = table_names.get(table_id) or table_id
        current_table_record_ids: set[str] = set()
        pagination_cursor: str | None = None

        while True:
            if check_active:
                check_active()

            try:
                res = client.get_table_records(
                    table_id=table_id,
                    field_key_type="name",
                    take=page_size,
                    cursor=pagination_cursor,
                )
            except TeableRateLimitError:
                stats["rate_limited"] = stats.get("rate_limited", 0) + 1
                logger.warning(
                    "Teable API rate limit reached during table %s sync; resuming next run.",
                    table_id,
                )
                break
            except TeableAPIError as exc:
                stats["errors"] = stats.get("errors", 0) + 1
                logger.error("Teable table %s error: %s", table_id, exc)
                raise

            records = res.get("records", [])
            for rec in records:
                rec_id = rec.get("id")
                if not rec_id:
                    continue
                current_table_record_ids.add(rec_id)

                row = render_teable_record(
                    record=rec,
                    table_id=table_id,
                    table_name=table_name,
                    base_url=client.base_url,
                )
                new_hash = row["content_hash"]
                previous_hash = committed_hashes.get(rec_id)

                # Content-hash comparison: yield only when content actually changed
                if previous_hash != new_hash:
                    committed_hashes[rec_id] = new_hash
                    stats["synced_records"] = stats.get("synced_records", 0) + 1
                    yield row
                else:
                    stats["skipped_unchanged"] = stats.get("skipped_unchanged", 0) + 1

            extra = res.get("extra", {})
            pagination_cursor = extra.get("nextCursor")
            if not pagination_cursor or not records:
                break

        # Check for deleted records missing from the complete listing
        previous_table_ids = set(known_records.get(table_id, []))
        missing_ids = previous_table_ids - current_table_record_ids

        # Check trash items for explicit confirmation
        trash_items = client.get_trash_items(table_id)
        for item in trash_items:
            t_rec_id = item.get("resourceId") or item.get("id")
            if t_rec_id and t_rec_id in previous_table_ids:
                missing_ids.add(t_rec_id)

        for del_id in missing_ids:
            stats["deleted_tombstones"] = stats.get("deleted_tombstones", 0) + 1
            committed_hashes.pop(del_id, None)
            yield {
                "id": f"teable:{table_id}:{del_id}",
                "_deleted": True,
            }

        # Update known records in state for future deletion reconciliation
        known_records[table_id] = list(current_table_record_ids)


def teable_source(
    *,
    api_key: str | None = None,
    base_url: str = DEFAULT_TEABLE_BASE_URL,
    table_ids: Sequence[str],
    table_names: dict[str, str] | None = None,
    resource_name: str | None = None,
    service: TeableClient | None = None,
    check_active: Callable[[], None] | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
):
    """Return a ``dlt`` resource yielding Teable documents and tombstones for ``remember``.

    Args:
        api_key: Teable Personal Access Token. Defaults to TEABLE_API_KEY env var.
        base_url: Teable API base URL (defaults to https://app.teable.ai/api).
        table_ids: Sequence of Teable table IDs to ingest.
        table_names: Optional dictionary mapping table_id to human-readable table name.
        resource_name: Stable connection identifier for cognee dataset staging.
        service: Optional pre-configured TeableClient (useful for testing and dependency injection).
        check_active: Optional callback to check credential freshness before each batch.
        page_size: Maximum records per page returned by Teable REST endpoints.

    Returns:
        A ``dlt`` resource configured with primary_key="id", write_disposition="merge",
        and tombstone support (_deleted=True).
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError("The Teable connector requires dlt (a cognee core dependency).") from exc

    resolved_token = (
        api_key or TeableSettings().teable_api_key or TeableSettings().teable_access_token
    )
    if not service and not resolved_token:
        raise ValueError("teable_source requires api_key or TEABLE_API_KEY env var.")

    client = service or TeableClient(
        api_key=resolved_token,  # type: ignore
        base_url=base_url,
    )

    if not table_ids:
        raise ValueError("teable_source requires at least one table_id in table_ids.")

    if resource_name is None:
        resource_name = f"teable_{table_ids[0]}"

    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
    )
    def teable_records():
        dlt_state = dlt.current.resource_state()
        yield from dlt_utils.guarded_rows(
            iter_teable_records(
                client=client,
                table_ids=table_ids,
                table_names=table_names,
                state=dlt_state,
                stats=stats,
                check_active=check_active,
                page_size=page_size,
            ),
            check_active=check_active,
        )

    resource = teable_records()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "teable")
    setattr(resource, PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
