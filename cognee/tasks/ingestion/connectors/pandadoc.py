"""PandaDoc data-source connector for cognee: turn proposals and contracts into memory.

One source ingests sales documents, quotes, agreements, and contracts from PandaDoc.
It extracts contract metadata, recipient signing states, commercial terms, pricing
tables, and custom fields into structured markdown documents so AI agents can answer:
"What did we quote Acme for the annual plan, and did they sign?"

Quickstart::

    import cognee
    from cognee.tasks.ingestion.connectors import pandadoc_source

    await cognee.add(
        pandadoc_source(
            api_key="your_pandadoc_api_key",
            statuses=["completed", "sent"],
        )
    )
    await cognee.cognify()

Design & Technical Principles
-----------------------------
* **Rich Commercial Intelligence**: Extracts document details, grand totals,
  currencies, recipient sign status, line-item pricing tables (item, price, qty,
  discount, total), and custom merge tokens.
* **Deterministic Output**: Markdown sections, pricing tables, and recipient lists
  are rendered deterministically so repeated runs produce byte-identical content
  and avoid creating duplicate graph nodes or redundant embeddings.
* **Status Name Mapping**: The PandaDoc REST API requires integer codes for status
  filters (`status=0` for draft, `status=1` for sent, `status=2` for completed),
  while returning string names in document payloads. The connector seamlessly
  translates friendly names (`"completed"`, `"sent"`, `"draft"`) to numeric codes.
* **Incremental Synchronization**: Uses `modified_from` ISO 8601 cursors. Because
  renaming, sending, or signing moves `date_modified`, status transitions and
  revisions are automatically detected on the next sync.
* **Forget-on-Delete Lifecycle**: PandaDoc's API exposes a `deleted=true` filter
  returning documents removed in the workspace. The connector emits `_deleted=True`
  tombstones so Cognee's `orphan_cleanup` clears deleted proposals from memory.
* **Security & Token Masking**: API keys are masked in `__repr__` and custom
  exceptions so secrets never leak into logs or telemetry.
* **Scoping**: Documents are scoped under ``cognee_node_set = ["pandadoc:documents"]``.
"""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.pandadoc.com/public/v1"
DEFAULT_PAGE_SIZE = 100
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 3

# Mapping of friendly and standard status names to PandaDoc numeric filter codes
STATUS_NAME_TO_CODE: dict[str, int] = {
    "document.draft": 0,
    "draft": 0,
    "document.sent": 1,
    "sent": 1,
    "document.completed": 2,
    "completed": 2,
    "document.uploaded": 3,
    "uploaded": 3,
    "document.error": 4,
    "error": 4,
    "document.viewed": 5,
    "viewed": 5,
    "document.waiting_approval": 6,
    "waiting_approval": 6,
    "document.approved": 7,
    "approved": 7,
    "document.rejected": 8,
    "rejected": 8,
    "document.waiting_pay": 9,
    "waiting_pay": 9,
    "document.paid": 10,
    "paid": 10,
    "document.voided": 11,
    "voided": 11,
    "document.declined": 12,
    "declined": 12,
    "document.external_review": 13,
    "external_review": 13,
}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class PandaDocSettings(BaseSettings):
    """Configuration settings for PandaDoc ingestion."""

    pandadoc_api_key: str = Field(
        default="",
        validation_alias="PANDADOC_API_KEY",
        description="PandaDoc API key or Sandbox API key.",
    )
    pandadoc_base_url: str = Field(
        default=DEFAULT_BASE_URL,
        validation_alias="PANDADOC_BASE_URL",
        description="Base URL for PandaDoc REST API v1.",
    )
    pandadoc_statuses: list[str] = Field(
        default_factory=list,
        validation_alias="PANDADOC_STATUSES",
        description="Optional list of document statuses to filter (e.g. ['sent', 'completed']).",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class PandaDocSourceError(RuntimeError):
    """Base error for PandaDoc connector operations."""


class PandaDocAPIError(PandaDocSourceError):
    """PandaDoc REST API returned an error or could not be reached."""


class PandaDocAuthError(PandaDocSourceError):
    """Authentication failed (HTTP 401 or 403)."""


class PandaDocRateLimitedError(PandaDocSourceError):
    """PandaDoc returned HTTP 429 Too Many Requests."""

    def __init__(self, retry_after: int | None = None):
        super().__init__("PandaDoc API rate limit exceeded")
        self.retry_after = retry_after


class PandaDocResourceNotFoundError(PandaDocAPIError):
    """Requested document was not found or has been purged (HTTP 404)."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class PandaDocClient:
    """Synchronous REST API v1 client for PandaDoc with key masking."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_RETRIES,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("PandaDoc api_key must not be empty")

        clean_url = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
        if not clean_url.startswith(("http://", "https://")):
            clean_url = f"https://{clean_url}"

        self.api_key = api_key.strip()
        self.base_url = clean_url
        self.timeout = timeout
        self.max_retries = max_retries

    def __repr__(self) -> str:
        return f"PandaDocClient(base_url={self.base_url!r}, api_key='***', timeout={self.timeout})"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"API-Key {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "cognee-pandadoc-connector/1.0",
        }

    def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if params:
            clean_params = {k: v for k, v in params.items() if v is not None}
            if clean_params:
                url = f"{url}?{urllib.parse.urlencode(clean_params)}"

        req = urllib.request.Request(url, headers=self._headers(), method="GET")

        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    if not raw:
                        return None
                    return json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise PandaDocAuthError(
                        f"PandaDoc authentication failed (HTTP {exc.code})"
                    ) from None
                if exc.code == 404:
                    raise PandaDocResourceNotFoundError(
                        f"PandaDoc resource not found (HTTP 404): {endpoint}"
                    ) from None
                if exc.code == 429:
                    retry_after = None
                    try:
                        retry_after = int(exc.headers.get("Retry-After", 1))
                    except (TypeError, ValueError):
                        pass
                    if attempt < self.max_retries:
                        continue
                    raise PandaDocRateLimitedError(retry_after=retry_after) from None
                if attempt == self.max_retries:
                    raise PandaDocAPIError(
                        f"PandaDoc API error (HTTP {exc.code}) on {endpoint}"
                    ) from None
            except urllib.error.URLError as exc:
                if attempt == self.max_retries:
                    raise PandaDocAPIError(
                        f"Network failure connecting to PandaDoc: {exc.reason}"
                    ) from None

        raise PandaDocAPIError(f"Exceeded retries requesting {endpoint}")

    def list_documents(
        self,
        modified_from: str | None = None,
        status: int | None = None,
        deleted: bool = False,
        page: int = 1,
        count: int = DEFAULT_PAGE_SIZE,
        tag: str | None = None,
        folder_uuid: str | None = None,
    ) -> dict[str, Any]:
        """List documents matching filter criteria."""
        params: dict[str, Any] = {
            "page": page,
            "count": count,
        }
        if modified_from:
            params["modified_from"] = modified_from
        if status is not None:
            params["status"] = status
        if deleted:
            params["deleted"] = "true"
        if tag:
            params["tag"] = tag
        if folder_uuid:
            params["folder_uuid"] = folder_uuid

        data = self._request("documents", params=params)
        return data if isinstance(data, dict) else {"results": [], "total": 0}

    def get_document_details(self, document_id: str) -> dict[str, Any]:
        """Fetch complete document details including recipients, tokens, and pricing."""
        data = self._request(f"documents/{document_id}/details")
        return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _render_document_markdown(details: dict[str, Any]) -> str:
    """Render PandaDoc document details into structured, deterministic markdown."""
    name = (details.get("name") or "Untitled Document").strip()
    doc_id = details.get("id", "unknown")
    status = details.get("status", "unknown")
    date_created = details.get("date_created") or "N/A"
    date_modified = details.get("date_modified") or "N/A"
    date_sent = details.get("date_sent") or "N/A"
    date_completed = details.get("date_completed") or "N/A"

    grand_total = details.get("grand_total") or {}
    total_amount = grand_total.get("amount") or details.get("total") or "0.00"
    currency = grand_total.get("currency") or "USD"

    lines = [
        f"# PandaDoc: {name}",
        "",
        "## Document Overview",
        f"- **Document ID**: `{doc_id}`",
        f"- **Status**: `{status}`",
        f"- **Grand Total**: `{currency} {total_amount}`",
        f"- **Created**: {date_created}",
        f"- **Last Modified**: {date_modified}",
        f"- **Date Sent**: {date_sent}",
        f"- **Date Completed / Signed**: {date_completed}",
    ]

    tags = details.get("tags") or []
    if tags:
        lines.append(f"- **Tags**: {', '.join(sorted(str(t) for t in tags))}")

    lines.append("")

    # Recipients
    recipients = details.get("recipients") or []
    if recipients:
        lines.append("## Parties & Recipients")
        # Deterministically sort by email
        sorted_recipients = sorted(
            recipients,
            key=lambda r: (r.get("email") or "", r.get("id") or ""),
        )
        for r in sorted_recipients:
            full_name = f"{r.get('first_name', '')} {r.get('last_name', '')}".strip() or "Unnamed"
            email = r.get("email") or "no-email"
            role = r.get("role") or "Recipient"
            has_completed = "Yes (Signed)" if r.get("has_completed") else "Pending"
            lines.append(
                f"- **{full_name}** (`{email}`) - Role: `{role}` | Signed: `{has_completed}`"
            )
        lines.append("")

    # Pricing Tables
    pricing = details.get("pricing") or {}
    tables = pricing.get("tables") or []
    if tables:
        lines.append("## Pricing & Commercial Terms")
        for table in tables:
            table_name = table.get("name") or "Pricing Items"
            lines.append(f"### {table_name}")
            lines.append("| Item | Description | Qty | Unit Price | Discount | Total |")
            lines.append("|---|---|---|---|---|---|")
            items = table.get("items") or []
            for itm in sorted(items, key=lambda i: (i.get("name") or "", str(i.get("id") or ""))):
                item_name = (itm.get("name") or "Item").replace("|", "/")
                desc = (itm.get("description") or "-").replace("|", "/")
                qty = itm.get("qty", 1)
                price = itm.get("price", "0.00")
                discount = itm.get("discount", "0%")
                subtotal = itm.get("subtotal") or itm.get("total") or "0.00"
                lines.append(
                    f"| {item_name} | {desc} | {qty} | {price} | {discount} | {subtotal} |"
                )
            lines.append("")

    # Fields & Tokens
    fields = details.get("fields") or []
    tokens = details.get("tokens") or []
    if fields or tokens:
        lines.append("## Contract Fields & Tokens")
        for f in sorted(fields, key=lambda x: str(x.get("name") or "")):
            fname = f.get("name") or f.get("title")
            fval = f.get("value")
            if fname and fval:
                lines.append(f"- **{fname}**: {fval}")
        for t in sorted(tokens, key=lambda x: str(x.get("name") or "")):
            tname = t.get("name")
            tval = t.get("value")
            if tname and tval:
                lines.append(f"- **[{tname}]**: {tval}")
        lines.append("")

    # Metadata
    metadata = details.get("metadata") or {}
    if metadata:
        lines.append("## Custom Metadata")
        for mk in sorted(metadata.keys()):
            lines.append(f"- **`{mk}`**: {metadata[mk]}")
        lines.append("")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Stream Iterator
# ---------------------------------------------------------------------------
def _iter_pandadoc_rows(
    client: PandaDocClient,
    status_codes: list[int] | None,
    tag: str | None,
    folder_uuid: str | None,
    state: dict[str, Any],
    stats: dict[str, int],
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Iterate through PandaDoc documents and yield document rows and tombstones."""
    cursor: str | None = state.get("cursor")
    seen_ids: set[str] = set(state.get("seen_ids", []))
    max_date_seen: str | None = cursor
    current_seen_at_max: set[str] = set(seen_ids)

    # If status_codes is provided, query each status; otherwise query all
    codes_to_query = status_codes if status_codes else [None]

    for status_code in codes_to_query:
        page = 1
        while True:
            if check_active is not None:
                check_active()

            try:
                res = client.list_documents(
                    modified_from=cursor,
                    status=status_code,
                    deleted=False,
                    page=page,
                    count=DEFAULT_PAGE_SIZE,
                    tag=tag,
                    folder_uuid=folder_uuid,
                )
            except PandaDocRateLimitedError:
                logger.warning("PandaDoc rate limit hit. Halting listing cleanly.")
                stats["rate_limited"] = stats.get("rate_limited", 0) + 1
                break

            results = res.get("results") or []
            if not results:
                break

            for doc in results:
                doc_id = doc.get("id")
                if not doc_id:
                    continue

                stats["scanned"] = stats.get("scanned", 0) + 1
                date_modified = doc.get("date_modified") or ""

                if cursor and date_modified:
                    if date_modified < cursor:
                        continue
                    if date_modified == cursor and doc_id in seen_ids:
                        continue

                # Fetch complete document details
                try:
                    details = client.get_document_details(doc_id)
                except PandaDocResourceNotFoundError:
                    # Document was deleted before details could be fetched
                    continue
                except PandaDocRateLimitedError:
                    logger.warning("PandaDoc rate limit hit during details fetch.")
                    stats["rate_limited"] = stats.get("rate_limited", 0) + 1
                    break

                rendered_content = _render_document_markdown(details)
                doc_name = details.get("name") or doc.get("name") or "Document"
                web_url = f"https://app.pandadoc.com/a/#/documents/{doc_id}"

                yield {
                    "id": f"pandadoc:doc:{doc_id}",
                    "title": f"PandaDoc: {doc_name}",
                    "content": rendered_content,
                    "url": web_url,
                    "_deleted": False,
                    "cognee_node_set": ["pandadoc:documents"],
                    "metadata": {
                        "source": "pandadoc",
                        "document_id": doc_id,
                        "status": details.get("status") or doc.get("status"),
                        "date_created": details.get("date_created"),
                        "date_modified": date_modified,
                        "date_completed": details.get("date_completed"),
                    },
                }
                stats["emitted"] = stats.get("emitted", 0) + 1

                if not max_date_seen or date_modified > max_date_seen:
                    max_date_seen = date_modified
                    current_seen_at_max = {doc_id}
                elif date_modified == max_date_seen:
                    current_seen_at_max.add(doc_id)

            if len(results) < DEFAULT_PAGE_SIZE:
                break
            page += 1

    # Deletions: query deleted documents since cursor
    page = 1
    while True:
        if check_active is not None:
            check_active()

        try:
            del_res = client.list_documents(
                modified_from=cursor,
                deleted=True,
                page=page,
                count=DEFAULT_PAGE_SIZE,
            )
        except PandaDocRateLimitedError:
            break

        del_results = del_res.get("results") or []
        if not del_results:
            break

        for ddoc in del_results:
            did = ddoc.get("id")
            if not did:
                continue
            yield {
                "id": f"pandadoc:doc:{did}",
                "_deleted": True,
            }
            stats["deleted"] = stats.get("deleted", 0) + 1

        if len(del_results) < DEFAULT_PAGE_SIZE:
            break
        page += 1

    # Update cursor state
    if max_date_seen:
        state["cursor"] = max_date_seen
        state["seen_ids"] = list(current_seen_at_max)


# ---------------------------------------------------------------------------
# Public DLT Source Factory
# ---------------------------------------------------------------------------
def pandadoc_source(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    statuses: list[str] | None = None,
    tag: str | None = None,
    folder_uuid: str | None = None,
    resource_name: str | None = None,
    client: PandaDocClient | None = None,
    check_active: Callable[[], None] | None = None,
):
    """Return a dlt resource that yields PandaDoc documents and proposals for ``remember``.

    Args:
        api_key: PandaDoc API Key or Sandbox Key. Defaults to ``PANDADOC_API_KEY`` from .env.
        base_url: Base URL of the PandaDoc REST API. Defaults to ``https://api.pandadoc.com/public/v1``.
        statuses: Optional list of document statuses (e.g. ``['sent', 'completed']``).
        tag: Optional tag to filter documents by.
        folder_uuid: Optional folder UUID to restrict document ingestion.
        resource_name: Stable resource table name (defaults to ``pandadoc_documents``).
        client: Optional pre-constructed ``PandaDocClient`` for custom transport or testing.
        check_active: Optional health/authorization callback called before network requests.

    Returns:
        A dlt resource configured with ``primary_key="id"``, ``write_disposition="merge"``,
        and ``_deleted`` tombstone tracking.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError(
            "The PandaDoc connector requires dlt. Install cognee dependencies."
        ) from exc

    if client is None:
        settings = PandaDocSettings()
        key = api_key or settings.pandadoc_api_key
        url = base_url or settings.pandadoc_base_url
        client = PandaDocClient(api_key=key, base_url=url)
        raw_statuses = statuses or settings.pandadoc_statuses or None
    else:
        raw_statuses = statuses

    # Map status strings to numeric codes
    status_codes: list[int] | None = None
    if raw_statuses:
        status_codes = []
        for s in raw_statuses:
            s_clean = s.strip().lower()
            if s_clean in STATUS_NAME_TO_CODE:
                status_codes.append(STATUS_NAME_TO_CODE[s_clean])
            elif s_clean.isdigit():
                status_codes.append(int(s_clean))
            else:
                logger.warning("Unknown PandaDoc status name '%s', skipping filter.", s)

    table_name = resource_name or "pandadoc_documents"
    stats: dict[str, int] = {}

    @dlt.resource(
        name=table_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def pandadoc_documents():
        stats.clear()
        stats.update(scanned=0, emitted=0, deleted=0)
        state = dlt.current.resource_state()
        yield from _iter_pandadoc_rows(
            client=client,
            status_codes=status_codes,
            tag=tag,
            folder_uuid=folder_uuid,
            state=state,
            stats=stats,
            check_active=check_active,
        )

    resource = pandadoc_documents()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "pandadoc")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, table_name)
    resource.cognee_sync_stats = stats
    return resource
