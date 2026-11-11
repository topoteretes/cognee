"""Figma data-source connector for cognee: a ``dlt`` source that turns Figma files into memory.

One source connects to one or more Figma files. It yields documents for files and pages,
including text elements, component annotations, comments, and hierarchical relationships
using ``cognee_structure``, and is meant to be handed directly to :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import figma_source

    await cognee.remember(
        figma_source(
            resource_name="figma_designs",
            access_token="<personal access token or oauth token>",
            file_keys=["abcdef1234567890"],
        ),
        dataset_name="figma",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
"""

import re
import time
from collections.abc import Callable, Iterator
from typing import Any

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_COLUMN_HINTS,
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
    STRUCTURE_COLUMN,
)

logger = get_logger("figma_connector")

BASE_URL = "https://api.figma.com"
DEFAULT_MAX_REQUESTS = 1500
_TIMEOUT = (10, 30)
_RETRIES = 3
_RETRY_WAIT_BUDGET_SECONDS = 60

# Node types containing useful semantic design information
_SEMANTIC_CONTAINER_TYPES = {"CANVAS", "FRAME", "SECTION", "GROUP", "COMPONENT", "COMPONENT_SET"}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class FigmaError(RuntimeError):
    """Base class for Figma connector errors."""


class FigmaAPIError(FigmaError):
    """An error response from Figma with status code and endpoint (tokens and IDs redacted)."""

    def __init__(self, status: int, endpoint: str, detail: str = ""):
        answer = f"HTTP {status}" if status else "no answer"
        super().__init__(f"Figma {endpoint}: {answer}")
        self.status = status
        self.endpoint = endpoint
        self.detail = detail


class FigmaAuthError(FigmaAPIError):
    """Figma rejected the access token (HTTP 401/403)."""

    def __init__(self, endpoint: str, detail: str = ""):
        super().__init__(401, endpoint, detail)


class FigmaRateLimitedError(FigmaAPIError):
    """Figma returned HTTP 429 Too Many Requests."""

    def __init__(self, endpoint: str, detail: str = ""):
        super().__init__(429, endpoint, detail)


class FigmaNotFoundError(FigmaAPIError):
    """The requested Figma file was not found (HTTP 404)."""

    def __init__(self, endpoint: str, detail: str = ""):
        super().__init__(404, endpoint, detail)


class FigmaBudgetExceededError(FigmaError):
    """The connector run spent its max_requests quota."""


# ---------------------------------------------------------------------------
# Transport & Client
# ---------------------------------------------------------------------------
def _endpoint(path: str) -> str:
    """Mask dynamic identifiers in endpoints for safe logging."""
    return re.sub(r"/files/[^/?]+", "/files/{key}", path)


class FigmaClient:
    """Synchronous REST client for Figma API with rate limit, retry, and token safety."""

    def __init__(
        self,
        access_token: str | Callable[[], str],
        *,
        auth_type: str = "auto",
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._token_provider = access_token
        self._auth_type = auth_type
        self._sleep = sleep

        # Validate initial token if string
        if isinstance(access_token, str):
            token = access_token.strip()
            if (
                not token
                or not token.isascii()
                or not token.isprintable()
                or any(char.isspace() for char in token)
            ):
                raise ValueError("The Figma access token is empty or has an invalid format")

    def __repr__(self) -> str:
        return "FigmaClient(access_token=<redacted>)"

    def _get_token(self) -> str:
        token = self._token_provider() if callable(self._token_provider) else self._token_provider
        if not isinstance(token, str) or not token.strip():
            raise ValueError("The Figma access token is empty or invalid")
        token = token.strip()
        if not token.isascii() or not token.isprintable() or any(char.isspace() for char in token):
            raise ValueError("The Figma access token contains invalid characters")
        return token

    def _headers(self, token: str) -> dict[str, str]:
        if self._auth_type == "pat" or (self._auth_type == "auto" and token.startswith("figd_")):
            return {"X-Figma-Token": token, "Accept": "application/json"}
        return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        check_active: Callable[[], None] | None = None,
        stats: dict[str, int] | None = None,
        max_requests: int | None = None,
    ) -> dict:
        """Execute a request against the Figma API, retrying transient errors."""
        import requests

        if stats is not None and max_requests is not None:
            if stats.get("requests", 0) >= max_requests:
                raise FigmaBudgetExceededError("Figma request budget exhausted")
            stats["requests"] = stats.get("requests", 0) + 1

        token = self._get_token()
        headers = self._headers(token)
        endpoint_name = _endpoint(path)
        slept_seconds = 0
        response = None

        for attempt in range(_RETRIES):
            if check_active is not None:
                check_active()
            if attempt > 0:
                delay = 2 ** (attempt - 1)
                if slept_seconds + delay > _RETRY_WAIT_BUDGET_SECONDS:
                    raise FigmaAPIError(0, endpoint_name, "Retry wait budget exhausted")
                slept_seconds += delay
                self._sleep(delay)

            try:
                response = requests.request(
                    method,
                    f"{BASE_URL}{path}",
                    params=params,
                    headers=headers,
                    timeout=_TIMEOUT,
                )
            except requests.RequestException:
                response = None
                continue

            if response.status_code < 500:
                break

        if response is None:
            raise FigmaAPIError(0, endpoint_name, "No response from Figma")

        return self._parse(response, endpoint_name)

    def _parse(self, response: Any, endpoint_name: str) -> dict:
        status = response.status_code
        try:
            body = response.json()
        except ValueError:
            body = {}

        detail = ""
        if isinstance(body, dict):
            detail = str(body.get("err") or body.get("message") or "")

        if status in (401, 403):
            raise FigmaAuthError(endpoint_name, detail)
        if status == 404:
            raise FigmaNotFoundError(endpoint_name, detail)
        if status == 429:
            raise FigmaRateLimitedError(endpoint_name, detail)
        if status != 200:
            raise FigmaAPIError(status, endpoint_name, detail)
        if not isinstance(body, dict):
            raise FigmaAPIError(status, endpoint_name, "Expected JSON object")

        return body


def build_figma_service(access_token: str) -> FigmaClient:
    """Build a default FigmaClient from an access token string."""
    return FigmaClient(access_token)


# ---------------------------------------------------------------------------
# Node Tree Traversal & Document Rendering
# ---------------------------------------------------------------------------
def _extract_page_content(
    node: dict[str, Any],
    components_map: dict[str, Any],
    comments_by_node: dict[str, list[dict]],
    lines: list[str],
    current_frame: str | None = None,
) -> None:
    """Recursively collect text elements, component annotations, and node comments."""
    node_type = node.get("type", "")
    node_name = node.get("name", "").strip()
    node_id = str(node.get("id", ""))

    # Frame/Section boundary
    if node_type in ("FRAME", "SECTION") and node_name:
        current_frame = node_name
        lines.append(f"\n### {node_type.capitalize()}: {node_name}")

    # Reusable component reference
    if node_type in ("COMPONENT", "COMPONENT_SET") and node_name:
        comp_info = components_map.get(node_id, {})
        desc = comp_info.get("description", "").strip()
        comp_line = f"Component: {node_name}"
        if desc:
            comp_line += f" - {desc}"
        lines.append(comp_line)

    # Text element
    if node_type == "TEXT":
        chars = node.get("characters", "").strip()
        if chars:
            prefix = f"[{current_frame}] " if current_frame else ""
            lines.append(f"{prefix}{chars}")

    # Node-specific comments
    if node_id in comments_by_node:
        lines.append(f"\nComments on {node_name or node_id}:")
        for comment in comments_by_node[node_id]:
            user = comment.get("user", {}).get("handle") or "Unknown"
            msg = comment.get("message", "").strip()
            if msg:
                lines.append(f"- {user}: {msg}")

    # Recurse children
    for child in node.get("children", []):
        if isinstance(child, dict):
            _extract_page_content(
                child,
                components_map,
                comments_by_node,
                lines,
                current_frame=current_frame,
            )


def _render_file_documents(
    file_key: str,
    file_data: dict[str, Any],
    comments_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Figma file data and comments into document rows with structure."""
    file_name = file_data.get("name") or file_key
    file_doc_id = f"file:{file_key}"
    version = file_data.get("version") or "unknown"
    last_modified = file_data.get("lastModified") or ""
    components = file_data.get("components") or {}
    documents: list[dict[str, Any]] = []

    # Map comments to nodes
    comments_by_node: dict[str, list[dict]] = {}
    file_comments: list[dict] = []
    for c in comments_data:
        client_meta = c.get("client_meta") or {}
        node_id = client_meta.get("node_id")
        if node_id:
            comments_by_node.setdefault(str(node_id), []).append(c)
        else:
            file_comments.append(c)

    # 1. File overview document
    file_content_parts = [
        f"File: {file_name}",
        f"Key: {file_key}",
        f"Version: {version}",
        f"Last Modified: {last_modified}",
    ]
    if components:
        file_content_parts.append("\nComponents:")
        for comp_id, comp in sorted(components.items()):
            comp_name = comp.get("name") or comp_id
            desc = comp.get("description", "").strip()
            file_content_parts.append(f"- {comp_name}" + (f": {desc}" if desc else ""))
    if file_comments:
        file_content_parts.append("\nFile Comments:")
        for c in file_comments:
            user = c.get("user", {}).get("handle") or "Unknown"
            msg = c.get("message", "").strip()
            if msg:
                file_content_parts.append(f"- {user}: {msg}")

    documents.append(
        {
            "id": file_doc_id,
            "title": file_name,
            "content": "\n".join(file_content_parts),
            "url": f"https://www.figma.com/design/{file_key}",
            "_deleted": False,
            NODE_SET_COLUMN: [f"figma:{file_key}"],
            STRUCTURE_COLUMN: {"ancestors": []},
        }
    )

    # 2. Page documents (children of root document)
    doc_node = file_data.get("document", {})
    canvases = [
        child
        for child in doc_node.get("children", [])
        if isinstance(child, dict) and child.get("type") == "CANVAS"
    ]

    for canvas in canvases:
        canvas_id = str(canvas.get("id", ""))
        canvas_name = canvas.get("name", "").strip() or "Untitled Page"
        page_doc_id = f"page:{file_key}:{canvas_id}"

        lines: list[str] = [f"Page: {canvas_name}"]
        _extract_page_content(canvas, components, comments_by_node, lines)

        documents.append(
            {
                "id": page_doc_id,
                "title": f"{file_name} / {canvas_name}",
                "content": "\n".join(lines).strip(),
                "url": f"https://www.figma.com/design/{file_key}?node-id={canvas_id}",
                "_deleted": False,
                NODE_SET_COLUMN: [f"figma:{file_key}"],
                STRUCTURE_COLUMN: {
                    "ancestors": [
                        {
                            "kind": "file",
                            "id": file_doc_id,
                            "name": file_name,
                            "document": True,
                        }
                    ]
                },
            }
        )

    return documents


# ---------------------------------------------------------------------------
# Incremental Walk & State Management
# ---------------------------------------------------------------------------
def new_stats() -> dict[str, int]:
    """Fresh counters for a figma_source run."""
    return {
        "scanned": 0,
        "skipped": 0,
        "failed": 0,
        "deleted": 0,
        "failed_rate_limit": 0,
        "failed_budget": 0,
        "requests": 0,
    }


def _clean_file_keys(file_keys: list[str] | tuple[str, ...]) -> list[str]:
    """Validate and clean list of Figma file keys, extracting keys if URLs are provided."""
    if not isinstance(file_keys, (list, tuple)) or not file_keys:
        raise ValueError(
            'file_keys must be a non-empty list or tuple of string keys, like ["<file_key>"]'
        )
    keys = []
    for k in file_keys:
        if not isinstance(k, str) or not k.strip():
            raise ValueError("file_keys must contain non-empty strings")
        item = k.strip()
        url_match = re.search(r"/(?:design|file|proto)/([0-9a-zA-Z]+)", item)
        if url_match:
            item = url_match.group(1)
        keys.append(item)
    return sorted(set(keys))


def _iter_rows(
    client: FigmaClient,
    state: dict,
    stats: dict[str, int],
    file_keys: list[str],
    *,
    depth: int | None = None,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict]:
    """Yield documents for the selected Figma files, updating incremental state."""
    files_state = state.setdefault("files", {})

    # Tombstones for files previously synced but removed from current selection
    deselected_keys = sorted(set(files_state) - set(file_keys))
    for rem_key in deselected_keys:
        for old_id in files_state[rem_key].get("seen_ids", []):
            if check_active is not None:
                check_active()
            yield {"id": old_id, "title": "", "content": "", "url": "", "_deleted": True}
            stats["deleted"] += 1
        del files_state[rem_key]

    for file_key in file_keys:
        if check_active is not None:
            check_active()

        file_state = files_state.setdefault(file_key, {})
        stored_modified = file_state.get("last_modified")

        # 1. Inspect file header / tree
        params = {"depth": depth} if depth is not None else None
        try:
            file_data = client.request(
                "GET",
                f"/v1/files/{file_key}",
                params=params,
                check_active=check_active,
                stats=stats,
                max_requests=max_requests,
            )
        except (FigmaRateLimitedError, FigmaBudgetExceededError) as exc:
            stats["failed"] += 1
            if isinstance(exc, FigmaRateLimitedError):
                stats["failed_rate_limit"] += 1
            else:
                stats["failed_budget"] += 1
            logger.warning("Figma sync stopped prematurely", reason=str(exc))
            return
        except FigmaNotFoundError:
            stats["failed"] += 1
            logger.warning("Figma file not found; omitting from sync", file_key=file_key)
            continue
        except FigmaError as exc:
            stats["failed"] += 1
            logger.warning("Failed to fetch Figma file", file_key=file_key, reason=str(exc))
            continue

        current_modified = file_data.get("lastModified")

        # 2. Incremental check: skip if unmodified
        if stored_modified and current_modified and stored_modified == current_modified:
            stats["skipped"] += 1
            continue

        # 3. Retrieve comments
        comments_data: list[dict[str, Any]] = []
        try:
            comments_resp = client.request(
                "GET",
                f"/v1/files/{file_key}/comments",
                params={"as_md": "true"},
                check_active=check_active,
                stats=stats,
                max_requests=max_requests,
            )
            comments_data = comments_resp.get("comments") or []
        except (FigmaRateLimitedError, FigmaBudgetExceededError) as exc:
            stats["failed"] += 1
            if isinstance(exc, FigmaRateLimitedError):
                stats["failed_rate_limit"] += 1
            else:
                stats["failed_budget"] += 1
            logger.warning("Figma sync comments stopped prematurely", reason=str(exc))
            return
        except FigmaError as exc:
            # Comments are best-effort; failure should not drop the file document
            logger.warning(
                "Could not retrieve comments for Figma file", file_key=file_key, reason=str(exc)
            )

        # 4. Render and emit documents
        docs = _render_file_documents(file_key, file_data, comments_data)
        current_ids = {doc["id"] for doc in docs}
        seen_ids = set(file_state.get("seen_ids", []))

        # Deletions: tombstones for nodes that disappeared
        for deleted_id in sorted(seen_ids - current_ids):
            if check_active is not None:
                check_active()
            yield {"id": deleted_id, "title": "", "content": "", "url": "", "_deleted": True}
            stats["deleted"] += 1

        # Emit active documents
        for doc in docs:
            if check_active is not None:
                check_active()
            yield doc
            stats["scanned"] += 1

        # Commit state for this file
        file_state["last_modified"] = current_modified
        file_state["seen_ids"] = sorted(current_ids)


# ---------------------------------------------------------------------------
# Public DLT Source
# ---------------------------------------------------------------------------
def figma_source(
    *,
    resource_name: str,
    file_keys: list[str] | tuple[str, ...],
    access_token: str | None = None,
    service: Any = None,
    depth: int | None = None,
    auth_type: str = "auto",
    check_active: Callable[[], None] | None = None,
    max_requests: int = DEFAULT_MAX_REQUESTS,
):
    """Return a ``dlt`` resource that yields the selected Figma files' documents.

    Args:
        resource_name: Unique identifier for this connection staging table and state.
        file_keys: List of Figma file keys to sync.
        access_token: Figma Personal Access Token or OAuth access token.
        service: Optional pre-built FigmaClient (injection point for tests/hosts).
        depth: Traversal depth for node tree retrieval (None for entire tree).
        auth_type: "auto", "pat", or "bearer".
        check_active: Optional cancellation checkpoint.
        max_requests: Maximum HTTP requests allowed per sync run (default: 1500).

    Returns:
        Configured DLT resource for document-mode ingestion.
    """
    import dlt
    from dlt.common.normalizers.naming.snake_case import NamingConvention

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 3:
        raise RuntimeError(
            "Figma sync requires a Cognee build supporting per-row structure "
            "(DOCUMENT_SYNC_VERSION >= 3)."
        )

    if not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("figma_source needs a resource_name, one per Figma connection")
    if NamingConvention().normalize_table_identifier(resource_name).startswith("_dlt_"):
        raise ValueError("resource_name must not start with _dlt_, which dlt keeps for itself")

    if not isinstance(max_requests, int) or isinstance(max_requests, bool) or max_requests < 1:
        raise ValueError("max_requests must be a positive int")

    clean_keys = _clean_file_keys(file_keys)

    if service is None and not access_token:
        raise ValueError("figma_source needs a service or an access_token")

    client = (
        service if service is not None else FigmaClient(access_token or "", auth_type=auth_type)
    )
    stats = new_stats()

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={
            "id": {"data_type": "text", "nullable": False},
            "title": {"data_type": "text", "nullable": True},
            "content": {"data_type": "text", "nullable": False},
            "url": {"data_type": "text", "nullable": True},
            "_deleted": {"data_type": "bool", "hard_delete": True},
            **DOCUMENT_COLUMN_HINTS,
        },
    )
    def figma_documents():
        stats.clear()
        stats.update(new_stats())
        rows = _iter_rows(
            client,
            dlt.current.resource_state(),
            stats,
            clean_keys,
            depth=depth,
            max_requests=max_requests,
            check_active=check_active,
        )
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = figma_documents()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "figma")
    setattr(resource, PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
