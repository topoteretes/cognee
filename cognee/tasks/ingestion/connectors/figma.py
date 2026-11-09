"""Figma data-source connector for cognee: turn design files, UI components and comments into memory.

One source handles one or more Figma file keys. It yields pages, design components, text frames,
and associated design review comments as flat document rows for :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import figma_source

    await cognee.remember(
        figma_source(
            file_keys=["aB1cD2eF3gH4"],
            access_token="<figma-personal-access-token>",
        ),
        dataset_name="figma_design",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``.
  Ids are prefixed by file and node (``figma:<file_key>:<node_id>``).
* **Hierarchy**: Walks document root -> canvas (page) -> frame / component -> text elements.
* **Comments**: Comments and annotations attached to specific coordinates or nodes are folded
  into the document context.
* **Deterministic**: Omit volatile viewport positions and zoom levels to ensure stable hashing.
"""

import logging
import re
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

BASE_URL = "https://api.figma.com/v1"
_TIMEOUT = (10, 30)
NODE_SET_COLUMN = "cognee_node_set"
DOCUMENT_SOURCE_ATTR = "document_source"


class FigmaError(Exception):
    """Base exception for Figma connector errors."""


class FigmaAuthError(FigmaError):
    """Raised when authentication credentials fail."""


class FigmaRateLimitError(FigmaError):
    """Raised when Figma API rate limits are hit."""


class FigmaClient:
    """HTTP client for Figma REST API."""

    def __init__(
        self,
        access_token: str | None = None,
        oauth_token: str | None = None,
        session: Any = None,
    ):
        if oauth_token:
            self.headers = {"Authorization": f"Bearer {oauth_token}"}
        elif access_token:
            self.headers = {"X-Figma-Token": access_token}
        else:
            raise ValueError("Provide either access_token or oauth_token")

        self.headers.update({
            "Accept": "application/json",
            "User-Agent": "Cognee-Figma-Connector/1.0",
        })
        self.session = session

    def fetch_file(self, file_key: str) -> dict[str, Any]:
        """Fetches Figma file document JSON."""
        clean_key = re.sub(r"[^a-zA-Z0-9_-]", "", file_key.strip())
        if not clean_key:
            raise ValueError("Invalid Figma file_key")

        if self.session is None:
            import requests
            session = requests.Session()
        else:
            session = self.session

        url = f"{BASE_URL}/files/{clean_key}"
        resp = session.get(url, headers=self.headers, timeout=_TIMEOUT)

        if resp.status_code == 401 or resp.status_code == 403:
            raise FigmaAuthError(f"Figma auth failed: HTTP {resp.status_code}")
        if resp.status_code == 429:
            raise FigmaRateLimitError("Figma rate limit hit")
        if not resp.ok:
            raise FigmaError(f"Figma request failed: HTTP {resp.status_code}")

        return resp.json()

    def fetch_comments(self, file_key: str) -> list[dict[str, Any]]:
        """Fetches comments for a Figma file."""
        clean_key = re.sub(r"[^a-zA-Z0-9_-]", "", file_key.strip())
        if not clean_key:
            return []

        if self.session is None:
            import requests
            session = requests.Session()
        else:
            session = self.session

        url = f"{BASE_URL}/files/{clean_key}/comments"
        try:
            resp = session.get(url, headers=self.headers, timeout=_TIMEOUT)
            if resp.ok:
                return resp.json().get("comments", [])
        except Exception as e:
            logger.warning("Failed to fetch comments for Figma file %s: %s", file_key, e)
        return []


def _extract_text_nodes(node: dict[str, Any], accumulated: list[dict[str, Any]]) -> None:
    """Recursively traverses a Figma node tree to extract text content."""
    node_type = node.get("type")
    if node_type == "TEXT":
        chars = (node.get("characters") or "").strip()
        if chars:
            accumulated.append({
                "id": node.get("id"),
                "name": node.get("name"),
                "text": chars,
            })

    for child in node.get("children", []):
        _extract_text_nodes(child, accumulated)


def render_figma_page(
    file_name: str,
    page_name: str,
    page_id: str,
    components: list[dict[str, Any]],
    text_nodes: list[dict[str, Any]],
    comments: list[dict[str, Any]] | None = None,
) -> str:
    """Renders structured design metadata for vector embeddings."""
    lines = [
        f"Figma Design Document: {file_name}",
        f"Page: {page_name} (ID: {page_id})",
    ]

    if components:
        lines.append("\nDesign Components:")
        for comp in components:
            lines.append(f"- {comp.get('name', 'Unnamed Component')}: {comp.get('description', 'No description')}")

    if text_nodes:
        lines.append("\nUI Text & Copy Elements:")
        for t in text_nodes:
            lines.append(f"• [{t.get('name')}]: \"{t.get('text')}\"")

    if comments:
        lines.append("\nDesign Review Comments:")
        for c in comments:
            user = c.get("user", {}).get("handle", "Contributor")
            msg = c.get("message", "").strip()
            if msg:
                lines.append(f"- @{user}: {msg}")

    return "\n".join(lines)


def _iter_figma_files(
    client: FigmaClient,
    file_keys: list[str],
    include_comments: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yields flat documents representing pages and components in Figma files."""
    for key in file_keys:
        try:
            data = client.fetch_file(key)
        except Exception as e:
            logger.error("Error fetching Figma file %s: %s", key, e)
            continue

        file_name = data.get("name") or f"Figma File {key}"
        doc_root = data.get("document", {})
        pages = doc_root.get("children", [])
        components_map = data.get("components", {})

        comments = []
        if include_comments:
            comments = client.fetch_comments(key)

        for page in pages:
            page_id = page.get("id")
            page_name = page.get("name") or "Page"

            text_nodes: list[dict[str, Any]] = []
            _extract_text_nodes(page, text_nodes)

            # Map page components
            page_components = []
            for comp_key, comp_info in components_map.items():
                page_components.append({
                    "key": comp_key,
                    "name": comp_info.get("name"),
                    "description": comp_info.get("description"),
                })

            rendered_content = render_figma_page(
                file_name=file_name,
                page_name=page_name,
                page_id=page_id,
                components=page_components,
                text_nodes=text_nodes,
                comments=comments,
            )

            file_url = f"https://www.figma.com/design/{key}?node-id={page_id}"

            yield {
                "id": f"figma:{key}:{page_id}",
                "title": f"{file_name} - {page_name}",
                "content": rendered_content,
                "url": file_url,
                "_deleted": False,
                NODE_SET_COLUMN: f"figma:{key}",
                DOCUMENT_SOURCE_ATTR: "figma",
            }


def figma_source(
    file_keys: list[str],
    access_token: str | None = None,
    oauth_token: str | None = None,
    include_comments: bool = True,
    client: FigmaClient | None = None,
):
    """Returns a DLT source for Figma file and component ingestion."""
    import dlt

    if client is None:
        client = FigmaClient(access_token=access_token, oauth_token=oauth_token)

    @dlt.source(name="figma_designs")
    def source():
        @dlt.resource(
            name="pages",
            primary_key="id",
            write_disposition="merge",
        )
        def pages_resource():
            yield from _iter_figma_files(
                client=client,
                file_keys=file_keys,
                include_comments=include_comments,
            )

        return pages_resource

    return source()


def build_figma_service(
    access_token: str | None = None,
    oauth_token: str | None = None,
) -> FigmaClient:
    """Helper to instantiate a standalone FigmaClient."""
    return FigmaClient(access_token=access_token, oauth_token=oauth_token)
