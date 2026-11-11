"""JSON Feed (jsonfeed.org) data-source connector for cognee.

A ``dlt`` source that ingests web feeds formatted according to the JSON Feed
specification (v1.0 and v1.1, https://jsonfeed.org) into searchable knowledge
graph documents.

Usage::

    import cognee
    from cognee.tasks.ingestion.connectors import jsonfeed_source

    await cognee.remember(
        jsonfeed_source(
            feed_urls=[
                "https://jsonfeed.org/feed.json",
                "https://daringfireball.net/feeds/json",
            ]
        ),
        dataset_name="web_feeds",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``.
* **Ids** are unique per feed and item: ``jsonfeed:<feed_url>#<item_id>``.
* **Node set** scopes retrieval to ``jsonfeed`` and the feed slug.
* **Content** converts item body (preferring ``content_text`` or converting
  ``content_html`` to plain text), metadata (author, publication dates, tags,
  attachments), and summary into structured Markdown.
* **Incremental**: Preserves the latest synced timestamp (``date_modified`` or
  ``date_published``) per feed in ``dlt.current.resource_state()`` to avoid
  re-processing older feed items.
* **Pagination**: Automatically traverses JSON Feed ``next_url`` pagination links.
* **Zero extra dependencies**: Uses standard Python libraries (``urllib``, ``json``,
  ``html.parser``) and ``dlt``.
"""

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from html.parser import HTMLParser
from typing import Any

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, NODE_SET_COLUMN

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "cognee-jsonfeed-connector/1.0 (https://cognee.ai)"
DEFAULT_TIMEOUT = 30.0
MAX_PAGINATION_PAGES = 50


class JSONFeedError(RuntimeError):
    """Base error for JSON Feed connector operations."""


class JSONFeedFetchError(JSONFeedError):
    """Error fetching a feed over HTTP."""


class JSONFeedParseError(JSONFeedError):
    """Error parsing JSON Feed content."""


class _HTMLToText(HTMLParser):
    """Lightweight, safe HTML to plain-text converter."""

    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []

    def handle_data(self, data: str) -> None:
        self._pieces.append(data)

    def handle_entityref(self, name: str) -> None:
        import html

        self._pieces.append(html.unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        import html

        self._pieces.append(html.unescape(f"&#{name};"))

    def get_text(self) -> str:
        text = "".join(self._pieces)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text.strip()


def html_to_text(html_content: str | None) -> str:
    """Strip HTML markup and return normalized plain text."""
    if not html_content:
        return ""
    parser = _HTMLToText()
    try:
        parser.feed(html_content)
        parser.close()
        return parser.get_text()
    except Exception:
        # Fallback to simple regex strip
        return re.sub(r"<[^<]+?>", " ", html_content).strip()


class JSONFeedService:
    """Fetches and parses JSON Feed (jsonfeed.org) data."""

    def __init__(
        self,
        feed_urls: list[str] | str,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if isinstance(feed_urls, str):
            feed_urls = [feed_urls]
        self.feed_urls = [url.strip() for url in feed_urls if url.strip()]
        self.user_agent = user_agent
        self.timeout = timeout

    def fetch_feed(self, url: str) -> dict[str, Any]:
        """Fetch and parse a JSON Feed document from a URL."""
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/feed+json, application/json;q=0.9, */*;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                content = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raise JSONFeedFetchError(f"HTTP {exc.code} fetching feed {url}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise JSONFeedFetchError(f"Failed to connect to feed {url}: {exc.reason}") from exc
        except Exception as exc:
            raise JSONFeedFetchError(f"Unexpected error fetching feed {url}: {exc}") from exc

        try:
            feed_data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise JSONFeedParseError(f"Invalid JSON at {url}: {exc}") from exc

        if not isinstance(feed_data, dict):
            raise JSONFeedParseError(f"Feed root at {url} must be a JSON object, got {type(feed_data).__name__}")

        version = feed_data.get("version", "")
        if not (isinstance(version, str) and "jsonfeed.org" in version.lower()):
            logger.warning("Feed at %s does not specify a standard jsonfeed.org version: %r", url, version)

        return feed_data

    def format_item_document(
        self,
        item: dict[str, Any],
        feed_meta: dict[str, Any],
        feed_url: str,
    ) -> dict[str, Any]:
        """Convert a single feed item into a cognee-compatible document row."""
        item_id = str(item.get("id") or item.get("url") or "")
        if not item_id:
            # Fallback to deterministic hash of title + date if no id provided
            title_hint = item.get("title") or ""
            date_hint = item.get("date_published") or item.get("date_modified") or ""
            item_id = f"{title_hint}_{date_hint}"

        title = item.get("title") or item.get("summary") or f"Feed Item {item_id}"
        # Normalize single-line title
        title = re.sub(r"\s+", " ", str(title)).strip()

        feed_title = feed_meta.get("title") or feed_url
        item_url = item.get("url") or item.get("external_url") or feed_meta.get("home_page_url") or feed_url

        # Author extraction (supports v1.1 authors array or v1.0 author object)
        author_names: list[str] = []
        if isinstance(item.get("authors"), list):
            for a in item["authors"]:
                if isinstance(a, dict) and a.get("name"):
                    author_names.append(a["name"])
        elif isinstance(item.get("author"), dict) and item["author"].get("name"):
            author_names.append(item["author"]["name"])
        elif isinstance(feed_meta.get("authors"), list):
            for a in feed_meta["authors"]:
                if isinstance(a, dict) and a.get("name"):
                    author_names.append(a["name"])
        elif isinstance(feed_meta.get("author"), dict) and feed_meta["author"].get("name"):
            author_names.append(feed_meta["author"]["name"])

        # Body content extraction
        body_text = item.get("content_text")
        if not body_text and item.get("content_html"):
            body_text = html_to_text(item["content_html"])
        if not body_text:
            body_text = item.get("summary") or ""

        date_published = item.get("date_published", "")
        date_modified = item.get("date_modified", "")
        tags = item.get("tags") if isinstance(item.get("tags"), list) else []

        # Render structured markdown
        lines: list[str] = [
            f"# {title}",
            "",
            f"- **Feed**: {feed_title}",
            f"- **Feed URL**: {feed_url}",
        ]
        if item_url:
            lines.append(f"- **URL**: {item_url}")
        if author_names:
            lines.append(f"- **Author**: {', '.join(author_names)}")
        if date_published:
            lines.append(f"- **Published**: {date_published}")
        if date_modified and date_modified != date_published:
            lines.append(f"- **Modified**: {date_modified}")
        if tags:
            lines.append(f"- **Tags**: {', '.join(str(t) for t in tags)}")

        summary = item.get("summary")
        if summary and summary.strip() != body_text.strip():
            lines.extend(["", "## Summary", "", summary.strip()])

        if body_text.strip():
            lines.extend(["", "## Content", "", body_text.strip()])

        content = "\n".join(lines)

        feed_slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", feed_title.lower()).strip("_")
        node_set = ["jsonfeed", f"jsonfeed:{feed_slug}"]

        doc_id = f"jsonfeed:{feed_url}#{item_id}"

        return {
            "id": doc_id,
            "title": title,
            "content": content,
            "url": item_url,
            "_deleted": False,
            NODE_SET_COLUMN: node_set,
            # Metadata fields for querying
            "feed_title": feed_title,
            "feed_url": feed_url,
            "date_published": date_published,
            "date_modified": date_modified,
        }

    def get_feed_items(
        self,
        feed_url: str,
        since_iso: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Fetch items from a feed, handling pagination and date filtering.

        Returns (items_list, latest_date_seen).
        """
        current_url = feed_url
        page_count = 0
        all_documents: list[dict[str, Any]] = []
        newest_date: str | None = since_iso

        while current_url and page_count < MAX_PAGINATION_PAGES:
            page_count += 1
            feed_data = self.fetch_feed(current_url)

            items = feed_data.get("items", [])
            if not isinstance(items, list):
                logger.warning("Feed at %s has non-list 'items' field", current_url)
                break

            for item in items:
                if not isinstance(item, dict):
                    continue

                item_date = item.get("date_modified") or item.get("date_published") or ""
                if since_iso and item_date and item_date <= since_iso:
                    # Item is older than or equal to cursor; skip
                    continue

                if item_date:
                    if newest_date is None or item_date > newest_date:
                        newest_date = item_date

                doc = self.format_item_document(item, feed_data, feed_url)
                all_documents.append(doc)

            # Check for next page
            next_url = feed_data.get("next_url")
            if next_url and isinstance(next_url, str) and next_url.strip() != current_url:
                current_url = next_url.strip()
            else:
                break

        return all_documents, newest_date


def build_jsonfeed_service(feed_urls: list[str] | str) -> JSONFeedService:
    """Build a JSONFeedService from a feed URL or list of feed URLs."""
    return JSONFeedService(feed_urls=feed_urls)


def _jsonfeed_resource_factory(
    service: JSONFeedService,
    feed_urls: list[str],
    sync_mode: str,
) -> Callable[..., Iterator[dict[str, Any]]]:
    """Build the dlt resource generator for JSON Feed items."""
    import dlt

    @dlt.resource(name="items", write_disposition="merge", primary_key="id")
    def items() -> Iterator[dict[str, Any]]:
        state = dlt.current.resource_state()
        feed_cursors = state.setdefault("feed_cursors", {})

        for url in feed_urls:
            cursor = feed_cursors.get(url) if sync_mode == "incremental" else None
            try:
                docs, newest_date = service.get_feed_items(url, since_iso=cursor)
                for doc in docs:
                    yield doc

                if newest_date:
                    feed_cursors[url] = newest_date
            except Exception as exc:
                logger.error("Failed to sync JSON Feed at %s: %s", url, exc, exc_info=True)

    return items


def jsonfeed_source(
    feed_urls: list[str] | str,
    endpoint_names: list[str] | None = None,
    sync_mode: str = "incremental",
) -> Any:
    """Create a dlt source for JSON Feed URLs.

    Args:
        feed_urls: A single feed URL string or list of feed URLs (jsonfeed.org compliant).
        endpoint_names: Optional endpoint filter list (e.g. ``["items"]``).
        sync_mode: ``"incremental"`` (default, skips previously synced items) or ``"full"``.

    Returns:
        A dlt source object configured for cognee document ingestion.
    """
    import dlt
    from cognee.tasks.ingestion.dlt_utils import PIPELINE_SCOPE_ATTR

    if isinstance(feed_urls, str):
        urls = [feed_urls]
    else:
        urls = list(feed_urls)

    service = build_jsonfeed_service(urls)

    @dlt.source(name="jsonfeed")
    def _source() -> list[Any]:
        items_res = _jsonfeed_resource_factory(service, urls, sync_mode)()
        return [items_res]

    source = _source()
    if endpoint_names:
        source = source.with_resources(*endpoint_names)

    setattr(source, DOCUMENT_SOURCE_ATTR, True)
    setattr(source, PIPELINE_SCOPE_ATTR, "jsonfeed")
    return source
