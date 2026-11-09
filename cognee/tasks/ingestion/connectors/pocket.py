"""Pocket reading list connector for cognee: turn bookmarks and research articles into memory.

One source handles a Pocket user account, pulling saved reading lists, article excerpts, and tagged
research items as flat document rows for :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import pocket_source

    await cognee.remember(
        pocket_source(
            consumer_key="<pocket-consumer-key>",
            access_token="<pocket-access-token>",
            state="all",
            limit=50,
        ),
        dataset_name="personal_reading_list",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``.
  Ids are formatted as ``pocket:item:<item_id>``.
* **Metadata & Tags**: User-applied tags, publication authors, resolved URLs, and article excerpts
  are structured cleanly within the document body for rich semantic search.
* **Incremental Sync**: Supports Pocket's ``since`` Unix timestamp parameter to only pull new
  or modified items on subsequent ingestion cycles.
* **Tombstones**: Items marked as deleted (status 2 in Pocket API) emit ``_deleted: True``
  to maintain synchronization with Cognee's graph memory.
"""

import logging
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

POCKET_API_URL = "https://getpocket.com/v3/get"
_TIMEOUT = (10, 30)
NODE_SET_COLUMN = "cognee_node_set"
DOCUMENT_SOURCE_ATTR = "document_source"


class PocketError(Exception):
    """Base exception for Pocket connector errors."""


class PocketAuthError(PocketError):
    """Raised when Pocket API authentication fails."""


class PocketRateLimitError(PocketError):
    """Raised when Pocket API rate limits are exceeded."""


class PocketClient:
    """HTTP Client for Pocket v3 REST API."""

    def __init__(self, consumer_key: str, access_token: str, session: Any = None):
        clean_consumer = consumer_key.strip()
        clean_token = access_token.strip()
        if not clean_consumer or not clean_token:
            raise ValueError("Pocket consumer_key and access_token must both be non-empty strings.")

        self.consumer_key = clean_consumer
        self.access_token = clean_token
        self._session = session

        if self._session is None:
            import requests
            self._session = requests.Session()

    def get_items(
        self,
        state: str = "all",
        favorite: int | None = None,
        tag: str | None = None,
        since: int | None = None,
        count: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Fetch items from user's Pocket reading list."""
        payload: dict[str, Any] = {
            "consumer_key": self.consumer_key,
            "access_token": self.access_token,
            "state": state,
            "detailType": "complete",
            "count": count,
            "offset": offset,
            "sort": "newest",
        }
        if favorite is not None:
            payload["favorite"] = favorite
        if tag is not None:
            payload["tag"] = tag
        if since is not None:
            payload["since"] = since

        headers = {
            "Content-Type": "application/json",
            "X-Accept": "application/json",
            "User-Agent": "cognee-pocket-connector/1.0",
        }

        try:
            resp = self._session.post(POCKET_API_URL, json=payload, headers=headers, timeout=_TIMEOUT)
            if resp.status_code == 401:
                raise PocketAuthError("Pocket authentication failed: Invalid consumer_key or access_token.")
            if resp.status_code == 403:
                raise PocketAuthError("Pocket access forbidden: Check app permissions.")
            if resp.status_code == 429:
                raise PocketRateLimitError("Pocket API rate limit exceeded.")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if not isinstance(e, PocketError):
                raise PocketError(f"Pocket API request error: {e}") from e
            raise


def format_pocket_document(item: dict[str, Any]) -> dict[str, Any]:
    """Format a Pocket item dictionary into a Cognee document row."""
    item_id = str(item.get("item_id") or item.get("resolved_id") or "")
    title = item.get("resolved_title") or item.get("given_title") or "Untitled Bookmark"
    url = item.get("resolved_url") or item.get("given_url") or ""
    excerpt = item.get("excerpt") or ""
    status = str(item.get("status", "0"))  # "0" = normal, "1" = archived, "2" = deleted

    time_added_ts = item.get("time_added")
    added_str = (
        datetime.fromtimestamp(int(time_added_ts), tz=timezone.utc).isoformat()
        if time_added_ts
        else datetime.now(timezone.utc).isoformat()
    )

    authors_dict = item.get("authors") or {}
    authors = [a.get("name") for a in authors_dict.values() if isinstance(a, dict) and a.get("name")]
    author_line = f"Authors: {', '.join(authors)}" if authors else None

    tags_dict = item.get("tags") or {}
    tags = [t for t in tags_dict.keys()] if isinstance(tags_dict, dict) else []
    tags_line = f"Tags: {', '.join(tags)}" if tags else None

    content_lines = [
        f"Title: {title}",
        f"URL: {url}",
        f"Saved At: {added_str}",
    ]
    if author_line:
        content_lines.append(author_line)
    if tags_line:
        content_lines.append(tags_line)
    content_lines.append("")
    if excerpt:
        content_lines.append("Article Excerpt:")
        content_lines.append(excerpt)

    full_content = "\n".join(content_lines).strip()
    is_deleted = status == "2"

    return {
        "id": f"pocket:item:{item_id}",
        "title": f"Pocket: {title}".strip(),
        "content": full_content,
        "url": url,
        "_deleted": is_deleted,
        NODE_SET_COLUMN: "pocket_reading_list",
    }


def pocket_source(
    consumer_key: str,
    access_token: str,
    state: str = "all",
    tag: str | None = None,
    favorite: int | None = None,
    since: int | None = None,
    limit: int = 100,
    session: Any = None,
) -> Iterator[dict[str, Any]]:
    """Yield standardized document rows for Pocket reading lists and articles.

    Parameters
    ----------
    consumer_key : str
        Pocket application consumer key.
    access_token : str
        Pocket user OAuth access token.
    state : str, default "all"
        Which items to retrieve: "unread", "archive", or "all".
    tag : str, optional
        Filter bookmarks by a specific tag.
    favorite : int, optional
        Filter by favorite status (0 = un-favorited, 1 = favorited).
    since : int, optional
        Unix timestamp to only pull items added or modified after this time.
    limit : int, default 100
        Maximum number of items to yield.
    session : Any, optional
        Optional HTTP session for custom connections / test mocking.

    Yields
    ------
    dict[str, Any]
        Flat document row with ``{id, title, content, url, _deleted, cognee_node_set}``.
    """
    client = PocketClient(consumer_key=consumer_key, access_token=access_token, session=session)

    yielded_count = 0
    offset = 0
    batch_size = min(50, limit)

    while yielded_count < limit:
        resp = client.get_items(
            state=state,
            favorite=favorite,
            tag=tag,
            since=since,
            count=batch_size,
            offset=offset,
        )

        items_dict = resp.get("list")
        if not items_dict or not isinstance(items_dict, dict):
            break

        for item_data in items_dict.values():
            if not isinstance(item_data, dict):
                continue
            doc = format_pocket_document(item_data)
            yield doc
            yielded_count += 1
            if yielded_count >= limit:
                break

        offset += len(items_dict)
        if len(items_dict) < batch_size:
            break
