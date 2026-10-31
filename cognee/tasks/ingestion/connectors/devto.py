"""Dev.to connector for cognee: a ``dlt`` source that reads one author's published articles.

One source is one Dev.to (Forem) author or organization. It yields one document per published
article, with the article's Markdown as content and its tags as node sets, and is meant to be
handed directly to :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import devto_source

    await cognee.remember(
        devto_source(username="ben"),
        dataset_name="devto",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Every run lists the author's articles in full, then fetches the body of an article only when
it is new or its ``published_at`` / ``edited_at`` stamps changed. An article that left the
listing (unpublished or deleted) is forgotten. Only public data is read, so no key is needed.
See ``devto.md`` for what is read, deletion and its gaps, and privacy.
"""

import itertools
import logging
import re
import time
from collections.abc import Callable, Iterator
from typing import Any

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
)

logger = logging.getLogger(__name__)

API_URL = "https://dev.to/api"
# The largest page the articles endpoint serves.
PAGE_SIZE = 1000
_TIMEOUT = (10, 30)
_RETRIES = 3
# A Retry-After longer than this is not waited for: the run raises and the next one continues.
_MAX_RETRY_AFTER = 30
_HEADERS = {
    "Accept": "application/vnd.forem.api-v1+json",
    "User-Agent": "cognee-devto-connector",
}
# Forem usernames are letters, digits and underscores; organizations may also use "-" and ".".
_USERNAME = re.compile(r"[A-Za-z0-9_.-]{1,64}")


# ---------------------------------------------------------------------------
# Errors: messages carry codes and article ids only, never response bodies.
# ---------------------------------------------------------------------------
class DevToSourceError(RuntimeError):
    """Base class of the errors this source raises."""


class DevToAPIError(DevToSourceError):
    """Dev.to answered with an error, an unexpected shape, or could not be reached."""


class DevToNotFoundError(DevToAPIError):
    """Dev.to answered HTTP 404."""


class DevToRateLimitedError(DevToAPIError):
    """Dev.to kept answering HTTP 429."""


class DevToEmptyListingError(DevToSourceError):
    """The author lists no articles at all, though earlier runs synced some."""


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
def _retry_after(headers: Any) -> float | None:
    try:
        return float(headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None


class DevToClient:
    """Minimal synchronous client for the public Dev.to API.

    ``get`` returns the decoded JSON. Tests and hosts can pass any object with the same
    ``get`` method instead.
    """

    def __init__(self, *, sleep: Callable[[float], None] = time.sleep):
        self._sleep = sleep

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        import requests

        for attempt in range(_RETRIES):
            last = attempt == _RETRIES - 1
            try:
                response = requests.get(
                    f"{API_URL}{path}", params=params, headers=_HEADERS, timeout=_TIMEOUT
                )
            except requests.RequestException as exc:
                if last:
                    raise DevToAPIError(f"Dev.to request failed: {type(exc).__name__}") from None
                self._sleep(2**attempt)
                continue
            status = response.status_code
            if status == 429:
                wait = _retry_after(response.headers)
                if last or (wait is not None and wait > _MAX_RETRY_AFTER):
                    raise DevToRateLimitedError("Dev.to rate limit exceeded (HTTP 429)")
                self._sleep(2**attempt if wait is None else max(wait, 0))
                continue
            if status >= 500 and not last:
                self._sleep(2**attempt)
                continue
            if status == 404:
                raise DevToNotFoundError("Dev.to answered HTTP 404")
            if status != 200:
                raise DevToAPIError(f"Dev.to request failed: HTTP {status}")
            try:
                return response.json()
            except ValueError:
                raise DevToAPIError("Dev.to response was not JSON") from None
        raise DevToAPIError("Dev.to request failed")  # pragma: no cover


# ---------------------------------------------------------------------------
# Rendering: deterministic, no volatile fields (reactions, comment counts).
# ---------------------------------------------------------------------------
def _article_id(article_id: int) -> str:
    return f"devto:article:{article_id}"


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _tags(article: dict) -> list[str]:
    """The article's tags, lower-cased, sorted and without repeats.

    The list endpoint returns ``tag_list`` as a list and ``tags`` as a comma-separated string;
    the single-article endpoint swaps them. Whichever field is a list wins.
    """
    value = next(
        (article.get(key) for key in ("tag_list", "tags") if isinstance(article.get(key), list)),
        None,
    )
    if value is None:
        text = next((article.get(key) for key in ("tag_list", "tags") if article.get(key)), "")
        value = text.split(",") if isinstance(text, str) else []
    return sorted({_text(tag).lower() for tag in value if _text(tag)})


def _fingerprint(summary: dict) -> str:
    """What decides whether an article's body is fetched again."""
    return f"{_text(summary.get('published_at'))}|{_text(summary.get('edited_at'))}"


def _article_row(article: dict) -> dict[str, Any]:
    """One document row. The content is a short header and the article's Markdown as written."""
    user = article.get("user") if isinstance(article.get("user"), dict) else {}
    name, handle = _text(user.get("name")), _text(user.get("username"))
    author = f"{name} (@{handle})" if name and handle else name or (f"@{handle}" if handle else "")
    url = _text(article.get("url"))
    canonical = _text(article.get("canonical_url"))
    tags = _tags(article)
    header = [
        f"{label}: {value}"
        for label, value in [
            ("Author", author),
            ("Published", _text(article.get("published_at"))),
            ("Tags", ", ".join(tags)),
            ("Canonical URL", canonical if canonical != url else ""),
        ]
        if value
    ]
    body = _text(article.get("body_markdown"))
    row: dict[str, Any] = {
        "id": _article_id(article["id"]),
        "title": _text(article.get("title")) or f"Dev.to article {article['id']}",
        "content": "\n".join(header) + ("\n\n" + body if body else ""),
        "url": url,
        "_deleted": False,
    }
    if tags:
        row[NODE_SET_COLUMN] = [f"devto:tag:{tag}" for tag in tags]
    return row


# ---------------------------------------------------------------------------
# Walk
# ---------------------------------------------------------------------------
def _list_articles(client: Any, username: str) -> list[dict]:
    """Every published article of the author, by id, as the list endpoint returns them.

    The listing is complete before anything is yielded. An entry without an integer id raises:
    skipping it would make that article look deleted.
    """
    articles: dict[int, dict] = {}
    for page in itertools.count(1):
        batch = client.get("/articles", {"username": username, "per_page": PAGE_SIZE, "page": page})
        if not isinstance(batch, list):
            raise DevToAPIError("Dev.to article listing was not a list")
        for item in batch:
            article_id = item.get("id") if isinstance(item, dict) else None
            if not isinstance(article_id, int) or isinstance(article_id, bool):
                raise DevToAPIError(f"Dev.to listed an article without an id on page {page}")
            # A page boundary can shift while listing; the first copy is kept.
            articles.setdefault(article_id, item)
        if len(batch) < PAGE_SIZE:
            return [articles[article_id] for article_id in sorted(articles)]
    raise DevToAPIError("Dev.to article listing did not end")  # pragma: no cover


def _iter_rows(
    client: Any,
    username: str,
    state: dict,
    stats: dict[str, int],
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield a row for every new or changed article, then a tombstone for every one that is gone.

    ``state["articles"]`` maps each synced article id to its fingerprint. An empty listing is
    refused when articles were synced before: Dev.to answers an unknown username with an empty
    list, so a typo or a renamed account would otherwise forget everything. ``state`` is written
    after the last row.
    """
    listed = _list_articles(client, username)
    previous: dict[str, str] = dict(state.get("articles", {}))
    if previous and not listed:
        raise DevToEmptyListingError(
            "Dev.to lists no articles for this username, but earlier runs synced some; nothing "
            "was synced and nothing was forgotten. Check the username, or pass a new "
            "resource_name to start over."
        )
    present: dict[str, str] = {}
    for summary in listed:
        if check_active is not None:
            check_active()  # outside any try: a revoked connection is not a vanished article
        key = str(summary["id"])
        fingerprint = _fingerprint(summary)
        if previous.get(key) == fingerprint:
            present[key] = fingerprint
            stats["unchanged"] += 1
            continue
        try:
            article = client.get(f"/articles/{summary['id']}")
        except DevToNotFoundError:
            # Unpublished between the listing and this fetch: treated as gone.
            logger.warning("Dev.to article %s disappeared while syncing.", summary["id"])
            stats["vanished"] += 1
            continue
        if not isinstance(article, dict) or article.get("id") != summary["id"]:
            raise DevToAPIError(f"Dev.to returned an unexpected article for id {summary['id']}")
        yield _article_row(article)
        present[key] = fingerprint
        # After the yield: a row that check_active stopped before it was published is not counted.
        stats["synced"] += 1

    for key in sorted(set(previous) - set(present), key=int):
        yield {"id": _article_id(int(key)), "_deleted": True}
        stats["deleted"] += 1
    state.update(v=1, articles=dict(sorted(present.items(), key=lambda item: int(item[0]))))
    logger.info(
        "Dev.to: %d article(s) synced, %d unchanged, %d forgotten.",
        stats["synced"],
        stats["unchanged"],
        stats["deleted"],
    )


def _username(username: Any) -> str:
    """The username without a leading ``@``, lower-cased; anything else is refused."""
    name = username.strip().removeprefix("@").lower() if isinstance(username, str) else ""
    if not _USERNAME.fullmatch(name):
        raise ValueError(
            "username must be a Dev.to username or organization, like 'ben' or '@ben'."
        )
    return name


def devto_source(
    *,
    username: str,
    resource_name: str | None = None,
    client: Any = None,
    check_active: Callable[[], None] | None = None,
):
    """Return a ``dlt`` resource that yields one document per published article of an author.

    Hand the result to ``cognee.remember(...)`` with ``write_disposition="merge"`` and
    ``primary_key="id"``. A run lists every article; one whose stamps did not change costs no
    request and no extraction. Read ``cognee_sync_stats`` afterwards.

    Args:
        username: The Dev.to username or organization, with or without a leading ``@``.
        resource_name: Names the staging table and the sync state. Defaults to
            ``devto_<username>``. Two sources in one dataset need different names.
        client: An object with a ``get(path, params)`` method; defaults to :class:`DevToClient`.
        check_active: Host authorization checkpoint, called before each article and before
            each row is published.
    """
    import dlt

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 2:
        raise RuntimeError(
            "Dev.to sync requires a Cognee build that reads per-row node sets "
            "(DOCUMENT_SYNC_VERSION >= 2). Upgrade Cognee before syncing."
        )
    name = _username(username)
    if resource_name is None:
        resource_name = f"devto_{re.sub(r'[^a-z0-9_]', '_', name)}"
    elif not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("resource_name must be a non-empty string.")
    client = DevToClient() if client is None else client
    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def devto_articles():
        stats.update(synced=0, unchanged=0, deleted=0, vanished=0)
        rows = _iter_rows(client, name, dlt.current.resource_state(), stats, check_active)
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = devto_articles()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "devto")
    setattr(resource, PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
