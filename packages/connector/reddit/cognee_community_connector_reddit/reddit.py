"""Reddit data-source connector for cognee — subreddits as long-term memory.

Design notes
------------
* Reddit closed its self-service "Data API" in November 2025. This connector
  uses the supported OAuth 2.0 **password** grant for registered "script" apps,
  which the Reddit Admin API team still requires (no public client-credentials
  or public data endpoints anymore). Credentials come from a developer app you
  register at https://www.reddit.com/prefs/apps (type "script").
* The connector is read-only and never posts, votes, or signs up for anything.
  Every network exchange goes through the same strict allow-listing and
  truncation rules described below.
* Every record is rendered to a data item and handed to cognee *as a document*
  (see ``DOCUMENT_SOURCE_ATTR``), so cognee extracts entities across your
  community's conversations instead of treating data rows as a schema.

Full-snapshot syncing
---------------------
Each run pulls the top ``limit`` submissions per subreddit (``after`` cursor
pagination) plus a **bounded** comment tree per post, and writes the whole view
with ``write_disposition="replace"``. A post that disappears from the listing
simply is not part of the new snapshot, and cognee's ``orphan_cleanup`` forgets
it from the graph/vector indices on the next sync. Unchanged posts keep a stable
content-hash ``data_id`` and are not re-ingested. A fetch error aborts the run
instead of letting a partial snapshot forget anything.
"""

import logging
import os
import re
import time
from collections import deque
from typing import Any

import httpx

logger = logging.getLogger("reddit_connector")

REDDIT_SOURCE_NAME = "reddit"
REDDIT_TABLE_POSTS = "reddit_posts"

_OAUTH_URL = "https://www.reddit.com/api/v1/access_token"
_API_BASE = "https://oauth.reddit.com"
_MORE_URL = _API_BASE + "/api/morechildren"

_DEFAULT_USER_AGENT = "cognee-reddit-connector/0.1 (by u/cognee)"

_EXTRA_HINT = (
    'The Reddit connector requires the "reddit" extra: '
    'pip install "cognee[reddit]" (provides dlt, httpx, cognee).'
)


class RedditAPIError(httpx.HTTPStatusError):
    """Raised for HTTP errors after non-transient retries are exhausted."""


def _one_line(text: str | None, *, limit: int = 500) -> str:
    """Collapse whitespace and cap a comment body so documents stay tidy."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.replace("&amp;", "&")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class RedditClient:
    """Thin read-only OAuth client for the Reddit JSON API (script app)."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        username: str | None = None,
        password: str | None = None,
        user_agent: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 4,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._username = username
        self._password = password
        self._user_agent = user_agent or _DEFAULT_USER_AGENT
        self._max_retries = max_retries
        self._token: str | None = None
        self._http = httpx.Client(
            headers={"User-Agent": self._user_agent},
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Basic {_b64(f'{self._client_id}:{self._client_secret}')}",
            "User-Agent": self._user_agent,
        }

    def _freshen_token(self) -> None:
        if not self._username or not self._password:
            raise ValueError(
                "Reddit script apps use the OAuth password grant: pass username= and "
                "password= (or REDDIT_USERNAME / REDDIT_PASSWORD)."
            )
        form = {
            "grant_type": "password",
            "username": self._username,
            "password": self._password,
        }
        resp = self._http.post(_OAUTH_URL, headers=self._auth_headers(), data=form)
        resp.raise_for_status()
        body = resp.json()
        self._token = body["access_token"]
        # Script-app tokens last ~1 hour; re-fetching is cheap.

    def _request(
        self, method: str, url: str, *, params: dict[str, Any] | None = None
    ) -> httpx.Response:
        if self._token is None:
            self._freshen_token()
        attempt = 0
        backoff = 1.0
        while True:
            resp = self._http.request(
                method,
                url,
                params=params,
                headers={"Authorization": f"Bearer {self._token}", "User-Agent": self._user_agent},
            )
            if resp.status_code == 401 and attempt == 0:
                self._token = None
                self._freshen_token()
                attempt += 1
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt >= self._max_retries:
                    self._raise_if_error(resp)
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                logger.info("Reddit: rate-limited (%s); retrying in %.1fs", resp.status_code, delay)
                time.sleep(delay)
                backoff = min(backoff * 2, 30.0)
                attempt += 1
                continue

            self._raise_if_error(resp)
            return resp

    @staticmethod
    def _raise_if_error(resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        raise RedditAPIError(
            f"Reddit API error {resp.status_code}: {body}",
            request=resp.request,
            response=resp,
        )

    def get_listing(
        self,
        path: str,
        *,
        after: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch one Listing page and return its ``children``."""
        params: dict[str, Any] = {"limit": min(limit, 100), "raw_json": "1", "sr_detail": "1"}
        if after:
            params["after"] = after
        resp = self._request("GET", _API_BASE + path, params=params)
        payload = resp.json()
        return payload["data"]["children"] if payload.get("kind") == "Listing" else []

    def get_raw(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        """Return parsed JSON for an arbitrary GET (path or absolute URL)."""
        url = path if path.startswith("http") else _API_BASE + path
        resp = self._request("GET", url, params=params)
        return resp.json()


def _b64(value: str) -> str:
    import base64

    return base64.b64encode(value.encode("ascii")).decode("ascii")


def _iter_submissions(client, subreddit, sort, limit):
    """Yield submission dicts for one subreddit, following the ``after`` cursor."""
    after = None
    collected = 0
    while True:
        children = client.get_listing(f"/r/{subreddit}/{sort}", after=after)
        for child in children:
            if child.get("kind") != "t3":
                continue
            yield child["data"]
            collected += 1
            if limit and collected >= limit:
                return
        if not children:
            return
        after = children[-1]["data"].get("name")
        if not after:
            return


def _get_comment_children(client, subreddit, post_id):
    """Return the top-level comment nodes (with nested replies) for a post."""
    path = f"/r/{subreddit}/comments/{post_id}"
    resp = client.get_raw(path, params={"limit": 100, "showmore": "1", "sort": "top"})
    listings = resp if isinstance(resp, list) else [resp]
    for entry in listings:
        if entry.get("kind") == "Listing":
            return entry["data"]["children"]
    return []


def _get_more_replies(client, link_id: str, children: list[str]) -> list[dict[str, Any]]:
    """Expand a ``more`` chunk into its comment nodes via /api/morechildren."""
    if not children:
        return []
    resp = client.get_raw(
        _MORE_URL,
        params={"api_type": "json", "link_id": link_id, "children": ",".join(children)},
    )
    payload = resp.get("json", resp) if isinstance(resp, dict) else {}
    return payload.get("data", {}).get("things", [])


def _flatten_comments(
    client,
    root_children: list[dict[str, Any]],
    *,
    post_permalink: str,
    max_count: int,
    max_depth: int,
) -> list[str]:
    """Breadth-first flatten of a comment tree with explicit ceilings."""
    lines: list[str] = []
    budget = max_count
    queue: deque[tuple[dict[str, Any], int]] = deque((child, 0) for child in root_children)
    while queue and budget > 0:
        node, depth = queue.popleft()
        kind = node.get("kind")
        data = node.get("data", {})

        if kind == "more":
            remaining = data.get("count") or len(data.get("children", []))
            if remaining <= 0:
                continue
            # Bounded expansion: fetch at most max_depth more-chunks worth of ids.
            more_ids = data.get("children", [])[: max(4, max_depth)]
            if depth + 1 <= max_depth:
                for extra in _get_more_replies(client, data.get("parent_id", ""), more_ids):
                    extra_data = extra.get("data", {})
                    queue.append(({"kind": extra.get("kind", "t1"), "data": extra_data}, depth + 1))
            if data.get("children", []):
                lines.append(f"{'  ' * depth}[more comment… truncated: {remaining} left]")
                budget -= 1
            continue

        author = data.get("author", "[deleted]")
        body = _one_line(data.get("body"))
        if not body:
            continue
        lines.append(f"{'  ' * depth}[+{data.get('score', 0)}] u/{author}: {body}")
        budget -= 1

        replies = data.get("replies")
        if isinstance(replies, dict) and replies.get("children"):
            for reply in replies["children"]:
                if depth + 1 <= max_depth:
                    queue.append((reply, depth + 1))

    if budget <= 0 or queue:
        lines.append("[more comment… truncated: more replies remain]")
    else:
        lines.append(f"(all {len(root_children)} comment thread(s) shown; see {post_permalink})")
    return lines


def _post_to_row(
    post: dict[str, Any],
    comments: list[str],
    *,
    permalink: str,
) -> dict[str, Any]:
    """Render a submission into a document row (allowlist fields only)."""

    def when_utc(value) -> str:
        return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(value)) if value else ""

    lines = [
        f"Subreddit: r/{post.get('subreddit')}",
        f"Author: u/{post.get('author')}",
        f"Score: {post.get('score', '?')}",
        f"Comments: {post.get('num_comments', '?')}",
        "Posted: " + (when_utc(post.get("created_utc")) or "—"),
    ]
    body = post.get("selftext") or post.get("selftext_html") or ""
    if body:
        lines += ["Body:", _one_line(body)]
    if comments:
        lines += ["", "Comments:", *comments]
    return {
        "id": post.get("name"),
        "url": permalink,
        "title": post.get("title"),
        "content": "\n".join(lines),
    }


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def reddit_source(
    client=None,
    *,
    client_id: str | None = None,
    client_secret: str | None = None,
    username: str | None = None,
    password: str | None = None,
    user_agent: str | None = None,
    subreddits: list[str] | None = None,
    sort: str = "hot",
    limit_per_subreddit: int = 50,
    max_comments_per_post: int = 100,
    max_comment_depth: int = 6,
):
    """Create a dlt source that yields Reddit posts as documents for ``remember``.

    Args:
        client: Pre-built :class:`RedditClient` (test injection point); when
            omitted one is built from the kwargs/env below.
        client_id: OAuth app client id (env ``REDDIT_CLIENT_ID``).
        client_secret: OAuth app secret (env ``REDDIT_CLIENT_SECRET``).
        username/password: Reddit account credentials (env ``REDDIT_USERNAME`` /
            ``REDDIT_PASSWORD``). Required for the script-app password grant.
        user_agent: Reddit User-Agent header (env ``REDDIT_USER_AGENT``).
        subreddits: Subreddits to crawl (default ``["all"]``).
        sort: Listing sort — hot, new, top, rising.
        limit_per_subreddit: Max submissions pulled per subreddit.
        max_comments_per_post: Max comment nodes included in each post document.
        max_comment_depth: Max reply depth rendered per thread.

    Returns:
        A dlt source suitable for ``cognee.add(...)`` / ``cognee.remember(...)``.
        Resources: ``reddit_posts``.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError(_EXTRA_HINT) from exc

    if client is None:
        client = _env_client(client_id, client_secret, username, password, user_agent)

    subs = subreddits or ["all"]

    @dlt.resource(name=REDDIT_TABLE_POSTS, primary_key="id", write_disposition="replace")
    def reddit_posts():
        count = 0
        for subreddit in subs:
            clean_sub = re.sub(r"^r/", "", subreddit.strip().lower())
            for post in _iter_submissions(client, clean_sub, sort, limit_per_subreddit):
                permalink = f"https://www.reddit.com{post.get('permalink', '')}"
                comments = _flatten_comments(
                    client,
                    _get_comment_children(client, clean_sub, post.get("id", "")),
                    post_permalink=permalink,
                    max_count=max_comments_per_post,
                    max_depth=max_comment_depth,
                )
                yield _post_to_row(post, comments, permalink=permalink)
                count += 1
        logger.info("Reddit: synced %d post(s).", count)

    @dlt.source(name=REDDIT_SOURCE_NAME)
    def _reddit():
        return [reddit_posts]

    source = _reddit()
    from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

    setattr(source, DOCUMENT_SOURCE_ATTR, REDDIT_SOURCE_NAME)
    return source


def _env_client(client_id, client_secret, username, password, user_agent) -> RedditClient:
    cid = client_id or os.environ.get("REDDIT_CLIENT_ID")
    secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET")
    user = username or os.environ.get("REDDIT_USERNAME")
    pwd = password or os.environ.get("REDDIT_PASSWORD")
    ua = user_agent or os.environ.get("REDDIT_USER_AGENT")
    if not cid or not secret:
        raise ValueError(
            "Reddit app credentials required: set REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET "
            "(and REDDIT_USERNAME/REDDIT_PASSWORD) or pass them to `reddit_source(...)`."
        )
    return RedditClient(cid, secret, username=user, password=pwd, user_agent=ua)


__all__ = [
    "REDDIT_SOURCE_NAME",
    "REDDIT_TABLE_POSTS",
    "RedditAPIError",
    "RedditClient",
    "reddit_source",
]
