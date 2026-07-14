"""Confluence connector for cognee — a ``dlt`` source that turns a wiki into memory.

Pull Confluence Cloud pages (and, optionally, their footer comments) into
cognee, incrementally and with forget-on-deletion — "ask my wiki".  Like the
sibling Gmail connector this builds entirely on the existing DLT ingestion
subsystem; the source produced here is handed directly to
:func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import confluence_source

    await cognee.remember(
        confluence_source(
            base_url="https://your-domain.atlassian.net",
            email="you@example.com",
            api_token="…",
            space_keys=["ENG"],
        ),
        dataset_name="my_wiki",
        primary_key="id",
        write_disposition="merge",   # incremental upsert by page id
        max_rows_per_table=0,        # 0 = no row cap (see note below)
    )

Design
------
* **Auth** — Confluence Cloud API token.  Pass the account ``email`` and an
  `API token <https://id.atlassian.com/manage-profile/security/api-tokens>`_;
  they are sent as HTTP Basic auth.  Access is read-only — the connector only
  issues ``GET`` requests.
* **Primary key** — the Confluence page ``id``.  Combined with
  ``write_disposition="merge"`` this gives idempotent upserts.
* **Incremental cursor** — the page's last-version timestamp
  (``version.createdAt``, a.k.a. ``version.when``).  Each run lists the current
  pages and emits only those modified since the highest timestamp seen so far;
  the cursor is persisted in dlt's per-resource state, so re-running
  ``remember`` resumes where it left off and re-embeds only the delta.
* **Forget-on-delete** — Confluence has no deletion feed, so each run does a
  lightweight id sweep of the space(s) and compares it against the ids seen on
  the previous run (also kept in resource state).  Pages that vanished are
  emitted with the ``_deleted`` hard-delete marker; dlt removes those rows on
  ``merge`` and cognee's existing ``orphan_cleanup`` then purges them from the
  graph + vector + relational stores.  No parallel cleanup path.

.. note::
   cognee's ``ingest_dlt_source`` reads at most ``max_rows_per_table`` rows
   from the dlt destination (default 50).  For a real space pass
   ``max_rows_per_table=0`` (unlimited) so orphan-cleanup compares against the
   *whole* synced corpus rather than a truncated window.

.. note::
   Footer comments are folded into their page's text and are (re)captured
   whenever that page is synced.  A comment added without any page edit does
   not bump ``version.createdAt``, so it is picked up on the page's next change.
"""

from __future__ import annotations

import html
import re
from typing import Any, Dict, Iterator, List, Optional, Set

from cognee.shared.logging_utils import get_logger

logger = get_logger("confluence_connector")

# Confluence Cloud REST API v2 lives under this path on the site.
_API_BASE = "/wiki/api/v2"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Auth / HTTP helpers
# ---------------------------------------------------------------------------
def _make_session(email: str, api_token: str) -> Any:
    """Build a ``requests`` session authenticated with a Confluence API token.

    ``requests`` is imported lazily so it stays an optional dependency
    (``pip install "cognee[confluence]"``).
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            'The Confluence connector requires "requests". Install the extra:\n'
            '    pip install "cognee[confluence]"'
        ) from exc

    session = requests.Session()
    session.auth = (email, api_token)
    session.headers.update({"Accept": "application/json"})
    return session


def _api_get(session: Any, base_url: str, path_or_url: str, params: Optional[dict] = None) -> dict:
    """GET a Confluence API path (or a ready-made pagination URL) and return JSON."""
    url = path_or_url if path_or_url.startswith("http") else f"{base_url}{path_or_url}"
    response = session.get(url, params=params or {})
    response.raise_for_status()
    return response.json()


def _paginate(session: Any, base_url: str, path: str, params: dict) -> Iterator[dict]:
    """Yield ``results`` items across all pages, following ``_links.next``.

    The v2 ``next`` link is a site-relative path that already carries the
    cursor, so subsequent requests drop the initial query params.
    """
    next_path: Optional[str] = path
    next_params = dict(params)
    while next_path:
        data = _api_get(session, base_url, next_path, next_params)
        for item in data.get("results", []) or []:
            yield item
        next_path = (data.get("_links") or {}).get("next")
        next_params = {}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _clean_html(raw: Optional[str]) -> str:
    """Strip Confluence storage-format markup down to plain text.

    Storage format is XHTML with macros; feeding raw tags into entity
    extraction is noisy, so we drop tags, unescape entities, and collapse
    whitespace. Dependency-free on purpose — no HTML parser required.
    """
    if not raw:
        return ""
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", raw))).strip()


def _version_when(page: dict) -> str:
    """Return the page's last-version timestamp (``version.createdAt``).

    Falls back to the v1 ``version.when`` field name so the cursor works
    regardless of which representation the site returns.
    """
    version = page.get("version") or {}
    return version.get("createdAt") or version.get("when") or ""


def _page_to_row(page: dict, body: str, base_url: str) -> Dict[str, Any]:
    """Flatten a Confluence page (+ its resolved body text) into a dlt row."""
    webui = (page.get("_links") or {}).get("webui") or ""
    if webui and not webui.startswith("http"):
        webui = f"{base_url}/wiki{webui}"

    return {
        "id": str(page.get("id")),
        "title": page.get("title") or "",
        "space_id": str(page.get("spaceId") or ""),
        "url": webui,
        "version_when": _version_when(page),
        "body": body,
        # Hard-delete marker (always False for live pages). Vanished pages are
        # emitted separately with _deleted=True.
        "_deleted": False,
    }


def _deleted_row(page_id: str) -> Dict[str, Any]:
    """Build a minimal row that instructs dlt to hard-delete a page by id."""
    return {"id": str(page_id), "_deleted": True}


# ---------------------------------------------------------------------------
# Confluence API reads
# ---------------------------------------------------------------------------
def _resolve_space_ids(session: Any, base_url: str, space_keys: Optional[List[str]]) -> List[str]:
    """Resolve space keys to numeric v2 space ids (all accessible spaces if None)."""
    params: Dict[str, Any] = {"limit": 250}
    if space_keys:
        params["keys"] = ",".join(space_keys)
    return [
        str(space["id"]) for space in _paginate(session, base_url, f"{_API_BASE}/spaces", params)
    ]


def _page_body(session: Any, base_url: str, page_id: str) -> str:
    """Fetch a single page's storage-format body and return it as plain text."""
    data = _api_get(session, base_url, f"{_API_BASE}/pages/{page_id}", {"body-format": "storage"})
    storage = (data.get("body") or {}).get("storage") or {}
    return _clean_html(storage.get("value"))


def _page_comments(session: Any, base_url: str, page_id: str) -> List[str]:
    """Fetch a page's footer comments as plain text."""
    texts: List[str] = []
    for comment in _paginate(
        session,
        base_url,
        f"{_API_BASE}/pages/{page_id}/footer-comments",
        {"body-format": "storage"},
    ):
        storage = (comment.get("body") or {}).get("storage") or {}
        text = _clean_html(storage.get("value"))
        if text:
            texts.append(text)
    return texts


# ---------------------------------------------------------------------------
# Sync (pure given a session + state dict — unit-testable)
# ---------------------------------------------------------------------------
def sync_pages(
    session: Any,
    base_url: str,
    state: dict,
    *,
    space_keys: Optional[List[str]] = None,
    include_comments: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Yield changed pages since the last run, plus hard-delete markers.

    One listing pass per space enumerates the *current* pages (cheap, no
    bodies): that set drives deletion detection, while pages newer than the
    stored cursor have their body (and comments) fetched and emitted.  The
    cursor (``last_when``) and the id set (``known_ids``) are advanced in
    ``state`` so the next run is a no-op when nothing changed.
    """
    known_ids: Set[str] = set(state.get("known_ids", []))
    last_when: str = state.get("last_when", "")
    newest_when = last_when
    current_ids: Set[str] = set()
    changed = 0

    for space_id in _resolve_space_ids(session, base_url, space_keys):
        for page in _paginate(
            session, base_url, f"{_API_BASE}/spaces/{space_id}/pages", {"limit": 250}
        ):
            page_id = str(page["id"])
            current_ids.add(page_id)

            when = _version_when(page)
            # Skip only pages we have already ingested and that have not changed.
            # A page absent from known_ids is fetched regardless of timestamp, so
            # pages new to the corpus but with an old version.when (moved into a
            # tracked space, restored, or tied at the cursor boundary) are not lost.
            if page_id in known_ids and when <= last_when:
                continue
            if when > newest_when:
                newest_when = when

            body = _page_body(session, base_url, page_id)
            if include_comments:
                comments = _page_comments(session, base_url, page_id)
                if comments:
                    body = f"{body}\n\nComments:\n" + "\n\n".join(comments)
            yield _page_to_row(page, body, base_url)
            changed += 1

    # Deletion detection relies on the sweep enumerating every current page. An
    # empty sweep while pages were previously known almost always means a
    # transient/failed listing (network blip, renamed/typo'd space key, momentary
    # zero-space enumeration) rather than a genuine wipe — treating it as "all
    # deleted" would purge the whole dataset and overwrite known_ids with [],
    # making the loss permanent. Skip deletion and preserve state in that case.
    if known_ids and not current_ids:
        logger.warning(
            "Confluence: page sweep returned 0 pages but %d were known; skipping "
            "deletion this run to avoid a mass forget-on-delete on a transient sweep.",
            len(known_ids),
        )
        state["last_when"] = newest_when
        logger.info("Confluence: %d changed page(s), 0 deletion(s).", changed)
        return

    deleted = known_ids - current_ids
    for page_id in sorted(deleted):
        yield _deleted_row(page_id)

    state["known_ids"] = sorted(current_ids)
    state["last_when"] = newest_when
    logger.info("Confluence: %d changed page(s), %d deletion(s).", changed, len(deleted))


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def confluence_source(
    *,
    base_url: str,
    email: Optional[str] = None,
    api_token: Optional[str] = None,
    space_keys: Optional[List[str]] = None,
    include_comments: bool = True,
    session: Any = None,
):
    """Return a ``dlt`` resource that yields Confluence pages for ``remember``.

    Args:
        base_url: Site base URL, e.g. ``https://your-domain.atlassian.net``.
        email: Atlassian account email (Basic-auth username).
        api_token: Confluence API token (Basic-auth password).
        space_keys: Restrict to these space keys (e.g. ``["ENG"]``). ``None``
            syncs every space the token can read.
        include_comments: Fold each page's footer comments into its text.
        session: Pre-built ``requests`` session. Mainly an injection point for
            tests; when omitted one is built from ``email`` / ``api_token``.

    Returns:
        A ``dlt`` resource (``confluence_pages``) configured with
        ``primary_key="id"``, ``write_disposition="merge"`` and an ``_deleted``
        hard-delete column. Hand it to ``cognee.remember(...)``.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError(
            'The Confluence connector requires the dlt extra: pip install "cognee[confluence]".'
        ) from exc

    base_url = base_url.rstrip("/")
    if session is None and not (email and api_token):
        raise ValueError("confluence_source requires email and api_token (or an injected session).")

    @dlt.resource(
        name="confluence_pages",
        primary_key="id",
        write_disposition="merge",
        # _deleted is a boolean hard-delete marker: rows where it is True are
        # removed from the dlt destination on merge, which propagates the
        # deletion through cognee's orphan_cleanup.
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def confluence_pages():
        client = session or _make_session(email, api_token)
        resource_state = dlt.current.resource_state()
        yield from sync_pages(
            client,
            base_url,
            resource_state,
            space_keys=space_keys,
            include_comments=include_comments,
        )

    return confluence_pages
