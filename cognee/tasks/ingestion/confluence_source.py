"""Confluence DLT source for cognee.

Two separate paths, intentionally decoupled:

1. `pages` resource  -> INCREMENTAL. Only yields pages changed since last
   sync (cursor: version.when). This is what gets embedded/graphed.

2. `get_all_current_page_ids()` -> FULL SWEEP, not incremental. Fetches
   *every* page id currently in Confluence (cheap: fields=id only).
   This is what orphan_cleanup diffs against — NEVER feed it the
   incremental delta, or every unchanged page looks "deleted" and gets
   wiped on the next sync.
"""

from typing import List, Optional, Set, Callable, Any

class CogneeDltSourceWrapper:
    """Safely wraps a DltSource alongside a full-ID sweep function for accurate orphan cleanup."""
    def __init__(self, source: Any, sweep_fn: Callable[[], Set[str]]):
        self.source = source
        self.get_all_current_page_ids = sweep_fn


def create_confluence_source(
    base_url: str,
    email: str,
    api_token: str,
    space_keys: Optional[List[str]] = None,
):
    """Create a dlt source yielding Confluence pages (incremental) + comments."""
    import dlt

    @dlt.source(name="confluence")
    def confluence_source():

        @dlt.resource(
            name="pages",
            primary_key="id",
            write_disposition="merge",
        )
        def pages(
            updated_at=dlt.sources.incremental(
                "version.when",
                initial_value="1970-01-01T00:00:00.000Z",
            ),
        ):
            """Yield only pages whose version.when is newer than last sync."""
            session = _make_session(email, api_token)
            spaces = space_keys or _get_all_space_keys(session, base_url)

            for space_key in spaces:
                url = f"{base_url}/wiki/api/v2/spaces/{space_key}/pages"
                params = {
                    "limit": 250,
                    "sort": "modified-date",
                    "body-format": "storage",
                }
                yield from _paginate(session, url, params)

        @dlt.resource(
            name="comments",
            primary_key="id",
            write_disposition="merge",
        )
        def comments():
            """Yield footer comments for synced pages."""
            session = _make_session(email, api_token)
            spaces = space_keys or _get_all_space_keys(session, base_url)

            for space_key in spaces:
                page_ids = _get_page_ids_for_space(session, base_url, space_key)
                for page_id in page_ids:
                    url = f"{base_url}/wiki/api/v2/pages/{page_id}/footer-comments"
                    params = {"limit": 250, "body-format": "storage"}
                    yield from _paginate(session, url, params)

        return pages, comments
    
    # Return the safe wrapper rather than monkey-patching the DLT object
    return CogneeDltSourceWrapper(
        source=confluence_source(),
        sweep_fn=lambda: get_all_current_page_ids(base_url, email, api_token, space_keys)
    )


def get_all_current_page_ids(
    base_url: str,
    email: str,
    api_token: str,
    space_keys: Optional[List[str]] = None,
) -> Set[str]:
    """Full, non-incremental sweep of every page id Confluence currently has."""
    session = _make_session(email, api_token)
    spaces = space_keys or _get_all_space_keys(session, base_url)

    all_ids: Set[str] = set()
    for space_key in spaces:
        all_ids.update(_get_page_ids_for_space(session, base_url, space_key))
    return all_ids


def _get_page_ids_for_space(session, base_url: str, space_key: str) -> Set[str]:
    """Id-only listing for one space — used by both comments() and the orphan sweep."""
    ids: Set[str] = set()
    url = f"{base_url}/wiki/api/v2/spaces/{space_key}/pages"
    params = {"limit": 250, "select": "id"}  # id-only, no body/version payload
    while url:
        resp = session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("results", []):
            ids.add(item["id"])
        next_link = data.get("_links", {}).get("next")
        if next_link:
            url = next_link if next_link.startswith("http") else f"{base_url}{next_link}"
            params = {}  # next link already carries query params
        else:
            url = None
    return ids


def _make_session(email: str, api_token: str):
    import requests
    session = requests.Session()
    session.auth = (email, api_token)
    session.headers.update({"Accept": "application/json"})
    return session


def _get_all_space_keys(session, base_url: str) -> List[str]:
    """Fetch every space key the authenticated user can access."""
    keys: List[str] = []
    url = f"{base_url}/wiki/api/v2/spaces"
    params = {"limit": 250}
    while url:
        resp = session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        for space in data.get("results", []):
            keys.append(space["key"])
        next_link = data.get("_links", {}).get("next")
        if next_link:
            url = next_link if next_link.startswith("http") else f"{base_url}{next_link}"
            params = {}
        else:
            url = None
    return keys


def _paginate(session, url, params):
    """Generic Confluence v2 paginator. dlt's incremental() filters transparently
    when this feeds the `pages` resource; used as a plain paginator elsewhere."""
    while url:
        resp = session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("results", []):
            yield item
        next_link = data.get("_links", {}).get("next")
        if next_link:
            url = next_link if next_link.startswith("http") else f"{url.split('/wiki')[0]}{next_link}"
            params = {}
        else:
            url = None
