"""arXiv connector for cognee: a ``dlt`` source that turns arXiv papers into memory.

One source is a selection of arXiv categories and/or authors. It yields one document
per paper (metadata and abstract) and is meant to be handed directly to
:func:`cognee.remember`::

    import cognee
    from cognee_connector_arxiv import arxiv_source

    await cognee.remember(
        arxiv_source(categories=["cs.AI"], resource_name="arxiv_cs_ai"),
        dataset_name="arxiv",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``. Ids are
  ``arxiv:<arXiv id>`` without a version suffix, so a new version of a paper replaces
  its document. Every column this source emits feeds the stored document identity, so
  the rendering is frozen and deterministic: collections are sorted, no volatile
  fields. A tombstone carries no node set.
* **Node set** ``arxiv:category:<code>`` per category the paper lists (sorted), so a
  ``recall`` can be scoped to one category.
* **Auth: none.** The arXiv API is public; the source sends no credential. The caller
  selects what to ingest with ``categories`` (arXiv category codes, ``cs.AI``) and/or
  ``authors`` (normalized arXiv author names, ``lecun_y``); at least one is required,
  and an empty selection can never mean everything.
* **Incremental** by ``submittedDate``. The first run queries the whole selection (or
  from ``submitted_after`` when given); every later run queries only
  ``submittedDate:[cursor TO now]``, where the cursor is the newest ``published``
  timestamp of the previous completed run. A paper already synced is re-rendered only
  when its ``updated`` timestamp changed. The range is inclusive on both ends and a
  run dedupes what it already saw, so a paper that shares the cursor's second costs
  one upsert, never a duplicate document.
* **Forget-on-delete.** arXiv has no deletion feed, so a run settles deletions only
  inside the window it re-queried: a previously synced paper whose ``published``
  falls into the window and that the listing no longer contains is emitted as a
  ``_deleted`` tombstone. A paper deleted long after sync, behind the cursor, is not
  re-checked (arXiv almost never removes papers; see the package README).
* **Rate limit is built in, not discovered.** arXiv asks for roughly one request
  every three seconds; the client waits ``request_delay`` seconds (default 3.0)
  before every request but the first of a run. There is no retry: a failed request
  aborts the run, ``dlt`` rolls the resource state back and the next run starts over.
* **Pagination.** A listing pages through the API with ``max_results`` (default 100,
  the page size), sorted by ``submittedDate`` ascending, until the feed reports no
  further entries. A non-200 answer, an unparsable feed, or a feed missing the fields
  a row is built from raises :class:`ArxivSourceError`: an incomplete listing must
  never look like deletions.
"""

import logging
import re
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any
from xml.etree import ElementTree

from cognee_connectors_core import dlt_conventions as dlt_utils
from cognee_connectors_core.dlt_conventions import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
)

logger = logging.getLogger(__name__)

API_BASE_URL = "https://export.arxiv.org/api/query"
# arXiv's fair-use limit, built in: one request every ~3 seconds.
DEFAULT_REQUEST_DELAY = 3.0
DEFAULT_MAX_RESULTS = 100
# Version of the resource state; a later release reads it to migrate an older layout.
STATE_VERSION = 1

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"
_OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"


class ArxivSourceError(RuntimeError):
    """Base error. Messages carry the endpoint and the reason, never a query's content."""


class ArxivResponseError(ArxivSourceError):
    """The API answered something other than a parsable Atom feed of the expected shape."""


def _bare_id(entry_id: str) -> str:
    """``http://arxiv.org/abs/2401.12345v2`` -> ``2401.12345`` (version stripped)."""
    bare = entry_id.rstrip("/").rsplit("/", 1)[-1].removesuffix(".pdf").removesuffix(".abs")
    return re.sub(r"v\d+$", "", bare)


def _to_arxiv_time(value: datetime) -> str:
    """A timezone-aware datetime as arXiv's ``submittedDate`` format (UTC ``YYYYMMDDHHMM``)."""
    return value.astimezone(timezone.utc).strftime("%Y%m%d%H%M")


def parse_window_start(submitted_after: str | datetime | None) -> str | None:
    """``submitted_after`` as an arXiv ``submittedDate`` bound; ``None`` is the whole history.

    A string is read as an ISO date or datetime (``2024-01-01``, ``2024-01-01T12:30``);
    a naive one is UTC. Anything unparseable is a ``ValueError``, because a mistyped
    bound would silently re-query the whole history.
    """
    if submitted_after is None:
        return None
    if isinstance(submitted_after, datetime):
        parsed = submitted_after
    elif isinstance(submitted_after, str):
        try:
            parsed = datetime.fromisoformat(submitted_after)
        except ValueError:
            raise ValueError(
                f"submitted_after {submitted_after!r} is not an ISO date or datetime "
                "(like '2024-01-01' or '2024-01-01T12:30')."
            ) from None
    else:
        raise ValueError("submitted_after must be an ISO date/datetime string or a datetime.")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return _to_arxiv_time(parsed)


class ArxivClient:
    """The arXiv API client: one request at a time, ``request_delay`` seconds apart.

    ``requests`` is imported on the first call, not at module import, so the package
    imports without it. ``sleep`` is injectable so tests need no real waiting.
    """

    def __init__(
        self,
        *,
        base_url: str = API_BASE_URL,
        request_delay: float = DEFAULT_REQUEST_DELAY,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base_url = base_url
        self._request_delay = request_delay
        self._sleep = sleep
        self._requested = False

    def get_feed(self, params: dict[str, Any]) -> ElementTree.Element:
        """One ``GET /api/query`` as a parsed Atom feed, rate-limited."""
        if self._requested:
            self._sleep(self._request_delay)
        self._requested = True
        import requests

        try:
            response = requests.get(self._base_url, params=params, timeout=60)
        except requests.RequestException as error:
            raise ArxivResponseError(
                f"arXiv API request failed ({type(error).__name__}); nothing was synced "
                "and nothing was forgotten."
            ) from error
        if response.status_code != 200:
            raise ArxivResponseError(
                f"arXiv API answered HTTP {response.status_code}; nothing was synced and "
                "nothing was forgotten."
            )
        try:
            return ElementTree.fromstring(response.text)
        except ElementTree.ParseError as error:
            raise ArxivResponseError(
                f"arXiv API answered an unparsable feed ({error}); nothing was synced and "
                "nothing was forgotten."
            ) from error


def build_arxiv_service(
    *, request_delay: float = DEFAULT_REQUEST_DELAY, sleep: Callable[[float], None] = time.sleep
) -> ArxivClient:
    """The default client a host or a caller hands to :func:`arxiv_source` as ``service=``."""
    return ArxivClient(request_delay=request_delay, sleep=sleep)


def _text(element: ElementTree.Element | None) -> str:
    """An element's text, whitespace-normalized; empty when the element is absent."""
    if element is None or element.text is None:
        return ""
    return " ".join(element.text.split())


def _parse_timestamp(element: ElementTree.Element | None, context: str) -> str:
    value = _text(element)
    if not value:
        raise ArxivResponseError(f"arXiv feed entry is missing {context}; the run stops.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ArxivResponseError(
            f"arXiv feed entry has an unparseable {context} ({value!r}); the run stops."
        ) from None
    return parsed.astimezone(timezone.utc).isoformat()


def _parse_entry(entry: ElementTree.Element) -> dict[str, Any]:
    """One feed entry as the record a row is rendered from."""
    entry_id = _text(entry.find(f"{_ATOM}id"))
    if not entry_id:
        raise ArxivResponseError("arXiv feed entry has no id; the run stops.")
    authors = sorted(
        {_text(author.find(f"{_ATOM}name")) for author in entry.findall(f"{_ATOM}author")}
    )
    authors = [author for author in authors if author]
    categories = sorted(
        {category.get("term", "") for category in entry.findall(f"{_ATOM}category")}
    )
    categories = [category for category in categories if category]
    return {
        "bare_id": _bare_id(entry_id),
        "title": _text(entry.find(f"{_ATOM}title")),
        "abstract": _text(entry.find(f"{_ATOM}summary")),
        "authors": authors,
        "categories": categories,
        "published": _parse_timestamp(entry.find(f"{_ATOM}published"), "published"),
        "updated": _parse_timestamp(entry.find(f"{_ATOM}updated"), "updated"),
        "comment": _text(entry.find(f"{_ARXIV}comment")),
        "journal_ref": _text(entry.find(f"{_ARXIV}journal_ref")),
        "doi": _text(entry.find(f"{_ARXIV}doi")),
    }


def _parse_page(feed: ElementTree.Element) -> tuple[int, list[dict[str, Any]]]:
    """A feed as ``(total_results, entries)``; a feed of the wrong shape raises."""
    if feed.tag != f"{_ATOM}feed":
        raise ArxivResponseError(
            f"arXiv API answered a {feed.tag!r} document, not an Atom feed; nothing was "
            "synced and nothing was forgotten."
        )
    total = _text(feed.find(f"{_OPENSEARCH}totalResults"))
    try:
        total_results = int(total) if total else None
    except ValueError:
        total_results = None
    if total_results is None:
        # Without the total a truncated listing could look like deletions, so a feed
        # that does not report it is treated as a wrong shape, not as an empty page.
        raise ArxivResponseError(
            f"arXiv feed reports a missing or unparseable totalResults ({total!r}); the run stops."
        )
    return total_results, [_parse_entry(entry) for entry in feed.findall(f"{_ATOM}entry")]


def _search_query(categories: list[str], authors: list[str], window: str) -> str:
    terms = [
        *(f"cat:{category}" for category in categories),
        *(f"au:{author}" for author in authors),
    ]
    return " AND ".join([*terms, f"submittedDate:[{window}]"])


def _paper_id(bare_id: str) -> str:
    return f"arxiv:{bare_id}"


def _paper_row(paper: dict[str, Any]) -> dict[str, Any]:
    """One document row. The content is the paper's metadata and abstract, deterministically."""
    lines = [
        f"arXiv paper {paper['bare_id']}"
        + (f" in {', '.join(paper['categories'])}." if paper["categories"] else "."),
        "",
        f"Title: {paper['title']}",
        f"Authors: {', '.join(paper['authors'])}",
        "",
        f"Abstract: {paper['abstract']}",
        "",
        f"Submitted: {paper['published'][:10]}",
        f"Updated: {paper['updated'][:10]}",
    ]
    if paper["comment"]:
        lines.append(f"Comment: {paper['comment']}")
    if paper["journal_ref"]:
        lines.append(f"Journal reference: {paper['journal_ref']}")
    if paper["doi"]:
        lines.append(f"DOI: {paper['doi']}")
    row: dict[str, Any] = {
        "id": _paper_id(paper["bare_id"]),
        "title": paper["title"],
        "content": "\n".join(lines),
        "url": f"https://arxiv.org/abs/{paper['bare_id']}",
        "_deleted": False,
    }
    if paper["categories"]:
        row[NODE_SET_COLUMN] = [f"arxiv:category:{category}" for category in paper["categories"]]
    return row


def _list_papers(
    service: Any,
    search_query: str,
    max_results: int,
    stats: dict[str, int],
) -> list[dict[str, Any]]:
    """The full listing of one run's window, paged until the feed runs dry."""
    papers: list[dict[str, Any]] = []
    start = 0
    while True:
        feed = service.get_feed(
            {
                "search_query": search_query,
                "start": start,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "ascending",
            }
        )
        total_results, entries = _parse_page(feed)
        papers.extend(entries)
        stats["requests"] += 1
        start += len(entries)
        if not entries or start >= total_results:
            return papers


def _validate_selection(
    categories: list[str] | None, authors: list[str] | None
) -> tuple[list[str], list[str]]:
    """The selection, sorted and without repeats; a bare string is refused.

    An empty selection would match nothing and the next run would forget every synced
    paper, so it is refused. arXiv category codes and author names are passed through
    as written; a typo matches nothing (nothing is forgotten on an empty listing).
    """

    def checked(values: list[str] | None, label: str) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str) or not isinstance(values, list) or not values:
            raise ValueError(f"{label} must be a list of non-empty strings, not {values!r}.")
        selection = sorted({value.strip() for value in values})
        if not selection or not all(selection):
            raise ValueError(f"{label} must be a list of non-empty strings, not {values!r}.")
        return selection

    selected_categories = checked(categories, "categories")
    selected_authors = checked(authors, "authors")
    if not selected_categories and not selected_authors:
        raise ValueError(
            "arxiv_source needs at least one of categories or authors; an empty "
            "selection would forget every synced paper on the next run."
        )
    return selected_categories, selected_authors


def _iter_rows(
    service: Any,
    categories: list[str],
    authors: list[str],
    window_start: str | None,
    max_results: int,
    state: dict,
    stats: dict[str, int],
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield a row for every new or changed paper of the window, then the window's tombstones.

    The stored cursor, once a completed run wrote one, overrides the factory's
    ``window_start``: every later run re-queries only ``[cursor TO now]``. The listing
    is complete before the first row is emitted, so an error mid-listing yields
    nothing and, because ``dlt`` rolls a failed run's state back, forgets nothing.
    The cursor is written only after the last row: a run cut short re-runs its window
    on the next call, and the overlap is settled by the seen set.
    """
    cursor = state.get("cursor")
    if cursor is not None:
        try:
            window_start = _to_arxiv_time(datetime.fromisoformat(cursor))
        except ValueError:
            raise ArxivSourceError(
                f"The stored sync cursor {cursor!r} is not a timestamp; the state is "
                "corrupt. Nothing was synced and nothing was forgotten."
            ) from None
    window_end = _to_arxiv_time(datetime.now(timezone.utc))
    if window_start is not None and window_start > window_end:
        raise ArxivSourceError(
            "The stored sync cursor is in the future; the clock or the state is wrong. "
            "Nothing was synced and nothing was forgotten."
        )
    window = f"{window_start or '000000000000'} TO {window_end}"
    query = _search_query(categories, authors, window)

    papers = _list_papers(service, query, max_results, stats)
    # A copy: a run that fails after this point must leave the caller's state untouched
    # (dlt rolls a resource's state back on failure; a direct caller keeps its dict).
    members: dict[str, dict[str, str]] = dict(state.get("members", {}))
    window_start_iso = _cursor_iso(window_start) if window_start is not None else None
    seen: set[str] = set()
    newest: str | None = None
    for paper in papers:
        if check_active is not None:
            check_active()  # outside any try: a revoked connection is not a failed paper
        paper_key = _paper_id(paper["bare_id"])
        seen.add(paper_key)
        # The cursor advances over every paper the window returned, unchanged ones too:
        # they are confirmed present, so the next run's window can start past them.
        if newest is None or paper["published"] > newest:
            newest = paper["published"]
        member = members.get(paper_key)
        if member is not None and member.get("updated") == paper["updated"]:
            continue  # unchanged since the run that synced it
        yield _paper_row(paper)
        stats["synced"] += 1
        members[paper_key] = {"submitted": paper["published"], "updated": paper["updated"]}

    for paper_key in sorted(
        key
        for key, member in members.items()
        if key not in seen
        and (window_start_iso is None or member.get("submitted", "") >= window_start_iso)
    ):
        yield {"id": paper_key, "_deleted": True}
        stats["deleted"] += 1
        del members[paper_key]

    if newest is not None:
        state["cursor"] = newest
    state.update(v=STATE_VERSION, members=members)
    logger.info("arXiv: %d paper(s) synced, %d forgotten.", stats["synced"], stats["deleted"])


def _cursor_iso(window_start: str) -> str:
    """A ``YYYYMMDDHHMM`` window bound as the ISO instant it names, for member comparison."""
    return datetime.strptime(window_start, "%Y%m%d%H%M").replace(tzinfo=timezone.utc).isoformat()


def arxiv_source(
    *,
    categories: list[str] | None = None,
    authors: list[str] | None = None,
    resource_name: str = "arxiv_papers",
    submitted_after: str | datetime | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    request_delay: float = DEFAULT_REQUEST_DELAY,
    service: Any | None = None,
    check_active: Callable[[], None] | None = None,
):
    """Return a ``dlt`` resource that yields one document per arXiv paper of the selection.

    Hand the result to ``cognee.remember(...)`` with ``write_disposition="merge"`` and
    ``primary_key="id"``. The first run ingests the selection (or only what
    ``submitted_after`` covers); every later run queries only the papers submitted
    since the previous completed run and forgets the ones that disappeared from that
    window.

    Args:
        categories: arXiv category codes to ingest, like ``["cs.AI", "cs.LG"]``.
        authors: Normalized arXiv author names, like ``["lecun_y"]``. At least one of
            ``categories`` and ``authors`` is required; both combine with AND.
        resource_name: Names the staging table and the sync state. Two selections in
            one dataset need different names.
        submitted_after: Only papers submitted at or after this instant on the first
            run; an ISO date/datetime string or a datetime. Later runs use the stored
            cursor instead.
        max_results: Page size of the listing requests (arXiv's ``max_results``).
        request_delay: Seconds the client waits between API requests; arXiv asks for
            roughly one request every 3 seconds.
        service: A pre-built client (see :func:`build_arxiv_service`) for tests or a
            host; a default one is built when omitted.
        check_active: Host authorization checkpoint, called before each row is
            published.
    """
    import dlt

    dlt_utils.require_document_sync_version(3, provider="arXiv", feature="reads per-row node sets")
    selected_categories, selected_authors = _validate_selection(categories, authors)
    window_start = parse_window_start(submitted_after)
    if not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("resource_name must be a non-empty string.")
    if max_results < 1:
        raise ValueError("max_results must be a positive integer.")
    client = service if service is not None else build_arxiv_service(request_delay=request_delay)
    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def arxiv_papers():
        stats.update(synced=0, deleted=0, requests=0)
        rows = _iter_rows(
            client,
            selected_categories,
            selected_authors,
            window_start,
            max_results,
            dlt.current.resource_state(),
            stats,
            check_active,
        )
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = arxiv_papers()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "arxiv")
    setattr(resource, PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
