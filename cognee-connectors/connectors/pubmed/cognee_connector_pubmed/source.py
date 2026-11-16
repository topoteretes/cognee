"""PubMed connector for cognee: a ``dlt`` source that turns the literature into memory.

One source is one PubMed query: the articles whose *entry date* (``edat``) falls in a date
window, matching a caller-supplied search term, each rendered as one document row. It is
meant to be handed directly to :func:`cognee.remember`::

    import cognee
    from cognee_connector_pubmed import pubmed_source

    await cognee.remember(
        pubmed_source(
            api_key="<ncbi api key, optional>",
            resource_name="pubmed_alzheimers",
            query="alzheimer disease[title/abstract]",
        ),
        dataset_name="pubmed",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}`` with the id
  prefixed ``pmid:<pmid>``. Every column feeds the stored document identity, so the rendering
  and the node set are frozen and deterministic: nothing volatile is rendered and lists
  (authors, keywords, publication types) keep the order PubMed sends, which is stable for a
  given record.
* **Node set** ``pubmed:<resource_name>`` scopes a ``recall`` to this connection.
* **What is rendered** is what ``efetch`` returns as article metadata: title, authors,
  journal, year, DOI, publication types, keywords and the abstract. Each text opens with one
  sentence naming its kind. Nothing else of the record (chemicals, MeSH headings, grants,
  references) is rendered.
* **Incremental by fingerprint.** PubMed indexes entry dates, not change feeds. Each run
  asks ``esearch`` for the ids whose ``edat`` lies in the window — id lists are cheap — and
  fetches (``efetch``) only the ids it has no fingerprint for: the fingerprints, kept in
  dlt resource state, are the cursor. A correction that PubMed does not re-date is not
  re-synced: ``edat`` does not move on a revision, and this connector does not re-read what
  it has fingerprinted (see the README, "Limitations").
* **Forget-on-delete.** The window is listed in full on every run, so an id that stopped
  appearing is spotted however old it is. One confirming ``<pmid>[uid]`` lookup tells a
  genuine removal (count 0) from an inconsistent index, and the row is emitted as a
  ``_deleted`` tombstone; any other answer, an error included, keeps the record. An id
  whose entry date left the window (a narrowed ``start_date``) is deselected and forgotten
  at once, without a check.
* **A run is bounded and resumable.** On a 429, or a spent request budget after rows were
  emitted, the source stops cleanly, keeps the state it reached (listing offset, fetch
  queue, pending deletes) and reports why in ``cognee_sync_stats``
  (``failed_rate_limit``, ``failed_budget``); the next run resumes where it stopped. A
  listing that cannot be proven complete never forgets anything.
* **One query per resource_name.** The query term and the window bounds are hashed into the
  listing state: changing them restarts the listing from offset 0, and ids outside the new
  window are forgotten like deletions on the next complete run.
* **Auth** is an optional NCBI API key. Without one, NCBI allows about 3 requests a second;
  with one, 10. The client spaces its requests accordingly and never waits out a 429. The
  key rides in query parameters, so a redirect is an error, not followed — a hostile answer
  could otherwise harvest it.
"""

import hashlib
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from typing import Any

from cognee_connectors_core import dlt_conventions as dlt_utils
from cognee_connectors_core.dlt_conventions import DOCUMENT_SOURCE_ATTR, NODE_SET_COLUMN

logger = logging.getLogger(__name__)

EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_PATH = "/esearch.fcgi"
EFETCH_PATH = "/efetch.fcgi"
ARTICLE_URL = "https://pubmed.ncbi.nlm.nih.gov"
# esearch's page size; below its deep-paging cap NCBI documents at 10,000.
PAGE_SIZE = 500
# esearch answers at most 10,000 ids without a history cursor; a bigger window is listed as
# capped and forgets nothing, so no record can be mistaken for deleted.
SEARCH_CAP = 10_000
# PMIDs per efetch request: NCBI accepts hundreds of ids per call; this is request packing,
# not a data cutoff.
EFETCH_BATCH = 200
DEFAULT_MAX_REQUESTS = 500
# Seconds NCBI asks between requests: about 3/s without a key, 10/s with one.
_MIN_INTERVAL_NO_KEY = 0.4
_MIN_INTERVAL_WITH_KEY = 0.12
# One pace per clock: clients that read the same monotonic clock share their last-request
# timestamp (in production they all default to time.monotonic, and NCBI's limit is per
# caller, not per client object), while independently-clocked clients pace themselves.
_REQUEST_PACES: dict[Callable[[], float], float] = {}
# (connect, read) seconds; our own choice, NCBI documents no expected latency.
_TIMEOUT = (10, 60)
# Attempts per request on 5xx or network errors, backing off 1s then 2s.
_RETRIES = 3
# YYYY/MM/DD, the only spelling esearch's datetype=edat accepts.
_DATE = re.compile(r"\d{4}/\d{2}/\d{2}")
# A PMID goes into a search term (`123[uid]`), so only digits may get through.
_PMID = re.compile(r"[0-9]+")


class PubMedSourceError(RuntimeError):
    """Base class of the errors this source raises."""


class PubMedAPIError(PubMedSourceError):
    """NCBI answered with an error, or could not be reached."""


class PubMedAuthError(PubMedAPIError):
    """NCBI rejected the API key (HTTP 400 naming the key)."""


class PubMedRateLimitedError(PubMedAPIError):
    """NCBI answered HTTP 429."""


class _Transient(Exception):
    """One failed attempt worth another: a network error (``response`` None) or a 5xx answer."""

    def __init__(self, response: Any, reason: str):
        super().__init__(reason)
        self.response = response
        self.reason = reason


class PubMedClient:
    """Minimal synchronous client for NCBI E-utilities.

    Spaces its requests to NCBI's polite limits (3/s without an API key, 10/s with one),
    reading the clock and sleeping through the injectable ``monotonic`` and ``sleep``.
    The pace is shared between clients that read the same clock — in production they all
    default to ``time.monotonic``, and NCBI's limit is per caller, not per client object.
    Tests and hosts can pass any object with the same ``request`` method instead: ``request``
    takes a ``check_active`` keyword and must call it before each request attempt, retries
    and waits included.
    """

    def __init__(
        self,
        api_key: str = "",
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        key = (api_key or "").strip()
        # This check is what keeps the key out of an exception: requests names the value of
        # a header or param it rejects, and every key it would reject is refused here.
        if key and (not key.isascii() or not key.isprintable() or " " in key):
            raise ValueError("The PubMed API key is empty or has an invalid format")
        self._api_key = key
        self._sleep = sleep
        self._monotonic = monotonic
        self._min_interval = _MIN_INTERVAL_WITH_KEY if key else _MIN_INTERVAL_NO_KEY
        _REQUEST_PACES.setdefault(monotonic, 0.0)

    def __repr__(self) -> str:
        return "PubMedClient(api_key=<redacted>)"

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        as_text: bool = False,
        check_active: Callable[[], None] | None = None,
    ) -> Any:
        import requests

        query = {"tool": "cognee", **(params or {})}
        if self._api_key:
            query["api_key"] = self._api_key

        def attempt() -> Any:
            self._respect_interval()
            try:
                response = requests.request(
                    method,
                    EUTILS_URL + path,
                    params=query,
                    timeout=_TIMEOUT,
                    # The key rides in the query string: a hop would carry it to whatever
                    # host the answer names, unlike an Authorization header.
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                raise _Transient(None, type(exc).__name__) from None
            if response.status_code >= 500:
                raise _Transient(response, f"HTTP {response.status_code}")
            return self._parse(response, path, as_text)

        try:
            return dlt_utils.with_retries(
                attempt,
                retry_on=(_Transient,),
                delay=lambda error, number: 2**number,
                attempts=_RETRIES,
                check_active=check_active,
                sleep=self._sleep,
            )
        except _Transient as error:
            if error.response is None:
                raise PubMedAPIError(f"PubMed request failed: {error.reason}") from None
            return self._parse(error.response, path, as_text)

    def _respect_interval(self) -> None:
        now = self._monotonic()
        wait = self._min_interval - (now - _REQUEST_PACES[self._monotonic])
        if wait > 0:
            self._sleep(wait)
            now = self._monotonic()
        _REQUEST_PACES[self._monotonic] = now

    def _parse(self, response: Any, path: str, as_text: bool) -> Any:
        status = response.status_code
        if 300 <= status < 400:
            raise PubMedAPIError(f"PubMed answered {path} with a redirect (HTTP {status})")
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if status == 400 and isinstance(payload, dict):
            message = str(payload.get("error") or "")
            if "api key" in message.lower():
                raise PubMedAuthError("PubMed rejected the API key (HTTP 400)")
        if status == 429:
            raise PubMedRateLimitedError("PubMed rate limit exceeded (HTTP 429)")
        if status != 200:
            raise PubMedAPIError(f"PubMed request failed: HTTP {status}")
        if as_text:
            return response.text
        return payload


def build_pubmed_service(api_key: str = "") -> PubMedClient:
    """Wrap a caller-owned API key in a client, the way a host storing keys would."""
    return PubMedClient(api_key)


# ---------------------------------------------------------------------------
# Parsing and rendering: deterministic, metadata and abstract only.
# ---------------------------------------------------------------------------


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _pub_date(container: ET.Element | None) -> str:
    """The year of a PubDate, else its MedlineDate's leading year, else empty."""
    if container is None:
        return ""
    year = _text(container.find("Year"))
    if year:
        return year
    medline_date = _text(container.find("MedlineDate"))
    match = re.match(r"\d{4}", medline_date)
    return match.group(0) if match else ""


def parse_article(xml_text: str) -> dict[str, Any]:
    """One ``PubmedArticle`` element of an efetch answer, flattened to what is rendered.

    Raises :class:`PubMedAPIError` when the XML carries no article or no PMID: the batch it
    came from cannot be trusted, and forgetting nothing is better than trusting an
    unreadable answer.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise PubMedAPIError(f"PubMed sent unreadable article XML: {exc}") from None
    article = root if root.tag == "PubmedArticle" else root.find(".//PubmedArticle")
    if article is None:
        raise PubMedAPIError("PubMed sent an efetch answer without an article")
    pmid = _text(article.find("./MedlineCitation/PMID"))
    if not pmid:
        raise PubMedAPIError("PubMed sent an article without a PMID")

    citation = article.find("./MedlineCitation")
    body = citation.find("Article") if citation is not None else None
    journal = body.find("Journal") if body is not None else None
    authors = []
    for author in body.findall("./AuthorList/Author") if body is not None else []:
        collective = _text(author.find("CollectiveName"))
        if collective:
            authors.append(collective)
            continue
        name = " ".join(
            part
            for part in (_text(author.find("LastName")), _text(author.find("Initials")))
            if part
        )
        if name:
            authors.append(name)
    abstract_lines = []
    for part in body.findall("./Abstract/AbstractText") if body is not None else []:
        label = part.get("Label")
        text = _text(part)
        if text:
            abstract_lines.append(f"{label}: {text}" if label else text)
    keywords = [
        _text(keyword)
        for keyword in (citation.findall("./KeywordList/Keyword") if citation is not None else [])
        if _text(keyword)
    ]
    types = [
        _text(kind)
        for kind in (
            body.findall("./PublicationTypeList/PublicationType") if body is not None else []
        )
        if _text(kind)
    ]
    doi = ""
    for article_id in article.findall("./PubmedData/ArticleIdList/ArticleId"):
        if article_id.get("IdType") == "doi":
            doi = _text(article_id)
            break
    edat = ""
    for date in article.findall("./PubmedData/History/PubMedPubDate"):
        if date.get("PubStatus") == "entrez":
            year, month, day = (_text(date.find(tag)) for tag in ("Year", "Month", "Day"))
            if year and month and day:
                edat = f"{int(year):04d}/{int(month):02d}/{int(day):02d}"
            break
    return {
        "pmid": pmid,
        "title": _text(body.find("ArticleTitle")) if body is not None else "",
        "authors": authors,
        "journal": _text(journal.find("Title")) if journal is not None else "",
        "year": _pub_date(journal.find("./JournalIssue/PubDate")) if journal is not None else "",
        "doi": doi,
        "types": types,
        "keywords": keywords,
        "abstract": "\n".join(abstract_lines),
        "edat": edat,
    }


def _lines(fields: list[tuple[str, str]]) -> list[str]:
    return [f"{label}: {value}" for label, value in fields if value]


def render_article(article: dict[str, Any], node_set: list[str]) -> dict[str, Any]:
    """An article's row. The text opens with one sentence naming its kind ("PubMed article
    ..."), then ``Label: value`` lines; the abstract, which spans paragraphs, comes last."""
    title = article["title"] or "Untitled article"
    fields = [
        ("Authors", ", ".join(article["authors"])),
        ("Journal", article["journal"]),
        ("Year", article["year"]),
        ("DOI", article["doi"]),
        ("Publication types", ", ".join(article["types"])),
        ("Keywords", ", ".join(article["keywords"])),
    ]
    content = "\n".join([f'PubMed article "{title}".', *_lines(fields)])
    if article["abstract"]:
        content += f"\nAbstract: {article['abstract']}"
    return {
        "id": f"pmid:{article['pmid']}",
        "title": title,
        "content": content,
        "url": f"{ARTICLE_URL}/{article['pmid']}/",
        "_deleted": False,
        NODE_SET_COLUMN: node_set,
    }


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
class _CutShortError(Exception):
    """Internal: stop the run here, keeping the state reached so far."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _Walker:
    """One run over the window, writing progress into ``state``.

    Each run pages ``esearch`` through the window's id list (cheap: ids only), fetches
    (``efetch``) only ids it has no fingerprint for — draining the queue in batches after
    each listing page, so a cut run resumes its queue — then forgets ids that left the
    complete listing: verified one by one, or at once when a narrowed window deselected
    them.
    """

    def __init__(
        self,
        client: Any,
        state: dict,
        stats: dict[str, int],
        *,
        query: str | None,
        start_date: str | None,
        end_date: str,
        max_requests: int,
        check_active: Callable[[], None] | None,
    ):
        self.client = client
        self.state = state
        self.stats = stats
        for key in ("scanned", "skipped", "fetched", "deleted", "failed", "requests"):
            stats.setdefault(key, 0)
        self.query = query
        self.start_date = start_date
        self.end_date = end_date
        self.max_requests = max_requests
        self.check_active = check_active
        self.emitted = 0
        self._deselected: list[str] = []
        # Ids listed in the current cycle, their fingerprints and entry dates: dlt resource
        # state, so a run cut mid-listing resumes and a finished cycle is remembered.
        self.fingerprints: dict[str, str] = state.setdefault("fingerprints", {})
        self.edat: dict[str, str] = state.setdefault("edat", {})
        self.seen: list[str] = state.setdefault("seen", [])
        name = state.get("name")
        self.node_set = [f"pubmed:{name}"] if isinstance(name, str) and name else []

    # -- requests ----------------------------------------------------------
    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        # Before the budget, so a revoked run ends as revoked, not as a resumable cut.
        if self.check_active is not None:
            self.check_active()
        if self.stats["requests"] >= self.max_requests:
            raise _CutShortError("budget")
        # A call the client retries on a 5xx or a network error still counts once: the
        # budget and this count are about what the run asks for, and the client owns its
        # retries.
        self.stats["requests"] += 1
        try:
            return self.client.request(method, path, check_active=self.check_active, **kwargs)
        except PubMedRateLimitedError:
            raise _CutShortError("rate_limit") from None
        except PubMedAuthError:
            if self.emitted:
                raise _CutShortError("auth") from None
            raise

    def _esearch(self, params: dict[str, Any]) -> dict[str, Any]:
        query = {"db": "pubmed", "retmode": "json", **params}
        if "term" not in query and self.query:
            query["term"] = self.query
        payload = self._call("GET", ESEARCH_PATH, params=query)
        result = payload.get("esearchresult") if isinstance(payload, dict) else None
        if not isinstance(result, dict):
            raise PubMedAPIError("PubMed search sent an unreadable answer")
        return result

    # -- listing -----------------------------------------------------------
    def _window_params(self, extra: dict[str, Any]) -> dict[str, Any]:
        params = dict(extra)
        if self.start_date:
            params["mindate"] = self.start_date
        if self.end_date:
            params["maxdate"] = self.end_date
        params["datetype"] = "edat"
        return params

    def _listing(self) -> Iterator[list[str]]:
        """The window's PMIDs one page at a time, resuming at the offset a cut run stopped on."""
        progress = self.state.setdefault("listing", {})
        window = {"query": self.query, "start": self.start_date, "end": self.end_date}
        if progress.get("window") != window:
            progress.clear()
            progress.update({"window": window, "retstart": 0, "count": None})
            self.seen.clear()
            # The queue holds ids the old window listed; the new listing re-encounters the
            # ones still inside it, so nothing queued is trusted across a window change.
            # Cleared in place: rows() already holds a reference to this list.
            self.state.setdefault("pending_fetch", []).clear()
        count = progress.get("count")
        if count is None:
            first = self._esearch(self._window_params({"retstart": 0, "retmax": PAGE_SIZE}))
            ids = _idlist(first)
            count = _count(first)
            progress["count"] = count
            progress["retstart"] = len(ids)
            yield ids
        if count > SEARCH_CAP:
            progress["capped"] = True
        # Below the cap, a listing that stopped at ``count`` ids is complete; beyond it
        # esearch has no more ids to give, and the run says so instead of trusting it.
        while progress["retstart"] < min(count, SEARCH_CAP):
            page = self._esearch(
                self._window_params({"retstart": progress["retstart"], "retmax": PAGE_SIZE})
            )
            ids = _idlist(page)
            yield ids
            progress["retstart"] += len(ids)
            if not ids:
                break

    # -- fetching ----------------------------------------------------------
    def _fetch(self, queue: list[str]) -> Iterator[dict]:
        """Fetch and emit the first batch of the state-backed ``queue``.

        The batch leaves the queue only after it resolves, so a run cut by the request
        budget or a 429 retries the batch next run instead of losing it. Ids already
        fingerprinted (a resumed run re-draining its queue) are skipped without a request.
        The batch is trusted only whole: every asked id must answer before any of its
        rows is committed to state or emitted, so a listing-shaped hole in the answer
        leaves no partial fingerprint behind.
        """
        batch = queue[:EFETCH_BATCH]
        needed = [pmid for pmid in batch if f"pmid:{pmid}" not in self.fingerprints]
        if needed:
            payload = self._call(
                "GET",
                EFETCH_PATH,
                params={"db": "pubmed", "id": ",".join(needed), "retmode": "xml"},
                as_text=True,
            )
            if not isinstance(payload, str) or "<PubmedArticle" not in payload:
                raise PubMedAPIError("PubMed article fetch sent an unreadable answer")
            articles = {
                article["pmid"]: article
                for article in (parse_article(chunk) for chunk in _article_chunks(payload))
            }
            rendered = []
            for pmid in needed:
                article = articles.get(pmid)
                if article is None:
                    # The answer cannot be matched to what was asked; the run raises
                    # instead of trusting a listing-shaped hole, and nothing is forgotten.
                    raise PubMedAPIError(f"PubMed article fetch answered without PMID {pmid}")
                rendered.append((render_article(article, self.node_set), article["edat"]))
            for row, edat in rendered:
                self.fingerprints[row["id"]] = _hash(row)
                if edat:
                    self.edat[row["id"]] = edat
                self.seen.append(row["id"])
                self.emitted += 1
                self.stats["fetched"] += 1
                yield row
        del queue[: len(batch)]

    # -- forget-on-delete ---------------------------------------------------
    def _close_cycle(self) -> None:
        """Split the ids that left the listing into deselected and to-verify, then reset."""
        progress = self.state.get("listing") or {}
        if progress.get("capped"):
            # A capped listing is incomplete, so nothing missing can be trusted as deleted.
            self.stats["failed"] += 1
            self.stats["failed_cap"] = 1
            logger.warning(
                "PubMed search is capped at %d records; narrow the query or the date "
                "window. Deletions are not detected until then.",
                SEARCH_CAP,
            )
            self._deselected = []
        else:
            seen = set(self.seen)
            floor = self.start_date or ""
            self._deselected = []
            pending = []
            for row_id in sorted(self.fingerprints):
                if row_id in seen:
                    continue
                row_edat = self.edat.get(row_id, "")
                if row_edat and floor and row_edat < floor:
                    # Narrowed out of the window: like a filter a user dropped, forgotten
                    # at once and without a check.
                    self._deselected.append(row_id)
                else:
                    pending.append(row_id)
            self.state["pending_deletes"] = pending
        self.state["listing"] = {}
        self.seen.clear()

    def _forget(self) -> Iterator[dict]:
        pending = self.state.setdefault("pending_deletes", [])
        while pending:
            row_id = pending[0]
            if self._is_gone(row_id):
                self.fingerprints.pop(row_id, None)
                self.edat.pop(row_id, None)
                self.stats["deleted"] += 1
                yield {"id": row_id, "_deleted": True}
            pending.pop(0)

    def _is_gone(self, row_id: str) -> bool:
        """Whether the article left the index. A check that fails keeps the record."""
        pmid = row_id.removeprefix("pmid:")
        if not _PMID.fullmatch(pmid):
            self.stats["failed"] += 1
            self.stats["failed_deletion_check"] = self.stats.get("failed_deletion_check", 0) + 1
            logger.warning("PubMed deletion check refused a non-numeric id; the record is kept")
            return False
        try:
            # No window bounds and no query term: a uid lookup answers whether the index
            # still holds the article, whatever dates it carries.
            result = self._esearch({"term": f"{pmid}[uid]"})
        except PubMedAPIError as exc:
            self.stats["failed"] += 1
            self.stats["failed_deletion_check"] = self.stats.get("failed_deletion_check", 0) + 1
            logger.warning("PubMed deletion check failed for one record; it is kept: %s", exc)
            return False
        return _count(result) == 0

    # -- the run --------------------------------------------------------------
    def rows(self) -> Iterator[dict]:
        # Finish deletions a cut run left behind before starting a new listing.
        yield from self._forget()
        queue: list[str] = self.state.setdefault("pending_fetch", [])
        queued = set(queue)
        for page in self._listing():
            for pmid in page:
                row_id = f"pmid:{pmid}"
                self.stats["scanned"] += 1
                if row_id in self.fingerprints:
                    self.stats["skipped"] += 1
                    self.seen.append(row_id)
                    continue
                if pmid not in queued:
                    queue.append(pmid)
                    queued.add(pmid)
            # Fetch after each page, so a run cut mid-window keeps at most one page's
            # worth of queued ids and the next run drains the queue before re-listing.
            while len(queue) >= EFETCH_BATCH:
                yield from self._fetch(queue)
        while queue:
            yield from self._fetch(queue)
        self._close_cycle()
        for row_id in self._deselected:
            self.fingerprints.pop(row_id, None)
            self.edat.pop(row_id, None)
            self.stats["deleted"] += 1
            yield {"id": row_id, "_deleted": True}
        yield from self._forget()


def _count(result: dict[str, Any]) -> int:
    count = result.get("count")
    try:
        return int(count)
    except (TypeError, ValueError):
        raise PubMedAPIError("PubMed search answered without a count") from None


def _idlist(result: dict[str, Any]) -> list[str]:
    ids = result.get("idlist")
    if not isinstance(ids, list) or not all(isinstance(pmid, str) for pmid in ids):
        raise PubMedAPIError("PubMed search answered without an id list")
    return ids


def _article_chunks(payload: str) -> Iterator[str]:
    """Split an efetch answer into ``PubmedArticle`` elements for one-at-a-time parsing."""
    for match in re.finditer(r"<PubmedArticle>.*?</PubmedArticle>", payload, flags=re.DOTALL):
        yield match.group(0)


def _iter_rows(
    client: Any,
    state: dict,
    stats: dict[str, int],
    *,
    query: str | None = None,
    start_date: str | None = None,
    end_date: str,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict]:
    """Yield the changed rows of a query's window. Pure of dlt, so tests drive it with a dict."""
    walker = _Walker(
        client,
        state,
        stats,
        query=query,
        start_date=start_date,
        end_date=end_date,
        max_requests=max_requests,
        check_active=check_active,
    )
    try:
        yield from walker.rows()
    except _CutShortError as cut:
        stats["failed"] += 1
        stats[f"failed_{cut.reason}"] = 1
        logger.warning("PubMed sync stopped early (%s); it resumes on the next run.", cut.reason)
    except PubMedSourceError:
        # Every raise counts as a failure; a host reads these counts after the raise: dlt
        # drops the state, not this object.
        stats["failed"] += 1
        raise


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def _date_arg(name: str, value: str) -> str:
    if not _DATE.fullmatch(value):
        raise ValueError(f"{name} must be a date like YYYY/MM/DD, e.g. 2026/01/31")
    return value


def pubmed_source(
    *,
    resource_name: str,
    api_key: str | None = None,
    service: Any = None,
    query: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    today: Callable[[], str] | None = None,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    check_active: Callable[[], None] | None = None,
):
    """Return a ``dlt`` resource that yields a PubMed query's documents for ``remember``.

    Args:
        resource_name: Stable name of this connection, one per query you sync. It scopes
            the sync state, the node set and the cleanup of deleted records, so keep it
            fixed across runs.
        api_key: Optional NCBI API key. It raises NCBI's rate limit from 3 to 10 requests
            a second; without one the client spaces its requests for the lower limit.
        service: Pre-built client (see :func:`build_pubmed_service`). Mainly an injection
            point for tests and hosts. Its ``request`` takes a ``check_active`` keyword
            and must call it before each request attempt, retries and waits included.
        query: PubMed search term selecting what to ingest, in esearch syntax (for
            example ``"alzheimer disease"[title/abstract]``). None syncs every article
            the window's dates reach, so narrow wide corpora with the dates below.
        start_date: Lower bound of the entry-date window, ``YYYY/MM/DD``. None leaves
            the window open downwards (up to everything matching ``query``). The window
            is fixed for the life of a resource_name: narrowing it later forgets what
            falls outside, widening it syncs what the new floor reaches.
        end_date: Upper bound of the window, ``YYYY/MM/DD``; None means today, so the
            window grows with the calendar and each run picks up what entered the index
            since the last one — while fetching only what it has never fingerprinted.
        today: The clock the default ``end_date`` reads, as a ``YYYY/MM/DD`` string.
            Mainly for tests.
        max_requests: Budget of API calls in one run. A call the client retries counts
            once. A bigger window finishes over several runs.
        check_active: Optional host authorization checkpoint, called before each
            request.

    Returns:
        A ``dlt`` resource configured with ``primary_key="id"``,
        ``write_disposition="merge"`` and a ``_deleted`` hard-delete column.
    """
    import dlt

    dlt_utils.require_document_sync_version(2, provider="PubMed", feature="reads per-row node sets")
    if not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("pubmed_source needs a resource_name, one per PubMed query")
    if query is not None and not isinstance(query, str):
        raise ValueError("query must be a PubMed search term string or None")
    begin = _date_arg("start_date", start_date) if start_date is not None else None
    stop = _date_arg("end_date", end_date) if end_date is not None else None
    if stop is None:
        clock = today or (lambda: time.strftime("%Y/%m/%d"))
        stop = _date_arg("today", clock())
    if begin and stop and begin > stop:
        raise ValueError(f"start_date {begin} is after end_date {stop}")
    if not isinstance(max_requests, int) or max_requests < 1:
        raise ValueError("max_requests must be a positive integer")

    client = service if service is not None else PubMedClient(api_key or "")
    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def pubmed_query():
        stats.clear()
        state = dlt.current.resource_state()
        state["name"] = resource_name
        rows = _iter_rows(
            client,
            state,
            stats,
            query=query,
            start_date=begin,
            end_date=stop,
            max_requests=max_requests,
            check_active=check_active,
        )
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = pubmed_query()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "pubmed")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
