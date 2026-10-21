# Adapted from topoteretes/cognee-community (Apache-2.0), PR #301, commit 4339aa4.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""OneNote connector for cognee: a ``dlt`` source that syncs one notebook through Microsoft Graph.

One source is one explicit notebook. It yields one document row per new or changed page, with
the page's place in the notebook as ``cognee_structure``, and is meant to be handed directly to
:func:`cognee.remember`. See ``onenote.md`` for what is read, the
incremental model, deletion and its gaps, limits and privacy.
"""

import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote, urlsplit

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    PIPELINE_SCOPE_ATTR,
    STRUCTURE_COLUMN,
    STRUCTURE_COLUMN_HINT,
)

_BASE = "https://graph.microsoft.com/v1.0/me/onenote"
_RENDERER_VERSION = 2
# One run may sleep at most this long in total across all its retries. The walk runs under
# cognee's process-wide dlt staging lock, so a longer wait stops the run instead.
RETRY_WAIT_BUDGET_SECONDS = 60
# Microsoft's OneNote error codes: 20113 "the resource specified in the request was deleted",
# 20102 "you requested a resource that doesn't exist". Only 20113 counts as proof of a deletion;
# 20102 also fits a resource this token can no longer see (see onenote.md).
_DELETED_CODE = "20113"
_NOT_FOUND_CODE = "20102"
logger = get_logger("onenote_connector")
Token = str | Callable[[], str]


class OneNoteError(RuntimeError):
    """Graph failure with safe diagnostics (never response bodies or bearer tokens)."""

    def __init__(self, message: str, *, status: int = 0, code: str = ""):
        super().__init__(message)
        self.status = status
        self.code = code


class OneNoteRateLimitedError(OneNoteError):
    """Graph answered 429, or asked for a longer wait than the retry budget allows."""


class OneNoteBudgetExceededError(OneNoteError):
    """The run spent its ``max_requests`` budget; it resumes on the next run."""


def _identity(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _path(value: str) -> str:
    return quote(value, safe="")


class _Graph:
    def __init__(
        self,
        token: Token,
        client: Any,
        *,
        stats=None,
        check_active=None,
        max_requests=300,
    ):
        import httpx

        self.token = token
        self.client = client or httpx.Client()
        self.owned = client is None
        self.stats = stats if stats is not None else {"requests": 0}
        self.check_active = check_active
        self.max_requests = max_requests
        # Seconds this run has spent waiting to retry, against RETRY_WAIT_BUDGET_SECONDS.
        self.slept = 0

    def ensure_active(self):
        if self.check_active is not None:
            self.check_active()

    def close(self):
        if self.owned:
            self.client.close()

    def request(self, url: str, *, html: bool = False):
        import httpx

        parsed = urlsplit(url)
        # Dot segments (also percent-encoded) are normalized by httpx, so a link under
        # /v1.0/me/onenote/ could otherwise send the token to another Graph endpoint.
        # A backslash, plain or encoded, is refused for the same reason.
        if any(part in (".", "..") for part in parsed.path.split("/")) or re.search(
            r"%2e|%5c|\\", parsed.path, re.IGNORECASE
        ):
            raise OneNoteError("Unsafe Graph URL rejected")
        if (
            parsed.scheme != "https"
            or parsed.hostname != "graph.microsoft.com"
            or parsed.port not in (None, 443)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or not parsed.path.startswith("/v1.0/me/onenote/")
        ):
            raise OneNoteError("Unsafe Graph URL rejected")
        for attempt in range(5):
            self.ensure_active()
            if self.stats["requests"] >= self.max_requests:
                raise OneNoteBudgetExceededError("Graph request budget exhausted")
            token = self.token() if callable(self.token) else self.token
            if not isinstance(token, str) or not token.strip():
                raise OneNoteError("A delegated access token is required")
            token = token.strip()
            if any(char.isspace() for char in token):
                raise OneNoteError("The access token must not contain whitespace")
            if not token.isascii() or not token.isprintable():
                raise OneNoteError("The access token is not a valid header value")
            self.stats["requests"] += 1
            # The failure is raised after the except block so no exception that mentions the
            # request (and so possibly the token) stays attached as __context__.
            failure = None
            try:
                response = self.client.get(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=30,
                    follow_redirects=False,
                )
            except httpx.LocalProtocolError:
                failure = "protocol"
            except httpx.TransportError:
                failure = "transport"
            if failure == "protocol":
                raise OneNoteError("Graph request could not be sent; check the access token")
            if failure == "transport":
                if attempt == 4:
                    raise OneNoteError("Graph transport failed after five attempts")
                delay = 2**attempt
                if self.slept + delay > RETRY_WAIT_BUDGET_SECONDS:
                    raise OneNoteError("Graph retry wait budget exhausted")
                self.slept += delay
                time.sleep(delay)
                continue
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                try:
                    delay = float(response.headers.get("Retry-After", 2**attempt))
                    if not math.isfinite(delay) or delay < 0:
                        delay = 2**attempt
                except ValueError:
                    delay = 2**attempt
                if self.slept + delay > RETRY_WAIT_BUDGET_SECONDS:
                    error = OneNoteRateLimitedError if response.status_code == 429 else OneNoteError
                    raise error(
                        "Graph Retry-After exceeds wait budget", status=response.status_code
                    )
                self.slept += delay
                time.sleep(delay)
                continue
            if response.status_code != 200:
                code = ""
                try:
                    code = str(response.json().get("error", {}).get("code", ""))
                except (ValueError, AttributeError):
                    # An unreadable error body leaves the code empty, which no caller treats
                    # as proof of a deletion, so the request fails closed.
                    pass
                message = (
                    "Graph rejected the token; reauthenticate the selected account"
                    if response.status_code == 401
                    else "Graph request failed; snapshot was not authorized for replacement"
                )
                error = OneNoteRateLimitedError if response.status_code == 429 else OneNoteError
                raise error(message, status=response.status_code, code=code)
            if html:
                if "text/html" not in response.headers.get("content-type", "").lower():
                    raise OneNoteError("Graph page content is not HTML")
                return response.text
            try:
                result = response.json()
            except ValueError:
                raise OneNoteError("Malformed Graph JSON") from None
            if not isinstance(result, dict):
                raise OneNoteError("Expected a Graph object")
            return result

    def collection(self, path: str):
        base = f"{_BASE}/{path}"
        url = f"{base}?$orderby=createdDateTime%20asc&$top=100&$skip=0"
        seen_urls, seen_ids = set(), set()
        offset = 0
        while True:
            if url in seen_urls:
                raise OneNoteError("Graph pagination cycle")
            seen_urls.add(url)
            response = self.request(url)
            rows = response.get("value")
            if not isinstance(rows, list):
                raise OneNoteError("Graph collection has no value array")
            for row in rows:
                _require(row, "id")
                if row["id"] in seen_ids:
                    raise OneNoteError("Graph collection changed or repeated during pagination")
                seen_ids.add(row["id"])
                yield row
            offset += len(rows)
            next_link = response.get("@odata.nextLink")
            if next_link is not None:
                if not isinstance(next_link, str) or not rows:
                    raise OneNoteError("Invalid Graph next link")
                url = next_link
            elif len(rows) == 100:
                url = f"{base}?$orderby=createdDateTime%20asc&$top=100&$skip={offset}"
            else:
                return


def _require(row, *fields):
    if not isinstance(row, dict) or any(
        not isinstance(row.get(field), str) or not row[field].strip() for field in fields
    ):
        raise OneNoteError("Graph metadata is missing required fields")


def _require_text(row, field):
    # A page may legitimately have an empty title, but the field must still be a string.
    if not isinstance(row, dict) or not isinstance(row.get(field), str):
        raise OneNoteError("Graph metadata is missing required fields")


def list_onenote_notebooks(token: Token, *, http_client: Any = None) -> list[dict]:
    """List notebook metadata for explicit selection; does not download page content."""
    graph = _Graph(token, http_client)
    try:
        return list(graph.collection("notebooks"))
    finally:
        graph.close()


def _body(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for image in soup.find_all("img"):
        alt = image.get("alt", "")
        reference = image.get("data-fullres-src") or image.get("src", "")
        image.replace_with(f"Image: {alt} ({reference})")
    for obj in soup.find_all("object"):
        obj.replace_with(f"Attachment: {obj.get('data-attachment', '')} ({obj.get('data', '')})")
    for anchor in soup.find_all("a", href=True):
        anchor.append(f" ({anchor['href']})")
    return soup.get_text(separator="\n", strip=True)


def _fields(page: dict) -> dict:
    """The row columns that come from page metadata, validated."""
    _require(page, "id", "lastModifiedDateTime")
    _require_text(page, "title")
    links = page.get("links", {})
    if not isinstance(links, dict) or not isinstance(links.get("oneNoteWebUrl", {}), dict):
        raise OneNoteError("Invalid page source links")
    url = links.get("oneNoteWebUrl", {}).get("href", "")
    if not isinstance(url, str):
        raise OneNoteError("Invalid page source URL")
    return {"id": page["id"], "title": page["title"].strip() or "Untitled", "url": url}


def _fingerprint(page: dict, fields: dict, ancestors: tuple[dict, ...]) -> str:
    """Everything that decides a page's row except its body: the body only changes with the
    timestamp, and the structure is not hashed by cognee but must be published when it moves."""
    return _identity(
        json.dumps(
            [
                page["lastModifiedDateTime"],
                _RENDERER_VERSION,
                fields["title"],
                fields["url"],
                list(ancestors),
            ],
            sort_keys=True,
        )
    )


def _row(fields: dict, ancestors: tuple[dict, ...], body: str) -> dict:
    return {**fields, "content": body, STRUCTURE_COLUMN: {"ancestors": list(ancestors)}}


def _containers(listings):
    """The container resources behind the listing paths a walk read as gone."""
    return sorted({re.sub(r"/(pages|sections|sectionGroups)$", "", path) for path in listings})


def _container_is_live(graph, container):
    try:
        graph.request(f"{_BASE}/{container}")
    except OneNoteError as exc:
        if exc.status in (404, 410) and exc.code == _DELETED_CODE:
            return False
        raise
    return True


def _lookup(graph, page_id, notebook_id):
    """Ask Graph about a page the listing no longer shows. Returns ``True`` when it is
    confirmed gone (deleted, or moved to another notebook) and ``False`` when Graph only says
    it does not exist, which is no proof."""
    try:
        page = graph.request(f"{_BASE}/pages/{_path(page_id)}?$expand=parentNotebook,parentSection")
    except OneNoteError as exc:
        if exc.status in (404, 410) and exc.code == _DELETED_CODE:
            return True
        if exc.status in (404, 410) and exc.code == _NOT_FOUND_CODE:
            # "Does not exist" also fits a page the token can no longer see, so it is
            # not proof of deletion: keep the page, never tombstone it, and ask again later.
            return False
        raise
    _require(page, "id", "lastModifiedDateTime")
    _require_text(page, "title")
    if page["id"] != page_id:
        raise OneNoteError("Page identity mismatch")
    parent = page.get("parentNotebook", {})
    _require(parent, "id")
    if parent["id"] == notebook_id:
        raise OneNoteError("Page moved during traversal; retry a fresh complete snapshot")
    # A positive move outside this notebook authorizes removing its old copy.
    return True


def _snapshot(graph, notebook_id, cache):
    """Build an authoritative view of the notebook.

    Returns ``(rows, present, unresolved)``: the rows of pages that are new or whose
    fingerprint changed, ``{page_id: fingerprint}`` of every listed page, and the pages kept
    because Graph answered 20102 for them. Bodies fetched for the rows are kept in
    ``cache["pending"]`` until a run publishes them, so a failed run does not repeat them.
    """
    pending = cache.setdefault("pending", {})
    published = cache.get("published", {})
    rows = {}
    present = {}
    locations = {}
    deleted_parent = False
    # Containers whose listing answered 20113 and was read as "no children" in this walk.
    swallowed = set()

    def children(path):
        try:
            items = []
            for child in graph.collection(path):
                # Observed reappearance invalidates old removal evidence even if
                # a later page, pagination batch or container makes this run fail.
                if path.endswith("/pages"):
                    cache.setdefault("confirmed_missing", {}).pop(child["id"], None)
                items.append(child)
            return items
        except OneNoteError as exc:
            if exc.status in (404, 410) and exc.code == _DELETED_CODE:
                swallowed.add(path)
                return []
            raise

    notebooks = {n["id"]: n for n in graph.collection("notebooks")}
    notebook = notebooks.get(notebook_id)
    if notebook is None:
        try:
            notebook = graph.request(f"{_BASE}/notebooks/{_path(notebook_id)}")
        except OneNoteError as exc:
            if exc.status not in (404, 410) or exc.code != _DELETED_CODE:
                raise
            deleted_parent = True
            swallowed.add(f"notebooks/{_path(notebook_id)}")
    if not deleted_parent:
        _require(notebook, "id", "displayName")
        if notebook["id"] != notebook_id:
            raise OneNoteError("Notebook identity mismatch")
        # A container's ancestors are nearest parent first, as cognee_structure wants them.
        root = ({"kind": "notebook", "id": notebook_id, "name": notebook["displayName"]},)
        pending_containers = [(f"notebooks/{_path(notebook_id)}", root)]
        seen = set()
        while pending_containers:
            container, ancestors = pending_containers.pop()
            if container in seen:
                raise OneNoteError("Section group cycle")
            seen.add(container)
            for section in children(f"{container}/sections"):
                _require(section, "id", "displayName")
                page_ancestors = (
                    {"kind": "section", "id": section["id"], "name": section["displayName"]},
                    *ancestors,
                )
                for page in children(f"sections/{_path(section['id'])}/pages"):
                    fields = _fields(page)
                    page_id = page["id"]
                    if page_id in locations:
                        raise OneNoteError("Page appeared in multiple locations during traversal")
                    locations[page_id] = page
                    graph.stats["scanned"] += 1
                    fingerprint = _fingerprint(page, fields, page_ancestors)
                    present[page_id] = fingerprint
                    if published.get(page_id) == fingerprint:
                        graph.stats["skipped"] += 1
                        continue
                    held = pending.get(page_id)
                    if held is None or held["fingerprint"] != fingerprint:
                        held = {
                            "fingerprint": fingerprint,
                            "text": _body(
                                graph.request(f"{_BASE}/pages/{_path(page_id)}/content", html=True)
                            ),
                        }
                        pending[page_id] = held
                    rows[page_id] = _row(fields, page_ancestors, held["text"])
            for group in children(f"{container}/sectionGroups"):
                _require(group, "id", "displayName")
                group_ancestors = (
                    {"kind": "sectionGroup", "id": group["id"], "name": group["displayName"]},
                    *ancestors,
                )
                pending_containers.append((f"sectionGroups/{_path(group['id'])}", group_ancestors))

    # Only a fresh complete walk can validate these checkpoints. A page restored
    # to this notebook must be confirmed again if it disappears on a later run.
    missing = set(published) - set(present)
    confirmed = cache.setdefault("confirmed_missing", {})
    # A confirmation carried over from an earlier failed run (typically a budget cut) lets a
    # large deletion finish across runs. It is only trusted while this walk reads no container
    # as "gone" that the run which recorded it did not. Otherwise a page restored in between
    # and omitted by a flapping listing would be forgotten without a fresh check.
    recorded = set(cache.get("swallowed", []))
    for page_id in set(confirmed) - missing:
        del confirmed[page_id]
    if swallowed - recorded:
        confirmed.clear()
        cache["swallowed"] = sorted(swallowed)
    else:
        cache["swallowed"] = sorted(recorded | swallowed)
        # The same containers answered 20113 as when the confirmations were recorded, but a
        # container that was restored in between and still answers 20113 for its listing would
        # look the same. Ask each one directly (one request per container); any that resolves
        # now voids the carried confirmations.
        if confirmed and any(_container_is_live(graph, name) for name in _containers(swallowed)):
            confirmed.clear()
    # The pages Graph answered 20102 for, oldest check first. One that is listed again is no
    # longer in question. Each one is checkpointed the moment it is classified.
    unresolved = [page_id for page_id in cache.get("unresolved", []) if page_id in missing]
    cache["unresolved"] = unresolved
    earlier = set(unresolved)
    for page_id in sorted(missing - set(confirmed) - earlier):
        if _lookup(graph, page_id, notebook_id):
            confirmed[page_id] = True
        else:
            unresolved.append(page_id)
    # Pages already unresolved are asked about again only with the budget the walk and the
    # confirmations left, oldest check first, so they can never starve the notebook. Running
    # out of budget here fails nothing: they stay kept and are asked about next run.
    for page_id in [page_id for page_id in unresolved if page_id in earlier]:
        try:
            gone = _lookup(graph, page_id, notebook_id)
        except OneNoteBudgetExceededError:
            break
        unresolved.remove(page_id)
        if gone:
            confirmed[page_id] = True
        else:
            unresolved.append(page_id)
    return rows, present, set(unresolved)


def onenote_source(
    token: Token,
    *,
    notebook_id: str,
    account_id: str,
    resource_name: str | None = None,
    http_client: Any = None,
    service: Any = None,
    check_active: Callable[[], None] | None = None,
    max_requests: int = 300,
):
    """Return one merge resource that syncs one OneNote notebook.

    A normal return is not proof of a finished sync. Hosts must require
    ``cognee_sync_stats['failed'] == 0`` and ``cognee_sync_stats['complete'] == 1``, and
    should watch ``cognee_sync_stats['unresolved']``: pages Graph answered 20102 for are
    kept, not forgotten, and do not fail the run. Expected Graph failures, throttling and a
    spent ``max_requests`` budget return normally, publish nothing and checkpoint progress;
    failures of ``check_active`` or of a token callable raise unchanged.

    ``token`` is a delegated access token or a zero-argument callable returning one.
    ``http_client`` and ``service`` are two names for one ``httpx.Client`` (give at most
    one). ``resource_name`` is normalized to dlt's snake-case table name and must stay the
    same across runs. See ``onenote.md`` for what is read, deletion, limits and privacy.
    """
    import dlt
    from dlt.common.normalizers.naming.snake_case import NamingConvention

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 3:
        raise RuntimeError(
            "OneNote sync requires a Cognee build that reads per-row structure "
            "(DOCUMENT_SYNC_VERSION >= 3). Upgrade Cognee before syncing."
        )
    if not account_id.strip() or not notebook_id.strip():
        raise ValueError("Stable account_id and explicit notebook_id are required")
    if http_client is not None and service is not None:
        raise ValueError("Supply only one of http_client and service")
    if not isinstance(max_requests, int) or isinstance(max_requests, bool) or max_requests <= 0:
        raise ValueError("max_requests must be a positive integer")
    if resource_name is not None and (
        not isinstance(resource_name, str) or not resource_name.strip()
    ):
        raise ValueError("resource_name must be a nonempty, non-whitespace string")
    identity = _identity(json.dumps([account_id, notebook_id], ensure_ascii=True))
    name = NamingConvention().normalize_table_identifier(
        resource_name if resource_name is not None else f"onenote_{identity}"
    )
    stats: dict[str, int] = {}

    @dlt.resource(
        name=name,
        primary_key="id",
        write_disposition="merge",
        columns={
            **{
                field: {"data_type": "text", "nullable": False}
                for field in ("id", "title", "content", "url")
            },
            **STRUCTURE_COLUMN_HINT,
            "_deleted": {"data_type": "bool", "hard_delete": True},
        },
    )
    def records():
        # Pipeline-level hints may be applied to a cloned resource. Inspect the
        # extraction source as well as this public resource before any I/O.
        current = dlt.current.source().resources[name]
        if current.write_disposition != "merge" or resource.write_disposition != "merge":
            raise ValueError("OneNote requires write_disposition='merge'")
        stats.clear()
        stats.update(
            dict.fromkeys(
                (
                    "scanned",
                    "skipped",
                    "failed",
                    "deleted",
                    "failed_rate_limit",
                    "failed_budget",
                    "unresolved",
                    "requests",
                    "complete",
                ),
                0,
            )
        )
        state = dlt.current.resource_state()
        if state.get("connection", identity) != identity:
            raise ValueError("Resource name is already bound to another OneNote connection")
        state["connection"] = identity
        cache = state.setdefault("onenote_cache", {})
        graph = _Graph(
            token,
            http_client if http_client is not None else service,
            stats=stats,
            check_active=check_active,
            max_requests=max_requests,
        )
        try:
            try:
                rows, present, unresolved = _snapshot(graph, notebook_id, cache)
            except OneNoteError as exc:
                stats["failed"] += 1
                stats["failed_rate_limit"] += int(isinstance(exc, OneNoteRateLimitedError))
                stats["failed_budget"] += int(isinstance(exc, OneNoteBudgetExceededError))
                logger.warning(
                    "OneNote sync incomplete; retaining documents and checkpointing progress",
                    reason=str(exc),
                )
                yield dlt.mark.materialize_table_schema()
                return
            graph.ensure_active()
            if unresolved:
                stats["unresolved"] = len(unresolved)
                logger.warning(
                    "OneNote pages answered not found; keeping their memory and retrying",
                    pages=len(unresolved),
                )
            published = cache.get("published", {})
            deleted = set(published) - set(present) - unresolved
            for page_id in sorted(deleted):
                graph.ensure_active()
                yield {"id": page_id, "title": "", "content": "", "url": "", "_deleted": True}
            for page_id in sorted(rows):
                graph.ensure_active()
                yield rows[page_id]
            if not rows and not deleted:
                yield dlt.mark.materialize_table_schema()
            # Unresolved pages keep the fingerprint of the row that is still staged.
            cache["published"] = {
                **{page_id: published[page_id] for page_id in unresolved},
                **present,
            }
            cache.pop("pending", None)
            cache.pop("confirmed_missing", None)
            cache.pop("swallowed", None)
            stats["deleted"] = len(deleted)
            stats["complete"] = 1
        finally:
            graph.close()

    resource = records()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "onenote")
    setattr(resource, PIPELINE_SCOPE_ATTR, name)
    resource.cognee_sync_stats = stats
    return resource
