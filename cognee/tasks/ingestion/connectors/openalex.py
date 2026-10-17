"""OpenAlex scholarly graph connector for cognee.

The connector exposes OpenAlex works as document-mode dlt rows.  Each row keeps
the useful graph identifiers (authors, institutions, topics, venue, and cited
works) alongside readable work text, allowing normal cognee ingestion to build
the semantic graph without adding an OpenAlex SDK dependency.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from datetime import date, datetime, timezone
from typing import Any

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, NODE_SET_COLUMN

logger = get_logger("openalex_connector")

OPENALEX_API_BASE_URL = "https://api.openalex.org"
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 200
MAX_RETRIES = 5


class OpenAlexAPIError(RuntimeError):
    """An OpenAlex response that cannot be recovered by retrying."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def decode_abstract_inverted_index(index: dict[str, list[int]] | None) -> str:
    """Turn OpenAlex's word-to-token-position map into ordinary prose."""
    if not index:
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        words.extend((position, word) for position in positions if isinstance(position, int))
    return " ".join(word for _, word in sorted(words))


def _retry_after(headers: Any, attempt: int) -> float:
    value = headers.get("retry-after") if headers is not None else None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return float(2**attempt)


def _date_watermark(value: str | date | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError as exc:
        raise ValueError("from_updated_date must be an ISO date (YYYY-MM-DD)") from exc


class OpenAlexClient:
    """Small injectable HTTP client for OpenAlex's REST API."""

    def __init__(self, http_client: Any, api_key: str | None = None, mailto: str | None = None):
        self._http = http_client
        self._params = {}
        if api_key:
            self._params["api_key"] = api_key
        if mailto:
            self._params["mailto"] = mailto

    def list_works(
        self,
        *,
        cursor: str,
        per_page: int,
        filters: list[str] | None = None,
    ) -> dict:
        params = dict(self._params)
        params.update({"cursor": cursor, "per-page": per_page})
        if filters:
            params["filter"] = ",".join(filters)
        import httpx

        for attempt in range(MAX_RETRIES):
            try:
                response = self._http.get(f"{OPENALEX_API_BASE_URL}/works", params=params)
            except httpx.TransportError:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2**attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == MAX_RETRIES - 1:
                    raise OpenAlexAPIError(response.status_code, response.text)
                time.sleep(_retry_after(response.headers, attempt))
                continue
            if response.status_code >= 400:
                raise OpenAlexAPIError(response.status_code, response.text)
            return response.json()
        raise OpenAlexAPIError(0, "OpenAlex request failed after retries")


def _canonical_id(value: str) -> str:
    return value.rstrip("/").split("/")[-1]


def _work_row(work: dict, scope: str) -> dict:
    work_id = _canonical_id(str(work.get("id", "")))
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    authors = []
    institutions = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        author_id = author.get("id")
        if author_id:
            authors.append(_canonical_id(author_id))
        for institution in authorship.get("institutions") or []:
            institution_id = institution.get("id")
            if institution_id:
                institutions.append(_canonical_id(institution_id))
    topics = [
        _canonical_id(topic["id"])
        for topic in work.get("topics") or []
        if isinstance(topic, dict) and topic.get("id")
    ]
    title = (work.get("title") or "Untitled work").strip()
    abstract = decode_abstract_inverted_index(work.get("abstract_inverted_index"))
    content_parts = [title]
    if abstract:
        content_parts.append(abstract)
    if work.get("doi"):
        content_parts.append(f"DOI: {work['doi']}")
    if authors:
        content_parts.append("Authors: " + ", ".join(authors))
    return {
        "id": work_id,
        "title": title,
        "content": "\n\n".join(content_parts),
        "url": work.get("doi") or work.get("id"),
        "doi": work.get("doi"),
        "authors": sorted(set(authors)),
        "institutions": sorted(set(institutions)),
        "topics": sorted(set(topics)),
        "venue": _canonical_id(source["id"]) if source.get("id") else None,
        "referenced_works": [
            _canonical_id(item) for item in work.get("referenced_works") or []
        ],
        NODE_SET_COLUMN: [f"openalex:{scope}:works"],
        "_deleted": False,
    }


def _iter_work_rows(
    client: OpenAlexClient,
    *,
    scope: str,
    filters: list[str],
    state: dict,
    page_size: int,
    from_updated_date: str | None,
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict]:
    if from_updated_date:
        filters = [*filters, f"from_updated_date:{from_updated_date}"]
    cursor = "*"
    seen: set[str] = set()
    while cursor:
        if check_active:
            check_active()
        payload = client.list_works(cursor=cursor, per_page=page_size, filters=filters)
        for work in payload.get("results") or []:
            row = _work_row(work, scope)
            if row["id"]:
                seen.add(row["id"])
                yield row
        cursor = (payload.get("meta") or {}).get("next_cursor")
    state["seen_work_ids"] = sorted(seen)
    if seen:
        state["updated_through"] = datetime.now(timezone.utc).date().isoformat()
    # Deletions are safe only after a complete, unfiltered scope walk. An
    # incremental walk cannot distinguish an unchanged item from a deleted one.
    if not from_updated_date and seen:
        for old_id in sorted(set(state.get("previous_work_ids", [])) - seen):
            yield {"id": old_id, "_deleted": True}


def openalex_source(
    *,
    api_key: str | None = None,
    mailto: str | None = None,
    filters: list[str] | None = None,
    doi: str | None = None,
    orcid: str | None = None,
    ror: str | None = None,
    openalex_id: str | None = None,
    topic: str | None = None,
    resource_name: str = "openalex_works",
    from_updated_date: str | date | datetime | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    http_client: Any = None,
    check_active: Callable[[], None] | None = None,
):
    """Return a dlt resource yielding OpenAlex works as cognee documents.

    ``filters`` accepts OpenAlex filter expressions. Convenience arguments add
    the corresponding scoped filters. Supply a stable ``resource_name`` per
    dataset/scope so incremental state and deletion tombstones stay isolated.
    """
    try:
        import dlt
        import httpx
    except ImportError as exc:
        raise ImportError("The OpenAlex connector requires dlt and httpx.") from exc
    if not filters and not any((doi, orcid, ror, openalex_id, topic)):
        filters = []
    resolved_filters = list(filters or [])
    if doi:
        resolved_filters.append(f"doi:{doi}")
    if orcid:
        resolved_filters.append(f"author.orcid:{orcid}")
    if ror:
        resolved_filters.append(f"institutions.ror:{ror}")
    if openalex_id:
        resolved_filters.append(f"ids.openalex:{openalex_id}")
    if topic:
        resolved_filters.append(f"topics.id:{topic}")
    watermark = _date_watermark(from_updated_date)
    page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
    scope = resource_name

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def works():
        owned_client = None if http_client is not None else httpx.Client(timeout=30.0)
        try:
            client = OpenAlexClient(
                http_client or owned_client,
                api_key or os.getenv("OPENALEX_API_KEY"),
                mailto or os.getenv("OPENALEX_MAILTO"),
            )
            state = dlt.current.resource_state()
            previous_ids = list(state.get("seen_work_ids", []))
            state["previous_work_ids"] = previous_ids
            # After the first successful full walk, continue from the stored
            # watermark unless the caller explicitly requests another date.
            effective_watermark = watermark or state.get("updated_through")
            yield from dlt_utils.guarded_rows(
                _iter_work_rows(
                    client,
                    scope=scope,
                    filters=resolved_filters,
                    state=state,
                    page_size=page_size,
                    from_updated_date=effective_watermark,
                    check_active=check_active,
                ),
                check_active,
            )
        finally:
            if owned_client is not None:
                owned_client.close()

    resource = works()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "openalex")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    return resource


__all__ = [
    "OpenAlexAPIError",
    "OpenAlexClient",
    "decode_abstract_inverted_index",
    "openalex_source",
]
