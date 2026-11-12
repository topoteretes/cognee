"""FRED connector for cognee: a ``dlt`` source that turns economic series into memory.

FRED (Federal Reserve Economic Data, https://fred.stlouisfed.org) publishes US economic
series such as ``GDP``, ``UNRATE``, ``CPIAUCSL`` and ``FEDFUNDS``. One source is one
selection of series, and is meant to be handed directly to :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import fred_source

    await cognee.remember(
        fred_source(api_key="<free FRED api key>", series_ids=["GDP", "UNRATE"]),
        dataset_name="fred",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Design
------
* **Selection** is any of ``series_ids``, ``category_id`` (the series of one FRED category)
  and ``search_text`` (``/fred/series/search``); the union is synced, each series once.
* **Rows** are flat ``{id, title, content, url, _deleted}`` with ids ``fred:<series_id>``:
  one document per series with its title, units, frequency, seasonal adjustment, source
  agency, description and the latest ``max_observations`` observations as
  ``YYYY-MM-DD: value`` lines. Values are rendered exactly as FRED sends them (strings), and
  the realtime start/end stamps are not rendered, so the document only changes when the
  series does.
* **Incremental.** The latest observation date per series is kept in resource state. A
  cheap probe asks for observations from that date on; a series with nothing newer yields
  nothing, otherwise its document is re-rendered with the latest ``max_observations``.
* **Forget-on-delete.** A series FRED no longer knows (HTTP 400/404 on its metadata) or marks
  as discontinued in its title is yielded as a ``_deleted`` tombstone.
* **Rate limit.** FRED allows 120 requests per minute. HTTP 429 and 5xx are retried with
  backoff (honouring ``Retry-After``); after the last retry the run fails.
* **API key** lives in the client object only; errors carry status codes, never URLs.

Privacy
-------
This connector only reads public FRED data and is opt-in: nothing is fetched until you
construct a source and call ``remember``.
"""

import logging
import re
import time
from collections.abc import Callable, Iterator
from typing import Any

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

logger = logging.getLogger(__name__)

API_URL = "https://api.stlouisfed.org/fred"
SERIES_URL = "https://fred.stlouisfed.org/series"
DEFAULT_MAX_OBSERVATIONS = 60
# FRED caps a page of series listings at 1000 rows.
_PAGE_SIZE = 1000
_TIMEOUT = (10, 30)
_RETRIES = 4


class FredSourceError(RuntimeError):
    """Base class of the errors this source raises."""


class FredAPIError(FredSourceError):
    """FRED answered with an error, or could not be reached."""


class FredNotFoundError(FredAPIError):
    """FRED does not know the requested series (HTTP 400/404)."""


class FredClient:
    """Minimal synchronous FRED client. ``get`` returns the decoded JSON object.

    Tests and hosts can pass any object with the same ``get`` method instead.
    """

    def __init__(self, api_key: str, *, sleep: Callable[[float], None] = time.sleep):
        key = (api_key or "").strip()
        if not key or not key.isascii() or not key.isprintable() or " " in key:
            raise ValueError("The FRED api key is empty or has an invalid format")
        self._api_key = key
        self._sleep = sleep

    def __repr__(self) -> str:
        return "FredClient(api_key=<redacted>)"

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        import requests

        query = {**(params or {}), "api_key": self._api_key, "file_type": "json"}
        for attempt in range(_RETRIES):
            last = attempt == _RETRIES - 1
            try:
                response = requests.get(f"{API_URL}/{path}", params=query, timeout=_TIMEOUT)
            except requests.RequestException as exc:
                if last:
                    raise FredAPIError(f"FRED request failed: {type(exc).__name__}") from None
                self._sleep(2**attempt)
                continue
            status = response.status_code
            if (status == 429 or status >= 500) and not last:
                try:
                    delay = float(response.headers.get("Retry-After"))
                except (TypeError, ValueError):
                    delay = 2**attempt
                self._sleep(delay)
                continue
            if status in (400, 404):
                raise FredNotFoundError(f"FRED rejected the request (HTTP {status})")
            if status != 200:
                raise FredAPIError(f"FRED request failed (HTTP {status})")
            body = response.json()
            if not isinstance(body, dict):
                raise FredAPIError("FRED answered with an unexpected payload")
            return body
        raise FredAPIError("FRED request failed")  # pragma: no cover


def build_fred_service(api_key: str) -> FredClient:
    """Build the FRED client for ``api_key``."""
    return FredClient(api_key)


def render_series(
    series: dict[str, Any], source: str, observations: list[dict[str, Any]]
) -> dict[str, Any]:
    """Render one series and its observations (oldest first) as a flat document row."""
    sid = series["id"]
    fields = [
        ("Units", series.get("units")),
        ("Frequency", series.get("frequency")),
        ("Seasonal adjustment", series.get("seasonal_adjustment")),
        ("Source", source),
    ]
    lines = [f"{label}: {value}" for label, value in fields if value]
    notes = (series.get("notes") or "").strip()
    if notes:
        lines += ["", notes]
    lines += ["", "Observations:"]
    lines += [f"{o['date']}: {o['value']}" for o in observations]
    return {
        "id": f"fred:{sid}",
        "title": f"{sid}: {series.get('title') or sid}",
        "content": "\n".join(lines),
        "url": f"{SERIES_URL}/{sid}",
        "_deleted": False,
    }


def _tombstone(series_id: str) -> dict[str, Any]:
    return {
        "id": f"fred:{series_id}",
        "title": series_id,
        "content": "",
        "url": f"{SERIES_URL}/{series_id}",
        "_deleted": True,
    }


def _paged(client: Any, path: str, params: dict[str, Any], key: str) -> Iterator[dict[str, Any]]:
    offset = 0
    while True:
        rows = client.get(path, {**params, "limit": _PAGE_SIZE, "offset": offset}).get(key) or []
        yield from rows
        if len(rows) < _PAGE_SIZE:
            return
        offset += _PAGE_SIZE


def _selected_ids(
    client: Any, series_ids: list[str], category_id: int | None, search_text: str | None
) -> list[str]:
    ids = list(series_ids)
    if category_id is not None:
        ids += [
            s["id"]
            for s in _paged(client, "category/series", {"category_id": category_id}, "seriess")
        ]
    if search_text:
        ids += [
            s["id"]
            for s in _paged(client, "series/search", {"search_text": search_text}, "seriess")
        ]
    return list(dict.fromkeys(ids))


def _source_name(get: Callable[[str, dict[str, Any]], dict[str, Any]], series_id: str) -> str:
    """Name of the agency publishing the series, via its release."""
    releases = get("series/release", {"series_id": series_id}).get("releases") or []
    if not releases:
        return ""
    sources = get("release/sources", {"release_id": releases[0]["id"]}).get("sources") or []
    return sources[0]["name"] if sources else ""


def _iter_series_rows(
    client: Any,
    series_id: str,
    state: dict[str, Any],
    max_observations: int,
    check_active: Callable[[], None] | None,
) -> Iterator[dict[str, Any]]:
    last = state.setdefault("last_observation", {})

    def get(path: str, params: dict[str, Any]) -> dict[str, Any]:
        if check_active is not None:
            check_active()
        return client.get(path, params)

    try:
        listed = get("series", {"series_id": series_id}).get("seriess") or []
    except FredNotFoundError:
        listed = []
    if not listed or "DISCONTINUED" in (listed[0].get("title") or "").upper():
        if series_id in last:
            del last[series_id]
            yield _tombstone(series_id)
        else:
            logger.warning("FRED series %s does not exist or is discontinued; skipped", series_id)
        return
    series = listed[0]

    if series_id in last:
        # Inclusive start: the stored date itself is returned, so two rows mean news.
        probe = (
            get(
                "series/observations",
                {
                    "series_id": series_id,
                    "observation_start": last[series_id],
                    "sort_order": "asc",
                    "limit": 2,
                },
            ).get("observations")
            or []
        )
        if len(probe) < 2:
            return

    recent = (
        get(
            "series/observations",
            {"series_id": series_id, "sort_order": "desc", "limit": max_observations},
        ).get("observations")
        or []
    )
    if not recent:
        return
    observations = recent[::-1]
    row = render_series(series, _source_name(get, series_id), observations)
    last[series_id] = observations[-1]["date"]
    yield row


def _default_resource_name(selection: str) -> str:
    return "fred_" + re.sub(r"[^a-z0-9]+", "_", selection.lower()).strip("_")


def fred_source(
    *,
    series_ids: list[str] | None = None,
    category_id: int | None = None,
    search_text: str | None = None,
    resource_name: str | None = None,
    check_active: Callable[[], None] | None = None,
    service: Any = None,
    api_key: str | None = None,
    max_observations: int = DEFAULT_MAX_OBSERVATIONS,
):
    """Return a ``dlt`` resource that yields one document per FRED series for ``remember``.

    Args:
        series_ids: Series ids such as ``["GDP", "UNRATE"]``.
        category_id: A FRED category; all its series are synced.
        search_text: A ``/fred/series/search`` query; all matching series are synced.
        resource_name: Stable, connection-specific name; must start with ``fred_`` when a
            host tracks the table by that prefix. Defaults to one derived from the selection.
        check_active: Optional host authorization checkpoint, called before each request.
        service: Pre-built client (see :func:`build_fred_service`). Mainly an injection
            point for tests and hosts.
        api_key: Free FRED api key, used to build the client when ``service`` is omitted.
        max_observations: Latest observations rendered per series (daily series have long
            histories, so this bound is mandatory).

    Returns:
        A ``dlt`` resource configured with ``primary_key="id"``,
        ``write_disposition="merge"`` and a ``_deleted`` hard-delete column.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError("The FRED connector requires dlt (a cognee core dependency).") from exc

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 1:
        raise RuntimeError(
            "FRED sync requires a Cognee build with table-scoped DLT document cleanup."
        )
    if service is None and not api_key:
        raise ValueError("fred_source needs a service or an api_key")
    series_ids = [s.strip().upper() for s in (series_ids or []) if s.strip()]
    if not series_ids and category_id is None and not search_text:
        raise ValueError("fred_source needs series_ids, a category_id or search_text")
    if max_observations < 1:
        raise ValueError("max_observations must be at least 1")

    client = service if service is not None else FredClient(api_key or "")
    selection = "_".join(series_ids) or f"{category_id}_{search_text}"
    name = resource_name or _default_resource_name(selection)

    @dlt.resource(
        name=name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def fred_series():
        state = dlt.current.resource_state()
        for series_id in _selected_ids(client, series_ids, category_id, search_text):
            yield from _iter_series_rows(client, series_id, state, max_observations, check_active)

    resource = fred_series()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "fred")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, name)
    return resource
