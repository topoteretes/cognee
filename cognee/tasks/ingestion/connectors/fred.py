"""FRED (Federal Reserve Economic Data) connector for cognee.

A ``dlt`` source that ingests macroeconomic time-series data from the St. Louis Fed
into cognee memory documents::

    import cognee
    from cognee.tasks.ingestion.connectors import fred_source

    await cognee.remember(
        fred_source(
            series_ids=["GDP", "UNRATE", "CPIAUCSL"],
            api_key="<fred api key>",
            max_observations=60,
        ),
        dataset_name="economic_data",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, url, _deleted, cognee_node_set}``.
  Ids are formatted as ``fred:<series_id>``.
* **Rendered content** includes series metadata (title, units, frequency, seasonal adjustment,
  source agency, description) and the recent observations rendered in clean
  ``YYYY-MM-DD: value`` lines.
* **Incremental**: Stores the latest seen observation date in resource state. A subsequent
  run with no new observations yields no document updates.
* **Tombstoning**: Discontinued or deleted series are tombstoned via ``_deleted=True``.
* **API rate limits**: Defaults to graceful throttling and backoff against the 120 req/min limit.
"""

import logging
import os
import re
import time
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import urlencode

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

logger = logging.getLogger(__name__)

FRED_API_BASE_URL = "https://api.stlouisfed.org/fred"
DEFAULT_MAX_OBSERVATIONS = 60
DEFAULT_MAX_REQUESTS = 120
_TIMEOUT = (10, 30)
_RETRIES = 3


class FREDApiError(Exception):
    """Base error for FRED API communication."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class FREDAuthError(FREDApiError):
    """Raised when the FRED API key is invalid or rejected."""


class FREDRateLimitedError(FREDApiError):
    """Raised when rate limits (120 req/min) are encountered."""


class FREDSeriesNotFoundError(FREDApiError):
    """Raised when a requested series ID does not exist."""


class FREDClient:
    """Minimal HTTP client for the FRED REST API."""

    def __init__(self, api_key: str, http_client: Any = None):
        if not api_key or not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("FRED API key is required and must be a non-empty string.")
        self._api_key = api_key.strip()
        self._http_client = http_client

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        full_params = {
            **params,
            "api_key": self._api_key,
            "file_type": "json",
        }
        url = f"{FRED_API_BASE_URL}/{endpoint.lstrip('/')}"

        if self._http_client is not None:
            # Used for tests and custom injected clients
            resp = self._http_client.get(url, params=full_params)
            status_code = getattr(resp, "status_code", 200)
            if status_code == 401 or status_code == 403:
                raise FREDAuthError(f"FRED authentication failed: {status_code}", status_code)
            if status_code == 429:
                raise FREDRateLimitedError("FRED rate limit exceeded (120 req/min)", status_code)
            if status_code == 404:
                raise FREDSeriesNotFoundError(f"FRED resource not found: {endpoint}", status_code)
            if status_code >= 400:
                raise FREDApiError(f"FRED API error ({status_code})", status_code)
            if hasattr(resp, "json"):
                return resp.json()
            return resp

        # Default standard library / requests fallback
        import requests

        for attempt in range(1, _RETRIES + 1):
            try:
                resp = requests.get(url, params=full_params, timeout=_TIMEOUT)
                if resp.status_code == 401 or resp.status_code == 403:
                    raise FREDAuthError(f"FRED authentication failed: {resp.status_code}", resp.status_code)
                if resp.status_code == 429:
                    if attempt < _RETRIES:
                        time.sleep(2 ** attempt)
                        continue
                    raise FREDRateLimitedError("FRED rate limit exceeded (120 req/min)", resp.status_code)
                if resp.status_code == 404:
                    raise FREDSeriesNotFoundError(f"FRED series not found on {endpoint}", resp.status_code)
                if resp.status_code >= 400:
                    raise FREDApiError(f"FRED API returned HTTP {resp.status_code}", resp.status_code)
                return resp.json()
            except requests.RequestException as e:
                if attempt == _RETRIES:
                    raise FREDApiError(f"Network error communicating with FRED API: {e}") from e
                time.sleep(1)

        raise FREDApiError("Failed to reach FRED API after retries.")

    def get_series_info(self, series_id: str) -> dict[str, Any]:
        """Fetch series metadata."""
        data = self._get("series", {"series_id": series_id})
        series_list = data.get("seriess", [])
        if not series_list:
            raise FREDSeriesNotFoundError(f"No series found for id: {series_id}")
        return series_list[0]

    def get_series_observations(
        self,
        series_id: str,
        observation_start: str | None = None,
        limit: int | None = None,
        sort_order: str = "desc",
    ) -> list[dict[str, Any]]:
        """Fetch series observations."""
        params: dict[str, Any] = {
            "series_id": series_id,
            "sort_order": sort_order,
        }
        if observation_start:
            params["observation_start"] = observation_start
        if limit:
            params["limit"] = limit

        data = self._get("series/observations", params)
        return data.get("observations", [])


def render_fred_document(series_info: dict[str, Any], observations: list[dict[str, Any]]) -> str:
    """Render a clean, deterministic text document for cognify ingestion."""
    title = series_info.get("title", series_info.get("id", "Unknown Series"))
    series_id = series_info.get("id", "")
    units = series_info.get("units", "N/A")
    frequency = series_info.get("frequency", "N/A")
    seasonal = series_info.get("seasonal_adjustment", "N/A")
    notes = series_info.get("notes", "")

    lines = [
        f"Economic Data Series: {title} ({series_id})",
        f"Units: {units}",
        f"Frequency: {frequency}",
        f"Seasonal Adjustment: {seasonal}",
    ]
    if notes:
        # Keep clean without huge multi-line junk
        clean_notes = notes.strip().replace("\r\n", "\n")
        lines.append(f"Description: {clean_notes}")

    lines.append("\nObservations (Recent):")
    if not observations:
        lines.append("No observations available.")
    else:
        # Chronological rendering (oldest to newest for clarity)
        sorted_obs = sorted(observations, key=lambda o: o.get("date", ""))
        for obs in sorted_obs:
            date = obs.get("date", "")
            val = obs.get("value", ".")
            if val != ".":  # In FRED '.' denotes missing data
                lines.append(f"{date}: {val}")

    return "\n".join(lines)


def _iter_fred_rows(
    client: FREDClient,
    state: dict[str, Any],
    series_ids: list[str],
    max_observations: int = DEFAULT_MAX_OBSERVATIONS,
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield document rows for each series incrementally."""
    last_dates: dict[str, str] = state.setdefault("last_observation_dates", {})

    for s_id in series_ids:
        if check_active is not None:
            check_active()

        clean_id = s_id.strip().upper()
        if not clean_id:
            continue

        try:
            info = client.get_series_info(clean_id)
        except FREDSeriesNotFoundError:
            logger.warning("FRED series %s not found. Emitting tombstone if previously synced.", clean_id)
            if clean_id in last_dates:
                yield {
                    "id": f"fred:{clean_id}",
                    "title": f"Discontinued: {clean_id}",
                    "content": "",
                    "url": f"https://fred.stlouisfed.org/series/{clean_id}",
                    "_deleted": True,
                    "cognee_node_set": [f"fred:{clean_id}"],
                }
                del last_dates[clean_id]
            continue

        last_seen_date = last_dates.get(clean_id)

        # Fetch observations
        # When incremental, we ask for observations starting after the last date
        obs = client.get_series_observations(
            clean_id,
            observation_start=last_seen_date,
            limit=max_observations,
            sort_order="desc",
        )

        if not obs and last_seen_date is not None:
            # Incremental pass: no new observations published
            continue

        # Filter out observations that match the exact last_seen_date to prevent duplicates
        new_obs = [o for o in obs if o.get("date") != last_seen_date] if last_seen_date else obs
        if not new_obs and last_seen_date is not None:
            continue

        # Determine the latest date among the newly retrieved observations
        all_dates = [o.get("date") for o in obs if o.get("date")]
        if all_dates:
            latest_date = max(all_dates)
            last_dates[clean_id] = latest_date

        doc_content = render_fred_document(info, obs)
        yield {
            "id": f"fred:{clean_id}",
            "title": info.get("title", clean_id),
            "content": doc_content,
            "url": f"https://fred.stlouisfed.org/series/{clean_id}",
            "_deleted": False,
            "cognee_node_set": [f"fred:{clean_id}", "fred:economic_data"],
        }


def build_fred_service(api_key: str, http_client: Any = None) -> FREDClient:
    """Build a FREDClient service object."""
    return FREDClient(api_key=api_key, http_client=http_client)


def fred_source(
    *,
    series_ids: list[str],
    api_key: str | None = None,
    service: FREDClient | None = None,
    resource_name: str = "fred_economic_data",
    max_observations: int = DEFAULT_MAX_OBSERVATIONS,
    check_active: Callable[[], None] | None = None,
):
    """Return a ``dlt`` resource yielding macroeconomic series documents from FRED.

    Hand the result to ``cognee.remember(...)`` with
    ``write_disposition="merge"`` and ``primary_key="id"``.

    Args:
        series_ids: List of FRED series identifiers (e.g. ``["GDP", "UNRATE"]``).
        api_key: Free FRED API key.
        service: Optional pre-built FREDClient (mainly for tests).
        resource_name: Stable name of the dlt resource/table.
        max_observations: Max recent observations to include per series document.
        check_active: Optional host authorization check hook.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError("The FRED connector requires dlt (a cognee core dependency).") from exc

    resolved_key = api_key or os.getenv("FRED_API_KEY")
    if service is None and not resolved_key:
        raise ValueError(
            "FRED API key is required. Pass api_key or set the FRED_API_KEY environment variable."
        )

    client = service if service is not None else FREDClient(api_key=resolved_key or "")

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def fred_economic_series():
        state = dlt.current.resource_state()
        for row in _iter_fred_rows(
            client=client,
            state=state,
            series_ids=series_ids,
            max_observations=max_observations,
            check_active=check_active,
        ):
            yield row

    setattr(fred_economic_series, DOCUMENT_SOURCE_ATTR, "fred")
    return fred_economic_series
