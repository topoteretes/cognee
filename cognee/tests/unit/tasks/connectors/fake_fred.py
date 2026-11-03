"""In-memory stand-in for the FRED REST API, shaped like its real responses."""

from typing import Any
from cognee.tasks.ingestion.connectors.fred import (
    FRED_API_BASE_URL,
    FREDAuthError,
    FREDRateLimitedError,
    FREDSeriesNotFoundError,
)


class FakeFREDHttp:
    """Mock HTTP client answering requests like the FRED REST API."""

    def __init__(self, api_key: str = "valid_key"):
        self.api_key = api_key
        self.series_db: dict[str, dict[str, Any]] = {}
        self.observations_db: dict[str, list[dict[str, Any]]] = {}
        self.calls: list[str] = []
        self.fail_with_auth = False
        self.fail_with_rate_limit = False

    def add_series(
        self,
        series_id: str,
        title: str,
        units: str = "Billions of Dollars",
        frequency: str = "Quarterly",
        seasonal_adjustment: str = "Seasonally Adjusted Annual Rate",
        notes: str = "Gross Domestic Product",
    ):
        clean_id = series_id.upper()
        self.series_db[clean_id] = {
            "id": clean_id,
            "title": title,
            "units": units,
            "frequency": frequency,
            "seasonal_adjustment": seasonal_adjustment,
            "notes": notes,
        }

    def add_observations(self, series_id: str, observations: list[tuple[str, str]]):
        """Add list of (date, value) tuples."""
        clean_id = series_id.upper()
        if clean_id not in self.observations_db:
            self.observations_db[clean_id] = []
        for d, v in observations:
            self.observations_db[clean_id].append({"date": d, "value": v})

    def get(self, url: str, params: dict[str, Any]):
        self.calls.append(url)
        key = params.get("api_key")
        if self.fail_with_auth or key != self.api_key:
            return MockResponse(401, {"error_code": 401, "error_message": "Invalid API key"})

        if self.fail_with_rate_limit:
            return MockResponse(429, {"error_code": 429, "error_message": "Rate limit exceeded"})

        endpoint = url.replace(FRED_API_BASE_URL, "").lstrip("/")

        if endpoint == "series":
            series_id = params.get("series_id", "").upper()
            if series_id in self.series_db:
                return MockResponse(200, {"seriess": [self.series_db[series_id]]})
            return MockResponse(404, {"error_code": 404, "error_message": "Series not found"})

        if endpoint == "series/observations":
            series_id = params.get("series_id", "").upper()
            if series_id not in self.series_db:
                return MockResponse(404, {"error_code": 404, "error_message": "Series not found"})

            all_obs = self.observations_db.get(series_id, [])
            start_date = params.get("observation_start")
            limit = params.get("limit")

            filtered = all_obs
            if start_date:
                filtered = [o for o in filtered if o["date"] >= start_date]

            # sort descending
            sorted_obs = sorted(filtered, key=lambda x: x["date"], reverse=True)
            if limit:
                sorted_obs = sorted_obs[: int(limit)]

            return MockResponse(200, {"observations": sorted_obs})

        return MockResponse(404, {"error_code": 404, "error_message": "Endpoint not found"})


class MockResponse:
    def __init__(self, status_code: int, json_data: dict[str, Any]):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data
