"""FRED (Federal Reserve Economic Data) connector for cognee."""

from __future__ import annotations
from typing import Any, Iterable, Sequence
import httpx
import dlt

FRED_BASE_URL = "https://api.stlouisfed.org/fred"


class FredService:
    def __init__(self, api_key: str, base_url: str = FRED_BASE_URL, timeout: float = 30.0) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("FRED api_key must be a non-empty string")
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def get_series(self, series_id: str) -> dict[str, Any]:
        resp = self._client.get(
            f"{self.base_url}/series",
            params={"series_id": series_id, "api_key": self.api_key, "file_type": "json"},
        )
        resp.raise_for_status()
        seriess = resp.json().get("seriess") or []
        return seriess[0] if seriess else {"id": series_id}

    def get_observations(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
        limit: int = 120,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end
        resp = self._client.get(f"{self.base_url}/series/observations", params=params)
        resp.raise_for_status()
        return resp.json().get("observations") or []


def build_fred_service(api_key: str, base_url: str = FRED_BASE_URL) -> FredService:
    return FredService(api_key=api_key, base_url=base_url)


@dlt.source(name="fred")
def fred_source(
    series_ids: Sequence[str],
    api_key: str = dlt.secrets.value,
    observation_start: str | None = None,
    observation_end: str | None = None,
    max_observations: int = 120,
    service: FredService | None = None,
):
    if not series_ids:
        raise ValueError("series_ids must contain at least one FRED series ID")
    svc = service or build_fred_service(api_key=api_key)

    @dlt.resource(name="series_documents", primary_key="id", write_disposition="merge")
    def series_documents() -> Iterable[dict[str, Any]]:
        for sid in series_ids:
            meta = svc.get_series(sid)
            obs = svc.get_observations(
                sid,
                observation_start=observation_start,
                observation_end=observation_end,
                limit=max_observations,
            )
            lines = [
                f"# {meta.get('title', sid)} ({sid})",
                f"Units: {meta.get('units', 'N/A')} | Frequency: {meta.get('frequency', 'N/A')}",
                f"Seasonal Adjustment: {meta.get('seasonal_adjustment', 'N/A')}",
                f"Last Updated: {meta.get('last_updated', 'N/A')}",
                "",
                meta.get("notes", ""),
                "",
                "## Observations",
            ]
            for o in obs:
                lines.append(f"- {o.get('date')}: {o.get('value')}")
            yield {
                "id": f"fred:series:{sid}",
                "series_id": sid,
                "title": meta.get("title", sid),
                "updated_at": meta.get("last_updated"),
                "text": "\n".join(lines).strip(),
            }

    return [series_documents]
