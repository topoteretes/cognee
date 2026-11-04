"""FRED (Federal Reserve Economic Data) connector for Cognee.

Ingests macroeconomic time-series, indicator definitions, and observations
directly from the St. Louis Fed FRED API into Cognee memory.
"""

import logging
import os
from collections.abc import Iterable, Iterator
from typing import Any
import httpx

logger = logging.getLogger(__name__)

FRED_API_BASE = "https://api.stlouisfed.org/fred"

def fetch_fred_series_observations(series_id: str, api_key: str, client: httpx.Client | None = None) -> list[dict[str, Any]]:
    """Fetches raw observations from FRED API."""
    url = f"{FRED_API_BASE}/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    
    if client is not None:
        response = client.get(url, params=params, timeout=20.0)
    else:
        with httpx.Client() as c:
            response = c.get(url, params=params, timeout=20.0)
            
    response.raise_for_status()
    data = response.json()
    return data.get("observations", [])

def fred_source(series_ids: list[str], api_key: str | None = None) -> Iterator[dict[str, Any]]:
    """Yields normalized macroeconomic observation records from FRED for ingestion into Cognee memory.
    
    Args:
        series_ids: List of FRED indicator series IDs (e.g. ['GDP', 'CPIAUCSL', 'UNRATE', 'FEDFUNDS'])
        api_key: Optional FRED API key (defaults to FRED_API_KEY environment variable)
    """
    key = api_key or os.getenv("FRED_API_KEY")
    if not key:
        logger.warning("No FRED API key provided; skipping FRED data ingestion.")
        return

    with httpx.Client() as client:
        for sid in series_ids:
            try:
                observations = fetch_fred_series_observations(sid, key, client=client)
                recent = observations[-12:] if len(observations) >= 12 else observations
                
                obs_text = "\n".join([f"{o.get('date')}: {o.get('value')}" for o in recent])
                doc_text = f"FRED Series: {sid}\nRecent Observations:\n{obs_text}"
                
                yield {
                    "id": f"fred:{sid}",
                    "series_id": sid,
                    "count": len(observations),
                    "latest_date": recent[-1].get("date") if recent else None,
                    "latest_value": recent[-1].get("value") if recent else None,
                    "text": doc_text,
                }
            except Exception as e:
                logger.warning("Failed to fetch FRED series %s: %s", sid, e)
