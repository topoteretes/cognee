import dlt
import pytest

from cognee.tasks.ingestion.connectors import fred_source
from cognee.tasks.ingestion.connectors.fred import (
    FredClient,
    FredNotFoundError,
    render_series,
)


class FakeFred:
    def __init__(self, observations, missing=()):
        self.observations = observations  # oldest first
        self.missing = set(missing)
        self.calls = []

    def get(self, path, params=None):
        params = params or {}
        self.calls.append((path, params))
        if path == "series":
            if params["series_id"] in self.missing:
                raise FredNotFoundError("gone")
            return {
                "seriess": [
                    {
                        "id": params["series_id"],
                        "title": "Unemployment Rate",
                        "units": "Percent",
                        "frequency": "Monthly",
                        "seasonal_adjustment": "Seasonally Adjusted",
                        "notes": "Share of labor force.",
                    }
                ]
            }
        if path == "series/release":
            return {"releases": [{"id": 1}]}
        if path == "release/sources":
            return {"sources": [{"name": "U.S. Bureau of Labor Statistics"}]}
        if path == "series/observations":
            obs = self.observations
            if "observation_start" in params:
                obs = [o for o in obs if o["date"] >= params["observation_start"]]
                return {"observations": obs[: params["limit"]]}
            return {"observations": obs[::-1][: params["limit"]]}
        raise AssertionError(path)


def _obs(*pairs):
    return [{"date": d, "value": v, "realtime_start": "x", "realtime_end": "y"} for d, v in pairs]


def _run(source):
    pipeline = dlt.pipeline("fred_test", destination="duckdb", dataset_name="t")
    return pipeline, pipeline.extract(source)


def test_render_is_readable_and_hides_realtime():
    row = render_series(
        {"id": "UNRATE", "title": "T", "units": "Percent"},
        "BLS",
        _obs(("2024-01-01", "3.7"), ("2024-02-01", ".")),
    )
    assert row["id"] == "fred:UNRATE"
    assert "2024-01-01: 3.7" in row["content"] and "2024-02-01: ." in row["content"]
    assert "realtime" not in row["content"] and "Source: BLS" in row["content"]
    assert set(row) == {"id", "title", "content", "url", "_deleted"}


def test_max_observations_bounds_and_incremental(tmp_path, monkeypatch):
    monkeypatch.setenv("DLT_DATA_DIR", str(tmp_path))
    fake = FakeFred(_obs(("2024-01-01", "1"), ("2024-02-01", "2"), ("2024-03-01", "3")))
    items = list(fred_source(series_ids=["unrate"], service=fake, max_observations=2))
    assert len(items) == 1
    assert "2024-01-01" not in items[0]["content"] and "2024-03-01: 3" in items[0]["content"]


def test_requires_selection_and_key():
    with pytest.raises(ValueError):
        fred_source(service=object())
    with pytest.raises(ValueError):
        fred_source(series_ids=["GDP"])
    with pytest.raises(ValueError):
        FredClient("")


def test_missing_series_unknown_is_skipped():
    fake = FakeFred([], missing={"OLD"})
    assert list(fred_source(series_ids=["OLD"], service=fake)) == []
