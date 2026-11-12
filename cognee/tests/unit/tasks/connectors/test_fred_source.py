"""Unit tests for the FRED (Federal Reserve Economic Data) connector."""

import pytest
from cognee.tasks.ingestion.connectors.fred import (
    FREDClient,
    FREDAuthError,
    FREDRateLimitedError,
    FREDSeriesNotFoundError,
    _iter_fred_rows,
    build_fred_service,
    render_fred_document,
)
from cognee.tests.unit.tasks.connectors.fake_fred import FakeFREDHttp


def test_fred_client_auth_error():
    fake_http = FakeFREDHttp(api_key="valid_key")
    fake_http.fail_with_auth = True
    client = FREDClient(api_key="invalid_key", http_client=fake_http)

    with pytest.raises(FREDAuthError):
        client.get_series_info("GDP")


def test_fred_client_rate_limit():
    fake_http = FakeFREDHttp(api_key="valid_key")
    fake_http.fail_with_rate_limit = True
    client = FREDClient(api_key="valid_key", http_client=fake_http)

    with pytest.raises(FREDRateLimitedError):
        client.get_series_info("GDP")


def test_fred_client_series_not_found():
    fake_http = FakeFREDHttp(api_key="valid_key")
    client = FREDClient(api_key="valid_key", http_client=fake_http)

    with pytest.raises(FREDSeriesNotFoundError):
        client.get_series_info("NON_EXISTENT_SERIES")


def test_render_fred_document():
    series_info = {
        "id": "GDP",
        "title": "Gross Domestic Product",
        "units": "Billions of Dollars",
        "frequency": "Quarterly",
        "seasonal_adjustment": "Seasonally Adjusted Annual Rate",
        "notes": "GDP is the value of the goods and services produced by the nation's economy.",
    }
    observations = [
        {"date": "2026-01-01", "value": "28500.5"},
        {"date": "2026-04-01", "value": "28900.2"},
        {"date": "2026-07-01", "value": "."},  # missing value should be omitted
    ]
    doc = render_fred_document(series_info, observations)

    assert "Economic Data Series: Gross Domestic Product (GDP)" in doc
    assert "Units: Billions of Dollars" in doc
    assert "Frequency: Quarterly" in doc
    assert "2026-01-01: 28500.5" in doc
    assert "2026-04-01: 28900.2" in doc
    assert "2026-07-01: ." not in doc


def test_iter_fred_rows_initial_and_incremental_sync():
    fake_http = FakeFREDHttp(api_key="valid_key")
    fake_http.add_series("UNRATE", "Unemployment Rate", units="Percent", frequency="Monthly")
    fake_http.add_observations("UNRATE", [
        ("2026-01-01", "4.0"),
        ("2026-02-01", "4.1"),
        ("2026-03-01", "3.9"),
    ])

    client = build_fred_service(api_key="valid_key", http_client=fake_http)
    state = {}

    # Initial Run
    rows_1 = list(_iter_fred_rows(client, state, series_ids=["UNRATE"]))
    assert len(rows_1) == 1
    row_1 = rows_1[0]
    assert row_1["id"] == "fred:UNRATE"
    assert row_1["_deleted"] is False
    assert "2026-03-01: 3.9" in row_1["content"]
    assert state["last_observation_dates"]["UNRATE"] == "2026-03-01"

    # Incremental Run without new data -> yields nothing
    rows_2 = list(_iter_fred_rows(client, state, series_ids=["UNRATE"]))
    assert len(rows_2) == 0

    # Incremental Run with new data published
    fake_http.add_observations("UNRATE", [("2026-04-01", "3.8")])
    rows_3 = list(_iter_fred_rows(client, state, series_ids=["UNRATE"]))
    assert len(rows_3) == 1
    row_3 = rows_3[0]
    assert "2026-04-01: 3.8" in row_3["content"]
    assert state["last_observation_dates"]["UNRATE"] == "2026-04-01"


def test_tombstone_on_deleted_series():
    fake_http = FakeFREDHttp(api_key="valid_key")
    fake_http.add_series("DISCONTINUED_SERIES", "Discontinued Series")
    fake_http.add_observations("DISCONTINUED_SERIES", [("2026-01-01", "100")])

    client = build_fred_service(api_key="valid_key", http_client=fake_http)
    state = {}

    # Run 1: Ingest series
    rows = list(_iter_fred_rows(client, state, series_ids=["DISCONTINUED_SERIES"]))
    assert len(rows) == 1
    assert "DISCONTINUED_SERIES" in state["last_observation_dates"]

    # Now remove from upstream (series discontinued or deleted)
    del fake_http.series_db["DISCONTINUED_SERIES"]

    # Run 2: Should yield a tombstone with _deleted=True
    tombstone_rows = list(_iter_fred_rows(client, state, series_ids=["DISCONTINUED_SERIES"]))
    assert len(tombstone_rows) == 1
    assert tombstone_rows[0]["id"] == "fred:DISCONTINUED_SERIES"
    assert tombstone_rows[0]["_deleted"] is True
    assert "DISCONTINUED_SERIES" not in state["last_observation_dates"]
