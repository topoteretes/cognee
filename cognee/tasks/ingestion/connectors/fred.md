# FRED (Federal Reserve Economic Data) Connector

The FRED connector turns macroeconomic time-series data from the [Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org/) into searchable documents for cognee.

Finance, strategy, and research teams use it to track inflation (`CPIAUCSL`), GDP (`GDP`), unemployment (`UNRATE`), interest rates (`FEDFUNDS`), and thousands of other economic indicators.

## Prerequisites

1. Get a free FRED API key from [FRED API Keys](https://fred.stlouisfed.org/docs/api/api_key.html) (instant signup, zero cost).
2. Rate limit: 120 requests/minute.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import fred_source

# Remember key US economic indicators
await cognee.remember(
    fred_source(
        series_ids=["GDP", "UNRATE", "CPIAUCSL", "FEDFUNDS"],
        api_key="your_fred_api_key",
        max_observations=60,
    ),
    dataset_name="economic_data",
    primary_key="id",
    write_disposition="merge",
    max_rows_per_table=0,
)
```

## Features

- **Document Ingestion**: Each series becomes a clean document with metadata (title, units, frequency, seasonal adjustment, source) and recent observations rendered as `YYYY-MM-DD: value`.
- **Incremental Sync**: Remembers the latest observation date per series. Repeated runs fetch only newly published data points.
- **Tombstoning**: Discontinued or deleted series are tombstoned via `_deleted=True` with `write_disposition="merge"`.
- **Rate-Limit Aware**: Gracefully respects FRED's 120 req/min limit with backoff.
