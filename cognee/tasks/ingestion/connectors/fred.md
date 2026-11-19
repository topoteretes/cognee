# FRED (Federal Reserve Economic Data) Connector

The FRED connector allows Cognee to ingest macroeconomic time series data, financial indicators, and economic statistics from the Federal Reserve Bank of St. Louis.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors.fred import fred_source

# Ingest macroeconomic indicators into knowledge memory
await cognee.remember(
    fred_source(series_ids=["GDP", "CPIAUCSL", "UNRATE", "FEDFUNDS"], api_key="your_fred_api_key"),
    dataset_name="macroeconomic_context"
)
```
