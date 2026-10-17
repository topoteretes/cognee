# Firecrawl Connector for Cognee

A ``dlt`` source that turns website and web pages into clean, LLM-ready markdown and indexes item into cognee for memory, graph ingestion.

## Example Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import firecrawl_source

await cognee.remember(
    firecrawl_source(
        token="your-firecrawl-api-key",
        urls=["https://example.com"],
        dataset_name="my_web_dataset",
    ),
    dataset_name="my_web_dataset",
    primary_key="id",
    write_disposition="merge",
)