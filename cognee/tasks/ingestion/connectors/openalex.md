# OpenAlex connector

`openalex_source` syncs OpenAlex works into cognee's document ingestion path:

```python
from cognee.tasks.ingestion.connectors import openalex_source

source = openalex_source(
    doi="10.1038/s41586-020-2649-2",
    mailto="you@example.org",
    resource_name="openalex_nature_paper",
)
await cognee.remember(source, dataset_name="papers", primary_key="id")
```

The source supports OpenAlex filter expressions plus DOI, ORCID, ROR, OpenAlex
ID, and topic convenience filters. It uses cursor pagination, decodes abstract
inverted indexes, retries 429/5xx responses, persists an updated-date watermark,
and emits hard-delete tombstones after a complete unfiltered scope walk.

Set `OPENALEX_API_KEY` and `OPENALEX_MAILTO` to use environment configuration.
