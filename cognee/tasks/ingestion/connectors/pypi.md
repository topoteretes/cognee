# PyPI Data-Source Connector

Ingest package metadata, dependency trees, and README documentation directly from PyPI (Python Package Index) into Cognee's persistent memory and knowledge graph.

## Quickstart

```python
import cognee
from cognee.tasks.ingestion.connectors import pypi_source

# Ingest package documentation and dependency graphs
await cognee.remember(
    pypi_source(packages=["cognee", "fastapi", "pydantic"]),
    dataset_name="python_libraries",
    primary_key="id",
    write_disposition="merge",
    max_rows_per_table=0,
)

# Search your indexed Python package knowledge
results = await cognee.search("Which dependencies does FastAPI require?")
print(results)
```

## Features

- **Zero-Token Public Access**: Connects to the public PyPI JSON API (`https://pypi.org/pypi/<package>/json`) without requiring API keys or user credentials.
- **Structured Knowledge Graph Ingestion**: Ingests package name, version, summary, author, license, project URLs, Python constraints, dependencies (`requires_dist`), classifiers, and project descriptions into clean markdown documents.
- **Incremental & Version-Aware**: Saves indexed package versions in resource state, automatically skipping unchanged packages on subsequent sync runs.
- **Node-Set Scoping**: Scopes documents to `pypi:<package>` and `pypi` for precise multi-tenant or domain-filtered retrieval.
