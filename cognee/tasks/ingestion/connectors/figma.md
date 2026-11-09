# Figma Design Connector

A Figma data-source connector for [cognee](https://github.com/topoteretes/cognee):
turn your design systems, UI component specs, copy text, and review comments into searchable memory.

It exposes a `dlt` source you pass directly to `cognee.remember(...)`, integrating seamlessly into Cognee's incremental memory graphs.

---

## Features

- **Document Hierarchy**: Traverses design documents from root canvas down to pages, frames, components, and nested text nodes.
- **Copy & Text Extraction**: Automatically indexes all user-facing UI labels, button copy, and headings.
- **Design Review Comments**: Folds inline feedback and annotations into each page context to preserve design rationale.
- **Node-Scoped Sets**: Tags records with `cognee_node_set: figma:<file_key>` for isolated design queries.

---

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import figma_source

await cognee.remember(
    figma_source(
        file_keys=["aB1cD2eF3gH4"],
        access_token="<figma-personal-access-token>",
    ),
    dataset_name="design_system",
    primary_key="id",
    write_disposition="merge",
    max_rows_per_table=0,
)
```

Alternatively, authenticate via OAuth token:

```python
await cognee.remember(
    figma_source(
        file_keys=["aB1cD2eF3gH4"],
        oauth_token="<figma-oauth-token>",
    ),
    dataset_name="design_system",
    primary_key="id",
    write_disposition="merge",
    max_rows_per_table=0,
)
```

---

## Data Schema

Each ingested row contains:
- `id`: Unique identifier (`figma:<file_key>:<node_id>`)
- `title`: Combination of file name and page title
- `content`: Formatted markdown containing component metadata, copy text, and review comments
- `url`: Direct link to page or node in Figma Web Editor
- `_deleted`: Boolean flag indicating deleted or removed frames
- `cognee_node_set`: Scoped graph identifier (`figma:<file_key>`)
- `document_source`: Set to `"figma"`
