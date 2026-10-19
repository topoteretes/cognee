# PandaDoc SDK Connector

A native PandaDoc data-source connector for [cognee](https://github.com/topoteretes/cognee): ingest proposals, contracts, commercial quotes, and signing events into persistent AI memory.

It exposes a standard `dlt` source passed directly into `cognee.add()` and `cognee.cognify()`. Contract terms, recipients, line-item pricing tables, and signing states are extracted into structured markdown documents, synchronized incrementally, and cleaned up automatically when deleted in PandaDoc.

---

## Quickstart

```python
import cognee
from cognee.tasks.ingestion.connectors import pandadoc_source

# Ingest completed and sent sales proposals into Cognee
await cognee.add(
    pandadoc_source(
        api_key="your_pandadoc_api_key_or_sandbox_key",
        statuses=["sent", "completed"],
    )
)

# Build knowledge graph entities and vector index
await cognee.cognify()

# Ask questions across historical agreements
result = await cognee.recall(
    "What discount did we offer Acme in the last proposal, and is it signed?"
)
print(result)
```

---

## Authentication & Setup

1. **PandaDoc API Key / Sandbox Key**:
   - In PandaDoc, navigate to **Settings** → **API and Integrations**.
   - Create a new API Key (or Sandbox API Key for development/testing).
   - PandaDoc uses HTTP header authentication: `Authorization: API-Key <your_key>`.

Alternatively, configure credentials via `.env`:

```bash
PANDADOC_API_KEY=your_pandadoc_api_key
PANDADOC_BASE_URL=https://api.pandadoc.com/public/v1
PANDADOC_STATUSES=sent,completed
```

---

## What is Ingested

| Field | Description |
|---|---|
| **Document ID** | `pandadoc:doc:<document_id>` |
| **Title** | `PandaDoc: <document_name>` |
| **Content** | Structured markdown with document overview, grand total, recipients, signing state, pricing tables, and custom tokens |
| **URL** | Direct link to document in PandaDoc workspace (`https://app.pandadoc.com/a/#/documents/<id>`) |
| **Node Scope** | `cognee_node_set = ["pandadoc:documents"]` |

### Commercial Intelligence Extracted
- **Recipients & Signers**: Contact names, emails, roles, and whether each party has signed (`has_completed`).
- **Pricing Tables**: Line items, descriptions, quantities, unit prices, applied discounts, and subtotals.
- **Contract Fields & Tokens**: Custom merge fields like payment terms, contract durations, and client metadata.

---

## Incremental Synchronization & Deletions

1. **`modified_from` Cursor**:
   PandaDoc supports querying documents by `modified_from=<iso_timestamp>`. When a document is created, sent, viewed, signed, or edited, PandaDoc updates its `date_modified` timestamp, triggering automatic re-ingestion on the next sync.
2. **Status Code Translation**:
   PandaDoc filters require numeric status codes (e.g. `0` for draft, `1` for sent, `2` for completed), while payloads return string names (`document.completed`). The connector handles this translation transparently.
3. **Forget-on-Delete Lifecycle**:
   When documents are deleted upstream in PandaDoc, querying `deleted=true` returns the deleted document IDs. The connector emits a `_deleted=True` tombstone, triggering Cognee's `orphan_cleanup` to remove the contract from the memory graph.
