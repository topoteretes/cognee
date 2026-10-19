# Square Point-of-Sale (POS) Data-Source Connector

A Square POS data-source connector for [cognee](https://github.com/topoteretes/cognee): sync catalog products, item variations, customer profiles, staff notes, and point-of-sale orders into AI memory incrementally, with automatic forget-on-deletion.

Exposes a `dlt` resource intended for `cognee.remember(...)`, integrating seamlessly into Cognee's document-mode ingestion pipeline. Records flow through chunking and LLM knowledge-graph extraction, enabling AI agents to reason over physical retail and hospitality business operations.

## Features

- **Point-of-Sale Catalog**: Ingests items, item variations (size, pricing, SKU), categories, descriptions, and modifiers.
- **Customer Directory**: Ingests customer profiles, companies, segments, staff notes, and preferences with strict PII masking on email and phone numbers.
- **Point-of-Sale Orders**: Ingests orders, line items, itemized quantities, totals, taxes, tips, discounts, states (`OPEN`, `COMPLETED`, `CANCELED`), and links them to customer entities.
- **Incremental Synchronization**:
  - Catalog uses `SearchCatalogObjects` with `begin_time` timestamp cursors.
  - Customers and orders search endpoints filter and sort by `updated_at`.
  - Cursors persist within DLT resource state.
- **Forget-on-Delete**:
  - Deleted catalog items and variations returned with `is_deleted: true` are emitted as `_deleted=True` tombstones under `write_disposition="merge"`, enabling Cognee's orphan cleanup to remove purged products.
- **Production & Free Sandbox Ready**:
  - Supports both `environment="sandbox"` (free Square Developer sandbox) and `environment="production"`.
  - Zero external third-party SDK dependencies (uses `httpx`, a core Cognee dependency).
  - Bearer tokens are masked in strings and logs (`sq0at-****`).

## Setup & Credentials

1. Create a free developer account at [developer.squareup.com](https://developer.squareup.com).
2. Create an Application in your Developer Dashboard.
3. In the Sandbox or Production tab, copy your **Access Token**.
4. Set your environment variables:

```bash
export SQUARE_ACCESS_TOKEN="sq0at-your-access-token"
export SQUARE_ENVIRONMENT="sandbox" # or "production"
```

## Quickstart

```python
import asyncio
import cognee
from cognee.tasks.ingestion.connectors import square_source

async def main():
    # Ingest catalog, customers, and orders into a dedicated POS dataset
    await cognee.remember(
        square_source(
            access_token="sq0at-your-access-token",
            environment="sandbox",
            resources=["catalog", "customers", "orders"],
        ),
        dataset_name="pos_memory",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

    # Query your business memory with graph completion
    answer = await cognee.search(
        query_text="What are our top-selling beverages and which regular customers ordered them recently?",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        datasets=["pos_memory"],
    )
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `access_token` | `str \| None` | `None` | Square OAuth or Personal Access Token (defaults to `SQUARE_ACCESS_TOKEN`). |
| `environment` | `str` | `"sandbox"` | `"sandbox"` or `"production"`. Sets base URL accordingly. |
| `resources` | `Sequence[str]` | `("catalog", "customers", "orders")` | List of resources to extract and sync. |
| `location_ids` | `Sequence[str] \| None` | `None` | Optional list of Square location IDs for order scoping. |
| `resource_name` | `str \| None` | `"square_<env>"` | Stable staging table identifier. |
| `page_size` | `int` | `100` | Pagination page size for Square REST API endpoints. |
| `service` | `SquareClient \| None` | `None` | Pre-configured client instance for testing and injection. |

## Document Schemas & Graph Entities

1. **Catalog Items (`catalog:<item_id>`)**:
   - `id`: `catalog:<item_id>`
   - `title`: `Product: <item_name>`
   - `category`: Category name (resolved via category map)
   - `content`: Markdown document detailing item name, category, description, and list of variations with formatted pricing (e.g., `$4.50 USD`).
   - `_deleted`: `True` when deleted in Square.

2. **Customers (`customer:<customer_id>`)**:
   - `id`: `customer:<customer_id>`
   - `title`: `Customer: <name>`
   - `company`: Company name
   - `content`: Profile details, segment groups, staff notes/preferences, and masked contact info.

3. **Orders (`order:<order_id>`)**:
   - `id`: `order:<order_id>`
   - `title`: `Order #<order_id_prefix>`
   - `customer_id`: Reference `customer:<customer_id>` linking the purchase to the customer node.
   - `state`: `OPEN`, `COMPLETED`, `CANCELED`, etc.
   - `content`: Itemized list of purchased quantities, variation names, unit costs, gross total, taxes, tips, and discounts.
