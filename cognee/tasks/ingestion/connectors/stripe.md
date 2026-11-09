# Stripe Billing & Invoices Connector

A Stripe data-source connector for [cognee](https://github.com/topoteretes/cognee):
turn your enterprise billing data, commercial invoices, customer transactions, and charge histories into searchable memory.

It exposes a `dlt` source you pass directly to `cognee.remember(...)`, integrating seamlessly into Cognee's incremental memory graphs.

---

## Features

- **Commercial Invoices Ingestion**: Ingests invoice numbers, line item breakdowns, unit prices, and payment statuses.
- **Transaction Charges & Settlements**: Ingests charge amounts, payment rails (e.g. Visa ending in 4242), and refund statuses.
- **Incremental Cursor Sync**: Implements Stripe's native `starting_after` cursor pagination for high-volume billing accounts.
- **Tombstones & Status Reconciliation**: Emits `_deleted: True` for voided invoices or refunded charges to keep memory clean.
- **Enterprise PCI-DSS Safe**: Never renders or touches card numbers, CVVs, or secret tokens. Only sanitized summary metadata is vectorized.

---

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import stripe_source

await cognee.remember(
    stripe_source(
        api_key="rk_live_...",
        include_invoices=True,
        include_charges=True,
    ),
    dataset_name="enterprise_finance",
    primary_key="id",
    write_disposition="merge",
    max_rows_per_table=0,
)
```

---

## Data Schema

Each ingested row contains:
- `id`: Unique identifier (`stripe:invoice:<id>` or `stripe:charge:<id>`)
- `title`: Invoice number or charge description
- `content`: Formatted markdown containing amounts, currency, customer ID, line items, and settlement metadata
- `url`: Direct link to Stripe Dashboard object
- `_deleted`: Boolean flag indicating voided invoices or refunded charges
- `cognee_node_set`: Scoped graph domain (`stripe:invoices` or `stripe:charges`)
- `document_source`: Set to `"stripe"`
