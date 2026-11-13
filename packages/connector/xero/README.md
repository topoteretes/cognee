# cognee-community-connector-xero

A Xero data-source connector for [cognee](https://github.com/topoteretes/cognee):
sync your invoices (and contacts) into memory — "ask your accounts".

It exposes a `dlt` source you hand to `cognee.remember(...)` / `cognee.add(...)`.
Invoices are rendered to text documents and ingested as **normal documents**
through cognee's cognify entity-extraction pipeline (document-mode marker), not
the deterministic dlt-row path.

## Requirements

* A Xero developer app (https://developer.xero.com/app/manage) with the
  `accounting.transactions` scope, or a connection to a trial org.
* cognee ≥ 1.4.0 (document-mode `DOCUMENT_SOURCE_ATTR` / `resolve_dlt_sources`).

## Install

```bash
uv pip install cognee-community-connector-xero
# or, from this monorepo:
cd packages/connector/xero && uv sync --all-extras
```

## Usage

```python
import cognee
from cognee_community_connector_xero import xero_authenticate, xero_source

xero_authenticate()  # one-time browser login → xero_tokens.json

await cognee.remember(
    xero_source(),  # XERO_TOKEN_PATH / XERO_TENANT_ID from env
    dataset_name="xero",
)

answer = await cognee.search(
    query_text="Which customers have open invoices?",
    query_type=cognee.SearchType.GRAPH_COMPLETION,
    datasets=["xero"],
)
```

## Auth: rotating refresh tokens

`xero_authenticate()` runs the OAuth authorization-code flow and stores
`access_token` + `refresh_token` in a local JSON file (`XERO_TOKEN_PATH`,
default `./xero_tokens.json`). Every sync that gets a `401` refreshes the token
pair and **rotates** the refresh token (Xero invalidates the old one), so the
file on disk is always up to date. If you do not set `XERO_TENANT_ID`, the
connector discovers your organisation automatically via `/connections`.

## Resources

| Resource | Default | Notes |
|----------|---------|-------|
| `xero_invoices` | on | Customer, status/reference/dates, currency, totals, line-item notes |
| `xero_contacts` | **off** | **Opt-in** (`include_contacts=True`); name/email/contact person only |

## Privacy & read-only guarantees

* Read-only, single tenant — never writes to Xero.
* **Data minimisation:** invoice documents are rendered from an allow-list
  (customer name + email, reference, status, dates, currency, subtotal/tax/
  total, and line-item descriptions/quantities/amounts). Contacts are opt-in
  and carry name/email/contact person only. No bank-account details, payment
  terms, or `LineItem` metadata are pulled into memory.

## How sync + forget-on-delete work

Xero's list endpoints page via `?page=` (≤100 per page) and expose no deletion
feed. Each run is therefore a **full snapshot**: `write_disposition="replace"`
rewrites staging with exactly the invoices (and, if enabled, contacts) present
today. An invoice deleted, reversed, or removed from the targeting organisation
drops out of the snapshot, and cognee's existing `orphan_cleanup` forgets it
from the graph and vector stores on the next sync. Unchanged invoices keep a
stable content-hash `data_id`, so only genuinely new/changed content is
re-ingested. A fetch error aborts the run rather than let a partial snapshot
forget live invoices.

*Intentionally out of scope:* Xero's older incremental `If-Modified-Since` feed.
It does not report deletions, and its conditional-update semantics would fight
the forget-on-delete promise — the snapshot model is strictly simpler and
correct. (Flagged in the PR description.)

## Testing

```bash
uv run pytest tests/
```

All tests mock the Xero API (no live credentials) and cover rendering
allow-lists, line-item spelling, pagination, token rotation on `401`, the
contacts opt-in gate, credential/token guards, edit re-sync, and
forget-on-delete.