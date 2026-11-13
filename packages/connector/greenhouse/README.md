# cognee-community-connector-greenhouse

A Greenhouse data-source connector for [cognee](https://github.com/topoteretes/cognee):
sync your hiring data into memory — "ask my ATS".

It exposes a `dlt` source you hand to `cognee.remember(...)` / `cognee.add(...)`.
Greenhouse records (jobs, job posts, and opt-in interview scorecards) are rendered
to text documents and ingested as **normal documents** through cognee's cognify
entity-extraction pipeline (cognee's document-mode marker), not the deterministic
dlt-row path.

## Requirements

> Built against **Harvest v3**. Greenhouse retired Harvest v1/v2 on
> **August 31, 2026**, so this connector authenticates with OAuth 2.0 client
> credentials (client id + client secret) — never the legacy API key.
>
> Requires a cognee release that ships document-mode
> (`DOCUMENT_SOURCE_ATTR` / `resolve_dlt_sources` routing). Not in cognee 1.3.0.

## Install

```bash
uv pip install cognee-community-connector-greenhouse
# or, from this monorepo:
cd packages/connector/greenhouse && uv sync --all-extras
```

## Usage

```python
import cognee
from cognee_community_connector_greenhouse import greenhouse_source

await cognee.remember(
    greenhouse_source(),  # GREENHOUSE_CLIENT_ID/_SECRET from env, or pass them
    dataset_name="greenhouse",
)

answer = await cognee.search(
    query_text="What roles are we hiring for?",
    query_type=cognee.SearchType.GRAPH_COMPLETION,
    datasets=["greenhouse"],
)
```

Credentials fall back to `GREENHOUSE_CLIENT_ID` / `GREENHOUSE_CLIENT_SECRET` /
`GREENHOUSE_SUB` environment variables, matching the `sub` (user id) convention
of Harvest v3's client-credentials flow. See `examples/example.py` for the full flow.

## Resources

| Resource               | Default | Notes |
|------------------------|---------|-------|
| `greenhouse_jobs`      | on      | Status, department/office, dates — the job's text lives on its posts |
| `greenhouse_job_posts` | on      | Public title + HTML description reduced to plain text |
| `greenhouse_scorecards`| **off** | Interview feedback — **opt-in only**, see below |

Pass `job_statuses=["open"]` to narrow jobs, `include_scorecards=True` to enable
scorecards, and `scorecard_application_ids=[...]` to scope them to specific
applications.

## Privacy

Scorecards are candidate personal data. They are **not** synced by default and
must be explicitly opted in. Even when enabled, the rendered documents exclude
private/public notes, candidate contact details, and compensation — only the
interview, interviewer, overall recommendation, and ratings structure are kept.
The connector is read-only.

## How sync + forget-on-delete work

Harvest v3 has cursor-based pagination (`Link` header) but no change or delete
feed. Each run is a **full snapshot**: `write_disposition="replace"` rewrites
staging with exactly the records currently visible. A deleted job/post — or a
scorecard removed by GDPR right-to-be-forgotten anonymization — drops out of the
listing, so it is absent from the snapshot and cognee's existing `orphan_cleanup`
removes it from the graph and vector stores on the next sync. Unchanged records
keep a stable content-hash `data_id`, so only genuinely new/changed content is
re-ingested. A fetch error aborts the run (leaving memory untouched) rather than
let a partial snapshot forget live records.

## Testing

```bash
uv run pytest tests/
```

All tests mock the Greenhouse API (no live credentials) and cover rendering,
HTML→plain-text, Link-header pagination, privacy allowlists, the scorecards
opt-in gate, edit re-sync, forget-on-delete, and failure safety.