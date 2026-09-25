# Company brain: three sources, one graph

This guide builds a small company brain for a fictional company, Acorn Analytics. It
ingests three kinds of data into one cognee dataset, links them into one knowledge graph,
serves the graph through the API and the UI, and connects Claude Code or Codex to it.

| Source | Kind | File | Node set |
|---|---|---|---|
| HR and project database | relational | `data/company.db` (built from `data/schema.sql`) | `hr_database` |
| Support desk export | structured | `data/tickets.json` | `support_tickets` |
| Meeting notes, a postmortem, a planning memo | unstructured | `data/docs/*.md` | `company_docs` |

Some people, projects and customers appear in all three sources. For example, Dana Kim is
an employee in the database, the assignee of ticket T-1041, and the owner of a fix in the
Atlas meeting notes. The goal is for cognee to turn those three mentions into one node.

## 1. Set up

```bash
uv venv && source .venv/bin/activate
uv pip install cognee          # or, in this repo: uv pip install -e .
echo 'LLM_API_KEY="sk-..."' > .env
```

- The relational source uses `dlt`, which ships with cognee. No extra is needed.
- The UI needs Node.js 20+ and npm. Without them, `cognee.start_ui` falls back to Docker.
- The script sets `ENABLE_BACKEND_ACCESS_CONTROL=false` unless your `.env` sets it.
  This is local single-user mode: the script, the API server and the MCP server read the
  same databases, and the API needs no login.

## 2. The graph model

`models.py` defines what the LLM extracts from every source:

```
Person ──member_of──▶ Team ◀──owned_by── Project ──for_customer──▶ Customer
  │ ├──works_on──────────────────────────▶ Project                     ▲
  │ └──reports_to──▶ Person                                            │
  ▲                                                                    │
Ticket ──assigned_to──▶ Person, ──about──▶ Project, ──raised_by──▶ Customer
```

Two things make the sources link:

- **Identity fields.** Every type declares `identity_fields` (`name`, or `ticket_id` for
  tickets). cognee derives the node id from those values, so "Dana Kim" extracted from the
  database and "Dana Kim" extracted from a ticket get the same id and are stored as one
  node.
- **Consistent names.** Identity only ignores case, spaces-versus-underscores and
  apostrophes, so "Search" and "Search team" would be two different teams. Three things
  keep names consistent here:
  - the sample data spells every name the same way in every file;
  - `EXTRACTION_PROMPT` tells the LLM to copy names exactly and not to add words;
  - `Team` and `Project` have a pydantic `field_validator` that strips the words the LLM
    still adds now and then ("Billing team" becomes "Billing", "Project Atlas" becomes
    "Atlas"). The node id is computed from the validated name, so both spellings land on
    one node, and `FromIdentity` references go through the same validator.

  With your own data, add validators for the variants you see, or merge duplicates
  afterwards with `consolidate_entities_pipeline` (see
  `examples/guides/entity_deduplication.py`).

## 3. Ingest the three sources

`company_brain.py` stores each source with its own `remember()` call into the dataset
`company_brain`, using the same `graph_model` every time. `remember()` ingests the data,
extracts the graph with the model, and runs `improve()` on the result:

```python
# Relational: each row of the three *_profiles views becomes one text document.
hr_database = sql_database(
    credentials="sqlite:///data/company.db",
    table_names=["employee_profiles", "project_profiles", "customer_profiles"],
    include_views=True,
)
hr_database.cognee_document_source = "hr_database"
await cognee.remember(
    hr_database,
    dataset_name="company_brain",
    node_set=["hr_database"],
    graph_model=CompanyGraph,
    custom_prompt=EXTRACTION_PROMPT,
)

# Structured: the ticket export.
await cognee.remember(
    "data/tickets.json",
    dataset_name="company_brain",
    node_set=["support_tickets"],
    graph_model=CompanyGraph,
    custom_prompt=EXTRACTION_PROMPT,
)

# Unstructured: the markdown documents.
await cognee.remember(
    ["data/docs/atlas_weekly_sync_2026-09-19.md", "data/docs/q4_planning_memo.md", ...],
    dataset_name="company_brain",
    node_set=["company_docs"],
    graph_model=CompanyGraph,
    custom_prompt=EXTRACTION_PROMPT,
)
```

Why the relational source looks like this:

- **`cognee_document_source`.** Passing a connection string such as
  `cognee.remember("sqlite:///company.db")` also works, but it takes cognee's relational
  path. That path builds a fixed table-and-row graph and ignores `graph_model`, so its
  nodes would never link to the other sources. Setting `cognee_document_source` on a dlt
  source sends each row down the document path instead, where it is extracted with the
  graph model.
- **The views.** The document path reads a `title` and a `content` column from each row.
  The views in `schema.sql` join the normalized tables into sentences such as "Dana Kim works
  in the Search team as Senior Software Engineer", so the LLM sees names instead of foreign
  keys.

What the node sets do: `node_set` tags the documents and text chunks of each source. It
does not tag the `Person`, `Team`, `Project`, `Customer` and `Ticket` nodes; those are
shared by every source that mentions them, which is the point of linking. So a node set
answers "what does this source say", through a `CHUNKS` search (section 6), and the graph
answers "what do we know", across all sources.

Run it:

```bash
uv run python examples/demos/company_brain/company_brain.py
```

The script empties cognee's store with `cognee.prune`, ingests the three sources (about
two minutes), prints a verification report (section 6), then starts the API server on
http://localhost:8000 and the UI on http://localhost:3000. Stop it with Ctrl+C.

> **The prune deletes all data cognee has stored locally**, not only this demo's. It keeps
> the graph limited to the demo, which the UI needs: in single-user mode every dataset
> shares one graph, so older data would otherwise show up next to the company brain. To
> keep existing data, point `DATA_ROOT_DIRECTORY` and `SYSTEM_ROOT_DIRECTORY` in `.env` at
> empty folders before running.

- `--no-ui` builds and verifies without starting the servers.
- `--api-port 8010 --ui-port 3010` uses other ports when 8000 or 3000 is taken. Use the
  same API port in the MCP commands in section 5.

## 4. Open the UI

Open http://localhost:3000 and select the `company_brain` dataset.

- The graph view shows one `Dana Kim` node, connected to the Search team, the Atlas and
  Harbor projects, ticket T-1041, and her manager Marco Rossi.
- The document and chunk nodes of each source hang off a `NodeSet` node named after it
  (`hr_database`, `support_tickets`, `company_docs`).

To start the servers later without ingesting again, run `cognee-cli -ui`.

## 5. Connect Claude Code or Codex

The coding agents talk to cognee through the cognee MCP server. Run it in API mode,
pointed at the running API server, so the agent reads the same graph as the UI. Without
`--api-url`, the MCP server opens its own local databases and sees a different, empty
brain.

Keep `company_brain.py` (or `cognee-cli -ui`) running, then register the MCP server.

**Claude Code**

```bash
claude mcp add cognee -- uvx cognee-mcp --api-url http://localhost:8000
claude mcp list        # cognee: ... ✓ Connected
```

**Codex**

```bash
codex mcp add cognee -- uvx cognee-mcp --api-url http://localhost:8000
```

or add it to `~/.codex/config.toml`:

```toml
[mcp_servers.cognee]
command = "uvx"
args = ["cognee-mcp", "--api-url", "http://localhost:8000"]
```

Then ask the agent a question that needs all three sources:

> Use cognee to recall: who is handling Brightline Retail's open high-priority ticket,
> which team are they on, and what fix was decided for it?

The agent calls the `recall` tool and should answer: Dana Kim (ticket T-1041), on the
Search team, who will change the Atlas indexer to read every catalog feed file and
re-index Brightline Retail by 26 September.

If your API server requires authentication (`ENABLE_BACKEND_ACCESS_CONTROL=true`), pass
a token with `--api-token <token>`.

## 6. Check that it worked

The script prints this report after ingestion. The answers are LLM-written, so wording
varies between runs; the facts should not.

**Nodes per type.** One node per real entity, with no duplicates:

```
Customer  3
Person    8
Project   4
Team      4
Ticket    5
```

A higher count means one entity was stored under two names. Look in the graph view for
near-duplicates such as "Search" and "Search team", or a component extracted as its own
project ("Ledger email worker" next to "Ledger"). Fix spelling variants with a validator
in `models.py` and wrong kinds of entity with a rule in `EXTRACTION_PROMPT`.

**One node per person, linked across sources.** Dana Kim is one node, and its edges come
from different sources: team and manager from the HR database, the ticket from the
export.

```
Dana Kim: 1 Person node, connected to
  member_of    Team     Search
  works_on     Project  Atlas
  works_on     Project  Harbor
  reports_to   Person   Marco Rossi
  assigned_to  Ticket   T-1041
```

**Per-source search.** A `CHUNKS` search with `node_name` returns text from that source
only:

```python
from cognee.modules.search.types import SearchType

await cognee.recall(
    "Who works on the Atlas project?",
    query_type=SearchType.CHUNKS,
    datasets=["company_brain"],
    node_name=["support_tickets"],
)
```

`hr_database` returns the Atlas, Dana Kim and Marco Rossi profile rows,
`support_tickets` returns the ticket export, and `company_docs` returns the meeting
notes, the postmortem and the memo. Use `CHUNKS` for this: in this setup the completion
search types (`GRAPH_COMPLETION`, `RAG_COMPLETION`) answer from the whole graph even when
`node_name` is given.

**Cross-source question.** Without a filter, the question from section 5 needs one fact
from each source: the assignee comes from the ticket export, the team from the HR
database, and the decided fix from the meeting notes.

## Troubleshooting

- **"Another next dev server is already running."** Next.js runs one dev server per
  checkout of `cognee-frontend`. Stop the other one (the message prints its PID), or keep
  it and open the URL it prints; it talks to the API server on port 8000 by default.
- **Port 8000 or 3000 is taken.** Pass `--api-port` and `--ui-port`.
- **The agent says it found nothing.** Check that the MCP server was registered with
  `--api-url` and the same API port the script prints.

## Files

| File | What it is |
|---|---|
| `company_brain.py` | Ingests the three sources, prints the verification report, starts the API and UI |
| `models.py` | The `CompanyGraph` model and the extraction prompt |
| `data/schema.sql` | Tables, rows and profile views of the HR database |
| `data/company.db` | The database built from `schema.sql` (rebuilt automatically if deleted) |
| `data/tickets.json` | The support desk export |
| `data/docs/` | Meeting notes, a postmortem and a planning memo |
| `company_brain_demo.py` | A separate, shorter demo of text, code and session memory |
