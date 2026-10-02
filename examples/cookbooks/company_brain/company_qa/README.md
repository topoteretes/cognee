# Company brain: your database, tickets and docs in one graph

One memory for your company, built from the three kinds of data most companies already
have: a relational database (HR, projects, customers), a support ticket export, and a
folder of documents (meeting notes, postmortems, memos).

cognee remembers each source in one dataset (`company_brain`) and extracts all of them
with one graph model, so a person in the database, the assignee of a ticket and a name in
the meeting notes become one node. Then you ask questions that no single source can
answer, browse the graph in the UI, or let Claude Code or Codex query it over MCP.

Agents run this cookbook through the `company-brain-company-qa` skill,
[`.agents/skills/company-brain-company-qa/SKILL.md`](../../../../.agents/skills/company-brain-company-qa/SKILL.md).

## What it needs

| What | Why | Where |
|---|---|---|
| `LLM_API_KEY` | cognee extracts the graph and answers with an LLM (OpenAI by default) | `.env` at the repo root |
| A SQL database (optional) | Any database SQLAlchemy can reach, read through dlt's `sql_database` | `--database` URL |
| A ticket export (optional) | One JSON or CSV file from your support desk | `--tickets` path |
| A docs folder (optional) | Meeting notes, postmortems, memos: any documents cognee reads | `--docs` path |
| Node.js 20+ and npm (for `--ui`) | Runs the UI. Without them, cognee falls back to Docker | your machine |

Pass at least one source. Without data of your own, try the sample first.

## Try it on the sample company

The sample is Acorn Analytics, a fictional company. `setup.py` holds its three sources and
writes them to `sample/` (git-ignored): an HR and project database (`company.db`), a ticket
export (`tickets.json`) and three documents (`docs/`). Some people, projects and customers
appear in all three. No accounts or data of yours are needed, only `LLM_API_KEY`.

```bash
uv run python examples/cookbooks/company_brain/company_qa/company_qa.py
```

With no `--database`, `--tickets` or `--docs` given, the script runs `setup.py` itself and uses the sample, as below. Pass `--sample` to
use it even when your own sources are set up.

The sample run points every source at `sample/` and asks a question that needs all three:

```text
[ingest] Remembered the database (employee_profiles, project_profiles, customer_profiles)
[ingest] Remembered the tickets in .../sample/tickets.json
[ingest] Remembered the docs in .../sample/docs
[ask] Q: Who is handling Brightline Retail's open high-priority ticket, which team are they on, and what fix was decided for it?
[ask] A: Dana Kim is handling it. She's on the Search team (owner of Atlas). The agreed fix is to change the indexer to read every file in the Brightline catalog feed and re-index the catalog.
```

The ticket says who is assigned, the database says which team she is on, and the meeting
notes say what fix was decided. The answer joins them because Dana Kim is one node.

## Run it on your data

```bash
uv run python examples/cookbooks/company_brain/company_qa/company_qa.py --check \
    --database postgresql://user:password@host/hr --tickets ~/exports/tickets.json \
    --docs ~/Documents/company
uv run python examples/cookbooks/company_brain/company_qa/company_qa.py \
    --database postgresql://user:password@host/hr --tables employees,projects \
    --tickets ~/exports/tickets.json --docs ~/Documents/company \
    --ask "Who owns the open billing incident, and what did we decide to do about it?"
```

- `--database` takes a SQLAlchemy URL (`postgresql://...`, `mysql+pymysql://...`,
  `sqlite:///path/to.db`). Every row of `--tables` becomes one document; without
  `--tables`, every table and view is read. A row with `title` and `content` columns is
  used as it is; any other row is written out as one `column: value` line per column.
  Rows read best as sentences, so a view that joins your tables into readable text with
  `id`, `title` and `content` columns (see the `*_profiles` views in `SCHEMA` in
  `setup.py`) extracts better than raw foreign keys.
- Running it again re-remembers the same content; cognee skips content it already holds.
- Later questions don't need the sources again: `scripts/ask.py "question"`.

## Steps

```
company_qa/
├── README.md           this file
├── company_qa.py       checks setup, then calls the scripts in order
├── setup.py            writes the sample company, for --sample
├── models.py           the graph model: edit it to match your company
├── sample/             written by setup.py, git-ignored
└── scripts/
    ├── ingest.py
    ├── ask.py
    └── ui.py
```

`company_qa.py` imports each script and calls its function in one process. Each script
also runs alone with the same options.

| # | Command (`uv run python examples/cookbooks/company_brain/company_qa/...`) | Does | Writes |
|---|---|---|---|
| 0 | `company_qa.py --check [sources]` | Reports what is missing. Does no work | nothing |
| 1 | `scripts/ingest.py [--database URL [--tables a,b]] [--tickets FILE] [--docs FOLDER]` | Remembers each source given, under its own node set (`database`, `tickets`, `docs`), extracted with `models.py` | cognee dataset |
| 2 | `scripts/ask.py "question"` (or `--ask`) | Answers from the whole graph, across every source | nothing |
| 3 | `scripts/ui.py` (or `--ui`) | Starts cognee's API server in this process and the UI at http://localhost:3000. Ctrl+C stops both | nothing |

All scripts use the cognee dataset `company_brain`, named once in each script.

They also default `ENABLE_BACKEND_ACCESS_CONTROL` to `false` (a value in `.env` still
wins). That is local single-user mode: the scripts, the API server and the MCP server
read one set of databases, and the API needs no login.

## The graph model

`models.py` defines what the LLM extracts from every source:

```
Person ──member_of──▶ Team ◀──owned_by── Project ──for_customer──▶ Customer
  │ ├──works_on──────────────────────────▶ Project                     ▲
  │ └──reports_to──▶ Person                                            │
  ▲                                                                    │
Ticket ──assigned_to──▶ Person, ──about──▶ Project, ──raised_by──▶ Customer
```

Edit it to match your company: add the entities your data talks about, drop the ones it
doesn't. Two things make the sources link, so keep them when you edit:

- **Identity fields.** Every type declares `identity_fields` (`name`, or `ticket_id` for
  tickets). cognee derives the node id from those values, so "Dana Kim" extracted from the
  database and "Dana Kim" extracted from a ticket get the same id and are stored as one
  node.
- **Consistent names.** Identity only ignores case, spaces-versus-underscores and
  apostrophes, so "Search" and "Search team" would be two different teams.
  `EXTRACTION_PROMPT` tells the LLM to copy names exactly, and `Team` and `Project` have a
  pydantic `field_validator` that strips the words the LLM still adds now and then
  ("Billing team" becomes "Billing"). Add validators for the variants you see in your
  data, or merge duplicates afterwards with `consolidate_entities_pipeline` (see
  `examples/guides/entity_deduplication.py`).

Node sets tag the documents and chunks of each source, not the `Person`, `Team`, ...
nodes, which are shared by every source that mentions them. So a node set answers "what
does this source say" (a `CHUNKS` recall with `node_name=["tickets"]`), and the graph
answers "what do we know", across all sources.

## Open the UI

Add `--ui`, or run `scripts/ui.py` later. Open http://localhost:3000 and select the
`company_brain` dataset. The mind map shows the extracted entities grouped by type, with
one node per person connected to their team, projects, tickets and manager.

## Connect Claude Code or Codex

The coding agents talk to cognee through the cognee MCP server. Run it in API mode,
pointed at the running API server, so the agent reads the same graph as the UI. Without
`--api-url`, the MCP server opens its own local databases and sees a different, empty
brain.

Keep `scripts/ui.py` running, then warm up the MCP server once. The first `uvx` run
downloads cognee, and even a cached start takes about 20 seconds, longer than an agent
waits on a first launch:

```bash
uvx cognee-mcp --help
```

**Claude Code**

```bash
claude mcp add --scope user cognee -- uvx cognee-mcp --api-url http://localhost:8000
claude mcp list        # cognee: ... ✓ Connected
```

`--scope user` makes the server available in every project. Without it, `claude mcp add`
registers it only for the directory you ran it in. If the server still times out on
start, raise the limit: `MCP_TIMEOUT=60000 claude`.

**Codex**

Add the server to `~/.codex/config.toml`. Codex waits 10 seconds for an MCP server to
start by default, so raise `startup_timeout_sec`:

```toml
[mcp_servers.cognee]
command = "uvx"
args = ["cognee-mcp", "--api-url", "http://localhost:8000"]
startup_timeout_sec = 60
```

Then ask the agent a question that needs several sources, for example on the sample:

> Use cognee to recall: who is handling Brightline Retail's open high-priority ticket,
> which team are they on, and what fix was decided for it?

## Clean up

```bash
uv run cognee-cli forget --dataset company_brain
```

The `follow_up_agent/` cookbook writes to the same `company_brain` dataset, so this also
removes what it remembered.
