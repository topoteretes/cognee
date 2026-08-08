---
name: cognee-cli
description: Use when the user wants to drive cognee from the terminal with cognee-cli — ingesting data, building and searching the knowledge graph, managing datasets and config, memory commands, or database migrations.
---

# Use the cognee CLI

`cognee-cli` ships with the package (entry point in `cognee/cli/_cognee.py`;
each command lives in `cognee/cli/commands/`). Every command has
`--help` with examples — prefer that over guessing flags. Needs
`LLM_API_KEY` configured, same as the SDK.

## Core flow

```bash
cognee-cli add "Your text here"              # also accepts file paths / URLs
cognee-cli add ./docs --dataset-name my_project
cognee-cli cognify                           # build the knowledge graph
cognee-cli search "Your question"            # GRAPH_COMPLETION by default
cognee-cli search "keyword" --query-type CHUNKS
cognee-cli delete --all                      # wipe local state
```

Search types match exactly 7 of the SDK's `SearchType` enum (`cognee/modules/search/types/SearchType.py`), those 7 being chosen in (`cognee/cli/config.py:SEARCH_TYPE_CHOICES`):
GRAPH_COMPLETION, RAG_COMPLETION, CHUNKS, SUMMARIES, CODE, CYPHER, GRAPH_REPORT
Others must be reached from the SDK, not CLI; e.g. call cognee.search with `query_type=SearchType.TEMPORAL`

## Management

```bash
cognee-cli datasets list                     # dataset operations
cognee-cli config get [key] [--show-secrets] # view one/all settings (API keys masked by default)
cognee-cli config set <key> <value>          # set + persist to ./.env in the cwd
cognee-cli config unset <key>                # reset a key to its default (also persisted)
cognee-cli -ui                               # launch API server + UI (see cognee-server skill)
cognee-cli serve --url http://localhost:8000 # connect CLI/SDK to a running instance
```

## Memory commands

Higher-level agent-memory surface on top of the core flow:

```bash
cognee-cli remember "fact to keep"           # store into permanent memory
cognee-cli recall "what do I know about X"   # retrieve
cognee-cli memify                            # enrich the graph
cognee-cli forget ...                        # remove memories
cognee-cli improve                           # bridge session memory into the graph
cognee-cli sessions get                      # retrieve session Q&A history
cognee-cli feedback ...                      # attach feedback to results
```

`remember`/`improve` build their graphs through `cognify()`, so cognify-level
settings (e.g. `CONTRADICTION_DETECTION=true`) apply to them too.

## Relational DB migrations (Alembic)

```bash
cognee-cli upgrade        # apply migrations
cognee-cli downgrade
cognee-cli history
cognee-cli current
```

Typically needed after version upgrades when the server refuses to start on
an old schema.

## Gotchas

- The CLI initializes cognee lazily; the first command in a fresh environment
  is slow (DB + model setup), later ones are fast.
- `add` without `--dataset-name` targets the default dataset `main_dataset`;
  cognify/search operate across your accessible datasets unless a dataset is
  given.
- `memify` requires one of the arguments -d/--dataset-name --dataset-id
- `config set`/`config unset` write to the `.env` file in whatever directory
  you run the command from (creating it if missing) — run later
  `cognee-cli`/SDK commands from that same directory to see the change take
  effect, same as editing `.env` by hand. `config reset` (reset *all* keys)
  is still not implemented.
