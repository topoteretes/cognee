---
name: cognee-cli
description: Use when the user wants to drive cognee from the terminal with cognee-cli — remember/recall/forget/improve memory commands, managing datasets and config, or database migrations.
---

# Use the cognee CLI

`cognee-cli` ships with the package (entry point in `cognee/cli/_cognee.py`;
each command lives in `cognee/cli/commands/`). Every command has
`--help` with examples — prefer that over guessing flags. Needs
`LLM_API_KEY` configured, same as the SDK.

## Core flow

The memory commands are the primary surface as of cognee 1.x:

```bash
cognee-cli remember "Your text here"         # also accepts file paths / URLs
cognee-cli remember ./docs --dataset-name my_project
cognee-cli recall "Your question"            # query the graph
cognee-cli recall "keyword" --query-type CHUNKS
cognee-cli forget --all                      # wipe local state
```

`remember` is ingest + graph build in one step (`add` + `cognify` under the
hood); `--background`/`-b` runs the cognify stage in the background, and
`--dry-run` estimates LLM tokens/cost without ingesting. `recall` takes
`--datasets`/`-d`, `--top-k`/`-k` (default 10), and `--session-id`/`-s`.

`forget` targets `--dataset`, `--dataset-id`, `--data-id` (needs a dataset), or
`--everything`/`--all` — one unified command covering what `delete`, `prune`,
and `empty_dataset` used to do separately.

> **`forget --all` does not ask for confirmation.** It deletes every dataset
> immediately, even on a non-interactive stdin. The legacy `delete --all`
> prompts `Delete ALL data from cognee? [y/N]` first, so switching to `forget`
> silently drops that safety net — script it with care.

Search types match exactly 7 of the SDK's `SearchType` enum (`cognee/modules/search/types/SearchType.py`), those 7 being chosen in (`cognee/cli/config.py:SEARCH_TYPE_CHOICES`):
GRAPH_COMPLETION, RAG_COMPLETION, CHUNKS, SUMMARIES, CODE, CYPHER, GRAPH_REPORT
Others must be reached from the SDK, not CLI; e.g. call cognee.recall with `query_type=SearchType.TEMPORAL`

Note the CLI defaults `--query-type` to `GRAPH_COMPLETION`, whereas the SDK's
`cognee.recall()` auto-routes when `query_type` is omitted.

## Session memory and enrichment

```bash
cognee-cli remember "fact" --dataset-name my_project
cognee-cli improve -d my_project             # enrich/index the graph
cognee-cli improve -d my_project -s chat_1   # + bridge session content into the graph
cognee-cli sessions get                      # retrieve session Q&A history
cognee-cli feedback ...                      # attach feedback to results
```

`improve` also takes `--node-name`, `--feedback-alpha` (default 0.1), and
`--background`/`-b`. `remember`/`improve` build their graphs through
`cognify()`, so cognify-level settings (e.g. `CONTRADICTION_DETECTION=true`)
apply to them too.

## Legacy / lower-level commands

`add`, `cognify`, `search`, `memify`, and `delete` still ship and are what the
memory commands call underneath. Use them only to drive a single stage in
isolation; prefer `remember`/`recall`/`forget`/`improve` otherwise.

```bash
cognee-cli add "text" && cognee-cli cognify  # what `remember` does in one step
cognee-cli search "question"                 # `recall` without session support
cognee-cli memify -d my_project              # custom extraction/enrichment tasks
cognee-cli delete --all                      # superseded by `forget --all`
```

## Management

```bash
cognee-cli datasets list                     # dataset operations
cognee-cli config get [key] [--show-secrets] # view one/all settings (API keys masked by default)
cognee-cli config set <key> <value>          # set + persist to ./.env in the cwd
cognee-cli config unset <key>                # reset a key to its default (also persisted)
cognee-cli -ui                               # launch API server + UI (see cognee-server skill)
cognee-cli serve --url http://localhost:8000 # connect CLI/SDK to a running instance
```

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
- `remember` (and `add`) without `--dataset-name` targets the default dataset
  `main_dataset`; `recall`/`search` operate across your accessible datasets
  unless a dataset is given.
- `forget` refuses to run bare — pass `--dataset`, `--dataset-id`, `--data-id`
  (with a dataset), or `--everything`/`--all`.
- `memify` requires one of the arguments -d/--dataset-name --dataset-id
- `config set`/`config unset` write to the `.env` file in whatever directory
  you run the command from (creating it if missing). `config reset` (reset
  *all* keys) is still not implemented.
- **Which `.env` actually wins is not always the cwd one.** At import, cognee
  calls `dotenv.load_dotenv(override=True)`, which resolves relative to the
  *cognee package location*, not your working directory. In a source/editable
  checkout (`uv pip install -e .`) a `.env` at the repo root therefore shadows
  the `.env` in the directory you ran from — and because `override=True`, it
  also beats variables you `export`ed. Symptom: `config set` appears to do
  nothing, or the CLI connects to a backend you thought you had overridden.
  To test against different settings, move the repo `.env` aside, or set
  values programmatically after import (`cognee.config.set_*`). (Under
  `python -c` the cwd `.env` does win, because dotenv falls back to the cwd
  when `__main__` has no `__file__` — which is why the same command can
  behave differently as a script vs. `-c`.)
