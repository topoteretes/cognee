---
name: company-brain-multi-source
description: Build one company memory from a SQL database, a support ticket export and a folder of documents, linked into one graph by cognee, then answer questions that need several of those sources together. Use when someone wants to connect their company's database, tickets and docs, or asks who owns, decided or is handling something across them. Runs locally; reads the sources, writes only to cognee's memory.
---

# Company brain: multi-source

Runs the company brain cookbook in `examples/cookbooks/company_brain/multi_source/`. It
remembers a SQL database, a ticket export and a docs folder in the cognee dataset
`company_brain`, extracts them with one graph model so the same person, team or project
becomes one node, then answers a question across them. Setup, the scripts and what each
one does are in the cookbook's
[`README.md`](../../../examples/cookbooks/company_brain/multi_source/README.md).

Run every command from the repo root.

## Steps

1. Work out the sources. Ask the user for the ones they want: a database URL (and tables),
   a ticket export file, a docs folder. If they have none, or want to see it work first,
   use the sample: run
   `uv run python examples/cookbooks/company_brain/multi_source/setup.py` and pass
   `--sample` in place of the sources below.
2. Run
   `uv run python examples/cookbooks/company_brain/multi_source/company_brain.py --check`
   with the sources (`--database URL [--tables a,b] --tickets FILE --docs FOLDER`, or
   `--sample`).
   - Exit 0: go to step 3.
   - Exit 2: each `[setup] MISSING:` line names one fix. Don't create credentials yourself.
     Tell the user which line to fix, using the README's "What it needs" table, and stop.
3. Run the same command without `--check`, adding the user's question as
   `--ask "..."`. With `--sample` and no `--ask`, it asks a question that needs all three
   sample sources.
4. The answer is everything after `[ask] A:`. Give it to the user. The `[ingest]` lines
   say which sources went in.
5. Later questions don't need the sources again:
   `uv run python examples/cookbooks/company_brain/multi_source/scripts/ask.py "..."`.
6. If a script fails, `company_brain.py` exits 1 with a line naming what went wrong. Run
   that script from `examples/cookbooks/company_brain/multi_source/scripts/` on its own to
   look closer.

## Rules

- The sources are the user's company data. Don't quote them beyond what answers the
  question, and don't print `.env` or a database URL's password.
- Every run uses LLM credits, so ingest only when the user asked for it.
- Don't pass `--ui` unless the user asked to browse the graph: it keeps running until
  Ctrl+C.

## Clean up

```bash
uv run cognee-cli forget --dataset company_brain
```

The `follow_up_agent/` cookbook writes to the same `company_brain` dataset, so this also
removes what it remembered.
