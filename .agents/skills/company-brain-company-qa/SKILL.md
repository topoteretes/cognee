---
name: company-brain-company-qa
description: Build one company memory from a SQL database, a support ticket export and a folder of documents, linked into one graph by cognee, then answer questions that need several of those sources together. Use when someone wants to connect their company's database, tickets and docs, or asks who owns, decided or is handling something across them. Runs locally; reads the sources, writes only to cognee's memory.
---

# Company brain: company Q&A

Runs the company brain cookbook in `examples/cookbooks/company_brain/company_qa/`. It
remembers a SQL database, a ticket export and a docs folder in the cognee dataset
`company_qa`, extracts them with one graph model so the same person, team or project
becomes one node, then answers a question across them. Setup, the scripts and what each
one does are in the cookbook's
[`README.md`](../../../examples/cookbooks/company_brain/company_qa/README.md).

Run every command from the repo root.

## Steps

1. Work out the sources. Ask the user for the ones they want: a database URL (and tables),
   one or more ticket export files (JSON or CSV), a docs folder. If they have none, or
   want to see it work first, pass no sources: the run then uses the sample company (the
   `[setup]` line says so). Tell the user the answer comes from sample data.
2. Run
   `uv run python examples/cookbooks/company_brain/company_qa/company_qa.py --check`
   with the sources (`--database URL [--tables a,b] --tickets FILE [FILE ...] --docs FOLDER`), or
   none for the sample.
   - Exit 0: go to step 3.
   - Exit 2: each `[setup] MISSING:` line names one fix. Don't create credentials yourself.
     Tell the user which line to fix, using the README's "What it needs" table, and stop.
3. Run the same command without `--check`, adding the user's question as
   `--ask "..."`. On the sample with no `--ask`, it asks a question that needs all three
   sample sources.
4. The answer is everything after `[ask] A:`. Give it to the user. The `[ingest]` lines
   say which sources went in.
5. Later questions don't need the sources again:
   `uv run python examples/cookbooks/company_brain/company_qa/scripts/ask.py "..."`.
6. If a script fails, `company_qa.py` exits 1 with a line naming what went wrong. Run
   that script from `examples/cookbooks/company_brain/company_qa/scripts/` on its own to
   look closer.

## Rules

- The sources are the user's company data. Don't quote them beyond what answers the
  question, and don't print `.env` or a database URL's password.
- Every run uses LLM credits, so ingest only when the user asked for it.
- A sample run forgets the cookbook's dataset `company_qa` before it ingests (the `[clear]`
  line). On the user's own data, pass `--clear` only when they asked to start over: it
  forgets everything the dataset holds.
- Don't pass `--ui` unless the user asked to browse the graph: it keeps running until
  Ctrl+C.

## Clean up

```bash
uv run cognee-cli forget --dataset company_qa
```
