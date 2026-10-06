---
name: personalized-email
description: Draft a reply to your newest Gmail email that knows what you discussed in your Granola meetings, what you promised the sender, and how you write, using cognee memory. Use when someone asks to answer or reply to their latest email in their own tone. Runs locally; reads Gmail read-only; the draft is printed, never sent.
---

# Personalized email

Runs the personalized email cookbook in `examples/cookbooks/personalized_email/`. It
remembers your Granola meeting notes and your Gmail inbox and sent mail in the cognee
dataset `personalized_email`, then drafts a reply to the newest inbox email. Setup, the
scripts and what each one does are in the cookbook's
[`README.md`](../../../examples/cookbooks/personalized_email/README.md).

Run every command from the repo root.

## Steps

With neither Gmail nor Granola set up, the commands below run on a sample mailbox (the
`[setup]` line says so) and draft a reply to a fictional email; only `LLM_API_KEY` is
needed. Tell the user the reply is to sample data. Add `--sample` to use the sample even
when their accounts are set up.

1. Run `uv run python examples/cookbooks/personalized_email/personalized_email.py --check`.
   - Exit 0: go to step 2.
   - Exit 2: each `[setup] MISSING:` line names one fix. Don't create credentials yourself.
     Tell the user exactly which line to fix, using the README's "What it needs" table,
     and stop. If only the Granola line is missing, ask whether to run with `--no-granola`
     instead.
2. Run `uv run python examples/cookbooks/personalized_email/personalized_email.py`. Add
   `--days N` if the user named a time range for meetings, and `--no-granola` if they
   don't want Granola used.
3. The draft is everything after the line `[draft] Reply:`. Give it to the user, and say
   which email it answers (the `[draft] Answering:` line) and which sources went in (the
   `[ingest_granola]` and `[ingest_email]` lines). The draft is never sent: hand it to the
   user to send.
4. If a script fails, `personalized_email.py` exits 1 with a line naming what went wrong.
   Run that script from `examples/cookbooks/personalized_email/scripts/` on its own to
   look closer.

## Rules

- Gmail access is read-only and nothing is sent.
- Don't print the contents of `credentials.json`, `token.json` or `.env`.
- A sample run forgets the cookbook's dataset `personalized_email` before it ingests (the `[clear]`
  line). On the user's own data, pass `--clear` only when they asked to start over: it
  forgets everything the dataset holds.
- Ingesting reads the user's real meetings and mailbox, and every run uses LLM credits, so
  run it only when the user asked for it.

## Clean up

```bash
uv run cognee-cli forget --dataset personalized_email
```
