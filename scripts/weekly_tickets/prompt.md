# Weekly ticket proposals from docs-support signal

You are analyzing real user pain from the last week of cognee docs-assistant
conversations (see `digest.md` in the working directory) against the cognee
codebase checked out in this repository.

## Task

Produce **3 to 6 ticket proposals** that would fix the most common problems
users hit this week. Quality bar: each proposal must be verified against the
code — find the actual file and line where the problem lives before proposing
anything. If a reported symptom does not correspond to a real defect in this
codebase (user error, already fixed on this branch, or third-party), say so in
a short "Not filed" list instead of inventing a ticket.

## Rules

- Work from `digest.md`: the theme counts tell you what is common; the raw
  error reports tell you what is actually breaking.
- Root-cause first: grep/read the code paths behind each error before writing
  the proposal. Every proposal must cite at least one `path/to/file.py:line`.
- Do not propose anything already covered by the open Linear tickets listed in
  the digest. If a proposal is adjacent to an existing ticket, name that ticket
  and explain the delta.
- Prefer small, high-leverage fixes (error-message quality, wrong defaults,
  retry misclassification, docs one-liners) over rewrites.
- No code changes — analysis and proposals only.

## Output

You write two files in the working directory. The workflow uploads both and
files the issue itself; you do **not** call `gh issue create` and you print
nothing of substance to stdout.

1. `proposals-report.md` — your full working notes, for the run artifact
   only: every proposal in the format below, the "Not filed" list, the
   duplicate checks against the digest's Linear tickets, and the code paths
   you read. There is no length limit on this file.
2. `proposals-issue.md` — what people read, so it is short: the top
   proposals as two tables with the same rows. **Write it only if at least
   one proposal survived verification**; otherwise leave it out and end the
   report with "No proposals this week."

### Report format (`proposals-report.md`)

```markdown
## Proposed tickets

### 1. <title, imperative, one line>
- **Priority:** High | Medium | Low
- **Evidence (docs digest):** <which reports/themes this addresses, with counts>
- **Root cause (verified):** <file:line + one-paragraph explanation>
- **Fix shape:** <2-4 sentences, concrete>

### 2. ...

## Not filed
- <symptom> — <why no ticket: duplicate of X / not reproducible in code / user error>
```

### Issue file format (`proposals-issue.md`, exact)

Two tables, same rows, nothing else — no title, no prose, no "Not filed".

```markdown
## Simply put

| # | Problem | Fix |
|---|---|---|
| 1 | <one sentence a non-engineer understands: what goes wrong for whom, and how often> | <one sentence: what we would change and how we would know it worked> |

## Details

| # | Evidence | Root cause | Fix shape |
|---|---|---|---|
| 1 | <one sentence: which reports or themes, with counts> | <one sentence: the defect, with `path/to/file.py:line`> | <one sentence: the concrete change> |
```

Example rows for one proposal:

`| 1 | About one in six questions this week were from people whose local model setup silently fell back to OpenAI and then failed for lack of a key. | Refuse to start with a clear message when only one of the two providers is configured, and watch that theme's count drop. |`

`| 1 | 41 of 260 conversations in "llm / model config"; 9 error reports mention an OpenAI 401 while using Ollama. | Embedding provider defaults to OpenAI when only LLM_PROVIDER is set (cognee/infrastructure/databases/vector/embeddings/config.py:88). | Raise a configuration error naming both providers when exactly one is set; add the pairing to the Ollama docs page. |`

Rules for the tables:

- **At most 3 proposals**, ordered by priority; the rest stay in the report.
  Both tables have the same rows in the same order, numbered 1, 2, 3.
- **One sentence, ≤ 20 words, per cell.**
- **"Simply put" is for someone who has never seen the code**: no file paths,
  function names, env var names, version strings or ticket numbers; say
  "installs", "local model setup", "the docs page" instead. Ratios ("one in
  six") beat percentages there.
- No `<br>`, no lists, no nested tables, no code fences, no `|` characters
  (write "or" instead). Everything beyond one sentence goes in the report.
- Nothing outside the two headings and the two tables.

The workflow fails the run if a table header is not exactly as above, a cell
is empty, or the two tables have different rows; it cuts any cell longer than
140 characters and drops proposals past the third.
