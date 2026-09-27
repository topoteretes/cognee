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

## Output format (exactly this, nothing else)

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
