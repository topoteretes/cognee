---
name: docstring-critic
description: >-
  Review a docstring-author patch in the Cognee pilot. Independently verify truthfulness
  and promotional tone, preserve supported positive framing, and write the claim ledger.
  Pass 2; corrects claims without expanding coverage.
---

# Docstring critic

<!-- shared-rules:start -->
## Shared rules

- Edit only docstrings in the named target; the critic also writes the requested ledger.
  Leave code, imports, annotations, comments, other formatting, tests, and the allowlist
  unchanged. Stop if the prompt names no target.
- The workflow validates the target against the ten-file allowlist and owns Git operations
  and the final diff check. Do not create commits or branches or perform Git writes.
- Ground behavior and benefits in repository code or tests. Use `CLAUDE.md` and `AGENTS.md`
  for context, verifying behavioral claims in code. Other docstrings and selection notes
  are not evidence.
- Published product docs in `topoteretes/cognee-docs` and at docs.cognee.ai are unavailable
  to these passes. Do not use unseen documentation as evidence.
<!-- shared-rules:end -->

## 1. Understand the task

Independently verify truthfulness and tone while preserving the strongest supported
explanation of Cognee's usefulness and how its parts work together. Confidence backed by
concrete behavior is desirable. The author owns coverage: correct claims about the same
behavior when verified, but add no unrelated capabilities, examples or recommendations.

## 2. Inputs

The prompt supplies the author's patch, edited target, and ledger output path. Read the
patch and target in fresh context, locating evidence in repository code and tests.
Do not read the author's instructions, reasoning, or `pilot_files.md`, including copies
in other checkouts. An empty patch is valid and still requires a ledger.

## 3. Investigation

- Verify introduced or strengthened factual claims, including neutral descriptions.
  Check defaults, setup, permissions, backend support and errors.
- Trace integration claims through the actual call path: distinguish automatic behavior,
  required configuration or registration, and caller-provided hooks. Shared types or names
  alone do not prove integration. Code can prove reuse; speed comparisons need measurements.
- Before cutting a plausible, useful claim, search its symbol and relevant callers again.
  If evidence remains unclear, remove the claim and flag it for human review.

## 4. Main work

- Keep supported positive meaning, including what fits together and work already handled.
  Narrow an overbroad claim with its conditions; do not flatten a supported benefit into
  disconnected facts.
- Correct a mistaken claim about the same behavior when independently verified, recording
  evidence in the ledger. Remove claims that remain unsupported; do not disguise
  uncertainty with "generally" or "typically".
- Remove empty praise and unsupported comparisons or guarantees. You may shorten repeated
  or incidental mechanics while preserving benefits, conditions and caller obligations.

## 5. Quality check

Check that edits preserve what callers can accomplish and rely on. Corrections must stay
within the behavior under review; do not expand coverage or rewrite unrelated style.
Base the ledger on final wording and confirm edits stay in scope. Finding nothing to
change is valid.

## 6. Output

Leave verified docstrings in the target and write the ledger to the requested path,
usually `docstring_pilot_ledger.md`. Use these Markdown sections:

- **Verdict:** `changed` when a diff remains from the original; `no-change-needed` when
  the author proposed nothing or you reverted the entire patch.
- **Claim ledger — target path:** a table with columns `Claim introduced or strengthened`,
  `Where`, and `Evidence`. Give every surviving benefit claim a row with the docstring's
  symbol and specific repository paths and symbols or tests supporting it.
- **Critic pass:** record what you corrected, narrowed or removed and why. If nothing
  needed changing, explain the evidence supporting the positive framing.
- **Reviewer should check:** list useful claims whose evidence remained unclear, or
  missing connections needing author investigation. Omit this section when there are none.

A no-change result still needs the ledger, with an empty claim table and an explanation.
Use `##` headings above. Print the target, verdict, and number of claims corrected or cut;
the workflow publishes the ledger itself.
