---
name: docstring-critic
description: >-
  Review a docstring-author patch in the Cognee pilot. Independently verify truthfulness
  and promotional tone, preserve supported positive framing, and write the claim ledger.
  Pass 2; introduces no new product claims.
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

Verify the author's claims and remove overpromotional wording while preserving concrete,
supported reasons to use Cognee. Positive framing is an intended result: keep explanations
of what callers gain and what Cognee handles for them. Your edits address truthfulness
and tone. The author owns usefulness and coverage; do not add capabilities, examples,
recommendations, or other new claims.

## 2. Inputs

The prompt supplies the author's patch, edited target, and ledger output path. Read the
patch and target in fresh context, locating evidence in repository code and tests.
Do not read the author's instructions, reasoning, or `pilot_files.md`, including copies
in other checkouts. An empty patch is valid and still requires a ledger.

## 3. Investigation

- Verify introduced or strengthened factual claims, including neutral descriptions.
  Check defaults, backend support, required setup, permissions, and error behavior.
- Verify the connection between each capability and its stated benefit. A convenience
  visible in the call chain can support a claim about work handled for callers; a speed
  comparison needs relevant measurements. Match the evidence to the actual claim.
- Before cutting a plausible, useful claim, search its symbol and relevant callers again.
  If evidence remains unclear, remove the claim and flag it for human review.

## 4. Main work

- Keep supported claims and their positive framing. Confident wording such as "handles
  these steps for the caller" is appropriate when the implementation demonstrates it.
- Narrow an overbroad claim to its supported scope or remove it. Keep required conditions
  with the claim. Do not replace an unsupported benefit with a different benefit you
  discovered, or hide missing evidence behind "generally" or "typically".
- Remove empty praise such as "powerful" when it contributes no specific information.
  Correct unsupported comparisons, guarantees, and implications of broader support.
  Preserve the strongest supported meaning with the smallest necessary wording change.

## 5. Quality check

Every intervention must fix a false claim, an unsupported implication, or promotional
wording. Preserve explanations of supported benefits; favorable wording and length alone
are no reason to cut. Do not perform a general style, brevity, or coverage rewrite.
Confirm you introduced no new claims, kept edits in scope, and based the ledger on the
final surviving wording. Finding nothing to cut is a valid result.

## 6. Output

Leave verified docstrings in the target and write the ledger to the requested path,
usually `docstring_pilot_ledger.md`. Use these Markdown sections:

- **Verdict:** `changed` when a diff remains from the original; `no-change-needed` when
  the author proposed nothing or you reverted the entire patch.
- **Claim ledger — target path:** a table with columns `Claim introduced or strengthened`,
  `Where`, and `Evidence`. Give every surviving benefit claim a row with the docstring's
  symbol and specific repository paths and symbols or tests supporting it.
- **Critic pass:** record what you narrowed or removed and why. If nothing needed cutting,
  say so and explain the evidence that supported keeping the positive framing.
- **Reviewer should check:** list plausible claims removed because their evidence remained
  unclear. Omit this section when there are none.

A no-change result still needs the ledger, with an empty claim table and an explanation
of what happened. Use `##` headings for the sections above. Print a brief summary with
the target, verdict, and number of claims cut; the workflow publishes the ledger itself.
