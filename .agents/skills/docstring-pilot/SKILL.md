---
name: docstring-pilot
description: Use when improving Python docstrings on one of the ten files listed in pilot_files.md — pass 1 (author.md) writes stronger module/class/function docstrings, pass 2 (critic.md) independently verifies them against the repo and writes a claim ledger. Invoked twice by .github/workflows/docstring_pilot.yml; also usable by hand on the same ten files.
---

# Evidence-led docstring pilot

Docstrings are product surface. IDE tooltips, `help()`, generated API docs, and coding
agents all show them before anyone reads the implementation. That gives them a specific
failure mode: an agent can make a docstring sound better than the code proves, and
nothing in CI catches it. Prose that overstates the design is worse than prose that is
missing, because a reader has no way to tell which sentences were earned.

## How this skill runs

**Two passes, two separate invocations, one target file.** This file holds what both
passes need. Each pass's checklist lives in its own file, and the prompt names the one
addressed to you:

| | Instructions | Inputs | Writes |
|---|---|---|---|
| **Pass 1 — author** | this file + `author.md` + `pilot_files.md` | the target file and the surrounding code | docstrings in the target file |
| **Pass 2 — critic** | this file + `critic.md` | the diff patch, the edited file, and the repo | cuts and rewrites in the target file, plus the claim ledger |

**Read only the files listed for your own pass.** Two things are deliberately withheld
from the critic:

- **`author.md`** — knowing what the author was told to do invites grading *"did it
  follow instructions?"* instead of *"is this claim true?"*, which is the question the
  author would most want asked in its favour.
- **`pilot_files.md`** — its reasons say what each file was picked to demonstrate ("a
  test of whether the critic leaves a strong surface alone"). That is the expected
  answer, and a critic that knows it is grading to the test rather than to the evidence.
  Whether a claim holds does not depend on why the file was selected.

The critic runs in **fresh context** and never sees the author's reasoning — only what
the author actually produced. That is the point of splitting them. An author asked to
critique itself remembers which of its claims it grounded in code and which it inferred
while reading, and that memory is exactly what can be confabulated; it will tend to
accept its own unsupported wording. A critic that has to re-locate every piece of
evidence from scratch has nothing to protect.

The known cost of that independence is the opposite error: the critic may cut a claim
that is true but whose evidence it failed to find. Absence of found evidence is not
absence of evidence. The **Reviewer should check** section of the ledger exists to keep
that visible rather than silently destructive.

## Shared rules

Both passes obey these, and they take priority over anything in the pass files.

1. **One file per run.** Edit only the file named as the target in the prompt.
2. **The target is already verified.** `tools/check_docstring_pilot_path.py` checks it
   against the ten files in `pilot_files.md` before either pass starts, so the target you
   were given is on the list. Never add a file to that list, and never work on a file you
   were not given — if the prompt names no target, stop and say so.
3. **Docstrings only.** Do not change code, imports, type annotations, comments,
   formatting, or tests. `tools/check_docstring_pilot_diff.py` runs after the critic and
   compares the file's AST before and after with docstrings stripped, so an edit outside
   a docstring fails the run for both passes rather than reaching a pull request.
4. **No commits, no branches, no `git` writes.** The workflow owns version control.
5. **Leaving the file unchanged is a valid, sometimes correct outcome.** Say why.
6. **Evidence means something a reviewer can open in this repository**: a symbol, a file,
   a test, or a statement in `CLAUDE.md` or `AGENTS.md`. "It is obvious from the design"
   is not evidence. Neither is another docstring — docstrings are the thing under review.
7. **The published documentation is not available to either pass.** `docs/` in this repo
   covers only Docker, Colima, Ollama, and Coolify setup; the product documentation lives
   in `topoteretes/cognee-docs` and is published at docs.cognee.ai, neither of which this
   run can read. A claim whose only support is "the docs say so" cannot be verified here.

## The four jobs of a docstring here

A docstring on these files should do as many of these as the code supports. The author
writes toward them; the critic judges against them.

1. **Purpose** — what this exists for, in terms a reader who has not seen the module
   can act on. Not a restatement of the name.
2. **Design context** — the surrounding facts a reader needs and cannot see from the
   signature: what calls this, what it delegates to, which config or env var changes its
   behavior, what it deliberately does not do.
3. **Intended use** — when to reach for this rather than a neighbouring API, and the
   constraints that decide it (required permissions, async-only, backend support,
   ordering requirements).
4. **Genuine strengths** — a real, load-bearing property of the design, stated only when
   the repo proves it. This is the job that most often goes wrong, and the one the critic
   scrutinises hardest.

## When you are done

Print a short summary: the target file, the verdict, how many docstrings changed, and —
if you are the critic — the count of claims cut. The workflow reads the ledger file, not
this summary, so keep it brief.
