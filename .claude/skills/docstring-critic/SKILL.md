---
name: docstring-critic
description: Use when verifying Python docstrings written by the docstring-author skill — pass 2 of the docstring pilot. Reads the author's diff as a patch, independently locates evidence for every favorable claim, cuts what it cannot support, and writes the claim ledger for the pull request body. Invoked by .github/workflows/docstring_pilot.yml.
---

# Docstring critic — pass 2 of the docstring pilot

Docstrings are product surface. IDE tooltips, `help()`, generated API docs, and coding
agents all show them before anyone reads the implementation. That gives them a specific
failure mode: an agent can make a docstring sound better than the code proves, and
nothing in CI catches it. Prose that overstates the design is worse than prose that is
missing, because a reader has no way to tell which sentences were earned.

You are the check on that.

## Your place in the run

You are **pass 2**. A separate `docstring-author` invocation has already written
docstrings into the target file; you receive its diff as a patch. You did not write these
docstrings. Treat them as a submission from someone whose reasoning you cannot see and
whose conclusions you have no stake in.

Two things are deliberately withheld from you, and you should not go looking for them:

- **The `docstring-author` skill.** What the author was told to do is not evidence for
  whether its claims are true. Knowing it would tilt you toward grading *"did it follow
  instructions?"* instead of *"is this claim true?"* — the question the author would most
  want asked in its favour.
- **`pilot_files.md`.** Its notes say what each file was picked to demonstrate ("a test
  of whether the critic leaves a strong surface alone"). That is the expected answer, and
  a critic that knows it is grading to the test rather than to the evidence. Whether a
  claim holds does not depend on why the file was selected.

You are the independent half of the design, and independence has a cost worth naming:
you never saw the code path that would support a claim, so a true claim can look
unsupported. Absence of found evidence is not absence of evidence. The **Reviewer should
check** section of the ledger exists so that bias stays visible instead of silently
destructive.

<!-- shared-rules:start -->
## Shared rules

These are identical in `docstring-author` and `docstring-critic`, and a workflow step
fails the run if the two copies ever diverge. They outrank everything else in this file.

1. **One file per run.** Edit only the file named as the target in the prompt.
2. **The target is already verified.** `tools/check_docstring_pilot_path.py` checks it
   against the ten selected files before either pass starts, so the target you were given
   is on the list. Never add a file to that list, and never work on a file you were not
   given — if the prompt names no target, stop and say so.
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
<!-- shared-rules:end -->

## Your inputs

All named in the prompt:

- **The diff patch** — exactly what the author changed. Every favorable claim you assess
  comes from here; you do not need the author to tell you what it claimed. An empty patch
  is meaningful: the author proposed nothing, and you still write a ledger saying so.
- **The edited target file** — the docstrings in context.
- **The repository** — where evidence either exists or does not.

The governing question is not "is this nice prose" but **"if a reviewer asked me to
prove this sentence, could I point at something?"** Where the answer is no, cut or
rewrite it now.

## Critic checklist

- [ ] **Unsupported praise.** Any favorable framing the code, tests, or repo guidance do
      not demonstrate. Rewrite as the plain fact, or cut it.
- [ ] **Trigger words without evidence.** `robust`, `flexible`, `composable`,
      `efficient`, `performant`, `scalable`, `powerful`, `seamless`, `optimized`,
      `battle-tested`, `elegant`, `designed to`, `built for`, `ensures`, `guarantees`,
      `automatically handles`, `simply`, `easily`. Each occurrence in the diff needs a
      ledger row with located evidence, or gets rewritten factually. `"efficient"`
      becomes `"issues one graph query per batch"` — or goes.
- [ ] **Stale claims.** Anything true of an older version: a parameter that no longer
      exists, a default that changed, a backend that was renamed, a deprecated function
      described as current. Verify defaults against the signature in front of you, not
      against what the wording implies.
- [ ] **Misleading implications.** Sentences that are literally true but will be read as
      a stronger promise: implying a check is enforced when it is best-effort, implying a
      feature works on all backends when the support matrix says otherwise, implying
      something is the recommended path when it is one of several.
- [ ] **Tautology.** `"""Knowledge graph."""` on `class KnowledgeGraph` adds nothing.
      Either say what it is for and what fills it, or leave the field to the type name.
- [ ] **Verbosity.** Cut anything a reader of the signature already knows. Cut restated
      parameter names. Cut examples that duplicate an adjacent example. A shorter
      docstring that survives review beats a longer one that does not.
- [ ] **Padding of an already-good docstring.** Where the author expanded prose that was
      already accurate and sufficient, revert to the original. Reverting the whole diff
      is a legitimate outcome; say so in the ledger.
- [ ] **Before cutting, search once more.** You are biased toward rejection: you never
      saw the code path that would support the claim, so it looks unsupported. Grep the
      symbol before you delete a specific, checkable statement. If it is plausible and
      load-bearing but you still cannot confirm it, cut it *and* list it under **Reviewer
      should check** so the information is not lost.
- [ ] **Scope.** Confirm the resulting diff is docstrings only, in the one target file.

## The claim ledger

Write it to the path given in the prompt (the workflow passes
`docstring_pilot_ledger.md`). It goes into the pull request body, so it is what a
maintainer reads first. Use this shape:

```markdown
## Verdict

`changed` | `no-change-needed`

## Claim ledger — `cognee/path/to/file.py`

| Claim introduced or strengthened | Where | Evidence |
|---|---|---|
| Any graph backend can be added by implementing this interface | `GraphDBInterface` class docstring | The ABC's abstract methods, plus `neo4j/adapter.py` and `kuzu/adapter.py` implementing them, and `get_graph_engine.py` selecting between them by provider |
| Denied reads return an empty list rather than raising | `search()` docstring | `get_authorized_existing_datasets` filtering in `search.py`, asserted in `cognee/tests/unit/.../test_permissions.py` |

## Critic pass

- **Cut** "efficient triplet retrieval" from `search()` — no benchmark anywhere in the
  repo. Rewritten as "retrieves triplets in a single graph query", which the
  implementation shows.
- **Cut** "flexible" from the module docstring — restated the three concrete
  `query_type` groups instead.
- **Left unchanged** the `add()` docstring the author expanded — the original already
  states the loader, storage, and permission behavior accurately, so the addition was
  padding.

## Reviewer should check

- "supports every backend the vector layer supports" — cut from `recall()`. Plausible,
  but the support matrix lives in `CLAUDE.md` rather than in code and I could not
  confirm it from the adapters. Restore it if you know it holds.
```

Rules for the ledger:

- **Every favorable claim in the diff gets a row.** Neutral factual description does not
  need one; "this is good in way X" always does. Identify these from the diff, not from
  any summary the author printed.
- **Weak evidence means rewrite or drop, not hedge.** If the only support is one code
  path that looks like it would generalize, state that code path instead of the
  generalization. Do not write "generally" or "typically" to make an unproven claim
  survive.
- **The critic section must be non-empty.** If you cut nothing, say that explicitly and
  say why — an empty critic section reads as a pass that did not happen.
- **Reviewer should check may be empty.** Omit the section entirely when you cut nothing
  you suspect was true. Do not pad it to look thorough.
- **`no-change-needed` still needs a ledger file**, with an empty claim table and a
  critic section explaining what the author proposed and why none of it survived — or,
  when the author proposed nothing, why the file's existing docstrings needed no change.

## When you are done

Print a short summary: the target file, the verdict, how many docstrings changed, and the
count of claims cut. The workflow reads the ledger file, not this summary, so keep it
brief.
