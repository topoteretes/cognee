---
name: docstring-pilot
description: Use when improving Python docstrings on one of the ten files listed in pilot_files.md — pass 1 writes stronger module/class/function docstrings, pass 2 independently verifies them against the repo and writes a claim ledger. Invoked twice by .github/workflows/docstring_pilot.yml; also usable by hand on the same ten files.
---

# Evidence-led docstring pilot

Docstrings are product surface. IDE tooltips, `help()`, generated API docs, and coding
agents all show them before anyone reads the implementation. That gives them a specific
failure mode: an agent can make a docstring sound better than the code proves, and
nothing in CI catches it. Prose that overstates the design is worse than prose that is
missing, because a reader has no way to tell which sentences were earned.

## How this skill runs

**Two passes, two separate invocations, one target file.** The prompt tells you which
pass you are. Follow the shared rules plus your own pass's section, and ignore the other
pass's section — it is not addressed to you.

| | Reads | Writes |
|---|---|---|
| **Pass 1 — author** | the target file and the surrounding code | docstrings in the target file |
| **Pass 2 — critic** | the diff patch, the edited file, and the repo | cuts and rewrites in the target file, plus the claim ledger |

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

Both passes obey these, and they take priority over anything else in this document.

Read `pilot_files.md` (next to this file) for the ten selected files and why each was
chosen.

1. **One file per run.** Edit only the file named as the target in the prompt.
2. **The target must be on the list in `pilot_files.md`.** If it is not, change nothing
   and say so. Never add a file to that list.
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

## Pass 1 — author checklist

Work through these in order.

- [ ] Read the whole target file. Not a grep window — the docstrings have to be true of
      the code below them.
- [ ] Read enough surrounding code to establish system context: the callers (grep the
      symbol across `cognee/`), the interfaces or base classes involved, the concrete
      implementations of anything abstract, and the tests that exercise it.
- [ ] Check the repo's own guidance for what this file is *for*: `CLAUDE.md` and
      `AGENTS.md` describe several of these files by name. Alignment with that guidance
      is evidence; contradiction with it is a signal you have the purpose wrong, not a
      licence to correct the guidance (editing `CLAUDE.md` or `AGENTS.md` is out of
      scope for this pilot). Treat a guidance claim you cannot find in the code as
      unverified — `CLAUDE.md` currently points at `docs/recall-vs-search.md`, which
      does not exist.
- [ ] Add or improve the **module** docstring when the file's role is not obvious from
      its path. Say what the module is responsible for and where it sits in the flow.
- [ ] Add or improve **class** docstrings for public classes: what the type represents,
      what invariants hold, and — for an ABC or base class — what an implementer must
      guarantee.
- [ ] Add or improve **public function and method** docstrings: purpose, the parameters
      whose meaning is not obvious from name and type, what is returned, and what is
      raised. Document the arguments a caller has to make a decision about; do not
      transcribe all thirty of them for the sake of coverage.
- [ ] For an enum, document the *choice*: what each member selects and when a caller
      would pick it over its neighbours. An enum whose members are only restated as
      prose has not been documented.
- [ ] Prefer a concrete, checkable statement over a general one. `"Returns [] when the
      user lacks read permission, rather than raising"` is worth five sentences of
      description.
- [ ] Match the file's existing docstring style — Google-style `Args:`/`Returns:`
      sections where the file already uses them, reST/`` `` ``-quoted references where it
      does. Do not restyle a file that is internally consistent.
- [ ] Keep lines within the repo's 100-character limit.
- [ ] Leave private helpers (leading underscore) alone unless their docstring is
      actively wrong.
- [ ] Where the existing docstring is already accurate and sufficient, leave it. Several
      pilot files were chosen precisely to test that restraint.

**Do not write the ledger.** The critic writes it, from your diff rather than from your
account of your diff. Finish by printing a two-line summary of what you touched.

## Pass 2 — critic checklist

You did not write these docstrings. Treat them as a submission from someone whose
reasoning you cannot see and whose conclusions you have no stake in.

Your inputs, all named in the prompt:

- **The diff patch** — exactly what the author changed. Every favorable claim you assess
  comes from here; you do not need the author to tell you what it claimed.
- **The edited target file** — the docstrings in context.
- **The repository** — where evidence either exists or does not.

The governing question is not "is this nice prose" but **"if a reviewer asked me to
prove this sentence, could I point at something?"** Where the answer is no, cut or
rewrite it now.

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

The critic writes it to the path given in the prompt (the workflow passes
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
  the author's summary.
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
