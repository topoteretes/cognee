---
name: docstring-pilot
description: Use when improving Python docstrings on one of the ten files listed in pilot_files.md — write stronger module/class/function docstrings, then critique them against the evidence in the repo and record a claim ledger. Invoked by .github/workflows/docstring_pilot.yml; also usable by hand on the same ten files.
---

# Evidence-led docstring pilot

Docstrings are product surface. IDE tooltips, `help()`, generated API docs, and coding
agents all show them before anyone reads the implementation. That gives them a specific
failure mode: an agent can make a docstring sound better than the code proves, and
nothing in CI catches it. Prose that overstates the design is worse than prose that is
missing, because a reader has no way to tell which sentences were earned.

This skill runs both passes in one session — **author first, then critic** — and makes
the critic's work visible in a claim ledger so a maintainer can check the reasoning
rather than re-derive it.

## Scope

Read `pilot_files.md` (next to this file) for the ten selected files and why each one
was chosen.

Hard rules, in priority order over anything else in this document:

1. **One file per run.** Edit only the file named as the target in the prompt.
2. **The target must be on the list in `pilot_files.md`.** If it is not, change nothing
   and say so. Never add a file to that list.
3. **Docstrings only.** Do not change code, imports, type annotations, comments,
   formatting, or tests. `tools/check_docstring_pilot_diff.py` compares the file's AST
   before and after with docstrings stripped and fails the run on any other change, so
   an edit outside a docstring is a wasted run, not a merged one.
4. **No commits, no branches, no `git` writes.** The workflow owns version control.
5. **Leaving the file unchanged is a valid, sometimes correct outcome.** Say why.

## The four jobs of a docstring here

A docstring on these files should do as many of these as the code supports:

1. **Purpose** — what this exists for, in terms a reader who has not seen the module
   can act on. Not a restatement of the name.
2. **Design context** — the surrounding facts a reader needs and cannot see from the
   signature: what calls this, what it delegates to, which config or env var changes its
   behavior, what it deliberately does not do.
3. **Intended use** — when to reach for this rather than a neighbouring API, and the
   constraints that decide it (required permissions, async-only, backend support,
   ordering requirements).
4. **Genuine strengths** — a real, load-bearing property of the design, stated only when
   the repo proves it. This is the job that most often goes wrong; see the ledger rules.

## Author checklist

Work through these in order.

- [ ] Read the whole target file. Not a grep window — the docstrings have to be true of
      the code below them.
- [ ] Read enough surrounding code to establish system context: the callers (grep the
      symbol across `cognee/`), the interfaces or base classes involved, the concrete
      implementations of anything abstract, and the tests that exercise it.
- [ ] Check the repo's own guidance for what this file is *for*: `CLAUDE.md`,
      `AGENTS.md`, and `docs/` describe several of these files by name. Alignment with
      that guidance is evidence; contradiction with it is a signal you have the purpose
      wrong, not a licence to correct the guidance (editing `CLAUDE.md` or `AGENTS.md`
      is out of scope for this pilot).
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

## Critic checklist

Now re-read every docstring you wrote or changed, adversarially. The question is not
"is this nice prose" but **"which sentence here would I have to delete if a reviewer
asked me to prove it?"** Delete those sentences now.

- [ ] **Unsupported praise.** Any favorable framing that the code, tests, or docs do not
      demonstrate. Rewrite as the plain fact, or cut it.
- [ ] **Trigger words without evidence.** `robust`, `flexible`, `composable`,
      `efficient`, `performant`, `scalable`, `powerful`, `seamless`, `optimized`,
      `battle-tested`, `elegant`, `designed to`, `built for`, `ensures`, `guarantees`,
      `automatically handles`, `simply`, `easily`. Each occurrence you introduced needs a
      ledger row or must be rewritten factually. `"efficient"` becomes
      `"issues one graph query per batch"` — or goes.
- [ ] **Stale claims.** Anything true of an older version: a parameter that no longer
      exists, a default that changed, a backend that was renamed, a deprecated function
      described as current. Verify defaults against the signature, not against memory.
- [ ] **Misleading implications.** Sentences that are literally true but will be read as
      a stronger promise: implying a check is enforced when it is best-effort, implying a
      feature works on all backends when the support matrix says otherwise, implying
      something is the recommended path when it is one of several.
- [ ] **Tautology.** `"""Knowledge graph."""` on `class KnowledgeGraph` adds nothing.
      Either say what it is for and what fills it, or leave the field to the type name.
- [ ] **Verbosity.** Cut anything a reader of the signature already knows. Cut restated
      parameter names. Cut examples that duplicate an adjacent example. A shorter
      docstring that survives review beats a longer one that does not.
- [ ] **Restraint.** Where the existing docstring was already accurate and sufficient,
      revert your change and record it as left unchanged. Several pilot files were
      chosen precisely to test this; a run that leaves a good file alone and explains
      why is a successful run.
- [ ] **Scope.** Confirm the diff is docstrings only, in the one target file.

## The claim ledger

Every run writes a ledger to the path given in the prompt (the workflow passes
`docstring_pilot_ledger.md`). It goes into the pull request body, so it is what a
maintainer reads first. Write it in this shape:

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
- **Left unchanged** the `add()` docstring — already states the loader, storage, and
  permission behavior accurately; nothing to add without padding.
```

Rules for the ledger:

- **Every favorable claim you introduced or strengthened gets a row.** Neutral factual
  description does not need one; "this is good in way X" always does.
- **Evidence must be something a reviewer can open**: a symbol, file, or test in this
  repo; a statement in `CLAUDE.md`, `AGENTS.md`, or `docs/`; or a documented product
  behavior. "It is obvious from the design" is not evidence. Neither is another
  docstring — docstrings are the thing under review.
- **Weak evidence means rewrite or drop, not hedge.** If the only support is one code
  path that looks like it would generalize, state that code path instead of the
  generalization. Do not write "generally" or "typically" to make an unproven claim
  survive.
- **The critic section must be non-empty.** If you cut nothing and left everything
  unchanged, say that explicitly and say why — an empty critic section reads as a pass
  that did not happen.
- **`no-change-needed` still needs a ledger file**, with an empty claim table and a
  critic section explaining what you considered adding and why it was not worth it.

## When you are done

Print a short summary: the target file, the verdict, how many docstrings you touched,
and the count of claims cut by the critic pass. The workflow reads the ledger file, not
this summary, so keep it brief.
