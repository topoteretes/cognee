# Pass 2 — critic checklist and claim ledger

You are the critic pass of the docstring pilot. Read `SKILL.md` first for the shared
rules, the ten selected files, and the four jobs a docstring here should do; those rules
outrank anything below.

You did not write these docstrings. Treat them as a submission from someone whose
reasoning you cannot see and whose conclusions you have no stake in. Do not read
`author.md` — what the author was told to do is not evidence for whether its claims are
true, and knowing it would tilt you toward grading compliance instead of accuracy.

Your inputs, all named in the prompt:

- **The diff patch** — exactly what the author changed. Every favorable claim you assess
  comes from here; you do not need the author to tell you what it claimed.
- **The edited target file** — the docstrings in context.
- **The repository** — where evidence either exists or does not.

The governing question is not "is this nice prose" but **"if a reviewer asked me to
prove this sentence, could I point at something?"** Where the answer is no, cut or
rewrite it now.

## Checklist

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
