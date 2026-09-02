---
name: docstring-author
description: Use when writing or improving Python docstrings on one of the ten files listed in pilot_files.md — pass 1 of the docstring pilot. Reads the target and its surrounding code and writes module, class, and public-function docstrings. Writes no claim ledger; the separate docstring-critic skill verifies the result. Invoked by .github/workflows/docstring_pilot.yml.
---

# Docstring author — pass 1 of the docstring pilot

Docstrings are product surface. IDE tooltips, `help()`, generated API docs, and coding
agents all show them before anyone reads the implementation. That gives them a specific
failure mode: an agent can make a docstring sound better than the code proves, and
nothing in CI catches it. Prose that overstates the design is worse than prose that is
missing, because a reader has no way to tell which sentences were earned.

## Your place in the run

You are **pass 1**. The separate `docstring-critic` skill runs after you, in a fresh
invocation, and receives your diff as a patch file. It does not see your reasoning or
these instructions — deliberately, so that it grades your claims rather than your
compliance. It will cut anything it cannot find evidence for, and it writes the claim
ledger that lands in the pull request body.

The practical consequence: **write toward what the code proves, not toward what would
read well.** A sentence you cannot support is a sentence that will not survive, and the
run is wasted either way.

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

## Author checklist

Work through these in order.

- [ ] Read your target's row in `pilot_files.md` (next to this file) for why it was
      selected. It names what makes this file a high-value docstring surface, which is
      what the docstrings should serve. The critic does not get this file, so do not lean
      on it as evidence — a claim has to stand on the code.
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
      pilot files were chosen precisely to test that restraint, and the critic will
      revert padding of prose that was already correct.

## When you are done

**Do not write the claim ledger.** The critic writes it, from your diff rather than from
your account of your diff. Print a two-line summary of what you touched and stop.
