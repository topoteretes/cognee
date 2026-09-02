# Pass 1 — author checklist

You are the author pass of the docstring pilot. Read `SKILL.md` first for the shared
rules, the ten selected files, and the four jobs a docstring here should do; those rules
outrank anything below. Then work through this checklist in order.

A separate critic pass runs after you, in fresh context, with your diff. It will cut any
claim it cannot find evidence for. Write toward what the code proves, not toward what
would read well — a sentence you cannot support is a sentence that will not survive, and
the run is wasted either way.

- [ ] Read your target's row in `pilot_files.md` for why it was selected. It names what
      makes this file a high-value docstring surface, which is what the docstrings should
      serve. The critic does not get this file, so do not lean on it as evidence — a
      claim has to stand on the code.
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

**Do not write the ledger.** The critic writes it, from your diff rather than from your
account of your diff. Finish by printing a two-line summary of what you touched.
