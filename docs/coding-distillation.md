# Coding distillation (`DISTILLATION_MODE=coding`)

An opt-in mode of `improve()` for coding agents, built for the Claude Code plugin
(SDK-1335; the plugin side is SDK-1336). Off by default. With it unset, extraction,
prompts and lesson documents are exactly the default ones.

```bash
DISTILLATION_MODE=coding   # no IMPROVE_ prefix
```

## Why

The default agent-context extraction runs while the agent works: every ten traces, over a
window of at most forty, with trace lines that omit tool inputs (which file was edited,
which command ran) and nothing that says how the task ended. In coding sessions that
yields vague lessons, failures recorded without their fix, lowercased symbol names copied
from the graph glossary, and outdated project state accumulating across sessions.

## What changes

**Nothing happens mid-session.** The per-trace hooks are off. `improve()` (the
`extract_agent_context` stage) extracts one *unit of work* at a time: the traces past the
session's extraction watermark (the newest 400 at most), plus the latest Q&A entry as the
user request and the agent's final answer. The stage no longer requires `AUTO_FEEDBACK`.

**Traces are split into parts.** Each trace is classified without an LLM
(`cognee/infrastructure/session/trace_phases.py`): *explore* for read and search tools,
read-only shell and git commands and inline Python that only reads; *change* for edit and
write tools, shell redirects, file-modifying and state-changing git commands and inline
Python that writes a file (targets under `/dev/null`, `/tmp`, `/var/folders` or a
`$variable` are scratch and do not count); *verify* for test runners and build tools,
`python -m` and script runs, inline Python that imports a third-party package or runs a
subprocess, and any program not listed (the project's own CLI). A chained command is
split on unquoted `&&`, `||`, `;`, `|` and newlines and takes its strongest segment
(change > verify > explore); a read command that only filters piped output does not
count. Setup and control commands and other tools (TodoWrite, Task, MCP) inherit the
previous trace's phase. Consecutive traces of one phase form blocks, merged one step at a
time (smallest block into its smaller neighbour, taking the larger side's phase) until
every part has at least 5 traces and there are at most 4 parts (up to 120 traces, else
`ceil(n / 30)`). A unit under 5 traces is one part.

**The LLM sees the whole record.** One input per part (a part over 120,000 chars is cut
into trace chunks, every chunk repeating every section), in this order:

1. `EXISTING LESSONS`: the session's 40 newest candidates.
2. `LESSONS ALREADY IN MEMORY`: up to 8 stored lessons, 1,500 chars in all, from a vector
   search over a query built from the part's file paths, first error lines, commands and
   patterns; this session's lessons first, then the dataset's, without duplicates. The
   search has a 30 s timeout and is fail-open: when it fails, the section is left out.
3. `TRACES (part k of N, phase <phase>)`: one line per trace, `- <tool> <input> [<status>] | feedback | error | output`,
   with the tool input (repo-relative file path, the command, the search pattern, or the
   old/new snippets of an edit), errors and outputs cut at 600 chars, secrets redacted.
4. `REPEATED ERRORS AND WHAT WORKED`: failed traces grouped by tool and normalized first
   error line (at least two failures, or one that a later call fixed), each paired with
   the first later successful call that changed the same file or reran the same program.
5. `FINAL MESSAGE`: the request (1,200 chars) and the final answer (6,000 chars).

Parts run in order so each sees the candidates of the earlier ones; up to 10 candidates per
call. After the last part, each fixed error group stores one deterministic `failure_lessons`
candidate (confidence 0.9) that opens with its pitfall key, so it survives the curator as a
pitfall and replaces its predecessor:

```
[state-key: pitfall:Edit:no-changes-to-make:py] Edit src/app.py 'foo' -> 'foo' failed with
"no changes to make: ..." → Bash python rewrite.py src/app.py
```

**Once per unit.** The extraction holds a per-(user, session) lock, reads the watermark
inside it and advances it to the trace count it loaded. A second trigger on the same unit,
sequential or concurrent, makes no LLM call. An LLM failure propagates before the watermark
moves, so the unit is retried by the next `improve()`.

**Coding prompts and typed lessons.** The `distill_sessions` stage uses the prompts under
`cognee/infrastructure/llm/prompts/coding/` (the default files are untouched) and typed
models. The curator sees the unit's request and final answer at the top of every batch of
six candidates. The writer additionally gets `PROPOSED STATE KEY`, a glossary recased from
the session's own text (the graph stores `marktaskcommand`; the lesson says
`MarkTaskCommand`) and a compact STYLE line (statements of about 25 words). Evidence rules
then settle the type: a lesson backed by a repeated-error candidate is a `pitfall` with that
candidate's key; a `pitfall` with no failed-trace candidate behind it becomes a `change`;
only `state` and `pitfall` lessons keep a state key.

## The lesson document

```
Lesson-Type: pitfall
State-Key: pitfall:Edit:no-changes-to-make:py
# Session learning (session claude_1234)

Edit on an unchanged string fails with "No changes to make" → rewrite the file with a script. (seen in 2 sessions)
```

- `Lesson-Type` is one of `location`, `pitfall`, `convention`, `constraint`, `change`,
  `state`, `other`.
- `State-Key` appears on state lessons (`test-baseline`, `footprint:<feature>`) and
  pitfalls (`pitfall:<tool>:<error-slug>[:<qualifier>]`, the qualifier being the file
  extension or the shell program).
- `(seen in N sessions)` appears on pitfalls only: one more than the highest count another
  session stored for the key. Re-distilling the same session keeps its number.

The document's `Data` row carries the same facts in `external_metadata`, which the dataset
listing exposes: `lesson_type`, `state_key`, `session_id`, `session_date`, `git_commit`,
`git_branch`, `files` (file names the statement mentions, in their original casing) and
`seen_in_sessions`. The date, commit and branch come from the Q&A entry's `context` when the
plugin sends it as JSON (`{"cwd", "transcript_path", "date", "start_commit",
"end_commit", "branch"}`); without it the lesson is stored with empty values.

**Replacement.** Once the new lessons are in the graph, every older row of the dataset with
one of their state keys is deleted, so a state key always names its newest lesson. A
re-distilled identical lesson has the same row id and is kept.

## Writer rules enforced in code

`already_known` never applies to a state-key update: when the writer still rejects one
as known, the curator's working statement is stored as the state lesson, since a state
lesson replaces the earlier one with its key by definition. The evidence rules above
settle the lesson type after the writer.

## Not in this mode

Trace persistence (`persist_agent_traces`) is unchanged; disable it with
`IMPROVE_STAGES_DISABLED=persist_agent_traces` if the cognified trace documents are not
wanted alongside the lessons. Phase-specific extraction prompts, post-distillation
cleanup passes and automatic retirement of lessons without a state key are out of scope.
