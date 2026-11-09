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
`extract_agent_context` stage) extracts the traces past the session's extraction
watermark, each paired with the *turn* it belongs to: the user request and the agent's
final answer it was made for. The plugin runs `improve()` at session end and on a
cooldown, so the pending traces usually span many turns; they are all extracted in one
pass, labelled per turn, so the number of LLM calls follows the amount of trace text, not
the number of turns. A backlog of more than 400 pending traces (improves that failed or
were off for a while) is walked oldest first in windows of 400, the watermark sealed after
each, so a failure in a later window never repeats an earlier one and no trace is skipped.
The stage no longer requires `AUTO_FEEDBACK`.

**Traces are paired to turns by time, not position**
(`cognee/infrastructure/session/coding_turns.py`). The plugin's Stop hook writes the
answer asynchronously, so the first traces of the next turn can land before it, and a
subagent left in the background keeps writing traces after the turn is over; the order
rows arrive in says nothing reliable. Instead every entry carries the moment of its event:
`QAEntry.time` (the turn's end) and `QAEntry.prompt_time` (its start), `TraceEntry.time`
(when the tool call finished), all stamped by the plugin on one clock, and by the server
at store time when a client sends none. A turn starts at its `prompt_time`, else at the
previous answer's time, and ends at its answer's time. A trace belongs to the latest turn
that had started when it happened (so a trace between one turn's answer and the next
turn's prompt is the earlier turn's late work), else to the first turn; a trace without a
time (a row from before the field existed) belongs to the newest turn, as every trace did
before. A trace newer than the last answer has no answer yet: it and everything stored
after it wait for the next `improve()`, which keeps the watermark a plain count.

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
into trace chunks, every chunk repeating every section; a turn's header and FINAL MESSAGE
block count only in the chunks that show the turn, so the cut follows the trace text, not
the number of turns), in this order:

1. `EXISTING LESSONS`: the session's 40 newest candidates.
2. `LESSONS ALREADY IN MEMORY`: up to 8 stored lessons, 1,500 chars in all, from a vector
   search over a query built from the part's file paths, first error lines, commands and
   patterns; this session's lessons first, then the dataset's, without duplicates. The
   search has a 30 s timeout and is fail-open: when it fails, the section is left out.
3. `TRACES (part k of N, phase <phase>)`: a `-- turn k: <request> --` line opens each
   turn's traces, then one line per trace, `- <tool> <input> [<status>] | feedback | error | output`,
   with the tool input (repo-relative file path, the command, the search pattern, or the
   old/new snippets of an edit), errors and outputs cut at 600 chars, secrets redacted.
4. `REPEATED ERRORS AND WHAT WORKED`: failed traces grouped by tool and normalized first
   error line (at least two failures, or one that a later call fixed), each paired with
   the first later successful call that changed the same file or reran the same program
   on the same target (`python broken.py` again, not `python --version`). The section
   shows the first 10 groups; the deterministic pitfall candidates cover every fixed group.
5. `FINAL MESSAGE`: for each turn the input covers, numbered as in `TRACES`, the request
   (1,200 chars) and the final answer (6,000 chars).

Parts run in order so each sees the candidates of the earlier ones; up to 10 candidates per
call, each naming the turn it was learned in (`CodingAgentContextExtraction`; a number
outside the input's turns counts as its newest). Every stored candidate links that turn's
Q&A row (`SessionContextEntry.source_qa_ids`), which is how the distillation finds the
request and answer to judge it against. After the last part, each fixed error group stores
one deterministic `failure_lessons` candidate (confidence 0.9) that opens with its pitfall
key, linked to its failing trace's turn, so it survives the curator as a pitfall and
replaces its predecessor:

```
[state-key: pitfall:Edit:no-changes-to-make:py] Edit src/app.py 'foo' -> 'foo' failed with
"no changes to make: ..." → Bash python rewrite.py src/app.py
```

**Once per trace.** The extraction holds a per-(user, session) lock, reads the watermark
inside it and advances it past each window it extracted. A second trigger on the same
traces, sequential or concurrent, makes no LLM call. An LLM failure propagates before the
window's watermark moves, so the window is retried by the next `improve()`.

**Coding prompts and typed lessons.** The `distill_sessions` stage uses the prompts under
`cognee/infrastructure/llm/prompts/coding/` (the default files are untouched) and typed
models. The candidates are batched six at a time in the order of their turns, and every
batch opens with the request and final answer of the turns its candidates were learned in
(a candidate that links no turn is judged against the newest one), so the call count is
the same as with one pinned turn while each candidate is judged against its own request
and answer. Each accepted lesson takes the date, commit and branch of its own turn (the
newest of its members' turns). The writer additionally gets `PROPOSED STATE KEY`, a glossary recased from
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
- `(seen in N sessions)` appears on pitfalls only: the number of distinct sessions that
  stored the key in this project. Each row carries that list (`seen_sessions`), so the count
  survives the replacement of older rows and re-distilling a session keeps its number.

The document's `Data` row carries the same facts in `external_metadata`, which the dataset
listing exposes: `lesson_type`, `state_key`, `session_id`, `session_date`, `git_commit`,
`git_branch`, `files` (file names the statement mentions, in their original casing),
`project_node_sets` (the lesson's project scope: its node sets without the distillation's
own `session_learnings` ones, empty when the session pinned none; the listing does not
expose `Data.node_set`, so this is how the plugin filters lessons by repository),
`seen_sessions` and `seen_in_sessions`. The date, commit and branch come from the `context` of the Q&A
entry of the lesson's own turn when the plugin sends it as JSON (`{"cwd", "transcript_path", "date", "start_commit",
"end_commit", "branch"}`); without it the lesson is stored with empty values.

**Replacement.** A state key names one lesson *per project*: the scope is the dataset's
coding lessons (rows whose metadata carries `lesson_type`, never a user's own documents)
with the same project node sets, meaning the row's node sets minus the distillation's own
`session_learnings` ones. The plugin pins one node set per repository, so two repositories
in one dataset keep separate `test-baseline` lessons, and a `node_name`-scoped recall still
finds its project's pitfalls. Once the new lessons are in the graph, the key's lessons in
that scope are reduced to the one with the newest `session_date`: normally the new lesson,
which deletes the older rows; when a later session already stored the key, the lesson of
the late-distilled older session is the one deleted. Without a date on both sides the
lesson stored last wins. Within one run, two accepted lessons with the same key store the
later one only (the candidates are curated in turn order). The replacement also runs when only the step after cognify failed
(`AfterRunHookError`), which is raised afterwards. Deleting needs delete permission on the
dataset, so coding mode requires it next to write permission before anything runs. A
re-distilled identical lesson has the same row id and is kept.

## Writer rules enforced in code

`already_known` never applies to a keyed lesson: a state lesson replaces the earlier one
with its key by definition, and a pitfall seen again is the recurrence its `(seen in N
sessions)` count exists for. When the writer still rejects one as known, the curator's
working statement is stored under the key, as the state lesson or as the pitfall its
repeated-error candidate names. The evidence rules above settle the lesson type after the
writer.

## Not in this mode

Trace persistence (`persist_agent_traces`) is unchanged; disable it with
`IMPROVE_STAGES_DISABLED=persist_agent_traces` if the cognified trace documents are not
wanted alongside the lessons. Phase-specific extraction prompts, post-distillation
cleanup passes and automatic retirement of lessons without a state key are out of scope.
