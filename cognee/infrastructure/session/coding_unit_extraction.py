"""Coding distillation: extract one unit of work's lessons at improve() time (SDK-1335).

Active with ``DISTILLATION_MODE=coding``. A *unit of work* is one user request, every
trace the agent produced for it, and the agent's final answer. The mid-session passes
of ``agent_context_extraction`` are off in this mode: they see ten traces at a time
without tool inputs and never see how the task ended, which is what produced vague
lessons and failures recorded without their fix.

An improve may run after one turn or after a whole session of them (the plugin runs it
at session end and on a cooldown), so the pending traces usually belong to several
turns. They are paired to their turns by time (``coding_turns``), and one extraction
pass covers the whole backlog: the number of LLM calls is set by the amount of trace
text, not by the number of turns. Within a call the traces are labelled per turn and
FINAL MESSAGE lists the request and answer of every turn the call covers, and each
lesson names the turn it comes from, so the distillation judges it against the right
request and answer.

Flow, from the ``extract_agent_context`` improve stage:

1. LOAD    the traces past ``TRACE_EXTRACTION_WATERMARK`` and every Q&A row of the
           session, under a per-(user, session) lock so two triggers never extract the
           same traces. Traces newer than the last answer wait for the next improve.
           More than ``TRACE_WINDOW_SIZE`` settled traces are walked oldest first in
           windows of that size, each sealed before the next, so nothing is skipped;
           a window's traces are grouped by the turns that own them.
2. SPLIT   the traces into explore, change and verify parts (``trace_phases``), and a
           part longer than one input into trace chunks.
3. RENDER  one extraction input per chunk, every chunk carrying all five sections:
           EXISTING LESSONS, LESSONS ALREADY IN MEMORY (a vector search over the
           dataset's stored lessons, fail-open), TRACES (with tool inputs, a header per
           turn), REPEATED ERRORS AND WHAT WORKED, FINAL MESSAGE (the request and answer
           of each covered turn). Inputs and outputs are redacted and paths made
           repo-relative.
4. EXTRACT one LLM call per chunk, in order, so each sees the candidates of the earlier
           ones; up to ``MAX_UNIT_LESSONS`` candidates per call go through the shared
           applier (``apply_candidate_updates``), each linked to its turn's Q&A row.
5. RECORD  one deterministic ``failure_lessons`` candidate per fixed error group, keyed
           ``[state-key: pitfall:<tool>:<slug>[:<qualifier>]]`` so distillation stores it
           as a pitfall that replaces its predecessor.
6. SEAL    advance the watermark to the end of the window.

Failures propagate, as in ``extract_pending_agent_context``: a window's watermark moves
only after every call of the window completed, and the improve stage reports ``errored``
per session; the windows sealed before the failure are not extracted again.
"""

import asyncio
import json
import re
import shlex
from dataclasses import dataclass
from typing import Any

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.session.agent_context_extraction import (
    TRACE_EXTRACTION_WATERMARK,
)
from cognee.infrastructure.session.agent_context_extraction import (
    _sanitize_error_text as redact_text,
)
from cognee.infrastructure.session.coding_turns import (
    Turn,
    build_turns,
    group_by_turn,
    parse_qa_context,
    settled_traces,
)
from cognee.infrastructure.session.session_context_builder import (
    apply_candidate_updates,
    coerce_active_context_entries,
)
from cognee.infrastructure.session.session_context_models import (
    MAX_CONTEXT_CONTENT_CHARS,
    CandidateFailureLessonUpdate,
    CodingAgentContextExtraction,
    ContextProfile,
)
from cognee.infrastructure.session.session_node_set import get_session_node_set
from cognee.infrastructure.session.session_persist_watermark import StateRowWatermark
from cognee.infrastructure.session.trace_phases import PhaseBlock, split_into_parts
from cognee.modules.agent_memory.sanitization import sanitize_value
from cognee.modules.preflight import llm_available
from cognee.shared.logging_utils import get_logger

logger = get_logger("coding_unit_extraction")

CODING_PROMPT_FILE = "coding/agent_context_extraction_system.txt"

# Pending traces are extracted oldest first in windows of this many; a backlog larger
# than one window (improves that failed or were off for a while) takes several in one run.
TRACE_WINDOW_SIZE = 400
# One LLM input; a longer unit is split into trace chunks that each repeat every section.
MAX_INPUT_CHARS = 120_000
MAX_UNIT_LESSONS = 10
EXISTING_LESSONS_SHOWN = 40
# LESSONS ALREADY IN MEMORY: stored lessons most similar to the part, session's first.
MEMORY_LESSONS_SHOWN = 8
MEMORY_LESSONS_CHARS = 1_500
MEMORY_SEARCH_TIMEOUT_SECONDS = 30

MAX_TRACE_INPUT_CHARS = 200
MAX_COMMAND_CHARS = 160
MAX_EDIT_SNIPPET_CHARS = 60
MAX_TRACE_TEXT_CHARS = 600  # error and output, each
MAX_ERROR_GROUPS = 10
MAX_REQUEST_CHARS = 1_200
MAX_FINAL_ANSWER_CHARS = 6_000
MAX_TURN_LABEL_CHARS = 120  # the request in a turn's TRACES header

# A fixed error group is direct evidence; it clears the 0.75 candidate gate.
FAILURE_CANDIDATE_CONFIDENCE = 0.9
STATE_KEY_PREFIX = "[state-key: "
PITFALL_KEY_PREFIX = "pitfall"
PITFALL_SLUG_MIN_WORDS = 3
PITFALL_SLUG_MAX_WORDS = 5

# Tools whose successful call changes the file it names.
CHANGE_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
# Shell words that only set up the command after them, and words that run no program.
_SHELL_WRAPPERS = frozenset({"sudo", "env", "time", "nohup"})
_SHELL_NO_PROGRAM = frozenset({"cd", "export", "source", "set", "echo", "sleep", "exit", "true"})
_SHELL_CHAIN_RE = re.compile(r"\s*(?:&&|\|\||;|\||\n)\s*")
_WORD_RE = re.compile(r"[a-z0-9_]{2,}")


@dataclass(frozen=True, slots=True)
class UnitOfWork:
    """What one extraction window learns from: its turns, oldest first, each one a
    request, the traces it owns and the agent's final answer."""

    turns: list[Turn]
    repo_root: str = ""

    @property
    def traces(self) -> list:
        """The window's traces, turn by turn."""
        return [trace for turn in self.turns for trace in turn.traces]

    def turn_numbers(self) -> dict[str, int]:
        """Trace id to the 1-based number of its turn, as the input labels them."""
        return {
            trace.trace_id: number
            for number, turn in enumerate(self.turns, start=1)
            for trace in turn.traces
        }


@dataclass(slots=True)
class ErrorGroup:
    """Failed traces sharing a tool and a normalized first error line."""

    tool: str
    error_line: str
    failures: list
    last_index: int = 0
    fix: Any = None  # the first later success on the same file or program, a trace entry

    @property
    def fixed(self) -> bool:
        return self.fix is not None


def relative_path(path: str, repo_root: str) -> str:
    """Strip the repository root (else the home directory) from an absolute path."""
    text = str(path or "").strip()
    root = (repo_root or "").rstrip("/")
    if root and text.startswith(root + "/"):
        return text[len(root) + 1 :]
    if root and text == root:
        return "."
    match = re.match(r"^(?:/Users|/home)/[^/]+/(.*)$", text)
    return match.group(1) if match else text


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def shell_program(command: str) -> str:
    """The program a shell command runs: the first segment's first word after wrappers."""
    for segment in _SHELL_CHAIN_RE.split(str(command or "").strip()):
        words = segment.split()
        while words and (words[0] in _SHELL_WRAPPERS or "=" in words[0]):
            words = words[1:]
        if not words or words[0] in _SHELL_NO_PROGRAM:
            continue
        return words[0]
    return ""


def _params(entry) -> dict:
    return entry.method_params if isinstance(entry.method_params, dict) else {}


def trace_file(entry) -> str:
    """The file a trace targets, from the tool input, or ``""``."""
    for key in ("file_path", "notebook_path"):
        value = _params(entry).get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def trace_command(entry) -> str:
    command = _params(entry).get("command")
    return command if isinstance(command, str) else ""


def render_trace_input(entry, repo_root: str = "") -> str:
    """The tool input in at most ``MAX_TRACE_INPUT_CHARS``: path, command, pattern or edit."""
    params = _params(entry)
    parts: list[str] = []
    path = trace_file(entry)
    if path:
        parts.append(relative_path(path, repo_root))
    command = trace_command(entry)
    if command.strip():
        parts.append(_clip(command, MAX_COMMAND_CHARS))
    pattern = params.get("pattern")
    if isinstance(pattern, str) and pattern.strip():
        parts.append(f"pattern={_clip(pattern, MAX_COMMAND_CHARS)}")
    old = params.get("old_string")
    new = params.get("new_string")
    if isinstance(old, str) or isinstance(new, str):
        old_snippet = _clip(old or "", MAX_EDIT_SNIPPET_CHARS)
        new_snippet = _clip(new or "", MAX_EDIT_SNIPPET_CHARS)
        parts.append(f"{old_snippet!r} -> {new_snippet!r}")
    if not parts:
        for value in params.values():
            if isinstance(value, str) and value.strip():
                parts.append(_clip(value, MAX_COMMAND_CHARS))
                break
    return redact_text(" ".join(parts), max_chars=MAX_TRACE_INPUT_CHARS)


def render_trace_line(entry, repo_root: str = "") -> str:
    """``- <tool> <input> [<status>] | feedback: ... | error: ... | output: ...``."""
    head = f"- {entry.origin_function}"
    rendered_input = render_trace_input(entry, repo_root)
    if rendered_input:
        head += f" {rendered_input}"
    parts = [f"{head} [{entry.status}]"]
    if entry.session_feedback:
        parts.append(
            f"feedback: {redact_text(entry.session_feedback, max_chars=MAX_TRACE_TEXT_CHARS)}"
        )
    if entry.error_message:
        parts.append(f"error: {redact_text(entry.error_message, max_chars=MAX_TRACE_TEXT_CHARS)}")
    sanitized_return = sanitize_value(entry.method_return_value)
    if sanitized_return not in (None, "", {}, []):
        serialized = (
            sanitized_return
            if isinstance(sanitized_return, str)
            else json.dumps(sanitized_return, ensure_ascii=False)
        )
        parts.append(f"output: {redact_text(serialized, max_chars=MAX_TRACE_TEXT_CHARS)}")
    return " | ".join(parts)


# -- Repeated errors ---------------------------------------------------------


def normalized_error_line(error_message: str) -> str:
    """The first line of an error, redacted, lowercased and whitespace-collapsed."""
    first = str(error_message or "").strip().splitlines()
    return " ".join(redact_text(first[0]).lower().split()) if first else ""


def pitfall_slug(error_line: str) -> str:
    """3–5 words of the error: the part before its first colon when that is enough."""
    lowered = error_line.lower()
    head = lowered.split(":", 1)[0]
    words = _WORD_RE.findall(head)
    if len(words) < PITFALL_SLUG_MIN_WORDS:
        words = _WORD_RE.findall(lowered)
    return "-".join(words[:PITFALL_SLUG_MAX_WORDS]).replace("_", "-")


def pitfall_qualifier(entry) -> str:
    """The file extension of the trace's file, else the shell program it ran."""
    path = trace_file(entry)
    basename = path.rsplit("/", 1)[-1]
    if "." in basename:
        return basename.rsplit(".", 1)[-1].lower()
    return shell_program(trace_command(entry))


def pitfall_key(group: ErrorGroup) -> str:
    slug = pitfall_slug(group.error_line) or "error"
    key = f"{PITFALL_KEY_PREFIX}:{group.tool}:{slug}"
    qualifier = pitfall_qualifier(group.failures[0])
    return f"{key}:{qualifier}" if qualifier else key


def _changes_file(entry, path: str) -> bool:
    """A change tool on the file, or a shell command that names it (a script rewriting it)."""
    if not path:
        return False
    if entry.origin_function in CHANGE_TOOLS:
        return trace_file(entry) == path
    command = trace_command(entry)
    basename = path.rsplit("/", 1)[-1]
    return bool(command) and (path in command or basename in command)


def _program_segment(command: str) -> list[str]:
    """The words of the first chained segment that runs a program, wrappers dropped."""
    for segment in _SHELL_CHAIN_RE.split(str(command or "").strip()):
        try:
            words = shlex.split(segment)
        except ValueError:
            words = segment.split()
        while words and (words[0] in _SHELL_WRAPPERS or "=" in words[0]):
            words = words[1:]
        if words and words[0] not in _SHELL_NO_PROGRAM:
            return words
    return []


def _command_target(command: str) -> str:
    """What the program ran on: its first argument that is not a flag (a script, a file,
    a subcommand), or "" when it ran bare."""
    for word in _program_segment(command)[1:]:
        if not word.startswith("-"):
            return word
    return ""


def _reruns_program(entry, failure) -> bool:
    """The same tool ran the same program on the same target: ``python broken.py`` after
    a failed ``python broken.py``, not ``python --version``."""
    if entry.origin_function != failure.origin_function:
        return False
    later, failed = trace_command(entry), trace_command(failure)
    program = shell_program(later)
    return (
        bool(program)
        and program == shell_program(failed)
        and _command_target(later) == _command_target(failed)
    )


def group_repeated_errors(traces: list) -> list[ErrorGroup]:
    """Failed traces grouped by tool and first error line, each paired with its fix.

    A group counts when it failed at least twice, or once and a later successful trace
    changed the same file or reran the same program. The fix is the first such success
    after the group's last failure. In order of first failure; the rendered section
    shows the first ``MAX_ERROR_GROUPS``, the deterministic candidates cover every
    fixed group.
    """
    groups: dict[tuple[str, str], ErrorGroup] = {}
    for index, entry in enumerate(traces):
        if entry.status != "error":
            continue
        line = normalized_error_line(entry.error_message)
        if not line:
            continue
        key = (entry.origin_function, line)
        group = groups.setdefault(
            key, ErrorGroup(tool=entry.origin_function, error_line=line, failures=[])
        )
        group.failures.append(entry)
        group.last_index = index

    for group in groups.values():
        first_failure = group.failures[0]
        failed_file = trace_file(first_failure)
        for later in traces[group.last_index + 1 :]:
            if later.status != "success":
                continue
            if _changes_file(later, failed_file) or _reruns_program(later, first_failure):
                group.fix = later
                break

    return [group for group in groups.values() if len(group.failures) >= 2 or group.fixed]


def render_error_groups(groups: list[ErrorGroup], repo_root: str = "") -> str:
    """The REPEATED ERRORS AND WHAT WORKED lines of the first ``MAX_ERROR_GROUPS`` groups."""
    lines = []
    for group in groups[:MAX_ERROR_GROUPS]:
        failure = group.failures[0]
        lines.append(
            f"- {group.tool} failed {len(group.failures)}x: "
            f"{render_trace_input(failure, repo_root) or '(no input)'} | "
            f"error: {_clip(group.error_line, MAX_TRACE_TEXT_CHARS)}"
        )
        if group.fixed:
            lines.append(
                f"  What worked: {group.fix.origin_function} "
                f"{render_trace_input(group.fix, repo_root) or '(no input)'}"
            )
        else:
            lines.append("  What worked: no later successful call")
    return "\n".join(lines)


def build_failure_candidates(
    groups: list[ErrorGroup], repo_root: str = ""
) -> list[tuple[str, CandidateFailureLessonUpdate]]:
    """``(source trace id, candidate)`` per fixed group; the content carries the pitfall key.

    The content is deterministic for a group, so the applier's exact-content dedup makes
    a repeated run link the same entry instead of creating another. The candidate model
    cuts the content at ``MAX_CONTEXT_CONTENT_CHARS``.
    """
    candidates = []
    for group in groups:
        if not group.fixed:
            continue
        failure = group.failures[0]
        problem = f"{group.tool} {render_trace_input(failure, repo_root)}".strip()
        worked = f"{group.fix.origin_function} {render_trace_input(group.fix, repo_root)}".strip()
        content = (
            f"{STATE_KEY_PREFIX}{pitfall_key(group)}] {problem} failed with "
            f'"{group.error_line}" → {worked}'
        )
        candidates.append(
            (
                failure.trace_id,
                CandidateFailureLessonUpdate(
                    content=content, confidence=FAILURE_CANDIDATE_CONFIDENCE
                ),
            )
        )
    return candidates


# -- Extraction input --------------------------------------------------------

# Room kept for EXISTING LESSONS and LESSONS ALREADY IN MEMORY at their largest, so the
# chunking of a unit never depends on how many lessons the earlier parts added.
_EXISTING_LESSONS_RESERVE = EXISTING_LESSONS_SHOWN * (MAX_CONTEXT_CONTENT_CHARS + 40) + 80
_MEMORY_LESSONS_RESERVE = MEMORY_LESSONS_CHARS + MEMORY_LESSONS_SHOWN * 4 + 80
_TRACES_HEADER_RESERVE = len("TRACES (part 9999 of 9999, phase explore):\n") + 4


def render_memory_lessons(lessons: list[str]) -> str:
    if not lessons:
        return ""
    return "LESSONS ALREADY IN MEMORY (stored for this repository — do not restate):\n" + "\n".join(
        f"- {lesson}" for lesson in lessons
    )


def memory_search_query(traces: list, repo_root: str = "") -> str:
    """What a part is about: its file paths, first error lines, commands and patterns."""
    pieces: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        text = " ".join(str(text or "").split())
        if text and text not in seen:
            seen.add(text)
            pieces.append(text)

    for entry in traces:
        path = trace_file(entry)
        if path:
            add(relative_path(path, repo_root))
        if entry.status == "error":
            add(normalized_error_line(entry.error_message))
        command = trace_command(entry)
        if command:
            add(_clip(command, MAX_COMMAND_CHARS))
        pattern = _params(entry).get("pattern")
        if isinstance(pattern, str):
            add(pattern)
    return "\n".join(redact_text(piece) for piece in pieces)


def select_memory_lessons(session_lessons: list[str], dataset_lessons: list[str]) -> list[str]:
    """Session lessons first, then the dataset's, no duplicates, within the two caps."""
    chosen: list[str] = []
    seen: set[str] = set()
    used = 0
    for text in [*session_lessons, *dataset_lessons]:
        normalized = " ".join(str(text or "").split())
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        if len(chosen) >= MEMORY_LESSONS_SHOWN or used + len(normalized) > MEMORY_LESSONS_CHARS:
            continue
        seen.add(key)
        chosen.append(normalized)
        used += len(normalized)
    return chosen


async def _search_memory_lessons(
    dataset, session_id: str, query: str, project: tuple[str, ...] = ()
) -> list[str]:
    """This session's stored lessons, then the dataset's lessons of the same project.

    ``project`` is the session's pinned node set (the plugin's repository). Repositories
    can share a dataset, and the prompt tells the model not to restate these lessons,
    so another project's lesson must not be shown: the dataset search asks for the
    lessons set and the project's sets together, and only hits scoped to exactly this
    project are kept (a session with no pin sees the unscoped lessons only).
    """
    # Imported here: distill imports this module, and the session package must not
    # load the vector stack on import.
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.modules.improve.constants import SESSION_LEARNINGS_NODE_SET
    from cognee.modules.session_distillation.distill import search_payload_texts
    from cognee.modules.session_distillation.lesson_store import project_node_sets
    from cognee.modules.truth_subspace.constants import truth_session_node_set

    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        vector_engine = await get_vector_engine_async()
        session_lessons = await search_payload_texts(
            vector_engine,
            "DocumentChunk_text",
            MEMORY_LESSONS_SHOWN,
            query_text=query,
            node_name=[truth_session_node_set(session_id)],
        )
        dataset_lessons = await search_payload_texts(
            vector_engine,
            "DocumentChunk_text",
            MEMORY_LESSONS_SHOWN,
            query_text=query,
            node_name=[SESSION_LEARNINGS_NODE_SET, *project],
            node_name_filter_operator="AND",
            project_scope=project_node_sets(list(project)),
        )
    return select_memory_lessons(session_lessons, dataset_lessons)


async def memory_lessons_for_part(
    dataset, session_id: str, traces: list, repo_root: str, project: tuple[str, ...] = ()
) -> list[str]:
    """Stored lessons most similar to the part; ``[]`` when there is no dataset, on a
    timeout (``MEMORY_SEARCH_TIMEOUT_SECONDS``) or on any failure, so the section is left out."""
    if dataset is None or not traces:
        return []
    query = memory_search_query(traces, repo_root)
    if not query:
        return []
    try:
        return await asyncio.wait_for(
            _search_memory_lessons(dataset, session_id, query, project),
            MEMORY_SEARCH_TIMEOUT_SECONDS,
        )
    except Exception as error:
        logger.warning(
            "Coding unit extraction: memory lessons search failed open: %s", error, exc_info=True
        )
        return []


def render_existing_lessons(lessons: list) -> str:
    if not lessons:
        return ""
    rendered = "\n".join(f"- [{entry.section}] {entry.content}" for entry in lessons)
    return "EXISTING LESSONS (already saved — do not repeat or reword these):\n" + rendered


def turn_header(number: int, turn: Turn) -> str:
    """The line that opens a turn's traces in TRACES."""
    return f"-- turn {number}: {_clip(turn.request, MAX_TURN_LABEL_CHARS) or '(no request)'} --"


def final_message_block(number: int, turn: Turn) -> str:
    """One turn's request and final answer, as FINAL MESSAGE shows them."""
    request = _clip(turn.request, MAX_REQUEST_CHARS) or "(none recorded)"
    answer = _clip(turn.final_answer, MAX_FINAL_ANSWER_CHARS) or "(none recorded)"
    return f"Turn {number}:\nUser request: {request}\nAgent's final answer: {answer}"


def render_final_message(unit: UnitOfWork, numbers: tuple[int, ...] | None = None) -> str:
    """FINAL MESSAGE: the request and final answer of the given turns (all by default)."""
    blocks = [
        final_message_block(number, turn)
        for number, turn in enumerate(unit.turns, start=1)
        if numbers is None or number in numbers
    ]
    return "FINAL MESSAGE:\n" + "\n\n".join(blocks)


def turn_chars(number: int, turn: Turn) -> int:
    """What an input grows by when it shows ``turn``: its TRACES header line and its
    FINAL MESSAGE block, with their separators."""
    return len(turn_header(number, turn)) + 1 + len(final_message_block(number, turn)) + 2


def render_tail_sections(
    unit: UnitOfWork, groups: list[ErrorGroup], numbers: tuple[int, ...] | None = None
) -> str:
    """REPEATED ERRORS AND WHAT WORKED (when any) followed by FINAL MESSAGE for ``numbers``."""
    sections = []
    if groups:
        sections.append(
            "REPEATED ERRORS AND WHAT WORKED:\n" + render_error_groups(groups, unit.repo_root)
        )
    sections.append(render_final_message(unit, numbers))
    return "\n\n".join(sections)


def pack_trace_lines(
    lines: list[str], turns: list[int], turn_costs: dict[int, int], budget: int
) -> list[list[str]]:
    """Pack lines in order into chunks that fit ``budget`` chars together with the
    header and FINAL MESSAGE block of every turn they show (``turns[i]`` is the turn of
    ``lines[i]``, ``turn_costs`` what each turn adds); an oversized line goes alone.

    A turn is paid for once per chunk that shows it, so a window of many turns is cut
    by its trace text and the turns it covers, never by every turn of the window.
    """
    chunks: list[list[str]] = [[]]
    used = 0
    shown: set[int] = set()
    for line, turn in zip(lines, turns):
        cost = len(line) + 1 + (0 if turn in shown else turn_costs[turn])
        if chunks[-1] and used + cost > budget:
            chunks.append([])
            used = 0
            shown = set()
            cost = len(line) + 1 + turn_costs[turn]
        chunks[-1].append(line)
        shown.add(turn)
        used += cost
    return chunks if chunks[0] else []


@dataclass(frozen=True, slots=True)
class ExtractionChunk:
    """One LLM input's traces: a part (or a piece of one) with its phase, and the
    numbers of the turns the traces belong to."""

    part: int
    parts: int
    phase: str
    traces: list
    lines: list[str]
    turns: tuple[int, ...] = ()


def plan_extraction_parts(
    unit: UnitOfWork, groups: list[ErrorGroup], *, max_chars: int = MAX_INPUT_CHARS
) -> list[ExtractionChunk]:
    """The unit's phase parts, each cut into chunks that fit ``max_chars`` with the
    other sections; a short unit is one part and one chunk. A turn's header and FINAL
    MESSAGE block count only in the chunks that show the turn."""
    fixed_tail = render_tail_sections(unit, groups, ())  # the tail with no turn in it
    budget = (
        max_chars
        - _EXISTING_LESSONS_RESERVE
        - _MEMORY_LESSONS_RESERVE
        - len(fixed_tail)
        - _TRACES_HEADER_RESERVE
    )
    turn_costs = {n: turn_chars(n, turn) for n, turn in enumerate(unit.turns, start=1)}
    all_traces = unit.traces
    numbers = unit.turn_numbers()
    parts: list[PhaseBlock] = split_into_parts(all_traces)
    chunks: list[ExtractionChunk] = []
    for index, part in enumerate(parts, start=1):
        traces = all_traces[part.start : part.end]
        lines = [render_trace_line(entry, unit.repo_root) for entry in traces]
        offset = 0
        turns = [numbers[entry.trace_id] for entry in traces]
        for chunk_lines in pack_trace_lines(lines, turns, turn_costs, budget):
            chunk_traces = traces[offset : offset + len(chunk_lines)]
            chunks.append(
                ExtractionChunk(
                    part=index,
                    parts=len(parts),
                    phase=part.phase,
                    traces=chunk_traces,
                    lines=chunk_lines,
                    turns=tuple(sorted({numbers[entry.trace_id] for entry in chunk_traces})),
                )
            )
            offset += len(chunk_lines)
    return chunks


def render_extraction_input(
    unit: UnitOfWork,
    existing_lessons: list,
    memory_lessons: list[str],
    groups: list[ErrorGroup],
    chunk: ExtractionChunk,
) -> str:
    """The five sections in order: EXISTING LESSONS, LESSONS ALREADY IN MEMORY, TRACES
    (a header opens each turn's traces), REPEATED ERRORS AND WHAT WORKED, FINAL MESSAGE
    (the chunk's turns)."""
    numbers = unit.turn_numbers()
    lines = []
    current = 0
    for entry, line in zip(chunk.traces, chunk.lines):
        number = numbers[entry.trace_id]
        if number != current:
            lines.append(turn_header(number, unit.turns[number - 1]))
            current = number
        lines.append(line)
    traces = f"TRACES (part {chunk.part} of {chunk.parts}, phase {chunk.phase}):\n" + "\n".join(
        lines
    )
    sections = [
        render_existing_lessons(existing_lessons),
        render_memory_lessons(memory_lessons),
        traces,
        render_tail_sections(unit, groups, chunk.turns),
    ]
    return "\n\n".join(section for section in sections if section)


def build_extraction_inputs(
    unit: UnitOfWork,
    existing_lessons: list,
    groups: list[ErrorGroup],
    *,
    memory_lessons: list[str] | None = None,
    max_chars: int = MAX_INPUT_CHARS,
) -> list[str]:
    """Every rendered input at once, with one memory section for all (tests, previews)."""
    return [
        render_extraction_input(unit, existing_lessons, memory_lessons or [], groups, chunk)
        for chunk in plan_extraction_parts(unit, groups, max_chars=max_chars)
    ]


# -- Orchestration -----------------------------------------------------------


async def recent_agent_lessons(session_manager, user_id: str, session_id: str) -> list:
    """The session's newest agent-profile lessons, oldest first, at most 40."""
    raw_entries = await session_manager.get_session_context_entries(
        user_id=user_id, session_id=session_id
    )
    agent_entries = [
        entry
        for entry in coerce_active_context_entries(raw_entries)
        if entry.context_profile == ContextProfile.AGENT.value
    ]
    agent_entries.sort(key=lambda entry: entry.created_at or "")
    return agent_entries[-EXISTING_LESSONS_SHOWN:]


@dataclass(frozen=True, slots=True)
class PendingBacklog:
    """The settled pending traces (oldest first, as stored), the session's turns, and
    the count already sealed."""

    traces: list
    turns: list[Turn]
    processed: int


async def load_pending_backlog(
    session_manager, user_id: str, session_id: str
) -> PendingBacklog | None:
    """Every pending trace that has its answer, with the session's turns, or None when
    nothing is pending. A trace newer than the last answer, and everything stored after
    it, waits for the next improve (``settled_traces``)."""
    total = await session_manager.get_agent_trace_count(user_id=user_id, session_id=session_id)
    stored = await TRACE_EXTRACTION_WATERMARK.read_count(session_manager, user_id, session_id)
    processed = StateRowWatermark.resolve_effective(stored, total, session_id=session_id)
    pending = total - processed
    if pending <= 0:
        return None

    traces = await session_manager.get_agent_trace_session(
        user_id=user_id, session_id=session_id, last_n=pending
    )
    rows = await session_manager.get_session(
        user_id=user_id, session_id=session_id, formatted=False
    )
    turns = build_turns(list(rows) if isinstance(rows, list) else [])
    settled = settled_traces(list(traces), turns)
    if not settled:
        return None
    return PendingBacklog(traces=settled, turns=turns, processed=processed)


def trace_windows(traces: list, size: int = TRACE_WINDOW_SIZE) -> list[list]:
    """The traces cut into windows of at most ``size``, oldest first."""
    return [traces[start : start + size] for start in range(0, len(traces), size)]


def window_unit(window: list, turns: list[Turn]) -> UnitOfWork:
    """One window's traces grouped by the turns that own them; the repository root is
    the newest covered turn's ``cwd``."""
    covered = group_by_turn(window, turns)
    return UnitOfWork(turns=covered, repo_root=str(covered[-1].qa_context.get("cwd") or ""))


async def extract_coding_unit_context(
    *,
    session_manager,
    user_id: str,
    session_id: str,
    dataset=None,
) -> list[str]:
    """Extract the session's pending traces once, paired to their turns. Returns the
    touched entry ids.

    ``dataset`` (id and owner_id) is where the stored lessons for the LESSONS ALREADY IN
    MEMORY section are searched; without it the section is left out. Serialized per
    (user, session): a concurrent trigger waits, then finds the watermark at the total
    and extracts nothing. The settled traces are walked in windows of
    ``TRACE_WINDOW_SIZE``, oldest first, the watermark sealed after each; an LLM failure
    propagates before its window's watermark moves, so that window and the ones after
    it are retried by the next improve.
    """
    if not llm_available():
        return []

    # Imported here, like the session manager does: the locks package must not
    # load with the session infrastructure.
    from cognee.infrastructure.locks import session_lock

    async with session_lock(f"{user_id}:{session_id}", "coding_unit_extraction"):
        backlog = await load_pending_backlog(session_manager, user_id, session_id)
        if backlog is None:
            return []
        sealed = backlog.processed

        system_prompt = read_query_prompt(CODING_PROMPT_FILE)
        if not system_prompt:
            raise RuntimeError("Coding unit extraction: system prompt not found")

        # The session's project, for the memory search; read strictly, like the
        # bridging stages do, so an unreadable pin is never taken for "no project".
        project: tuple[str, ...] = ()
        if dataset is not None:
            project = await get_session_node_set(session_manager, user_id, session_id)

        touched: list[str] = []
        for window in trace_windows(backlog.traces):
            sealed += len(window)
            touched += await _extract_window(
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
                dataset=dataset,
                unit=window_unit(window, backlog.turns),
                system_prompt=system_prompt,
                project=project,
            )
            await TRACE_EXTRACTION_WATERMARK.write_count(
                session_manager, user_id, session_id, count=sealed
            )
        return touched


def _section(candidate) -> str:
    """A candidate's section, whether it is a model or a plain dict."""
    if isinstance(candidate, dict):
        return str(candidate.get("section") or "")
    return str(getattr(candidate, "section", "") or "")


async def _extract_window(
    *,
    session_manager,
    user_id: str,
    session_id: str,
    dataset,
    unit: UnitOfWork,
    system_prompt,
    project: tuple[str, ...] = (),
) -> list[str]:
    """One window's extraction calls and failure candidates; returns the touched ids.

    Every candidate is linked to the Q&A row of its turn: the LLM names the turn of
    each lesson (a number outside the chunk's turns counts as its newest), a failure
    candidate takes its failing trace's turn. An LLM ``failure_lessons`` candidate is
    also linked to the errored traces of its turn, the evidence the distillation's
    pitfall rule looks for; with none it stays unlinked and its lesson is a change.
    """
    groups = group_repeated_errors(unit.traces)
    chunks = plan_extraction_parts(unit, groups)
    numbers = unit.turn_numbers()
    touched: list[str] = []
    for chunk in chunks:
        # Re-read per chunk, so each sees the candidates the earlier ones produced.
        existing = await recent_agent_lessons(session_manager, user_id, session_id)
        memory = await memory_lessons_for_part(
            dataset, session_id, chunk.traces, unit.repo_root, project
        )
        result = await LLMGateway.acreate_structured_output(
            text_input=render_extraction_input(unit, existing, memory, groups, chunk),
            system_prompt=system_prompt,
            response_model=CodingAgentContextExtraction,
        )
        by_turn: dict[int, list] = {}
        for item in list(result.lessons)[:MAX_UNIT_LESSONS]:
            number = item.turn if item.turn in chunk.turns else chunk.turns[-1]
            by_turn.setdefault(number, []).append(item.lesson)
        for number, candidates in by_turn.items():
            turn = unit.turns[number - 1]
            errored = [trace.trace_id for trace in turn.traces if trace.status == "error"]
            failures = [c for c in candidates if _section(c) == "failure_lessons"]
            others = [c for c in candidates if _section(c) != "failure_lessons"]
            for source_id, batch in ((errored, failures), ("", others)):
                if batch:
                    touched += await apply_candidate_updates(
                        session_manager=session_manager,
                        user_id=user_id,
                        session_id=session_id,
                        source_id=source_id,
                        candidates=batch,
                        qa_id=turn.qa_id,
                    )

    for source_id, candidate in build_failure_candidates(groups, unit.repo_root):
        touched += await apply_candidate_updates(
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
            source_id=source_id,
            candidates=[candidate],
            qa_id=unit.turns[numbers[source_id] - 1].qa_id,
        )

    logger.info(
        "Coding unit extraction: session '%s' window traces=%d turns=%d parts=%d inputs=%d "
        "error_groups=%d touched=%d",
        session_id,
        len(unit.traces),
        len(unit.turns),
        chunks[-1].parts if chunks else 0,
        len(chunks),
        len(groups),
        len(touched),
    )
    return touched
