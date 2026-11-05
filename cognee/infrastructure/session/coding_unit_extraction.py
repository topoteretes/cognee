"""Coding distillation: extract one unit of work's lessons at improve() time (SDK-1335).

Active with ``DISTILLATION_MODE=coding``. A *unit of work* is one user request, every
trace the agent produced for it, and the agent's final answer. The mid-session passes
of ``agent_context_extraction`` are off in this mode: they see ten traces at a time
without tool inputs and never see how the task ended, which is what produced vague
lessons and failures recorded without their fix.

Flow, once per unit, from the ``extract_agent_context`` improve stage:

1. LOAD    the traces past ``TRACE_EXTRACTION_WATERMARK`` (the previous unit's end) and
           the latest Q&A entry (request and final answer), under a per-(user, session)
           lock so two triggers never extract the same unit. More than
           ``TRACE_WINDOW_SIZE`` pending traces are walked oldest first in windows of
           that size, each sealed before the next, so nothing is skipped.
2. SPLIT   the traces into explore, change and verify parts (``trace_phases``), and a
           part longer than one input into trace chunks.
3. RENDER  one extraction input per chunk, every chunk carrying all five sections:
           EXISTING LESSONS, LESSONS ALREADY IN MEMORY (a vector search over the
           dataset's stored lessons, fail-open), TRACES (with tool inputs), REPEATED
           ERRORS AND WHAT WORKED, FINAL MESSAGE. Inputs and outputs are redacted and
           paths made repo-relative.
4. EXTRACT one LLM call per chunk, in order, so each sees the candidates of the earlier
           ones; up to ``MAX_UNIT_LESSONS`` candidates per call go through the shared
           applier (``apply_candidate_updates``).
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
from dataclasses import dataclass, field, replace
from typing import Any

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.session.agent_context_extraction import (
    TRACE_EXTRACTION_WATERMARK,
)
from cognee.infrastructure.session.agent_context_extraction import (
    _sanitize_error_text as redact_text,
)
from cognee.infrastructure.session.session_context_builder import (
    apply_candidate_updates,
    coerce_active_context_entries,
)
from cognee.infrastructure.session.session_context_models import (
    MAX_CONTEXT_CONTENT_CHARS,
    AgentContextExtraction,
    CandidateFailureLessonUpdate,
    ContextProfile,
)
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
MEMORY_QUERY_CHARS = 2_000

MAX_TRACE_INPUT_CHARS = 200
MAX_COMMAND_CHARS = 160
MAX_EDIT_SNIPPET_CHARS = 60
MAX_TRACE_TEXT_CHARS = 600  # error and output, each
MAX_ERROR_GROUPS = 10
MAX_REQUEST_CHARS = 1_200
MAX_FINAL_ANSWER_CHARS = 6_000

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
    """One request, its traces (oldest first) and the agent's final answer."""

    request: str
    final_answer: str
    traces: list
    repo_root: str = ""
    qa_context: dict = field(default_factory=dict)


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


def parse_qa_context(context: str | None) -> dict:
    """The structured context a coding plugin stores on its Q&A entry, or ``{}``.

    SDK-1336 sends ``{"cwd", "transcript_path", "date", "start_commit", "end_commit",
    "branch"}`` as the entry's ``context``; any other text is not an error.
    """
    if not context or not isinstance(context, str):
        return {}
    try:
        parsed = json.loads(context)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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
    return "\n".join(redact_text(piece) for piece in pieces)[:MEMORY_QUERY_CHARS]


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


async def _search_memory_lessons(dataset, session_id: str, query: str) -> list[str]:
    # Imported here: distill imports this module, and the session package must not
    # load the vector stack on import.
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.modules.improve.constants import SESSION_LEARNINGS_NODE_SET
    from cognee.modules.session_distillation.distill import search_payload_texts
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
            node_name=[SESSION_LEARNINGS_NODE_SET],
        )
    return select_memory_lessons(session_lessons, dataset_lessons)


async def memory_lessons_for_part(
    dataset, session_id: str, traces: list, repo_root: str
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
            _search_memory_lessons(dataset, session_id, query), MEMORY_SEARCH_TIMEOUT_SECONDS
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


def render_tail_sections(unit: UnitOfWork, groups: list[ErrorGroup]) -> str:
    """REPEATED ERRORS AND WHAT WORKED (when any) followed by FINAL MESSAGE."""
    sections = []
    if groups:
        sections.append(
            "REPEATED ERRORS AND WHAT WORKED:\n" + render_error_groups(groups, unit.repo_root)
        )
    request = _clip(unit.request, MAX_REQUEST_CHARS) or "(none recorded)"
    answer = _clip(unit.final_answer, MAX_FINAL_ANSWER_CHARS) or "(none recorded)"
    sections.append(f"FINAL MESSAGE:\nUser request: {request}\nAgent's final answer: {answer}")
    return "\n\n".join(sections)


def chunk_trace_lines(lines: list[str], budget: int) -> list[list[str]]:
    """Pack lines in order under ``budget`` chars per chunk; an oversized line goes alone."""
    chunks: list[list[str]] = [[]]
    used = 0
    for line in lines:
        cost = len(line) + 1
        if chunks[-1] and used + cost > budget:
            chunks.append([])
            used = 0
        chunks[-1].append(line)
        used += cost
    return chunks if chunks[0] else []


@dataclass(frozen=True, slots=True)
class ExtractionChunk:
    """One LLM input's traces: a part (or a piece of one) with its phase."""

    part: int
    parts: int
    phase: str
    traces: list
    lines: list[str]


def plan_extraction_parts(
    unit: UnitOfWork, groups: list[ErrorGroup], *, max_chars: int = MAX_INPUT_CHARS
) -> list[ExtractionChunk]:
    """The unit's phase parts, each cut into chunks that fit ``max_chars`` with the
    other sections; a short unit is one part and one chunk."""
    tail = render_tail_sections(unit, groups)
    budget = (
        max_chars
        - _EXISTING_LESSONS_RESERVE
        - _MEMORY_LESSONS_RESERVE
        - len(tail)
        - _TRACES_HEADER_RESERVE
    )
    parts: list[PhaseBlock] = split_into_parts(unit.traces)
    chunks: list[ExtractionChunk] = []
    for index, part in enumerate(parts, start=1):
        traces = unit.traces[part.start : part.end]
        lines = [render_trace_line(entry, unit.repo_root) for entry in traces]
        offset = 0
        for chunk_lines in chunk_trace_lines(lines, max(budget, 1)):
            chunks.append(
                ExtractionChunk(
                    part=index,
                    parts=len(parts),
                    phase=part.phase,
                    traces=traces[offset : offset + len(chunk_lines)],
                    lines=chunk_lines,
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
    """The five sections in order: EXISTING LESSONS, LESSONS ALREADY IN MEMORY, TRACES,
    REPEATED ERRORS AND WHAT WORKED, FINAL MESSAGE."""
    traces = f"TRACES (part {chunk.part} of {chunk.parts}, phase {chunk.phase}):\n" + "\n".join(
        chunk.lines
    )
    sections = [
        render_existing_lessons(existing_lessons),
        render_memory_lessons(memory_lessons),
        traces,
        render_tail_sections(unit, groups),
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


async def load_unit_of_work(
    session_manager, user_id: str, session_id: str
) -> tuple[UnitOfWork, int] | None:
    """Every pending trace as one unit and the count already sealed, or None when
    nothing is pending. ``trace_windows`` cuts the unit for extraction."""
    total = await session_manager.get_agent_trace_count(user_id=user_id, session_id=session_id)
    stored = await TRACE_EXTRACTION_WATERMARK.read_count(session_manager, user_id, session_id)
    processed = StateRowWatermark.resolve_effective(stored, total, session_id=session_id)
    pending = total - processed
    if pending <= 0:
        return None

    traces = await session_manager.get_agent_trace_session(
        user_id=user_id, session_id=session_id, last_n=pending
    )
    latest = await session_manager.get_session(
        user_id=user_id, session_id=session_id, last_n=1, formatted=False
    )
    qa = latest[-1] if isinstance(latest, list) and latest else None
    qa_context = parse_qa_context(getattr(qa, "context", "")) if qa is not None else {}
    unit = UnitOfWork(
        request=getattr(qa, "question", "") or "",
        final_answer=getattr(qa, "answer", "") or "",
        traces=list(traces),
        repo_root=str(qa_context.get("cwd") or ""),
        qa_context=qa_context,
    )
    return unit, processed


def trace_windows(unit: UnitOfWork, size: int = TRACE_WINDOW_SIZE) -> list[UnitOfWork]:
    """The unit cut into windows of at most ``size`` traces, oldest first, each with the
    unit's request, answer and context."""
    return [
        replace(unit, traces=unit.traces[start : start + size])
        for start in range(0, len(unit.traces), size)
    ]


async def extract_coding_unit_context(
    *,
    session_manager,
    user_id: str,
    session_id: str,
    dataset=None,
) -> list[str]:
    """Extract the session's pending unit of work once. Returns the touched entry ids.

    ``dataset`` (id and owner_id) is where the stored lessons for the LESSONS ALREADY IN
    MEMORY section are searched; without it the section is left out. Serialized per
    (user, session): a concurrent trigger waits, then finds the watermark at the total
    and extracts nothing. The pending traces are walked in windows of
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
        loaded = await load_unit_of_work(session_manager, user_id, session_id)
        if loaded is None:
            return []
        pending_unit, sealed = loaded

        system_prompt = read_query_prompt(CODING_PROMPT_FILE)
        if not system_prompt:
            raise RuntimeError("Coding unit extraction: system prompt not found")

        touched: list[str] = []
        for unit in trace_windows(pending_unit):
            sealed += len(unit.traces)
            touched += await _extract_window(
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
                dataset=dataset,
                unit=unit,
                system_prompt=system_prompt,
            )
            await TRACE_EXTRACTION_WATERMARK.write_count(
                session_manager, user_id, session_id, count=sealed
            )
        return touched


async def _extract_window(
    *, session_manager, user_id: str, session_id: str, dataset, unit: UnitOfWork, system_prompt
) -> list[str]:
    """One window's extraction calls and failure candidates; returns the touched ids."""
    groups = group_repeated_errors(unit.traces)
    chunks = plan_extraction_parts(unit, groups)
    touched: list[str] = []
    for chunk in chunks:
        # Re-read per chunk, so each sees the candidates the earlier ones produced.
        existing = await recent_agent_lessons(session_manager, user_id, session_id)
        memory = await memory_lessons_for_part(dataset, session_id, chunk.traces, unit.repo_root)
        result = await LLMGateway.acreate_structured_output(
            text_input=render_extraction_input(unit, existing, memory, groups, chunk),
            system_prompt=system_prompt,
            response_model=AgentContextExtraction,
        )
        candidates = list(result.lessons)[:MAX_UNIT_LESSONS]
        if candidates:
            touched += await apply_candidate_updates(
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
                source_id="",
                candidates=candidates,
            )

    for source_id, candidate in build_failure_candidates(groups, unit.repo_root):
        touched += await apply_candidate_updates(
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
            source_id=source_id,
            candidates=[candidate],
        )

    logger.info(
        "Coding unit extraction: session '%s' window traces=%d parts=%d inputs=%d "
        "error_groups=%d touched=%d",
        session_id,
        len(unit.traces),
        chunks[-1].parts if chunks else 0,
        len(chunks),
        len(groups),
        len(touched),
    )
    return touched
