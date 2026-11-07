"""Split a coding unit's traces into explore, change and verify parts (SDK-1335).

Standard library only, no LLM. Each trace is classified by its tool and, for a shell
call, by the programs the command runs; consecutive traces of one phase form a block,
and blocks are merged until every part is big enough and there are few enough parts.

Phases, from weakest to strongest (a chained command takes its strongest segment's):

- ``explore``: read and search tools, read-only shell and git commands, and inline
  Python that neither writes a file nor runs anything;
- ``verify``: test runners and build tools, ``python -m`` and script runs, inline Python
  that imports a third-party package or runs a subprocess, and any shell program not
  listed anywhere here (the project's own CLI, say);
- ``change``: write and edit tools, shell redirects and file-modifying commands, git
  commands that change state, and inline Python that writes a file. A target under
  ``/dev/null``, ``/tmp``, ``/var/folders`` or ``$variable`` is scratch and does not count.

A trace with no phase (``cd``, ``echo``, ``export``, shell keywords, TodoWrite, Task, MCP
tools) inherits the previous trace's phase, ``explore`` for the first. A read command that
only filters piped output (``cat x | grep y``) does not count.
"""

import re
import sys
from dataclasses import dataclass
from math import ceil

EXPLORE = "explore"
CHANGE = "change"
VERIFY = "verify"
PHASE_RANK = {EXPLORE: 1, VERIFY: 2, CHANGE: 3}

# Merge bounds: every part has at least MIN_PART_TRACES traces, and a unit has at most
# MAX_PARTS parts up to SMALL_UNIT_TRACES traces, ceil(n / TRACES_PER_PART) above.
MIN_PART_TRACES = 5
MAX_PARTS = 4
SMALL_UNIT_TRACES = 120
TRACES_PER_PART = 30

EXPLORE_TOOLS = frozenset({"Read", "Grep", "Glob", "LS", "NotebookRead", "WebFetch", "WebSearch"})
CHANGE_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
SHELL_TOOLS = frozenset({"Bash"})

READ_ONLY_PROGRAMS = frozenset(
    {
        "cat",
        "ls",
        "ll",
        "grep",
        "rg",
        "ag",
        "egrep",
        "fgrep",
        "find",
        "fd",
        "head",
        "tail",
        "less",
        "more",
        "wc",
        "which",
        "whereis",
        "pwd",
        "stat",
        "file",
        "diff",
        "tree",
        "du",
        "df",
        "type",
        "jq",
        "yq",
        "awk",
        "sort",
        "uniq",
        "cut",
        "tr",
        "column",
        "basename",
        "dirname",
        "realpath",
        "readlink",
        "md5sum",
        "sha256sum",
        "shasum",
        "env",
        "printenv",
        "date",
        "whoami",
        "hostname",
        "uname",
        "id",
        "ps",
        "top",
        "lsof",
        "nl",
        "od",
        "xxd",
        "hexdump",
        "strings",
        "tac",
        "rev",
        "cmp",
        "comm",
        "look",
        "locate",
        "mdfind",
        "bat",
        "exa",
        "eza",
        "test",
        "[",
        "printf",
    }
)
# Programs whose run is a check or build, not a read and not a change.
VERIFY_PROGRAMS = frozenset(
    {
        "pytest",
        "python",
        "python3",
        "uv",
        "uvx",
        "pip",
        "pip3",
        "npm",
        "npx",
        "pnpm",
        "yarn",
        "bun",
        "deno",
        "node",
        "cargo",
        "go",
        "make",
        "cmake",
        "mvn",
        "gradle",
        "gradlew",
        "tox",
        "nox",
        "ruff",
        "black",
        "isort",
        "flake8",
        "pylint",
        "mypy",
        "pyright",
        "ty",
        "pre-commit",
        "eslint",
        "prettier",
        "tsc",
        "jest",
        "vitest",
        "mocha",
        "rspec",
        "rake",
        "bundle",
        "dotnet",
        "swift",
        "xcodebuild",
        "gcc",
        "clang",
        "javac",
        "java",
        "docker",
        "kubectl",
        "terraform",
        "curl",
        "wget",
        "http",
        "httpie",
        "bash",
        "sh",
        "zsh",
        "poetry",
        "pipx",
        "conda",
        "pdm",
        "hatch",
        "coverage",
        "hypothesis",
        "behave",
        "playwright",
        "cypress",
        "ab",
        "wrk",
    }
)
FILE_MODIFYING_PROGRAMS = frozenset(
    {
        "rm",
        "mv",
        "cp",
        "mkdir",
        "rmdir",
        "touch",
        "chmod",
        "chown",
        "chgrp",
        "ln",
        "truncate",
        "patch",
        "tee",
        "dd",
        "install",
        "rsync",
        "unzip",
        "tar",
        "gunzip",
        "gzip",
        "zip",
        "xz",
        "bzip2",
        "shred",
        "sqlite3",
    }
)
READ_ONLY_GIT = frozenset(
    {
        "status",
        "log",
        "diff",
        "show",
        "blame",
        "ls-files",
        "ls-tree",
        "rev-parse",
        "describe",
        "remote",
        "config",
        "cat-file",
        "shortlog",
        "reflog",
        "grep",
        "name-rev",
        "merge-base",
        "rev-list",
        "count-objects",
        "for-each-ref",
        "var",
        "version",
        "help",
        "stash list",
        "worktree list",
        "branch --list",
        "tag --list",
        "diff-tree",
        "diff-index",
        "check-ignore",
    }
)
# Filters that only narrow what a previous command printed.
FILTER_PROGRAMS = frozenset(
    {
        "grep",
        "rg",
        "egrep",
        "fgrep",
        "head",
        "tail",
        "sort",
        "uniq",
        "wc",
        "cut",
        "tr",
        "awk",
        "sed",
        "less",
        "more",
        "jq",
        "yq",
        "column",
        "tac",
        "nl",
        "cat",
        "xargs",
        "tee",
        "fold",
    }
)
NO_PHASE_PROGRAMS = frozenset(
    {
        "cd",
        "echo",
        "export",
        "unset",
        "sleep",
        "source",
        ".",
        "set",
        "alias",
        "unalias",
        "exit",
        "return",
        "true",
        "false",
        ":",
        "wait",
        "pushd",
        "popd",
        "shift",
        "trap",
        "ulimit",
        "umask",
        "eval",
        "exec",
        "clear",
        "history",
        "read",
        "declare",
        "local",
        "typeset",
        "readonly",
        "let",
        "time",
        "nohup",
        "sudo",
        "doas",
        "command",
        "builtin",
        "if",
        "then",
        "else",
        "elif",
        "fi",
        "for",
        "while",
        "until",
        "do",
        "done",
        "case",
        "esac",
        "in",
        "select",
        "function",
        "{",
        "}",
        "!",
        "&&",
        "||",
    }
)
SHELL_WRAPPERS = frozenset({"sudo", "doas", "env", "time", "nohup", "command", "builtin", "exec"})

_SCRATCH_PREFIXES = ("/dev/null", "/tmp", "/var/folders", "/private/tmp", "/private/var/folders")
_REDIRECT_RE = re.compile(r"(?<![<>])(?:\d?>>?|&>|>\|)\s*([^\s;&|]+)")
_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)(\w+)\1")
_PY_WRITE_RE = re.compile(
    r"open\([^)]*,\s*['\"][wax]|\.write_text\(|\.write_bytes\(|\bshutil\.(?:copy|move|rmtree)|"
    r"\bos\.(?:remove|unlink|rename|replace|makedirs|mkdir|rmdir)\(|\.unlink\(|\.rename\(|"
    r"\.replace\(|\.touch\(|\.mkdir\(|\bjson\.dump\(|\bpickle\.dump\(|\byaml\.(?:safe_)?dump\("
)
_PY_SUBPROCESS_RE = re.compile(r"\bsubprocess\b|\bos\.system\(|\bos\.exec|\bos\.popen\(|\bpty\.")
_PY_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+(?:\s*,\s*[\w.]+)*))", re.MULTILINE
)
_STDLIB = frozenset(sys.stdlib_module_names) | {"__future__", "cognee_scratch"}


@dataclass(slots=True)
class PhaseBlock:
    """Consecutive traces of one phase: ``[start, end)`` indices into the unit."""

    phase: str
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


# -- Shell parsing ------------------------------------------------------------


def split_shell_command(command: str) -> list[tuple[str, bool]]:
    """Split a command on unquoted ``&&``, ``||``, ``;``, ``|`` and newlines.

    Returns ``(segment, after_pipe)`` pairs; a here-document body stays with its
    command, so an inline Python script is one segment.
    """
    text = str(command or "")
    segments: list[tuple[str, bool]] = []
    current: list[str] = []
    quote = ""
    after_pipe = False
    index = 0
    heredoc_terminators: list[str] = []

    def flush(next_after_pipe: bool) -> None:
        nonlocal current, after_pipe
        segment = "".join(current).strip()
        if segment:
            segments.append((segment, after_pipe))
        current = []
        after_pipe = next_after_pipe

    while index < len(text):
        char = text[index]
        if quote:
            current.append(char)
            if char == "\\" and quote == '"' and index + 1 < len(text):
                current.append(text[index + 1])
                index += 2
                continue
            if char == quote:
                quote = ""
            index += 1
            continue
        if char == "\\" and index + 1 < len(text):
            current.append(text[index : index + 2])
            index += 2
            continue
        if char in ("'", '"'):
            quote = char
            current.append(char)
            index += 1
            continue
        if char == "<" and text.startswith("<<", index):
            match = _HEREDOC_RE.match(text, index)
            if match:
                heredoc_terminators.append(match.group(2))
                current.append(match.group(0))
                index = match.end()
                continue
        if char == "\n":
            if heredoc_terminators:
                # Consume the body up to and including the terminator line.
                terminator = heredoc_terminators.pop(0)
                end = index + 1
                while end <= len(text):
                    line_end = text.find("\n", end)
                    line_end = len(text) if line_end == -1 else line_end
                    line = text[end:line_end].strip()
                    end = line_end + 1
                    if line == terminator:
                        break
                current.append(text[index:end])
                index = end
                if not heredoc_terminators:
                    flush(False)  # the terminator line ends the command
                continue
            flush(False)
            index += 1
            continue
        if text.startswith("&&", index) or text.startswith("||", index):
            flush(False)
            index += 2
            continue
        if char == ";":
            flush(False)
            index += 1
            continue
        if char == "|":
            flush(True)
            index += 1
            continue
        current.append(char)
        index += 1
    flush(False)
    return segments


def _words(segment: str) -> list[str]:
    words = segment.split()
    while words and (
        words[0] in SHELL_WRAPPERS or ("=" in words[0] and not words[0].startswith("="))
    ):
        words = words[1:]
    return words


def program_of(segment: str) -> str:
    """The program a segment runs, after wrappers and ``VAR=value`` assignments."""
    words = _words(segment)
    return words[0].rsplit("/", 1)[-1] if words else ""


def is_scratch_path(path: str) -> bool:
    text = path.strip().strip("'\"")
    return text.startswith("$") or any(
        text == prefix or text.startswith(prefix + "/") for prefix in _SCRATCH_PREFIXES
    )


def _redirect_targets(segment: str) -> list[str]:
    # The here-document body is not part of the command line.
    head = segment.split("\n", 1)[0]
    return [target for target in _REDIRECT_RE.findall(head) if not target.startswith("&")]


def _writes_outside_scratch(segment: str) -> bool:
    targets = _redirect_targets(segment)
    return any(not is_scratch_path(target) for target in targets)


def _file_arguments(words: list[str]) -> list[str]:
    return [word for word in words[1:] if not word.startswith("-")]


_LISTING_GIT = frozenset({"branch", "tag", "remote", "worktree"})
_LISTING_FLAGS = frozenset({"-l", "--list", "-a", "-r", "-v", "-vv", "--all", "list", "show"})


def _git_phase(words: list[str]) -> str | None:
    """Read-only git is explore; anything that changes the tree, index or refs is change."""
    args = words[1:]
    positional = [word for word in args if not word.startswith("-")]
    flags = [word for word in args if word.startswith("-")]
    if not positional:
        return None
    subcommand = positional[0]
    if subcommand in READ_ONLY_GIT:
        return EXPLORE
    rest = positional[1:]
    if subcommand == "stash":
        return EXPLORE if rest and rest[0] in ("list", "show") else CHANGE
    if subcommand in _LISTING_GIT:
        if not rest and all(flag in _LISTING_FLAGS for flag in flags):
            return EXPLORE
        if rest and rest[0] == "list" and subcommand == "worktree":
            return EXPLORE
        return CHANGE
    return CHANGE


def python_script_phase(script: str) -> str:
    """Phase of inline Python: writes a file, runs or imports beyond stdlib, or only reads."""
    if _PY_WRITE_RE.search(script):
        return CHANGE
    if _PY_SUBPROCESS_RE.search(script):
        return VERIFY
    for match in _PY_IMPORT_RE.finditer(script):
        names = match.group(1) or match.group(2) or ""
        for name in names.split(","):
            root = name.strip().split(".", 1)[0]
            if root and root not in _STDLIB:
                return VERIFY
    return EXPLORE


def _python_phase(segment: str, words: list[str]) -> str:
    args = words[1:]
    if not args:
        return VERIFY  # the REPL or stdin without a here-document: a run
    if args[0] == "-m":
        return VERIFY
    if args[0] == "-c":
        return python_script_phase(segment.split("-c", 1)[1].strip().strip("'\""))
    if args[0] == "-" or "<<" in segment:
        body = segment.split("\n", 1)[1] if "\n" in segment else ""
        return python_script_phase(body)
    if args[0].startswith("-"):
        return VERIFY
    return VERIFY  # python <script file>


def segment_phase(segment: str, after_pipe: bool = False) -> str | None:
    """The phase of one shell segment, or None for setup and control commands."""
    words = _words(segment)
    if not words:
        return None
    program = words[0].rsplit("/", 1)[-1]
    if program in NO_PHASE_PROGRAMS:
        if program == "echo" and _writes_outside_scratch(segment):
            return CHANGE
        return None
    if _writes_outside_scratch(segment):
        return CHANGE
    if (
        after_pipe
        and program in FILTER_PROGRAMS
        and not (
            program == "sed"
            and any(word in ("-i", "--in-place") or word.startswith("-i") for word in words)
        )
    ):
        return None  # only filters what the previous command printed
    if program in ("python", "python3", "python2", "pypy", "pypy3") or program.startswith(
        "python3."
    ):
        return _python_phase(segment, words)
    if program == "git":
        return _git_phase(words)
    if program == "sed":
        in_place = any(word in ("-i", "--in-place") or word.startswith("-i") for word in words[1:])
        if in_place:
            files = [word for word in _file_arguments(words) if "/" in word or "." in word]
            return CHANGE if not files or any(not is_scratch_path(f) for f in files) else None
        return EXPLORE
    if program in FILE_MODIFYING_PROGRAMS:
        files = _file_arguments(words)
        if files and all(is_scratch_path(f) for f in files):
            return None
        return CHANGE
    if program in READ_ONLY_PROGRAMS:
        return EXPLORE
    if program in VERIFY_PROGRAMS:
        return VERIFY
    return VERIFY  # any other program: the project's own CLI, a custom script


def command_phase(command: str) -> str | None:
    """The strongest phase of a chained shell command (change > verify > explore)."""
    strongest: str | None = None
    for segment, after_pipe in split_shell_command(command):
        phase = segment_phase(segment, after_pipe)
        if phase and (strongest is None or PHASE_RANK[phase] > PHASE_RANK[strongest]):
            strongest = phase
    return strongest


def classify_trace(entry) -> str | None:
    """The phase of one trace by its tool and input, or None (inherits the previous)."""
    tool = entry.origin_function
    if tool in EXPLORE_TOOLS:
        return EXPLORE
    if tool in CHANGE_TOOLS:
        return CHANGE
    if tool in SHELL_TOOLS:
        params = entry.method_params if isinstance(entry.method_params, dict) else {}
        command = params.get("command")
        return command_phase(command) if isinstance(command, str) else None
    return None


def assign_phases(traces: list) -> list[str]:
    """One phase per trace; a trace without one takes the previous trace's."""
    phases: list[str] = []
    previous = EXPLORE
    for entry in traces:
        phase = classify_trace(entry) or previous
        phases.append(phase)
        previous = phase
    return phases


# -- Blocks and merging ---------------------------------------------------------


def group_blocks(phases: list[str]) -> list[PhaseBlock]:
    blocks: list[PhaseBlock] = []
    for index, phase in enumerate(phases):
        if blocks and blocks[-1].phase == phase:
            blocks[-1].end = index + 1
        else:
            blocks.append(PhaseBlock(phase=phase, start=index, end=index + 1))
    return blocks


def max_parts_for(trace_count: int) -> int:
    if trace_count <= SMALL_UNIT_TRACES:
        return MAX_PARTS
    return ceil(trace_count / TRACES_PER_PART)


def _join_adjacent(blocks: list[PhaseBlock]) -> list[PhaseBlock]:
    joined: list[PhaseBlock] = []
    for block in blocks:
        if joined and joined[-1].phase == block.phase:
            joined[-1].end = block.end
        else:
            joined.append(PhaseBlock(block.phase, block.start, block.end))
    return joined


def merge_blocks(blocks: list[PhaseBlock], trace_count: int) -> list[PhaseBlock]:
    """Merge one step at a time until every part has MIN_PART_TRACES and parts fit the cap.

    Each step: take the smallest block (the earliest on ties); merge it into its smaller
    neighbour (the previous on ties); the merged block takes the phase of the larger side
    (the neighbour's on ties); join adjacent blocks that now share a phase. A unit with
    fewer than MIN_PART_TRACES traces stays one part.
    """
    blocks = _join_adjacent(blocks)
    if trace_count < MIN_PART_TRACES:
        return [PhaseBlock(blocks[0].phase, 0, trace_count)] if blocks else []
    limit = max_parts_for(trace_count)
    while len(blocks) > 1 and (
        len(blocks) > limit or any(block.size < MIN_PART_TRACES for block in blocks)
    ):
        smallest = min(range(len(blocks)), key=lambda i: (blocks[i].size, i))
        previous = blocks[smallest - 1] if smallest > 0 else None
        following = blocks[smallest + 1] if smallest + 1 < len(blocks) else None
        if previous is not None and (following is None or previous.size <= following.size):
            neighbour_index = smallest - 1
        else:
            neighbour_index = smallest + 1
        block = blocks[smallest]
        neighbour = blocks[neighbour_index]
        phase = neighbour.phase if neighbour.size >= block.size else block.phase
        merged = PhaseBlock(phase, min(block.start, neighbour.start), max(block.end, neighbour.end))
        low, high = sorted((smallest, neighbour_index))
        blocks = _join_adjacent(blocks[:low] + [merged] + blocks[high + 1 :])
    return blocks


def split_into_parts(traces: list) -> list[PhaseBlock]:
    """The unit's traces as ordered phase parts."""
    if not traces:
        return []
    return merge_blocks(group_blocks(assign_phases(traces)), len(traces))
