"""Terminal UI toolkit for the cognee CLI.

One place decides how the CLI looks and degrades:

- ``TermCaps``     — a single capability probe (TTY, color, width, CI,
                     progress mode) shared by every renderer, so the welcome
                     screen, progress, errors, and future hints all agree.
- ``Style``        — 16-color ANSI roles (user themes keep control).
- ``StageBoard``   — pipeline progress. On an interactive terminal it renders
                     a live stage checklist; everywhere else (pipes, CI,
                     NO_COLOR+dumb terminals, ``COGNEE_PROGRESS=plain``) it
                     falls back to append-only lines with a periodic
                     heartbeat, so output never turns into mangled redraws.
- ``error_block``  — the calm error anatomy: what happened, why, the exact
                     next command, one docs link.

All decoration goes to stderr; stdout stays reserved for answers so
``cognee-cli search ... | jq`` works while the terminal stays alive.
"""

import logging
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_CI_ENV_VARS = (
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "TRAVIS",
    "CIRCLECI",
    "BUILDKITE",
    "CIRRUS_CI",
    "TF_BUILD",
    "JENKINS_URL",
)


def _is_ci() -> bool:
    return any(os.environ.get(var) for var in _CI_ENV_VARS)


def _supports_unicode(stream) -> bool:
    encoding = getattr(stream, "encoding", None) or ""
    try:
        "✔⠋○".encode(encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


@dataclass(frozen=True)
class TermCaps:
    """What the current terminal (or pipe) can actually display."""

    stdout_tty: bool
    stderr_tty: bool
    color: bool
    unicode: bool
    ci: bool
    width: int
    # "live" = redrawing checklist, "plain" = append-only lines, "off" = nothing
    progress: str

    @property
    def interactive(self) -> bool:
        return self.stderr_tty and not self.ci


def detect_caps(quiet: bool = False) -> TermCaps:
    stdout_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    stderr_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    ci = _is_ci()

    no_color = bool(os.environ.get("NO_COLOR"))
    dumb = os.environ.get("TERM") == "dumb"
    color = stderr_tty and not no_color and not dumb

    try:
        width = shutil.get_terminal_size().columns
    except Exception:
        width = 80

    progress_env = os.environ.get("COGNEE_PROGRESS", "auto").lower()
    if quiet or progress_env == "off":
        progress = "off"
    elif progress_env == "plain":
        progress = "plain"
    elif progress_env == "live":
        progress = "live" if stderr_tty else "plain"
    else:  # auto
        if stderr_tty and not ci and not dumb:
            progress = "live"
        else:
            progress = "plain"

    return TermCaps(
        stdout_tty=stdout_tty,
        stderr_tty=stderr_tty,
        color=color,
        unicode=_supports_unicode(sys.stderr),
        ci=ci,
        width=max(40, width),
        progress=progress,
    )


class Glyphs:
    """One symbol per state, with an ASCII floor for terminals that need it."""

    def __init__(self, unicode_ok: bool) -> None:
        if unicode_ok:
            self.ok = "✔"
            self.fail = "✗"
            self.warn = "!"
            self.pending = "○"
            self.bullet = "•"
            self.sep = "·"
            self.spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        else:
            self.ok = "[ok]"
            self.fail = "[x]"
            self.warn = "[!]"
            self.pending = "[ ]"
            self.bullet = "*"
            self.sep = "-"
            self.spinner = ["|", "/", "-", "\\"]


class Style:
    """16-color ANSI roles. Colors enhance meaning, never carry it alone."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _c(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, t: str) -> str:
        return self._c("1", t)

    def dim(self, t: str) -> str:
        return self._c("2", t)

    def red(self, t: str) -> str:
        return self._c("31", t)

    def green(self, t: str) -> str:
        return self._c("32", t)

    def yellow(self, t: str) -> str:
        return self._c("33", t)

    def cyan(self, t: str) -> str:
        return self._c("36", t)


def format_duration(seconds: float) -> str:
    """Durations the way uv/cargo/buildx print them: 298ms · 2.1s · 1m 42s."""
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _stream_width(fallback: int = 80) -> int:
    """Current width of the stream live renderers draw to (stderr), re-queried
    per frame so a mid-run terminal resize doesn't wrap the redraw region."""
    try:
        return max(40, os.get_terminal_size(sys.stderr.fileno()).columns)
    except (ValueError, OSError, AttributeError):
        return fallback


def _visible_len(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


def _truncate(text: str, width: int) -> str:
    """Trim a (possibly styled) line so it never wraps a live redraw region."""
    if _visible_len(text) <= width:
        return text
    plain = _ANSI_RE.sub("", text)
    return plain[: max(0, width - 1)] + "…"


def error_block(
    title: str,
    why: Optional[str] = None,
    fixes: Optional[Sequence[Tuple[str, str]]] = None,
    footer: Optional[str] = None,
    caps: Optional[TermCaps] = None,
) -> None:
    """The calm error anatomy — what happened, why, and the exact next step.

    fixes are (label, text) pairs, e.g. ("Fix", "export LLM_API_KEY=sk-...").
    Everything goes to stderr; exit codes are the caller's job.
    """
    caps = caps or detect_caps()
    style = Style(caps.color)
    glyphs = Glyphs(caps.unicode)
    out = [f"{style.red(glyphs.fail)} {style.bold(title)}"]
    if why:
        out.append("")
        for line in why.splitlines():
            out.append(f"  {line}")
    if fixes:
        out.append("")
        label_width = max(len(label) for label, _ in fixes)
        for label, text in fixes:
            first, *rest = str(text).splitlines() or [""]
            out.append(f"  {style.dim(label.ljust(label_width))}  {first}")
            for cont in rest:
                out.append(f"  {' ' * label_width}  {cont}")
    if footer:
        out.append("")
        out.append(style.dim(footer))
    sys.stderr.write("\n".join(out) + "\n")


def guide_block(
    title: str,
    commands: Sequence[str],
    lead_in: Optional[str] = None,
    caps: Optional[TermCaps] = None,
) -> None:
    """An empty state that teaches: name what's empty, list the commands that fill it."""
    caps = caps or detect_caps()
    style = Style(caps.color)
    out = [f"{style.bold(title)}"]
    if lead_in:
        out.append(f"{lead_in}")
    out.append("")
    for command in commands:
        out.append(f"  {style.cyan(command)}")
    sys.stderr.write("\n".join(out) + "\n")


def success_line(message: str, caps: Optional[TermCaps] = None) -> None:
    caps = caps or detect_caps()
    style = Style(caps.color)
    glyphs = Glyphs(caps.unicode)
    sys.stderr.write(f"{style.green(glyphs.ok)} {message}\n")


def next_step(command: str, label: str = "Next:", caps: Optional[TermCaps] = None) -> None:
    """The onboarding rail: every success ends in exactly one next command."""
    caps = caps or detect_caps()
    style = Style(caps.color)
    sys.stderr.write(f"  {style.dim(label)} {style.cyan(command)}\n")


@dataclass
class _Stage:
    key: str
    label: str  # imperative, shown on the checklist: "Extract chunks"
    verb: str  # progressive, for plain mode: "Extracting chunks"
    status: str = "pending"  # pending | running | done | failed
    detail: str = ""
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return (self.finished_at or time.monotonic()) - self.started_at


class StageBoard:
    """Live stage checklist on a TTY; append-only rail with heartbeats elsewhere.

    The board renders on stderr only. In live mode a background thread redraws
    a small region ~10x/s (throttled — progress rendering must never compete
    with the work). In plain mode, stage transitions print one line each and a
    heartbeat line every ``heartbeat_secs`` proves liveness during long stages.
    """

    HEARTBEAT_SECS = 20

    def __init__(
        self,
        title: str,
        caps: Optional[TermCaps] = None,
        known_stages: Optional[Sequence[Tuple[str, str, str]]] = None,
    ) -> None:
        self.caps = caps or detect_caps()
        self.style = Style(self.caps.color)
        self.glyphs = Glyphs(self.caps.unicode)
        self.title = title
        self._stages: List[_Stage] = []
        self._by_key: Dict[str, _Stage] = {}
        self._lock = threading.Lock()
        self._ticker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._stopped = False
        self._rendered_lines = 0
        self._spin_index = 0
        self._last_heartbeat = time.monotonic()
        self._started_at: Optional[float] = None
        for key, label, verb in known_stages or []:
            self._add_stage(key, label, verb)

    # -- stage bookkeeping ------------------------------------------------

    def _add_stage(self, key: str, label: str, verb: str) -> _Stage:
        stage = _Stage(key=key, label=label, verb=verb)
        self._stages.append(stage)
        self._by_key[key] = stage
        return stage

    def stage_started(self, key: str, label: Optional[str] = None, verb: Optional[str] = None):
        with self._lock:
            stage = self._by_key.get(key)
            if stage is None:
                stage = self._add_stage(key, label or key, verb or (label or key))
            if stage.status in ("pending", "done"):
                stage.status = "running"
                if stage.started_at is None:
                    stage.started_at = time.monotonic()
                stage.finished_at = None
                if self.caps.progress == "plain":
                    self._plain_line(f"{stage.verb}...")

    def stage_completed(self, key: str, detail: str = ""):
        with self._lock:
            stage = self._by_key.get(key)
            if stage is None:
                return
            if stage.started_at is None:
                stage.started_at = time.monotonic()
            stage.status = "done"
            stage.finished_at = time.monotonic()
            if detail:
                stage.detail = detail
            if self.caps.progress == "plain":
                duration = format_duration(stage.elapsed)
                extra = f" — {stage.detail}" if stage.detail else ""
                self._plain_line(f"{_past_tense(stage.verb)}{extra} ({duration})")

    def stage_failed(self, key: str, detail: str = ""):
        with self._lock:
            stage = self._by_key.get(key)
            if stage is None:
                return
            stage.status = "failed"
            stage.finished_at = time.monotonic()
            if detail:
                stage.detail = detail
            if self.caps.progress == "plain":
                self._plain_line(f"{stage.verb} failed")

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self.caps.progress == "off":
            return
        self._started_at = time.monotonic()
        if self.caps.progress == "plain":
            if self.title:
                self._plain_line(self.title)
        if self.caps.progress in ("live", "plain"):
            self._stop.clear()
            self._ticker = threading.Thread(target=self._tick, daemon=True)
            self._ticker.start()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        if self._ticker is not None:
            self._ticker.join(timeout=1)
            self._ticker = None
        if self.caps.progress == "live":
            with self._lock:
                self._redraw(final=True)

    def log_line(self, text: str) -> None:
        """Print a normal line without corrupting the live redraw region:
        erase the board, emit the line into scrollback, repaint the board."""
        with self._lock:
            stream = sys.stderr
            if self.caps.progress == "live" and self._rendered_lines:
                stream.write(f"\033[{self._rendered_lines}A\033[0J")
                self._rendered_lines = 0
                stream.write(text.rstrip("\n") + "\n")
                self._redraw()
            else:
                stream.write(text.rstrip("\n") + "\n")
                stream.flush()

    def finish(self, summary: str, next_command: Optional[str] = None) -> None:
        """Freeze the board and land the durable one-line summary."""
        self.stop()
        stream = sys.stderr
        if self.caps.progress != "off":
            stream.write("\n")
        stream.write(f"{self.style.green(self.glyphs.ok)} {summary}\n")
        if next_command:
            stream.write(f"  {self.style.dim('Next:')} {self.style.cyan(next_command)}\n")

    def fail(self) -> None:
        """Stop rendering, leaving completed stages visible as receipts."""
        with self._lock:
            for stage in self._stages:
                if stage.status == "running":
                    stage.status = "failed"
                    stage.finished_at = time.monotonic()
        self.stop()
        if self.caps.progress != "off":
            sys.stderr.write("\n")

    # -- rendering ---------------------------------------------------------

    def _plain_line(self, text: str) -> None:
        sys.stderr.write(text + "\n")
        sys.stderr.flush()

    def _tick(self) -> None:
        interval = 0.1 if self.caps.progress == "live" else 1.0
        while not self._stop.wait(interval):
            with self._lock:
                if self.caps.progress == "live":
                    self._spin_index = (self._spin_index + 1) % len(self.glyphs.spinner)
                    self._redraw()
                else:
                    self._heartbeat()

    def _heartbeat(self) -> None:
        """terraform's rule: never silent longer than ~20s, in any mode."""
        now = time.monotonic()
        if now - self._last_heartbeat < self.HEARTBEAT_SECS:
            return
        running = [s for s in self._stages if s.status == "running"]
        if not running:
            return
        stage = running[0]
        elapsed = format_duration(stage.elapsed)
        detail = f" {stage.detail}" if stage.detail else ""
        self._plain_line(f"Still {stage.verb.lower()}... [{elapsed} elapsed]{detail}")
        self._last_heartbeat = now

    def _stage_line(self, stage: _Stage) -> str:
        glyphs, style = self.glyphs, self.style
        if stage.status == "done":
            mark = style.green(glyphs.ok)
        elif stage.status == "running":
            mark = style.yellow(glyphs.spinner[self._spin_index])
        elif stage.status == "failed":
            mark = style.red(glyphs.fail)
        else:
            mark = style.dim(glyphs.pending)

        label = stage.label
        if stage.status == "pending":
            return f"  {mark} {style.dim(label)}"

        label_col = max(26, max((len(s.label) for s in self._stages), default=26))
        padded = label.ljust(label_col)
        detail = stage.detail or ""
        elapsed = format_duration(stage.elapsed) if stage.started_at else ""
        line = f"  {mark} {padded}"
        if detail:
            line += f" {style.dim(detail)}" if stage.status == "done" else f" {detail}"
        if elapsed:
            line += f"  {style.dim(elapsed)}" if stage.status == "running" else f"  {elapsed}"
        return line

    def _redraw(self, final: bool = False) -> None:
        stream = sys.stderr
        lines = []
        if self.title:
            lines.append(self.style.bold(self.title))
        for stage in self._stages:
            lines.append(self._stage_line(stage))

        width = _stream_width(self.caps.width)
        rendered = [_truncate(line, width - 1) for line in lines]

        if self._rendered_lines:
            stream.write(f"\033[{self._rendered_lines}A")
        for line in rendered:
            stream.write("\033[2K" + line + "\n")
        # If the board shrank (it never should), clear leftovers defensively.
        extra = self._rendered_lines - len(rendered)
        if extra > 0:
            for _ in range(extra):
                stream.write("\033[2K\n")
            stream.write(f"\033[{extra}A")
        stream.flush()
        self._rendered_lines = len(rendered)


def _past_tense(verb_phrase: str) -> str:
    """'Extracting chunks' -> 'Extracted chunks' — covers the verbs we use."""
    specials = {
        "Storing": "Stored",
        "Building": "Built",
        "Writing": "Wrote",
        "Running": "Ran",
        "Classifying": "Classified",
    }
    word, _, rest = verb_phrase.partition(" ")
    if word in specials:
        past = specials[word]
    elif word.endswith("ing"):
        stem = word[:-3]
        if stem.endswith("y"):
            past = stem[:-1] + "ied"
        else:
            past = stem + "ed"
    else:
        return verb_phrase
    return f"{past} {rest}".strip()


class spinner_line:
    """A single self-erasing status line for short waits (search, add).

    TTY: dim spinner + text, redrawn in place, erased on exit so only the
    command's real output remains. Non-TTY/plain: prints the text once (with
    no animation); off: silent. Always stderr.
    """

    def __init__(self, text: str, caps: Optional[TermCaps] = None) -> None:
        self.caps = caps or detect_caps()
        self.style = Style(self.caps.color)
        self.glyphs = Glyphs(self.caps.unicode)
        self.text = text
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_at = 0.0

    def __enter__(self) -> "spinner_line":
        self._started_at = time.monotonic()
        if self.caps.progress == "live":
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        elif self.caps.progress == "plain":
            sys.stderr.write(f"{self.text}...\n")
            sys.stderr.flush()
        return self

    def _spin(self) -> None:
        index = 0
        while not self._stop.wait(0.1):
            index = (index + 1) % len(self.glyphs.spinner)
            elapsed = format_duration(time.monotonic() - self._started_at)
            line = self.style.dim(
                f"{self.glyphs.spinner[index]} {self.text} {self.glyphs.sep} {elapsed}"
            )
            sys.stderr.write("\r\033[2K" + _truncate(line, _stream_width(self.caps.width) - 1))
            sys.stderr.flush()

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            sys.stderr.write("\r\033[2K")
            sys.stderr.flush()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started_at


# --- pipeline event wiring --------------------------------------------------

# executable name -> (stage key, checklist label, progressive verb)
COGNIFY_STAGES: List[Tuple[str, str, str]] = [
    ("classify_documents", "Classify documents", "Classifying documents"),
    ("extract_chunks_from_documents", "Extract chunks", "Extracting chunks"),
    ("extract_graph_and_summarize", "Extract knowledge graph", "Extracting knowledge graph"),
    ("add_data_points", "Store graph + embeddings", "Storing graph + embeddings"),
    ("extract_dlt_fk_edges", "Link table references", "Linking table references"),
]

_HIDDEN_TASKS = {"check_permissions_on_dataset"}

_TASK_EVENT_RE = re.compile(r"task (started|completed): `([A-Za-z0-9_]+)`")


class _TaskEventHandler(logging.Handler):
    """Turns the pipeline runner's own log events into StageBoard updates.

    The task runner already logs "<type> task started/completed: `name`" for
    every task — the checklist is driven by real events, never by guesses.
    """

    def __init__(self, board: StageBoard) -> None:
        super().__init__(level=logging.INFO)
        self.board = board

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.msg
            if isinstance(message, dict):  # structlog wrap_for_formatter
                message = str(message.get("event", ""))
            else:
                message = record.getMessage()
            match = _TASK_EVENT_RE.search(message)
            if not match:
                return
            action, task_name = match.groups()
            if task_name in _HIDDEN_TASKS:
                return
            if action == "started":
                label = task_name.replace("_", " ").capitalize()
                self.board.stage_started(task_name, label=label, verb=f"Running {label.lower()}")
            else:
                self.board.stage_completed(task_name)
        except Exception:  # never let progress rendering break the pipeline
            pass


class pipeline_progress:
    """Context manager wiring a StageBoard to the pipeline runner's events.

    While the live board is on screen, the console log handler (warnings and
    errors that still reach the terminal) is routed through the board so an
    interleaved log line lands in scrollback above the board instead of being
    overwritten by the next redraw.
    """

    def __init__(
        self,
        title: str,
        known_stages: Sequence[Tuple[str, str, str]] = (),
        caps: Optional[TermCaps] = None,
    ) -> None:
        self.board = StageBoard(title, caps=caps, known_stages=known_stages)
        self._handler = _TaskEventHandler(self.board)
        self._logger = logging.getLogger("run_tasks_base")
        self._console_handler = None
        self._original_emit = None

    def _route_console_through_board(self) -> None:
        if self.board.caps.progress != "live":
            return
        for handler in logging.getLogger().handlers:
            if getattr(handler, "_cognee_console_handler", False):
                board = self.board
                original_format = handler.format

                def routed_emit(record, _fmt=original_format, _board=board):
                    try:
                        _board.log_line(_fmt(record))
                    except Exception:
                        pass

                self._console_handler = handler
                self._original_emit = handler.emit
                handler.emit = routed_emit
                return

    def __enter__(self) -> StageBoard:
        self._logger.addHandler(self._handler)
        self._route_console_through_board()
        self.board.start()
        return self.board

    def __exit__(self, exc_type, exc, tb) -> None:
        self._logger.removeHandler(self._handler)
        if self._console_handler is not None:
            self._console_handler.emit = self._original_emit
            self._console_handler = None
        if exc_type is not None:
            self.board.fail()
        else:
            # The command usually calls board.finish(...) right after; stop()
            # is idempotent, and this guarantees no ticker thread outlives the
            # block even if the caller forgets.
            self.board.stop()
