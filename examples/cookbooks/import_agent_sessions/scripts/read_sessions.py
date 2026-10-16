"""Read your Claude Code and Codex session history and write one Markdown file per session.

Claude Code keeps every session in ~/.claude/projects/<project>/<session id>.jsonl, and Codex
in ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl. Both are JSON lines full of tool calls,
system reminders and injected instructions. This keeps only what was said: each prompt you
typed and the agent's final answer to it. Text that looks like an API key or token is
replaced with [redacted] before anything is written.

The files go to sessions/<agent>/ next to this cookbook (git-ignored), named after the
session's start date and id, so a rerun rewrites the same files and cognee skips the ones
whose content did not change. Nothing here calls an LLM or cognee.

Run alone: uv run python examples/cookbooks/import_agent_sessions/scripts/read_sessions.py \
    [--agents claude,codex] [--limit 20] [--project cognee] [--since 2026-09-01]
"""

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

OUT = Path(__file__).parent.parent / "sessions"

AGENTS = {"claude": "Claude Code", "codex": "Codex"}

MAX_PROMPT_CHARS = 2_000
MAX_ANSWER_CHARS = 3_000
MAX_SESSION_CHARS = 30_000
RUNNING_SECONDS = 120  # a file written to this recently is a session still running

# Blocks the harness puts into a user turn; they are not what the user typed. The bash-*
# tags hold a `! command` run in Claude Code and its output, which can print secrets.
NOISE_TAGS = (
    "system-reminder",
    "ide_opened_file",
    "ide_selection",
    "command-name",
    "command-message",
    "command-args",
    "local-command-stdout",
    "local-command-stderr",
    "local-command-caveat",
    "task-notification",
    "bash-input",
    "bash-stdout",
    "bash-stderr",
    "environment_context",
    "user_instructions",
    "permissions instructions",
    "turn_aborted",
)
NOISE = re.compile(
    "|".join(rf"<{tag}[^>]*>.*?</{tag.split()[0]}>" for tag in NOISE_TAGS)
    + r"|\[Request interrupted by user[^\]]*\]",  # Claude Code's marker when you press Esc
    re.DOTALL,
)

SECRET = re.compile(
    r"sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}"
    r"|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
    r"|AKIA[0-9A-Z]{16}|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|(?i:bearer)\s+[A-Za-z0-9._~+/-]{20,}=*"
)


@dataclass
class Session:
    agent: str  # "claude" or "codex"
    id: str
    started: datetime
    cwd: str = ""
    branch: str = ""
    title: str = ""
    turns: list[tuple[str, str]] = field(default_factory=list)  # (prompt, answer)


def clean(text: str, limit: int) -> str:
    text = SECRET.sub("[redacted]", NOISE.sub("", text)).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " […]"


def parse_time(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone()
    except (AttributeError, ValueError):
        return None


def json_lines(path: Path):
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # a session still being written can end in a half line


def add_turn(session: Session, prompt: str, answer: str) -> None:
    prompt, answer = clean(prompt, MAX_PROMPT_CHARS), clean(answer, MAX_ANSWER_CHARS)
    if prompt:
        session.turns.append((prompt, answer))


def read_claude_session(path: Path) -> Session | None:
    """One Claude Code transcript: user prompts, and the last text the assistant wrote
    before the next prompt (its answer). Tool calls, tool results and sidechains
    (subagents) are skipped."""
    session, prompt, answer = None, None, ""
    for record in json_lines(path):
        kind = record.get("type")
        if kind == "ai-title" and session:
            session.title = clean(record.get("aiTitle") or "", 200)
        if kind not in ("user", "assistant") or record.get("isSidechain"):
            continue
        if session is None:
            started = parse_time(record.get("timestamp"))
            if not started:
                continue
            session = Session(
                "claude",
                record.get("sessionId") or path.stem,
                started,
                record.get("cwd", ""),
                record.get("gitBranch", ""),
            )
        content = (record.get("message") or {}).get("content")
        if isinstance(content, str):
            blocks = [{"type": "text", "text": content}]
        else:
            blocks = [b for b in content or [] if isinstance(b, dict)]
        texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]

        if kind == "user":
            # isCompactSummary: the summary /compact writes in place of the earlier turns
            if record.get("isMeta") or record.get("isCompactSummary"):
                continue
            if any(b.get("type") == "tool_result" for b in blocks):
                continue
            text = clean("\n".join(texts), MAX_PROMPT_CHARS)
            if not text:
                continue  # only harness noise, like a /clear
            if prompt is not None:
                add_turn(session, prompt, answer)
            prompt, answer = text, ""
        elif texts and prompt is not None:
            answer = "\n".join(texts)
    if session and prompt is not None:
        add_turn(session, prompt, answer)
    return session if session and session.turns else None


def codex_prompt(text: str) -> str:
    """What the user typed in one Codex input part. The injected AGENTS.md is dropped, and so
    are the open tabs the IDE extension puts before the request; the environment and abort
    blocks are NOISE tags."""
    if text.lstrip().startswith("# AGENTS.md"):
        return ""
    if text.lstrip().startswith("# Context from my IDE setup"):
        return text.partition("## My request for Codex:")[2]
    return text


def read_codex_session(path: Path) -> Session | None:
    """One Codex rollout: user prompts (without the injected AGENTS.md and environment
    blocks) and the final answer of each turn."""
    session, prompt, answer = None, None, ""
    for record in json_lines(path):
        payload = record.get("payload") or {}
        if record.get("type") == "session_meta":
            started = parse_time(payload.get("timestamp") or record.get("timestamp"))
            if started:
                session = Session(
                    "codex",
                    payload.get("id") or path.stem,
                    started,
                    payload.get("cwd", ""),
                    (payload.get("git") or {}).get("branch", "") or "",
                )
            continue
        if session is None or payload.get("type") != "message":
            continue
        role, parts = payload.get("role"), payload.get("content") or []
        if role == "user":
            texts = [
                codex_prompt(p.get("text", "")) for p in parts if p.get("type") == "input_text"
            ]
            text = clean("\n".join(texts), MAX_PROMPT_CHARS)
            if not text:
                continue
            if prompt is not None:
                add_turn(session, prompt, answer)
            prompt, answer = text, ""
        elif role == "assistant" and prompt is not None:
            text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "output_text")
            # a turn's commentary comes first and its final_answer last; keep the last
            if text and (payload.get("phase") != "commentary" or not answer):
                answer = text
    if session and prompt is not None:
        add_turn(session, prompt, answer)
    return session if session and session.turns else None


def history_files(agent: str) -> list[Path]:
    if agent == "claude":
        root = Path(os.environ.get("CLAUDE_CONFIG_DIR", "~/.claude")).expanduser() / "projects"
        return sorted(root.glob("*/*.jsonl"))  # subagent transcripts sit one level deeper
    root = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser() / "sessions"
    return sorted(root.glob("*/*/*/rollout-*.jsonl"))


def read_sessions(
    agent: str, limit: int = 20, project: str | None = None, since: str | None = None
) -> list[Session]:
    """The newest `limit` finished sessions of one agent that have at least one turn."""
    reader = read_claude_session if agent == "claude" else read_codex_session
    cutoff = datetime.fromisoformat(since).astimezone() if since else None
    files = sorted(history_files(agent), key=lambda p: p.stat().st_mtime, reverse=True)
    running = datetime.now().astimezone() - timedelta(seconds=RUNNING_SECONDS)
    sessions = []
    for path in files:
        if datetime.fromtimestamp(path.stat().st_mtime).astimezone() > running:
            continue  # skip the session running this import: it would only describe itself
        if cutoff and datetime.fromtimestamp(path.stat().st_mtime).astimezone() < cutoff:
            break  # files are newest first, so every later one is older too
        session = reader(path)
        if not session or (project and project.lower() not in session.cwd.lower()):
            continue
        if cutoff and session.started < cutoff:
            continue
        sessions.append(session)
        if len(sessions) >= limit:
            break
    return sessions


def render(session: Session) -> str:
    name = AGENTS[session.agent]
    title = session.title or session.turns[0][0].splitlines()[0][:80]
    lines = [
        f"# {name} session: {title}",
        "",
        f"- Agent: {name}",
        f"- Project: {session.cwd or 'unknown'}"
        + (f" (branch {session.branch})" if session.branch else ""),
        f"- Started: {session.started:%Y-%m-%d %H:%M}",
        f"- Session id: {session.agent}_{session.id}",
    ]
    for number, (prompt, answer) in enumerate(session.turns, 1):
        lines += ["", f"## Turn {number}", "", f"**User:** {prompt}", ""]
        lines.append(f"**{name}:** {answer or '(no answer recorded)'}")
        if sum(len(line) for line in lines) > MAX_SESSION_CHARS:
            lines += ["", f"(The session continues for {len(session.turns) - number} more turns.)"]
            break
    return "\n".join(lines) + "\n"


def write_sessions(sessions: list[Session], agent: str, out: Path = OUT) -> list[Path]:
    """Write one Markdown file per session to <out>/<agent>/, replacing an earlier run's."""
    folder = out / agent
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    paths = []
    for session in sessions:
        path = folder / f"{session.started:%Y-%m-%d}_{session.id}.md"
        path.write_text(render(session), encoding="utf-8")
        paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agents", default="claude,codex", help="claude, codex or both")
    parser.add_argument("--limit", type=int, default=20, help="newest sessions per agent")
    parser.add_argument("--project", help="only sessions whose folder contains this text")
    parser.add_argument("--since", help="only sessions started on or after this date")
    args = parser.parse_args()
    for agent in args.agents.split(","):
        sessions = read_sessions(agent, args.limit, args.project, args.since)
        paths = write_sessions(sessions, agent)
        turns = sum(len(s.turns) for s in sessions)
        print(
            f"[read_sessions] {AGENTS[agent]}: {len(paths)} sessions, {turns} turns -> {OUT / agent}"
        )
