"""Discovery of day-one seed sources.

Pure discovery: walk the workspace and known coding-agent homes and return a
:class:`SeedPlan` describing what a seed run *would* ingest — agent memory
files (``MEMORY.md``, ``SOUL.md``, …), the workspace README, recent session
transcripts (Claude Code, Codex, Gemini CLI, pi, Aider), and the workspace
codebase. Nothing here ingests or touches cognee storage — the only content
read is each Codex rollout's first metadata line, needed to match sessions to
this workspace; the runner in :mod:`cognee.modules.seeding.seed` decides what
to do with the plan.

Safety posture: only explicitly allowlisted dot-paths are ever picked up
(``.claude/CLAUDE.md`` and the Claude Code project home for *this* workspace),
so secrets in ``.env`` or other hidden files can never enter the plan. Session
transcripts are bounded by count and size.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Agent memory conventions at the workspace root. Covers the Claude Code /
# OpenClaw-style file set; missing files are simply not discovered.
MEMORY_FILE_NAMES = (
    "MEMORY.md",
    "SOUL.md",
    "AGENTS.md",
    "CLAUDE.md",
    "USER.md",
    "IDENTITY.md",
    "TOOLS.md",
)

# Dot-paths that are allowed despite the hidden-file rule.
ALLOWED_DOT_RELATIVE_PATHS = (Path(".claude") / "CLAUDE.md",)

DEFAULT_MAX_SESSION_LOGS = 3
DEFAULT_MAX_SESSION_LOG_BYTES = 5 * 1024 * 1024
DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class SeedPlan:
    """Everything a seed run would ingest, grouped by provenance."""

    workspace: Path
    memory_files: List[Path] = field(default_factory=list)
    readmes: List[Path] = field(default_factory=list)
    session_logs: List[Path] = field(default_factory=list)
    codebase: Optional[Path] = None
    skipped: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.memory_files or self.readmes or self.session_logs or self.codebase)

    def describe(self) -> str:
        """Human-readable plan, used by ``cognee-cli seed --dry-run`` and logs."""
        lines = [f"Seed plan for workspace: {self.workspace}"]

        def section(title: str, paths: List[Path]) -> None:
            lines.append(f"  {title}: {len(paths)}")
            for path in paths:
                lines.append(f"    - {path}")

        section("Agent memory files", self.memory_files)
        section("Workspace docs", self.readmes)
        section("Session logs", self.session_logs)
        if self.codebase:
            lines.append(f"  Codebase: {self.codebase} (resolved as a code project)")
        else:
            lines.append("  Codebase: none")
        for reason in self.skipped:
            lines.append(f"  (skipped) {reason}")
        return "\n".join(lines)


def find_workspace_root(start: Optional[Path] = None) -> Path:
    """Best-effort workspace root: the nearest ancestor that is a code project.

    Falls back to ``start`` itself when no ancestor carries a project marker,
    so seeding still works in a plain notes directory.
    """
    from cognee.tasks.code_graph.code_repo import detect_code_project

    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if detect_code_project(candidate):
            return candidate
    return start


def claude_code_project_dir(workspace: Path, claude_home: Optional[Path] = None) -> Path:
    """The Claude Code per-project directory for this workspace.

    Claude Code slugs the absolute workspace path by replacing every
    non-alphanumeric character with ``-`` (e.g. ``/Users/me/proj_x`` becomes
    ``-Users-me-proj-x``).
    """
    claude_home = claude_home or (Path.home() / ".claude")
    slug = re.sub(r"[^A-Za-z0-9-]", "-", str(workspace.resolve()))
    return claude_home / "projects" / slug


def _newest_first(paths: List[Path]) -> List[Path]:
    def mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    return sorted(paths, key=mtime, reverse=True)


def _claude_code_transcripts(
    workspace: Path, limit: int, claude_home: Optional[Path] = None
) -> List[Path]:
    """Claude Code: ``~/.claude/projects/<slug>/*.jsonl`` (slug = the
    workspace path with every non-alphanumeric character replaced by ``-``)."""
    project_dir = claude_code_project_dir(workspace, claude_home)
    transcripts = [path for path in project_dir.glob("*.jsonl") if path.is_file()]
    return _newest_first(transcripts)[:limit]


# Rollout meta lines are small; a sane first line fits well under this.
_CODEX_META_READ_BYTES = 64 * 1024
# Rollouts for every workspace share one tree, so bound how many meta lines
# one discovery pass will read.
_CODEX_MAX_SCANNED = 500


def _codex_transcripts(
    workspace: Path, limit: int, codex_home: Optional[Path] = None
) -> List[Path]:
    """Codex CLI: ``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``.

    Rollouts for every workspace live in one date-partitioned tree; the first
    line of each file is a session-meta record whose ``payload.cwd`` (``cwd``
    at the top level in older layouts) names the working directory, so each
    candidate's meta line is read to keep only this workspace's sessions.
    """
    import json

    codex_home = codex_home or (Path.home() / ".codex")
    sessions_dir = codex_home / "sessions"
    if not sessions_dir.is_dir():
        return []

    # The YYYY/MM/DD partitioning plus the timestamped filename make the
    # lexicographic path order chronological — reversed() is newest-first
    # without a stat() per file.
    candidates = sorted(sessions_dir.glob("*/*/*/rollout-*.jsonl"), reverse=True)
    workspace_str = str(workspace.resolve())

    matches: List[Path] = []
    for rollout in candidates[:_CODEX_MAX_SCANNED]:
        try:
            with rollout.open("r", encoding="utf-8", errors="replace") as stream:
                meta = json.loads(stream.readline(_CODEX_META_READ_BYTES))
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        payload = meta.get("payload")
        if not isinstance(payload, dict):
            payload = meta
        if payload.get("cwd") == workspace_str:
            matches.append(rollout)
            if len(matches) >= limit:
                break
    return matches


def _gemini_session_files(
    workspace: Path, limit: int, gemini_home: Optional[Path] = None
) -> List[Path]:
    """Gemini CLI: ``~/.gemini/tmp/<sha256(project_root)>/`` holds ``logs.json``
    and saved chats under ``chats/``."""
    import hashlib

    gemini_home = gemini_home or (Path.home() / ".gemini")
    project_hash = hashlib.sha256(str(workspace.resolve()).encode("utf-8")).hexdigest()
    project_dir = gemini_home / "tmp" / project_hash
    if not project_dir.is_dir():
        return []

    files: List[Path] = []
    logs = project_dir / "logs.json"
    if logs.is_file():
        files.append(logs)
    chats_dir = project_dir / "chats"
    if chats_dir.is_dir():
        files.extend(_newest_first([p for p in chats_dir.glob("*.json") if p.is_file()])[:limit])
    return files


def _pi_transcripts(workspace: Path, limit: int, pi_home: Optional[Path] = None) -> List[Path]:
    """pi: ``~/.pi/agent/sessions/--<cwd with / -> ->--/*.jsonl``.

    Directory naming observed in the wild (not officially documented): the
    workspace path with ``/`` replaced by ``-``, wrapped in ``--``. When the
    derived directory does not exist, nothing is discovered — harmless.
    """
    pi_home = pi_home or (Path.home() / ".pi")
    slug = "--" + str(workspace.resolve()).strip("/").replace("/", "-") + "--"
    session_dir = pi_home / "agent" / "sessions" / slug
    if not session_dir.is_dir():
        return []
    return _newest_first([p for p in session_dir.glob("*.jsonl") if p.is_file()])[:limit]


def _aider_history(workspace: Path) -> List[Path]:
    """Aider: ``.aider.chat.history.md`` at the workspace root (an explicitly
    allowlisted dotfile — markdown chat history, no secrets by design)."""
    history = workspace / ".aider.chat.history.md"
    return [history] if history.is_file() else []


def _file_within_cap(path: Path, cap: int, skipped: List[str], label: str) -> bool:
    try:
        size = path.stat().st_size
    except OSError:
        skipped.append(f"{label} {path}: unreadable")
        return False
    if size == 0:
        skipped.append(f"{label} {path}: empty")
        return False
    if size > cap:
        skipped.append(f"{label} {path}: {size} bytes exceeds cap of {cap}")
        return False
    return True


def discover_seed_plan(
    workspace: Optional[Path] = None,
    *,
    include_codebase: bool = True,
    include_session_logs: bool = True,
    claude_home: Optional[Path] = None,
    codex_home: Optional[Path] = None,
    gemini_home: Optional[Path] = None,
    pi_home: Optional[Path] = None,
    max_session_logs: Optional[int] = None,
    max_session_log_bytes: Optional[int] = None,
    max_file_bytes: Optional[int] = None,
) -> SeedPlan:
    """Discover what a day-one seed of ``workspace`` would ingest."""
    from cognee.tasks.code_graph.code_repo import detect_code_project

    workspace = (workspace or find_workspace_root()).resolve()
    include_codebase = include_codebase and _env_bool("COGNEE_SEED_CODEBASE", True)
    include_session_logs = include_session_logs and _env_bool("COGNEE_SEED_SESSION_LOGS", True)
    max_session_logs = max_session_logs or _env_int(
        "COGNEE_SEED_MAX_SESSION_LOGS", DEFAULT_MAX_SESSION_LOGS
    )
    max_session_log_bytes = max_session_log_bytes or _env_int(
        "COGNEE_SEED_MAX_SESSION_LOG_BYTES", DEFAULT_MAX_SESSION_LOG_BYTES
    )
    max_file_bytes = max_file_bytes or _env_int(
        "COGNEE_SEED_MAX_FILE_BYTES", DEFAULT_MAX_FILE_BYTES
    )

    plan = SeedPlan(workspace=workspace)

    # --- Agent memory files -------------------------------------------------
    memory_candidates: List[Path] = [workspace / name for name in MEMORY_FILE_NAMES]
    memory_candidates += [workspace / rel for rel in ALLOWED_DOT_RELATIVE_PATHS]
    memory_dir = workspace / "memory"
    if memory_dir.is_dir():
        memory_candidates += sorted(memory_dir.glob("*.md"))

    project_dir = claude_code_project_dir(workspace, claude_home)
    auto_memory_dir = project_dir / "memory"
    if auto_memory_dir.is_dir():
        memory_candidates += sorted(auto_memory_dir.glob("*.md"))

    for candidate in memory_candidates:
        if candidate.is_file() and _file_within_cap(
            candidate, max_file_bytes, plan.skipped, "memory file"
        ):
            plan.memory_files.append(candidate)

    # --- Workspace docs -----------------------------------------------------
    for candidate in sorted(workspace.glob("README*")):
        if candidate.is_file() and _file_within_cap(
            candidate, max_file_bytes, plan.skipped, "readme"
        ):
            plan.readmes.append(candidate)

    # --- Session logs -------------------------------------------------------
    # One adapter per coding agent, each scoped to this workspace and capped
    # to the newest max_session_logs entries. Adapters over free-form globs on
    # purpose: cognee loads .env from the working directory, so an env-driven
    # glob would let a hostile repo point the seeder at arbitrary files.
    if include_session_logs:
        adapters = (
            ("Claude Code", _claude_code_transcripts(workspace, max_session_logs, claude_home)),
            ("Codex", _codex_transcripts(workspace, max_session_logs, codex_home)),
            ("Gemini CLI", _gemini_session_files(workspace, max_session_logs, gemini_home)),
            ("pi", _pi_transcripts(workspace, max_session_logs, pi_home)),
            ("Aider", _aider_history(workspace)),
        )
        for agent_name, session_files in adapters:
            for session_file in session_files:
                if _file_within_cap(
                    session_file,
                    max_session_log_bytes,
                    plan.skipped,
                    f"{agent_name} session log",
                ):
                    plan.session_logs.append(session_file)
    else:
        plan.skipped.append("session logs: disabled")

    # --- Codebase -----------------------------------------------------------
    if include_codebase:
        if detect_code_project(workspace):
            plan.codebase = workspace
        else:
            plan.skipped.append("codebase: workspace is not a recognized code project")
    else:
        plan.skipped.append("codebase: disabled")

    return plan
