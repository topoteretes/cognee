"""Discovery of day-one seed sources.

Pure discovery: walk the workspace and known coding-agent homes and return a
:class:`SeedPlan` describing what a seed run *would* ingest — agent memory
files (``MEMORY.md``, ``SOUL.md``, …), the workspace README, recent session
transcripts, and the workspace codebase. Nothing here reads file contents or
touches cognee storage; the runner in :mod:`cognee.modules.seeding.seed`
decides what to do with the plan.

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
    if include_session_logs:
        transcripts = [path for path in project_dir.glob("*.jsonl") if path.is_file()]
        transcripts.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        if len(transcripts) > max_session_logs:
            plan.skipped.append(
                f"session logs: keeping newest {max_session_logs} of {len(transcripts)}"
            )
        for transcript in transcripts[:max_session_logs]:
            if _file_within_cap(transcript, max_session_log_bytes, plan.skipped, "session log"):
                plan.session_logs.append(transcript)

        extra_glob = os.getenv("COGNEE_SEED_SESSION_LOG_GLOB")
        if extra_glob:
            for path in sorted(Path("/").glob(extra_glob.lstrip("/"))):
                if path.is_file() and _file_within_cap(
                    path, max_session_log_bytes, plan.skipped, "session log"
                ):
                    plan.session_logs.append(path)
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
