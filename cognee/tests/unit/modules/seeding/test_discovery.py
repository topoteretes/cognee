import os
import re
import time
from pathlib import Path

from cognee.modules.seeding.discovery import (
    claude_code_project_dir,
    discover_seed_plan,
    find_workspace_root,
)


def make_workspace(tmp_path: Path, *, code_project: bool = True) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if code_project:
        (workspace / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (workspace / "MEMORY.md").write_text("# Memory\nProject uses uv.\n")
    (workspace / "SOUL.md").write_text("# Soul\nBe helpful.\n")
    (workspace / "README.md").write_text("# Demo\nA demo project.\n")
    memory_dir = workspace / "memory"
    memory_dir.mkdir()
    (memory_dir / "notes.md").write_text("- a fact\n")
    claude_dir = workspace / ".claude"
    claude_dir.mkdir()
    (claude_dir / "CLAUDE.md").write_text("Project instructions.\n")
    (claude_dir / "settings.json").write_text("{}")  # must never be discovered
    (workspace / ".env").write_text("SECRET=1\n")  # must never be discovered
    return workspace


def make_claude_home(tmp_path: Path, workspace: Path, transcripts: int = 5) -> Path:
    claude_home = tmp_path / "claude-home"
    project_dir = claude_code_project_dir(workspace, claude_home)
    project_dir.mkdir(parents=True)
    for index in range(transcripts):
        transcript = project_dir / f"session-{index}.jsonl"
        transcript.write_text('{"role": "user"}\n')
        stamp = time.time() - (transcripts - index) * 60
        os.utime(transcript, (stamp, stamp))
    auto_memory = project_dir / "memory"
    auto_memory.mkdir()
    (auto_memory / "MEMORY.md").write_text("# Auto memory\n")
    return claude_home


def test_discovers_memory_docs_sessions_and_codebase(tmp_path):
    workspace = make_workspace(tmp_path)
    claude_home = make_claude_home(tmp_path, workspace)

    plan = discover_seed_plan(workspace, claude_home=claude_home)

    memory_names = {path.name for path in plan.memory_files}
    assert "MEMORY.md" in memory_names
    assert "SOUL.md" in memory_names
    assert "CLAUDE.md" in memory_names  # .claude/CLAUDE.md allowlisted
    assert "notes.md" in memory_names
    # Two MEMORY.md entries: workspace root + Claude auto-memory.
    assert sum(1 for path in plan.memory_files if path.name == "MEMORY.md") == 2

    assert [path.name for path in plan.readmes] == ["README.md"]
    assert plan.codebase == workspace
    assert not plan.is_empty

    # Never pick up secrets or non-allowlisted dotfiles.
    discovered = {str(path) for path in plan.memory_files + plan.readmes + plan.session_logs}
    assert not any(item.endswith(".env") for item in discovered)
    assert not any(item.endswith("settings.json") for item in discovered)


def test_session_logs_keep_newest_n(tmp_path):
    workspace = make_workspace(tmp_path)
    claude_home = make_claude_home(tmp_path, workspace, transcripts=5)

    plan = discover_seed_plan(workspace, claude_home=claude_home, max_session_logs=3)

    assert len(plan.session_logs) == 3
    # Newest first: index 4 has the most recent mtime.
    assert [path.name for path in plan.session_logs] == [
        "session-4.jsonl",
        "session-3.jsonl",
        "session-2.jsonl",
    ]


def test_size_caps_and_empty_files_are_skipped(tmp_path):
    workspace = make_workspace(tmp_path)
    claude_home = make_claude_home(tmp_path, workspace, transcripts=1)
    (workspace / "MEMORY.md").write_text("")  # empty -> skipped

    plan = discover_seed_plan(
        workspace,
        claude_home=claude_home,
        max_session_log_bytes=4,  # transcript is larger than this
    )

    assert not plan.session_logs
    assert sum(1 for path in plan.memory_files if path.name == "MEMORY.md") == 1  # auto-memory only
    assert any("exceeds cap" in reason for reason in plan.skipped)
    assert any("empty" in reason for reason in plan.skipped)


def test_toggles_disable_categories(tmp_path):
    workspace = make_workspace(tmp_path)
    claude_home = make_claude_home(tmp_path, workspace)

    plan = discover_seed_plan(
        workspace,
        claude_home=claude_home,
        include_codebase=False,
        include_session_logs=False,
    )

    assert plan.codebase is None
    assert not plan.session_logs
    assert any("codebase: disabled" in reason for reason in plan.skipped)
    assert any("session logs: disabled" in reason for reason in plan.skipped)


def test_non_code_workspace_has_no_codebase(tmp_path):
    workspace = make_workspace(tmp_path, code_project=False)

    plan = discover_seed_plan(workspace, claude_home=tmp_path / "claude-home")

    assert plan.codebase is None
    assert any("not a recognized code project" in reason for reason in plan.skipped)
    assert plan.memory_files  # memory files still seed a plain notes directory


def test_env_toggles(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path)
    claude_home = make_claude_home(tmp_path, workspace)
    monkeypatch.setenv("COGNEE_SEED_CODEBASE", "false")
    monkeypatch.setenv("COGNEE_SEED_SESSION_LOGS", "0")

    plan = discover_seed_plan(workspace, claude_home=claude_home)

    assert plan.codebase is None
    assert not plan.session_logs


def test_claude_code_project_dir_slug(tmp_path):
    workspace = tmp_path / "my_proj.x"
    workspace.mkdir()
    project_dir = claude_code_project_dir(workspace, tmp_path / "home")

    # Every non-alphanumeric character in the absolute path becomes '-'
    # (path separators included, so the rule holds on Windows too).
    assert project_dir.parent == tmp_path / "home" / "projects"
    assert re.fullmatch(r"[A-Za-z0-9-]+", project_dir.name)
    assert project_dir.name.endswith("my-proj-x")


def test_find_workspace_root_walks_up_to_project_marker(tmp_path):
    workspace = make_workspace(tmp_path)
    nested = workspace / "src" / "pkg"
    nested.mkdir(parents=True)

    assert find_workspace_root(nested) == workspace


def test_find_workspace_root_falls_back_to_start(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()

    assert find_workspace_root(plain) == plain


def make_codex_home(tmp_path: Path, workspace: Path) -> Path:
    """Fake ~/.codex with rollouts for this workspace, another one, and junk."""
    import json

    codex_home = tmp_path / "codex-home"

    def rollout(day: str, name: str, first_line: str) -> None:
        day_dir = codex_home / "sessions" / "2026" / "08" / day
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / f"rollout-2026-08-{day}T10-00-00-{name}.jsonl").write_text(first_line + "\n")

    meta = {"timestamp": "t", "type": "session_meta", "payload": {"cwd": str(workspace)}}
    other = {"timestamp": "t", "type": "session_meta", "payload": {"cwd": "/elsewhere"}}
    legacy = {"timestamp": "t", "cwd": str(workspace)}  # cwd at top level (older layout)
    rollout("18", "old-match", json.dumps(meta))
    rollout("19", "other-ws", json.dumps(other))
    rollout("19", "legacy-match", json.dumps(legacy))
    rollout("20", "malformed", "{not json")
    rollout("20", "new-match", json.dumps(meta))
    return codex_home


def test_codex_adapter_filters_by_cwd_newest_first(tmp_path):
    from cognee.modules.seeding.discovery import _codex_transcripts

    workspace = make_workspace(tmp_path)
    codex_home = make_codex_home(tmp_path, workspace)

    matches = _codex_transcripts(workspace, 10, codex_home)

    names = [path.name for path in matches]
    assert [name.split("-")[-1].replace(".jsonl", "") for name in names] == [
        "match",
        "match",
        "match",
    ]
    # Newest day first; the other-workspace and malformed rollouts are excluded.
    assert "new-match" in names[0]
    assert "legacy-match" in names[1]
    assert "old-match" in names[2]

    assert [path.name for path in _codex_transcripts(workspace, 1, codex_home)] == names[:1]


def test_gemini_adapter_uses_project_hash(tmp_path):
    import hashlib

    from cognee.modules.seeding.discovery import _gemini_session_files

    workspace = make_workspace(tmp_path)
    gemini_home = tmp_path / "gemini-home"
    project_hash = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()
    project_dir = gemini_home / "tmp" / project_hash
    (project_dir / "chats").mkdir(parents=True)
    (project_dir / "logs.json").write_text("[]")
    (project_dir / "chats" / "session.json").write_text("{}")
    # A different project's directory must not be picked up.
    other_dir = gemini_home / "tmp" / ("0" * 64)
    other_dir.mkdir(parents=True)
    (other_dir / "logs.json").write_text("[]")

    files = _gemini_session_files(workspace, 5, gemini_home)

    assert {path.name for path in files} == {"logs.json", "session.json"}
    assert all(project_hash in str(path) for path in files)


def test_pi_and_aider_adapters(tmp_path):
    from cognee.modules.seeding.discovery import (
        _aider_history,
        _pi_session_dir,
        _pi_transcripts,
    )

    workspace = make_workspace(tmp_path)
    pi_home = tmp_path / "pi-home"
    session_dir = _pi_session_dir(workspace, pi_home)
    session_dir.mkdir(parents=True)
    (session_dir / "0001_abc.jsonl").write_text("{}\n")
    (workspace / ".aider.chat.history.md").write_text("# aider chat\n")

    assert [path.name for path in _pi_transcripts(workspace, 3, pi_home)] == ["0001_abc.jsonl"]
    assert [path.name for path in _aider_history(workspace)] == [".aider.chat.history.md"]
    assert _pi_transcripts(workspace, 3, tmp_path / "missing") == []


def test_discover_plan_collects_all_agent_sessions(tmp_path):
    import hashlib

    workspace = make_workspace(tmp_path)
    claude_home = make_claude_home(tmp_path, workspace, transcripts=1)
    codex_home = make_codex_home(tmp_path, workspace)
    gemini_home = tmp_path / "gemini-home"
    project_dir = (
        gemini_home / "tmp" / hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()
    )
    project_dir.mkdir(parents=True)
    (project_dir / "logs.json").write_text("[]")
    (workspace / ".aider.chat.history.md").write_text("# aider chat\n")

    plan = discover_seed_plan(
        workspace,
        claude_home=claude_home,
        codex_home=codex_home,
        gemini_home=gemini_home,
        pi_home=tmp_path / "pi-home",
        max_session_logs=2,
    )

    names = {path.name for path in plan.session_logs}
    assert "session-0.jsonl" in names  # Claude Code
    assert any("rollout-" in name for name in names)  # Codex
    assert "logs.json" in names  # Gemini CLI
    assert ".aider.chat.history.md" in names  # Aider
    # Codex respects the per-agent newest-N cap.
    assert sum(1 for name in names if "rollout-" in name) == 2
