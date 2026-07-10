"""Offline unit tests for `cognee install <harness>`.

All tests use monkeypatched HOME / USERPROFILE / APPDATA / XDG_CONFIG_HOME
pointing to tmp_path so no real config files are touched.
No live binary, no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_env(monkeypatch, tmp_path: Path) -> None:
    """Redirect all home / config env vars to tmp_path."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))


# ---------------------------------------------------------------------------
# 1. list
# ---------------------------------------------------------------------------


def test_list_harnesses(capsys):
    from cognee.cli.commands.install_command import InstallCommand

    cmd = InstallCommand()
    cmd.execute(
        SimpleNamespace(
            harness=None,
            list=True,
            uninstall=False,
            dry_run=False,
            mcp_dir=None,
            scope="user",
        )
    )
    out = capsys.readouterr().out
    assert "claude-code" in out
    assert "cursor" in out
    assert "opencode" in out


# ---------------------------------------------------------------------------
# 2. install cursor
# ---------------------------------------------------------------------------


def test_install_cursor(monkeypatch, tmp_path):
    _patch_env(monkeypatch, tmp_path)
    mcp_dir = str(tmp_path / "cognee-mcp-src")

    with mock.patch("cognee.cli.commands.install_command._verify"):
        from cognee.cli.commands.install_command import InstallCommand

        cmd = InstallCommand()
        cmd.execute(
            SimpleNamespace(
                harness="cursor",
                list=False,
                uninstall=False,
                dry_run=False,
                mcp_dir=mcp_dir,
                scope="user",
            )
        )

    config_path = tmp_path / ".cursor" / "mcp.json"
    assert config_path.exists(), f"Expected config at {config_path}"
    cfg = json.loads(config_path.read_text())
    assert "cognee" in cfg["mcpServers"]
    entry = cfg["mcpServers"]["cognee"]
    assert entry["command"] == "uv"
    assert "--directory" in entry["args"]


# ---------------------------------------------------------------------------
# 3. install claude-code — JSON fallback (no claude binary)
# ---------------------------------------------------------------------------


def test_install_claude_code_json_fallback(monkeypatch, tmp_path):
    _patch_env(monkeypatch, tmp_path)
    mcp_dir = str(tmp_path / "cognee-mcp-src")

    monkeypatch.setattr(
        "cognee.cli.commands.installers.claude_code.ClaudeCodeInstaller._claude_binary",
        staticmethod(lambda: None),
    )

    with mock.patch("cognee.cli.commands.install_command._verify"):
        from cognee.cli.commands.install_command import InstallCommand

        cmd = InstallCommand()
        cmd.execute(
            SimpleNamespace(
                harness="claude-code",
                list=False,
                uninstall=False,
                dry_run=False,
                mcp_dir=mcp_dir,
                scope="user",
            )
        )

    config_path = tmp_path / ".claude.json"
    assert config_path.exists(), f"Expected config at {config_path}"
    cfg = json.loads(config_path.read_text())
    assert "cognee" in cfg["mcpServers"]
    entry = cfg["mcpServers"]["cognee"]
    assert entry["command"] == "uv"
    assert "--directory" in entry["args"]


# ---------------------------------------------------------------------------
# 4. install claude-code — native path (claude binary on PATH, mocked)
# ---------------------------------------------------------------------------


def test_install_claude_code_native(monkeypatch, tmp_path):
    _patch_env(monkeypatch, tmp_path)
    mcp_dir = str(tmp_path / "cognee-mcp-src")

    fake_claude = str(tmp_path / "claude")
    Path(fake_claude).touch(mode=0o755)

    monkeypatch.setattr(
        "cognee.cli.commands.installers.claude_code.ClaudeCodeInstaller._claude_binary",
        staticmethod(lambda: fake_claude),
    )

    mock_result = mock.MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    with (
        mock.patch("subprocess.run", return_value=mock_result) as mock_sub,
        mock.patch("cognee.cli.commands.install_command._verify"),
    ):
        from cognee.cli.commands.install_command import InstallCommand

        cmd = InstallCommand()
        cmd.execute(
            SimpleNamespace(
                harness="claude-code",
                list=False,
                uninstall=False,
                dry_run=False,
                mcp_dir=mcp_dir,
                scope="user",
            )
        )

    mock_sub.assert_called_once()
    call_args = mock_sub.call_args[0][0]
    assert fake_claude in call_args
    assert "mcp" in call_args
    assert "add" in call_args
    assert "cognee" in call_args


# ---------------------------------------------------------------------------
# 5. install opencode
# ---------------------------------------------------------------------------


def test_install_opencode(monkeypatch, tmp_path):
    _patch_env(monkeypatch, tmp_path)
    mcp_dir = str(tmp_path / "cognee-mcp-src")

    with mock.patch("cognee.cli.commands.install_command._verify"):
        from cognee.cli.commands.install_command import InstallCommand

        cmd = InstallCommand()
        cmd.execute(
            SimpleNamespace(
                harness="opencode",
                list=False,
                uninstall=False,
                dry_run=False,
                mcp_dir=mcp_dir,
                scope="user",
            )
        )

    config_path = tmp_path / ".config" / "opencode" / "opencode.json"
    assert config_path.exists(), f"Expected config at {config_path}"
    cfg = json.loads(config_path.read_text())
    assert "cognee" in cfg["mcp"]
    entry = cfg["mcp"]["cognee"]
    assert entry["type"] == "local"
    assert isinstance(entry["command"], list)
    assert "uv" in entry["command"]
    assert "--directory" in entry["command"]
    assert entry.get("enabled") is True


# ---------------------------------------------------------------------------
# 6. idempotent
# ---------------------------------------------------------------------------


def test_install_idempotent(monkeypatch, tmp_path, capsys):
    _patch_env(monkeypatch, tmp_path)
    mcp_dir = str(tmp_path / "cognee-mcp-src")

    with mock.patch("cognee.cli.commands.install_command._verify"):
        from cognee.cli.commands.install_command import InstallCommand

        def _run():
            InstallCommand().execute(
                SimpleNamespace(
                    harness="cursor",
                    list=False,
                    uninstall=False,
                    dry_run=False,
                    mcp_dir=mcp_dir,
                    scope="user",
                )
            )

        _run()
        capsys.readouterr()  # discard first run output

        _run()  # second run — should be a no-op
        out = capsys.readouterr().out

    assert "Already installed" in out


# ---------------------------------------------------------------------------
# 7. uninstall
# ---------------------------------------------------------------------------


def test_uninstall(monkeypatch, tmp_path):
    _patch_env(monkeypatch, tmp_path)

    config_path = tmp_path / ".cursor" / "mcp.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "cognee": {"command": "cognee-mcp", "args": []},
                    "other-tool": {"command": "other", "args": []},
                }
            }
        )
    )

    from cognee.cli.commands.install_command import InstallCommand

    InstallCommand().execute(
        SimpleNamespace(
            harness="cursor",
            list=False,
            uninstall=True,
            dry_run=False,
            mcp_dir=None,
            scope="user",
        )
    )

    cfg = json.loads(config_path.read_text())
    assert "cognee" not in cfg.get("mcpServers", {})
    assert "other-tool" in cfg.get("mcpServers", {}), "uninstall must not touch other entries"


# ---------------------------------------------------------------------------
# 8. dry-run
# ---------------------------------------------------------------------------


def test_dry_run(monkeypatch, tmp_path, capsys):
    _patch_env(monkeypatch, tmp_path)
    mcp_dir = str(tmp_path / "cognee-mcp-src")

    from cognee.cli.commands.install_command import InstallCommand

    InstallCommand().execute(
        SimpleNamespace(
            harness="cursor",
            list=False,
            uninstall=False,
            dry_run=True,
            mcp_dir=mcp_dir,
            scope="user",
        )
    )

    out = capsys.readouterr().out
    assert "dry-run" in out.lower()

    config_path = tmp_path / ".cursor" / "mcp.json"
    assert not config_path.exists(), "dry-run must not write any files"


# ---------------------------------------------------------------------------
# 9. preserves existing config
# ---------------------------------------------------------------------------


def test_preserves_existing_config(monkeypatch, tmp_path):
    _patch_env(monkeypatch, tmp_path)
    mcp_dir = str(tmp_path / "cognee-mcp-src")

    config_path = tmp_path / ".cursor" / "mcp.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "my-other-tool": {"command": "other-tool", "args": ["--flag"]},
                }
            }
        )
    )

    with mock.patch("cognee.cli.commands.install_command._verify"):
        from cognee.cli.commands.install_command import InstallCommand

        InstallCommand().execute(
            SimpleNamespace(
                harness="cursor",
                list=False,
                uninstall=False,
                dry_run=False,
                mcp_dir=mcp_dir,
                scope="user",
            )
        )

    cfg = json.loads(config_path.read_text())
    assert "cognee" in cfg["mcpServers"], "cognee entry should be added"
    assert "my-other-tool" in cfg["mcpServers"], "existing entry must be preserved"


# ---------------------------------------------------------------------------
# 10. --mcp-dir flag
# ---------------------------------------------------------------------------


def test_mcp_dir_flag(monkeypatch, tmp_path):
    _patch_env(monkeypatch, tmp_path)
    mcp_dir = str(tmp_path / "my-mcp-source")

    with mock.patch("cognee.cli.commands.install_command._verify"):
        from cognee.cli.commands.install_command import InstallCommand

        InstallCommand().execute(
            SimpleNamespace(
                harness="cursor",
                list=False,
                uninstall=False,
                dry_run=False,
                mcp_dir=mcp_dir,
                scope="user",
            )
        )

    config_path = tmp_path / ".cursor" / "mcp.json"
    cfg = json.loads(config_path.read_text())
    entry = cfg["mcpServers"]["cognee"]
    assert entry["command"] == "uv"
    assert "--directory" in entry["args"]
    assert mcp_dir in entry["args"]
    assert "run" in entry["args"]
    assert "cognee-mcp" in entry["args"]


# ---------------------------------------------------------------------------
# 11. verify — success
# ---------------------------------------------------------------------------


def test_verify_success(capsys):
    import cognee.cli.commands.install_command as mod

    fake_response = json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"tools": []}})
    mock_result = mock.MagicMock()
    mock_result.stdout = fake_response + "\n"
    mock_result.stderr = ""
    mock_result.returncode = 0

    with mock.patch("subprocess.run", return_value=mock_result):
        mod._verify({"command": "cognee-mcp", "args": []})

    out = capsys.readouterr().out
    assert "cognee-mcp responded" in out or "tools/list" in out


# ---------------------------------------------------------------------------
# 12. verify — timeout
# ---------------------------------------------------------------------------


def test_verify_timeout(capsys):
    import subprocess
    import cognee.cli.commands.install_command as mod

    with mock.patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["cognee-mcp"], timeout=5),
    ):
        mod._verify({"command": "cognee-mcp", "args": []})

    out = capsys.readouterr().out
    assert "timed out" in out.lower() or "config is written" in out.lower()


# ---------------------------------------------------------------------------
# 13. corrupt config raises RuntimeError (not silently overwritten)
# ---------------------------------------------------------------------------


def test_corrupt_config_raises(monkeypatch, tmp_path):
    """A malformed config file must raise RuntimeError, not be silently wiped.

    We test the installer directly (not through InstallCommand.execute) because
    execute() catches RuntimeError and converts it to fmt.error() output.
    The important contract is that the *installer* raises — execute() will
    then surface it as a user-facing error message, which is the correct behaviour.
    """
    _patch_env(monkeypatch, tmp_path)
    mcp_dir = str(tmp_path / "cognee-mcp-src")

    # Write invalid JSON to the cursor config path
    config_path = tmp_path / ".cursor" / "mcp.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{NOT VALID JSON}", encoding="utf-8")

    from cognee.cli.commands.installers.cursor import CursorInstaller

    installer = CursorInstaller()
    with pytest.raises(RuntimeError, match="not valid JSON"):
        installer.install(mcp_dir=mcp_dir, scope="user")

    # The corrupt file must NOT have been overwritten
    assert config_path.read_text(encoding="utf-8") == "{NOT VALID JSON}", (
        "corrupt config was silently overwritten — data loss bug still present"
    )


# ---------------------------------------------------------------------------
# 14. corrupt claude_code config raises RuntimeError (not silently overwritten)
# ---------------------------------------------------------------------------


def test_corrupt_config_raises_claude_code(monkeypatch, tmp_path):
    """Malformed ~/.claude.json must raise RuntimeError, not be silently wiped.

    ~/.claude.json holds all Claude Code state — silent data loss here would be
    especially painful. This test mirrors test_corrupt_config_raises for cursor.
    """
    _patch_env(monkeypatch, tmp_path)
    mcp_dir = str(tmp_path / "cognee-mcp-src")

    # Write invalid JSON to the claude-code user config path
    config_path = tmp_path / ".claude.json"
    config_path.write_text("{NOT VALID JSON}", encoding="utf-8")

    monkeypatch.setattr(
        "cognee.cli.commands.installers.claude_code.ClaudeCodeInstaller._claude_binary",
        staticmethod(lambda: None),  # force JSON fallback path
    )

    from cognee.cli.commands.installers.claude_code import ClaudeCodeInstaller

    with pytest.raises(RuntimeError, match="not valid JSON"):
        ClaudeCodeInstaller().install(mcp_dir=mcp_dir, scope="user")

    # The corrupt file must NOT have been overwritten
    assert config_path.read_text(encoding="utf-8") == "{NOT VALID JSON}", (
        "corrupt ~/.claude.json was silently overwritten — data loss bug still present"
    )


# ---------------------------------------------------------------------------
# 15. corrupt opencode config raises RuntimeError (not silently overwritten)
# ---------------------------------------------------------------------------


def test_corrupt_config_raises_opencode(monkeypatch, tmp_path):
    """Malformed opencode config must raise RuntimeError, not be silently wiped."""
    _patch_env(monkeypatch, tmp_path)
    mcp_dir = str(tmp_path / "cognee-mcp-src")

    # Write invalid JSON to the opencode config path
    config_path = tmp_path / ".config" / "opencode" / "opencode.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{NOT VALID JSON}", encoding="utf-8")

    from cognee.cli.commands.installers.opencode import OpenCodeInstaller

    with pytest.raises(RuntimeError, match="not valid JSON"):
        OpenCodeInstaller().install(mcp_dir=mcp_dir, scope="user")

    # The corrupt file must NOT have been overwritten
    assert config_path.read_text(encoding="utf-8") == "{NOT VALID JSON}", (
        "corrupt opencode config was silently overwritten — data loss bug still present"
    )


# ---------------------------------------------------------------------------
# 16. claude-code --scope project writes to .mcp.json (not ~/.claude.json)
# ---------------------------------------------------------------------------


def test_install_claude_code_project_scope(monkeypatch, tmp_path):
    """--scope project for claude-code must write to Path.cwd()/.mcp.json.

    This covers the F2 fix: config_path() is now scope-aware. Before the fix,
    config_path() always returned ~/.claude.json regardless of scope, so the
    idempotency pre-check looked in the wrong file on re-run.
    """
    _patch_env(monkeypatch, tmp_path)
    mcp_dir = str(tmp_path / "cognee-mcp-src")

    # chdir to tmp_path so Path.cwd() / ".mcp.json" lands in our temp dir
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        "cognee.cli.commands.installers.claude_code.ClaudeCodeInstaller._claude_binary",
        staticmethod(lambda: None),  # force JSON fallback; avoids spawning subprocess
    )

    from cognee.cli.commands.installers.claude_code import ClaudeCodeInstaller

    result = ClaudeCodeInstaller().install(mcp_dir=mcp_dir, scope="project")

    # Must write to .mcp.json in cwd, NOT to ~/.claude.json
    project_config = tmp_path / ".mcp.json"
    user_config = tmp_path / ".claude.json"

    assert project_config.exists(), (
        f"Expected project-scope config at {project_config}, but it was not created."
    )
    assert not user_config.exists(), (
        "user-scope ~/.claude.json was written even though --scope project was requested."
    )

    cfg = json.loads(project_config.read_text())
    assert "cognee" in cfg["mcpServers"]
    assert ".mcp.json" in result
