"""Tests for the cognee attach CLI command."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from cognee.cli.commands.attach_command import (
    AttachCommand,
    _detect_llm_key,
    _get_mcp_config_path,
    _mcp_server_config,
    _write_mcp_config,
    MCP_TARGETS,
    TOOLS_TARGETS,
    ALL_TARGETS,
)


class TestDetectLlmKey:
    def test_finds_llm_api_key(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "sk-test"}, clear=False):
            assert _detect_llm_key() == "LLM_API_KEY"

    def test_finds_openai_api_key(self):
        env = {"OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            assert _detect_llm_key() == "OPENAI_API_KEY"

    def test_finds_anthropic_api_key(self):
        env = {"ANTHROPIC_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            assert _detect_llm_key() == "ANTHROPIC_API_KEY"

    def test_returns_none_when_no_key(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _detect_llm_key() is None

    def test_priority_order(self):
        env = {
            "LLM_API_KEY": "a",
            "OPENAI_API_KEY": "b",
            "ANTHROPIC_API_KEY": "c",
        }
        with patch.dict(os.environ, env, clear=True):
            assert _detect_llm_key() == "LLM_API_KEY"


class TestGetMcpConfigPath:
    def test_cursor_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = _get_mcp_config_path("cursor")
        assert path == tmp_path / ".cursor" / "mcp.json"

    def test_vscode_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = _get_mcp_config_path("vscode")
        assert path == tmp_path / ".vscode" / "mcp.json"

    def test_claude_path(self):
        path = _get_mcp_config_path("claude")
        assert "Claude" in str(path)
        assert path.name == "claude_desktop_config.json"

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match="Unknown MCP target"):
            _get_mcp_config_path("unknown")


class TestMcpServerConfig:
    def test_returns_dict_with_command(self):
        config = _mcp_server_config()
        assert "command" in config
        assert "args" in config

    def test_uses_cognee_mcp_when_available(self):
        with patch("cognee.cli.commands.attach_command.shutil.which", return_value="/usr/bin/cognee-mcp"):
            config = _mcp_server_config()
            assert config["command"] == "cognee-mcp"
            assert "--transport" in config["args"]
            assert "stdio" in config["args"]

    def test_falls_back_to_uv_when_not_found(self):
        with patch("cognee.cli.commands.attach_command.shutil.which", return_value=None):
            config = _mcp_server_config()
            assert config["command"] == "uv"


class TestWriteMcpConfig:
    def test_writes_cursor_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = _write_mcp_config("cursor")
        assert path.exists()
        data = json.loads(path.read_text())
        assert "mcpServers" in data
        assert "cognee" in data["mcpServers"]

    def test_dry_run_does_not_write(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = _write_mcp_config("cursor", dry_run=True)
        assert not path.exists()

    def test_preserves_existing_servers(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / ".cursor" / "mcp.json"
        config_path.parent.mkdir(parents=True)
        existing = {
            "mcpServers": {
                "other-server": {"command": "other", "args": []},
            }
        }
        config_path.write_text(json.dumps(existing))

        _write_mcp_config("cursor")
        data = json.loads(config_path.read_text())
        assert "other-server" in data["mcpServers"]
        assert "cognee" in data["mcpServers"]

    def test_overwrites_existing_cognee_entry(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / ".cursor" / "mcp.json"
        config_path.parent.mkdir(parents=True)
        existing = {
            "mcpServers": {
                "cognee": {"command": "old", "args": []},
            }
        }
        config_path.write_text(json.dumps(existing))

        _write_mcp_config("cursor")
        data = json.loads(config_path.read_text())
        assert data["mcpServers"]["cognee"]["command"] != "old"

    def test_vscode_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = _write_mcp_config("vscode")
        assert path.exists()
        data = json.loads(path.read_text())
        assert "cognee" in data["mcpServers"]


class TestTargetSets:
    def test_mcp_targets(self):
        assert "cursor" in MCP_TARGETS
        assert "claude" in MCP_TARGETS
        assert "vscode" in MCP_TARGETS

    def test_tools_targets(self):
        assert "openai" in TOOLS_TARGETS
        assert "langchain" in TOOLS_TARGETS
        assert "crewai" in TOOLS_TARGETS
        assert "anthropic" in TOOLS_TARGETS

    def test_all_targets_is_union(self):
        assert ALL_TARGETS == MCP_TARGETS | TOOLS_TARGETS

    def test_no_overlap(self):
        assert MCP_TARGETS & TOOLS_TARGETS == set()


class TestAttachCommandProtocol:
    def test_has_required_attributes(self):
        cmd = AttachCommand()
        assert cmd.command_string == "attach"
        assert cmd.help_string
        assert hasattr(cmd, "configure_parser")
        assert hasattr(cmd, "execute")
