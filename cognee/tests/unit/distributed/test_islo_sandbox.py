"""Offline unit tests for the Islo CLI deployment script."""

import importlib
import json
import os
import subprocess
from pathlib import Path

import pytest


def _import_module():
    return importlib.import_module("distributed.deploy.islo_sandbox")


def _completed(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=args, returncode=returncode, stdout=stdout, stderr=stderr
    )


class FakeIslo:
    def __init__(self, *, existing=False, deploy_returncode=0, health_returncode=0):
        self.existing = existing
        self.created = False
        self.deploy_returncode = deploy_returncode
        self.health_returncode = health_returncode
        self.calls = []
        self.env_file_contents = None
        self.env_file_mode = None

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        args = command[1:]

        if args[:3] == ["status", "--output", "json"]:
            return _completed(command, stdout=json.dumps({"auth": {"authenticated": True}}))

        if args[:2] == ["status", "cognee-api"]:
            if self.existing or self.created:
                return _completed(
                    command,
                    stdout=json.dumps({"id": "sb-1", "name": "cognee-api", "status": "running"}),
                )
            return _completed(command, returncode=1, stderr="sandbox not found")

        if args and args[0] == "use" and _is_health_command(args):
            if self.health_returncode == 0:
                return _completed(command, stdout="OK\n")
            return _completed(command, returncode=self.health_returncode, stderr="not ready")

        if args and args[0] == "use" and _is_log_command(args):
            return _completed(command, stdout="server log\n")

        if args and args[0] == "use":
            env_path = Path(args[args.index("--env-file") + 1])
            self.env_file_contents = env_path.read_text()
            self.env_file_mode = os.stat(env_path).st_mode & 0o777
            if self.deploy_returncode != 0:
                return _completed(
                    command, returncode=self.deploy_returncode, stderr="deploy failed"
                )
            self.created = True
            return _completed(command, stdout="started\n")

        if args and args[0] == "share":
            return _completed(
                command,
                stdout=json.dumps(
                    {
                        "url": "https://share.example",
                        "expires_at": "2026-07-17T00:00:00Z",
                    }
                ),
            )

        raise AssertionError(f"Unexpected Islo command: {command!r}")


def _is_health_command(args):
    return "python3" in args and "urllib.request" in args[-1]


def _is_log_command(args):
    return "tail -n 30 /tmp/cognee-server.log" in args


def _configure(monkeypatch, fake):
    module = _import_module()
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/local/bin/islo")
    monkeypatch.setattr(module.subprocess, "run", fake)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    return module


def test_run_islo_returns_completed_process(monkeypatch):
    module = _import_module()
    expected = _completed(["islo", "status"], stdout="ok")
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: expected)

    result = module.run_islo(["status"], echo=False)

    assert result is expected


def test_run_islo_raises_on_nonzero_exit(monkeypatch):
    module = _import_module()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: _completed([], returncode=2, stderr="boom"),
    )

    with pytest.raises(RuntimeError, match="boom"):
        module.run_islo(["use", "broken"], echo=False)


def test_run_islo_raises_on_timeout(monkeypatch):
    module = _import_module()

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="islo", timeout=3)

    monkeypatch.setattr(module.subprocess, "run", timeout)

    with pytest.raises(RuntimeError, match="timed out"):
        module.run_islo(["use", "slow"], timeout=3)


def test_deploy_requires_cli(monkeypatch):
    module = _import_module()
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="Islo CLI is required"):
        module.deploy_cognee()


def test_deploy_requires_llm_api_key(monkeypatch):
    module = _import_module()
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/local/bin/islo")

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        module.deploy_cognee()


def test_deploy_requires_authentication(monkeypatch):
    module = _import_module()
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/local/bin/islo")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **_kwargs: _completed(
            command, stdout=json.dumps({"auth": {"authenticated": False}})
        ),
    )

    with pytest.raises(RuntimeError, match="not authenticated"):
        module.deploy_cognee()


def test_deploy_happy_path_uses_cli_and_protects_credentials(monkeypatch):
    fake = FakeIslo()
    module = _configure(monkeypatch, fake)

    sandbox = module.deploy_cognee()

    assert sandbox["id"] == "sb-1"
    commands = [call[0] for call in fake.calls]
    deploy_command = next(command for command in commands if "--env-file" in command)
    assert deploy_command[:3] == ["islo", "use", module.SANDBOX_NAME]
    assert "--cpu" in deploy_command
    assert "--memory" in deploy_command
    assert "--disk" in deploy_command
    assert "--run-as-user" in deploy_command
    assert "test-llm-key" not in deploy_command
    assert fake.env_file_contents is not None
    assert 'LLM_API_KEY="test-llm-key"' in fake.env_file_contents
    assert fake.env_file_mode == 0o600
    assert any(command[1] == "share" for command in commands)


def test_deploy_removes_temporary_env_file(monkeypatch):
    fake = FakeIslo()
    module = _configure(monkeypatch, fake)

    module.deploy_cognee()

    deploy_call = next(call[0] for call in fake.calls if "--env-file" in call[0])
    env_path = Path(deploy_call[deploy_call.index("--env-file") + 1])
    assert not env_path.exists()


def test_deploy_prints_share_and_cli_cleanup_commands(monkeypatch, capsys):
    module = _configure(monkeypatch, FakeIslo())

    module.deploy_cognee()

    output = capsys.readouterr().out
    assert "https://share.example/health" in output
    assert "https://share.example/docs" in output
    assert "islo stop cognee-api" in output
    assert "islo rm cognee-api --force" in output


def test_deploy_reports_existing_sandbox(monkeypatch):
    module = _configure(monkeypatch, FakeIslo(existing=True))

    with pytest.raises(RuntimeError, match="islo rm cognee-api --force"):
        module.deploy_cognee()


def test_deploy_aborts_when_cli_deploy_fails(monkeypatch):
    fake = FakeIslo(deploy_returncode=1)
    module = _configure(monkeypatch, fake)

    with pytest.raises(RuntimeError, match="deploy failed"):
        module.deploy_cognee()

    assert not any(call[0][1] == "share" for call in fake.calls)


def test_deploy_aborts_when_server_never_becomes_healthy(monkeypatch):
    fake = FakeIslo(health_returncode=1)
    module = _configure(monkeypatch, fake)
    monkeypatch.setattr(module, "wait_for_server_health", lambda *_args: False)

    with pytest.raises(RuntimeError, match="did not become healthy"):
        module.deploy_cognee()

    assert not any(call[0][1] == "share" for call in fake.calls)


def test_wait_for_server_health_retries(monkeypatch):
    fake = FakeIslo(health_returncode=1)
    module = _configure(monkeypatch, fake)

    assert module.wait_for_server_health(retries=2, delay=0) is False
    health_calls = [call for call in fake.calls if _is_health_command(call[0][1:])]
    assert len(health_calls) == 2
