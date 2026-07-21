"""Offline unit tests for distributed/deploy/islo_sandbox.py.

The ``islo`` SDK is a documented script prerequisite (like ``daytona`` for
daytona_sandbox.py), not a project dependency, so these tests inject fake
``islo`` modules before importing the deploy script. No network access and no
API key are required.
"""

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _import_module(monkeypatch):
    """Import distributed.deploy.islo_sandbox with fake islo modules installed."""
    fake_islo = types.ModuleType("islo")
    setattr(fake_islo, "Islo", MagicMock(name="Islo"))

    fake_islo_types = types.ModuleType("islo.types")

    class LifecyclePolicy:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    setattr(fake_islo_types, "LifecyclePolicy", LifecyclePolicy)
    setattr(fake_islo, "types", fake_islo_types)

    fake_islo_errors = types.ModuleType("islo.errors")

    class ConflictError(Exception):
        pass

    setattr(fake_islo_errors, "ConflictError", ConflictError)
    setattr(fake_islo, "errors", fake_islo_errors)

    monkeypatch.setitem(sys.modules, "islo", fake_islo)
    monkeypatch.setitem(sys.modules, "islo.types", fake_islo_types)
    monkeypatch.setitem(sys.modules, "islo.errors", fake_islo_errors)
    sys.modules.pop("distributed.deploy.islo_sandbox", None)
    return importlib.import_module("distributed.deploy.islo_sandbox")


def _exec_result(status, stdout="", stderr="", exit_code=None):
    return SimpleNamespace(
        exec_id="e1",
        status=status,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        truncated=False,
    )


class FakeSandboxes:
    def __init__(self, exec_results=None, sandbox_statuses=None):
        self.exec_results = list(exec_results or [])
        self.sandbox_statuses = list(sandbox_statuses or [])
        self.create_calls = []
        self.exec_calls = []
        self.get_result_calls = 0

    def create_sandbox(self, **kwargs):
        self.create_calls.append(kwargs)
        return SimpleNamespace(id="sb-1", name=kwargs.get("name"), status="starting")

    def get_sandbox(self, sandbox_name):
        return SimpleNamespace(id="sb-1", name=sandbox_name, status=self.sandbox_statuses.pop(0))

    def exec_in_sandbox(self, sandbox_name, *, command, env=None, timeout_secs=None, workdir=None):
        self.exec_calls.append(
            {
                "sandbox_name": sandbox_name,
                "command": list(command),
                "env": env,
                "workdir": workdir,
                "timeout_secs": timeout_secs,
            }
        )
        return SimpleNamespace(exec_id="e1", sandbox_id="sb-1", status="running")

    def get_exec_result(self, sandbox_name, exec_id):
        self.get_result_calls += 1
        return self.exec_results.pop(0)


class FakeShares:
    def __init__(self):
        self.calls = []

    def create_share(self, sandbox_name, *, port, ttl_seconds=None):
        self.calls.append((sandbox_name, port, ttl_seconds))
        return SimpleNamespace(
            url="https://share.example",
            port=port,
            share_id="sh1",
            expires_at=None,
            created_at="2026-01-01T00:00:00Z",
        )


class FakeClient:
    def __init__(self, sandboxes, shares=None):
        self.sandboxes = sandboxes
        self.shares = shares or FakeShares()


def test_run_command_polls_until_completed(monkeypatch):
    module = _import_module(monkeypatch)
    fake = FakeSandboxes(
        exec_results=[
            _exec_result("running"),
            _exec_result("running"),
            _exec_result("completed", stdout="hi", exit_code=0),
        ]
    )
    client = FakeClient(fake)

    result = module.run_command(client, "cognee-api", ["echo", "hi"], poll_interval=0)

    assert result.status == "completed"
    assert result.stdout == "hi"
    assert result.exit_code == 0
    assert fake.get_result_calls == 3
    assert len(fake.exec_calls) == 1
    assert fake.exec_calls[0]["sandbox_name"] == "cognee-api"
    assert fake.exec_calls[0]["command"] == ["echo", "hi"]


@pytest.mark.parametrize("terminal_status", ["failed", "timeout"])
def test_run_command_returns_on_failed_and_timeout_statuses(monkeypatch, terminal_status):
    module = _import_module(monkeypatch)
    fake = FakeSandboxes(
        exec_results=[
            _exec_result("running"),
            _exec_result(terminal_status, stderr="boom", exit_code=1),
        ]
    )
    client = FakeClient(fake)

    result = module.run_command(client, "cognee-api", ["false"], poll_interval=0)

    assert result.status == terminal_status
    assert fake.get_result_calls == 2


def test_run_command_raises_when_exec_never_terminal(monkeypatch):
    module = _import_module(monkeypatch)
    fake = FakeSandboxes(exec_results=[_exec_result("running"), _exec_result("running")])
    client = FakeClient(fake)

    with pytest.raises(RuntimeError, match="terminal status"):
        module.run_command(client, "cognee-api", ["sleep", "inf"], poll_interval=0, max_wait=0)

    assert fake.get_result_calls == 1


def test_wait_for_sandbox_running_success(monkeypatch):
    module = _import_module(monkeypatch)
    fake = FakeSandboxes(sandbox_statuses=["starting", "running"])
    client = FakeClient(fake)

    sandbox = module.wait_for_sandbox_running(client, "cognee-api", poll_interval=0)

    assert sandbox.status == "running"
    assert fake.sandbox_statuses == []


def test_wait_for_sandbox_running_raises_on_failed(monkeypatch):
    module = _import_module(monkeypatch)
    fake = FakeSandboxes(sandbox_statuses=["failed"])
    client = FakeClient(fake)

    with pytest.raises(RuntimeError, match="failed"):
        module.wait_for_sandbox_running(client, "cognee-api", poll_interval=0)


def test_deploy_requires_env_vars(monkeypatch):
    module = _import_module(monkeypatch)

    monkeypatch.delenv("ISLO_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ISLO_API_KEY"):
        module.deploy_cognee()

    monkeypatch.setenv("ISLO_API_KEY", "test-key")
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        module.deploy_cognee()


def _deploy_with_fakes(monkeypatch):
    """Run deploy_cognee() against fully scripted fakes; returns (module, client)."""
    module = _import_module(monkeypatch)
    monkeypatch.setenv("ISLO_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")

    fake_sandboxes = FakeSandboxes(
        sandbox_statuses=["running"],
        exec_results=[
            _exec_result("completed", stdout="installed", exit_code=0),
            _exec_result("completed", stdout="started", exit_code=0),
            _exec_result("completed", stdout="OK", exit_code=0),
        ],
    )
    client = FakeClient(fake_sandboxes)
    monkeypatch.setattr(module, "Islo", lambda: client)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    sandbox = module.deploy_cognee()
    return module, client, sandbox


def test_deploy_happy_path(monkeypatch):
    module, client, sandbox = _deploy_with_fakes(monkeypatch)

    assert sandbox.id == "sb-1"

    create_call = client.sandboxes.create_calls[0]
    assert create_call["name"] == module.SANDBOX_NAME
    assert create_call["image"] == module.DEFAULT_IMAGE
    assert create_call["env"]["LLM_API_KEY"] == "test-llm-key"
    assert create_call["env"]["HOST"] == "0.0.0.0"

    commands = [call["command"] for call in client.sandboxes.exec_calls]
    assert any("pip install 'cognee[api]'" in " ".join(command) for command in commands)
    assert any("uvicorn" in " ".join(command) for command in commands)

    assert client.shares.calls == [(module.SANDBOX_NAME, module.API_PORT, 86400)]


def test_deploy_share_url_printed(monkeypatch, capsys):
    _deploy_with_fakes(monkeypatch)

    output = capsys.readouterr().out
    assert "https://share.example" in output
    assert "https://share.example/health" in output
    assert "https://share.example/docs" in output


@pytest.mark.parametrize(
    ("install_status", "install_exit_code"),
    [("failed", None), ("timeout", None), ("completed", 1)],
)
def test_deploy_aborts_when_install_fails(monkeypatch, install_status, install_exit_code):
    module = _import_module(monkeypatch)
    monkeypatch.setenv("ISLO_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")

    fake_sandboxes = FakeSandboxes(
        sandbox_statuses=["running"],
        exec_results=[
            _exec_result(install_status, stderr="pip boom", exit_code=install_exit_code),
        ],
    )
    client = FakeClient(fake_sandboxes)
    monkeypatch.setattr(module, "Islo", lambda: client)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="Failed to install cognee"):
        module.deploy_cognee()

    assert len(fake_sandboxes.exec_calls) == 1
    assert client.shares.calls == []


def test_deploy_aborts_when_server_start_fails(monkeypatch):
    module = _import_module(monkeypatch)
    monkeypatch.setenv("ISLO_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")

    fake_sandboxes = FakeSandboxes(
        sandbox_statuses=["running"],
        exec_results=[
            _exec_result("completed", stdout="installed", exit_code=0),
            _exec_result("failed", stderr="uvicorn boom", exit_code=1),
        ],
    )
    client = FakeClient(fake_sandboxes)
    monkeypatch.setattr(module, "Islo", lambda: client)

    with pytest.raises(RuntimeError, match="Failed to start cognee API server"):
        module.deploy_cognee()

    assert len(fake_sandboxes.exec_calls) == 2
    assert client.shares.calls == []


def test_deploy_aborts_when_server_never_becomes_healthy(monkeypatch):
    module = _import_module(monkeypatch)
    monkeypatch.setenv("ISLO_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")

    fake_sandboxes = FakeSandboxes(
        sandbox_statuses=["running"],
        exec_results=[
            _exec_result("completed", stdout="installed", exit_code=0),
            _exec_result("completed", stdout="started", exit_code=0),
        ],
    )
    client = FakeClient(fake_sandboxes)
    monkeypatch.setattr(module, "Islo", lambda: client)
    monkeypatch.setattr(module, "wait_for_server_health", lambda *_args: False)

    with pytest.raises(RuntimeError, match="did not become healthy"):
        module.deploy_cognee()

    assert client.shares.calls == []


def test_deploy_reports_existing_sandbox_conflict(monkeypatch):
    module = _import_module(monkeypatch)
    monkeypatch.setenv("ISLO_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")

    class ConflictingSandboxes(FakeSandboxes):
        def create_sandbox(self, **kwargs):
            raise module.ConflictError("sandbox name already exists")

    client = FakeClient(ConflictingSandboxes())
    monkeypatch.setattr(module, "Islo", lambda: client)

    with pytest.raises(RuntimeError, match="already exists") as excinfo:
        module.deploy_cognee()

    assert "delete_sandbox('cognee-api')" in str(excinfo.value)
