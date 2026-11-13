"""Offline unit tests for distributed/deploy/boxd_sandbox.py.

The ``boxd`` SDK is a documented script prerequisite (like ``islo`` for
islo_sandbox.py), not a project dependency, so these tests inject fake
``boxd`` modules before importing the deploy script. No network access and no
API key are required.
"""

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _import_module(monkeypatch):
    """Import distributed.deploy.boxd_sandbox with fake boxd modules installed."""
    fake_boxd = types.ModuleType("boxd")
    fake_boxd.Boxd = MagicMock(name="Boxd")

    fake_boxd_errors = types.ModuleType("boxd.errors")

    class ConflictError(Exception):
        pass

    fake_boxd_errors.ConflictError = ConflictError
    fake_boxd.errors = fake_boxd_errors

    monkeypatch.setitem(sys.modules, "boxd", fake_boxd)
    monkeypatch.setitem(sys.modules, "boxd.errors", fake_boxd_errors)
    sys.modules.pop("distributed.deploy.boxd_sandbox", None)
    return importlib.import_module("distributed.deploy.boxd_sandbox")


def _exec_result(exit_code=0, stdout="", stderr=""):
    return SimpleNamespace(exit_code=exit_code, stdout=stdout, stderr=stderr)


class FakeFs:
    def __init__(self):
        self.writes = {}
        self.modes = {}

    def write(self, machine_id, path, data, *, mode=None):
        self.writes[path] = data
        self.modes[path] = mode
        return SimpleNamespace(path=path)


class FakeMachines:
    def __init__(self, exec_results=None):
        self.exec_results = list(exec_results or [])
        self.create_calls = []
        self.exec_calls = []
        self.ready_calls = []
        self.bot_protection_calls = []
        self.fs = FakeFs()

    def create(self, name, **kwargs):
        self.create_calls.append({"name": name, **kwargs})
        return SimpleNamespace(id="m-1", name=name, status="starting")

    def wait_until_ready(self, machine_id, *, timeout=90.0):
        self.ready_calls.append((machine_id, timeout))
        return SimpleNamespace(id=machine_id, status="running")

    def exec(self, machine_id, command, *, timeout=None):
        self.exec_calls.append({"machine_id": machine_id, "command": command, "timeout": timeout})
        return self.exec_results.pop(0)

    def set_bot_protection(self, machine_id, *, mode=None):
        self.bot_protection_calls.append((machine_id, mode))

    def get(self, machine_id):
        return SimpleNamespace(
            id=machine_id, access=SimpleNamespace(url="https://cognee-api.boxd.example")
        )


class FakeClient:
    def __init__(self, machines):
        self.machines = machines


def test_run_command_runs_bash_and_returns_result(monkeypatch, capsys):
    module = _import_module(monkeypatch)
    fake = FakeMachines(exec_results=[_exec_result(0, stdout="hi", stderr="warn")])

    result = module.run_command(FakeClient(fake), "m-1", "echo hi", timeout=30, label="x")

    assert result.exit_code == 0
    assert fake.exec_calls == [
        {"machine_id": "m-1", "command": ["bash", "-lc", "echo hi"], "timeout": 30}
    ]
    captured = capsys.readouterr()
    assert "[x] hi" in captured.out
    assert "[x] warn" in captured.err


def test_build_env_file_forwards_only_set_values(monkeypatch):
    module = _import_module(monkeypatch)
    for name in module.FORWARDED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_MODEL", "gemini/gemini-2.5-flash")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "")

    env_file = module.build_env_file("test-llm-key")

    assert env_file == (
        "LLM_API_KEY='test-llm-key'\nHOST='0.0.0.0'\nLLM_MODEL='gemini/gemini-2.5-flash'\n"
    )


@pytest.mark.parametrize(
    "password", ["abc #frag", "'quoted'", '"dq', "it's", "back\\slash", "a b=c", "$HOME"]
)
def test_build_env_file_values_round_trip_through_dotenv(monkeypatch, password):
    import io

    from dotenv import dotenv_values

    module = _import_module(monkeypatch)
    for name in module.FORWARDED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEFAULT_USER_PASSWORD", password)

    parsed = dotenv_values(stream=io.StringIO(module.build_env_file("k")))

    assert parsed["DEFAULT_USER_PASSWORD"] == password
    assert parsed["LLM_API_KEY"] == "k"


@pytest.mark.parametrize("password", ["line1\nline2", "x${HOME}y"])
def test_build_env_file_rejects_values_dotenv_would_change(monkeypatch, password):
    module = _import_module(monkeypatch)
    for name in module.FORWARDED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEFAULT_USER_PASSWORD", password)

    with pytest.raises(ValueError, match="DEFAULT_USER_PASSWORD"):
        module.build_env_file("k")


def test_wait_for_server_health_retries_until_ok(monkeypatch):
    module = _import_module(monkeypatch)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    fake = FakeMachines(exec_results=[_exec_result(7), _exec_result(0, stdout="ok")])

    assert module.wait_for_server_health(FakeClient(fake), "m-1") is True
    assert len(fake.exec_calls) == 2
    assert fake.exec_calls[0]["command"] == [
        "curl",
        "-fsS",
        "--max-time",
        "10",
        "http://localhost:8000/health",
    ]


def test_wait_for_server_health_prints_log_when_never_healthy(monkeypatch):
    module = _import_module(monkeypatch)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    fake = FakeMachines(exec_results=[_exec_result(7), _exec_result(7), _exec_result(0)])

    assert module.wait_for_server_health(FakeClient(fake), "m-1", retries=2) is False
    assert "journalctl -u cognee-api" in fake.exec_calls[-1]["command"][-1]


def test_deploy_requires_env_vars(monkeypatch):
    module = _import_module(monkeypatch)

    monkeypatch.delenv("BOXD_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="BOXD_API_KEY"):
        module.deploy_cognee()

    monkeypatch.setenv("BOXD_API_KEY", "test-key")
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        module.deploy_cognee()


def _set_env(monkeypatch, module):
    monkeypatch.setenv("BOXD_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    monkeypatch.delenv("BOXD_ORG", raising=False)
    monkeypatch.delenv("BOXD_MACHINE_NAME", raising=False)
    for name in module.FORWARDED_ENV:
        monkeypatch.delenv(name, raising=False)


def _deploy_with_fakes(monkeypatch, exec_results):
    """Run deploy_cognee() against scripted fakes; returns (module, machines, result)."""
    module = _import_module(monkeypatch)
    _set_env(monkeypatch, module)
    fake = FakeMachines(exec_results=exec_results)
    monkeypatch.setattr(module, "Boxd", lambda: FakeClient(fake))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    return module, fake


HAPPY_EXECS = [
    _exec_result(0, stdout="installed"),
    _exec_result(0, stdout="started"),
    _exec_result(0, stdout='{"status":"ready"}'),
]


def test_deploy_happy_path(monkeypatch):
    module, fake = _deploy_with_fakes(monkeypatch, list(HAPPY_EXECS))

    machine = module.deploy_cognee()

    assert machine.id == "m-1"
    assert fake.create_calls == [
        {"name": "cognee-api", "org": None, "isolated": True, "vcpu": 2, "memory": "8G"}
    ]
    assert fake.ready_calls == [("m-1", 300)]

    install, start, health = (call["command"] for call in fake.exec_calls)
    assert "uv pip install 'cognee[api]'" in install[-1]
    assert "systemctl enable --now cognee-api" in start[-1]
    assert health[-1] == "http://localhost:8000/health"

    env_file = fake.fs.writes["/home/boxd/cognee/.env"]
    assert "LLM_API_KEY='test-llm-key'\n" in env_file
    assert "HOST='0.0.0.0'\n" in env_file
    assert fake.fs.modes["/home/boxd/cognee/.env"] == 0o600
    unit = fake.fs.writes["/home/boxd/cognee/cognee-api.service"]
    assert "uvicorn cognee.api.client:app --host 0.0.0.0 --port 8000" in unit
    assert "Restart=on-failure" in unit

    assert fake.bot_protection_calls == [("m-1", "off")]


def test_deploy_never_sends_the_llm_key_as_create_env(monkeypatch):
    module, fake = _deploy_with_fakes(monkeypatch, list(HAPPY_EXECS))

    module.deploy_cognee()

    assert "env" not in fake.create_calls[0]
    assert all("test-llm-key" not in str(call["command"]) for call in fake.exec_calls)


def test_deploy_uses_org_and_machine_name_from_env(monkeypatch):
    module, fake = _deploy_with_fakes(monkeypatch, list(HAPPY_EXECS))
    monkeypatch.setenv("BOXD_ORG", "my-org")
    monkeypatch.setenv("BOXD_MACHINE_NAME", "cognee-staging")

    module.deploy_cognee()

    assert fake.create_calls[0]["name"] == "cognee-staging"
    assert fake.create_calls[0]["org"] == "my-org"


def test_deploy_url_printed(monkeypatch, capsys):
    module, _fake = _deploy_with_fakes(monkeypatch, list(HAPPY_EXECS))

    module.deploy_cognee()

    output = capsys.readouterr().out
    assert "https://cognee-api.boxd.example" in output
    assert "https://cognee-api.boxd.example/health" in output
    assert "https://cognee-api.boxd.example/docs" in output
    assert "Boxd().machines.delete('m-1')" in output


def test_deploy_aborts_when_install_fails(monkeypatch):
    module, fake = _deploy_with_fakes(monkeypatch, [_exec_result(1, stderr="pip boom")])

    with pytest.raises(RuntimeError, match="Failed to install cognee"):
        module.deploy_cognee()

    assert len(fake.exec_calls) == 1
    assert fake.fs.writes == {}
    assert fake.bot_protection_calls == []


def test_deploy_failure_names_the_machine_left_running(monkeypatch, capsys):
    module, _fake = _deploy_with_fakes(monkeypatch, [_exec_result(1, stderr="pip boom")])

    with pytest.raises(RuntimeError, match="Failed to install cognee"):
        module.deploy_cognee()

    err = capsys.readouterr().err
    assert "cognee-api (m-1) is still running" in err
    assert "Boxd().machines.delete('m-1')" in err


def test_deploy_interrupt_names_the_machine_left_running(monkeypatch, capsys):
    module, fake = _deploy_with_fakes(monkeypatch, [])

    def interrupted(machine_id, command, *, timeout=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(fake, "exec", interrupted)

    with pytest.raises(KeyboardInterrupt):
        module.deploy_cognee()

    err = capsys.readouterr().err
    assert "cognee-api (m-1) is still running" in err
    assert "Boxd().machines.delete('m-1')" in err


def test_deploy_names_the_machine_when_bot_protection_fails(monkeypatch, capsys):
    module, fake = _deploy_with_fakes(monkeypatch, list(HAPPY_EXECS))

    def refused(machine_id, *, mode=None):
        raise RuntimeError("bot protection boom")

    monkeypatch.setattr(fake, "set_bot_protection", refused)

    with pytest.raises(RuntimeError, match="bot protection boom"):
        module.deploy_cognee()

    captured = capsys.readouterr()
    assert "cognee-api (m-1) is still running" in captured.err
    assert "Boxd().machines.delete('m-1')" in captured.err
    assert "Cognee is running!" not in captured.out


def test_deploy_rejects_bad_env_value_before_creating_a_machine(monkeypatch):
    module, fake = _deploy_with_fakes(monkeypatch, [])
    monkeypatch.setenv("DEFAULT_USER_PASSWORD", "line1\nline2")

    with pytest.raises(ValueError, match="DEFAULT_USER_PASSWORD"):
        module.deploy_cognee()

    assert fake.create_calls == []


def test_deploy_aborts_when_server_start_fails(monkeypatch):
    module, fake = _deploy_with_fakes(
        monkeypatch, [_exec_result(0, stdout="installed"), _exec_result(1, stderr="unit boom")]
    )

    with pytest.raises(RuntimeError, match="Failed to start cognee API server"):
        module.deploy_cognee()

    assert len(fake.exec_calls) == 2
    assert fake.bot_protection_calls == []


def test_deploy_aborts_when_server_never_becomes_healthy(monkeypatch):
    module, fake = _deploy_with_fakes(
        monkeypatch, [_exec_result(0, stdout="installed"), _exec_result(0, stdout="started")]
    )
    monkeypatch.setattr(module, "wait_for_server_health", lambda *_args: False)

    with pytest.raises(RuntimeError, match="did not become healthy"):
        module.deploy_cognee()

    assert fake.bot_protection_calls == []


def test_deploy_reports_existing_machine_conflict(monkeypatch):
    module = _import_module(monkeypatch)
    _set_env(monkeypatch, module)

    class ConflictingMachines(FakeMachines):
        def create(self, name, **kwargs):
            raise module.ConflictError("name 'cognee-api' is already taken")

    monkeypatch.setattr(module, "Boxd", lambda: FakeClient(ConflictingMachines()))

    with pytest.raises(RuntimeError, match="is already taken") as excinfo:
        module.deploy_cognee()

    assert "c.machines.get('cognee-api').id" in str(excinfo.value)
