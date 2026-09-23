"""Optional MCP failures must not prevent launching the UI and backend."""

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ui = importlib.import_module("cognee.api.v1.ui.ui")


@pytest.fixture
def launch(monkeypatch):
    backend = MagicMock(pid=101)
    backend.poll.return_value = None
    frontend = MagicMock(pid=102)
    frontend.poll.return_value = None
    mcp = MagicMock(pid=103)
    mcp.poll.return_value = None

    def spawn(command, **kwargs):
        if command[0] == "docker":
            return mcp
        if "uvicorn" in command:
            return backend
        return frontend

    popen = MagicMock(side_effect=spawn)
    run = MagicMock()
    docker = MagicMock(return_value=(True, "ok"))
    logger = MagicMock()
    ports = MagicMock(return_value=True)
    monkeypatch.setattr(ui, "_is_port_available", ports)
    monkeypatch.setattr(ui, "_check_docker_available", docker)
    monkeypatch.setattr(ui, "find_frontend_path", lambda: Path("/tmp/frontend"))
    monkeypatch.setattr(ui, "check_node_npm", lambda: (True, "ok"))
    monkeypatch.setattr(ui, "install_frontend_dependencies", lambda path: True)
    monkeypatch.setattr(ui, "get_nvm_sh_path", lambda: MagicMock(exists=lambda: False))
    monkeypatch.setattr(ui, "_stream_process_output", MagicMock())
    monkeypatch.setattr(ui.time, "sleep", lambda _: None)
    monkeypatch.setattr(ui.subprocess, "Popen", popen)
    monkeypatch.setattr(ui.subprocess, "run", run)
    monkeypatch.setattr(ui, "logger", logger)
    return SimpleNamespace(
        frontend=frontend,
        backend=backend,
        mcp=mcp,
        popen=popen,
        run=run,
        docker=docker,
        ports=ports,
        logger=logger,
    )


@pytest.mark.parametrize("mcp_port", [8001, 18001])
def test_busy_mcp_port_starts_ui_and_backend_without_docker(launch, mcp_port):
    launch.ports.side_effect = lambda port: port != mcp_port
    callbacks = []
    result = ui.start_ui(
        callbacks.append,
        start_backend=True,
        start_mcp=True,
        mcp_port=mcp_port,
        open_browser=False,
    )
    assert result is launch.frontend
    assert callbacks == [101, 102]
    assert launch.popen.call_count == 2
    launch.run.assert_not_called()
    launch.docker.assert_not_called()
    launch.logger.warning.assert_called_once()
    launch.logger.error.assert_not_called()


@pytest.mark.parametrize("busy_port", [3000, 8000])
def test_required_port_conflict_aborts_before_any_process(launch, busy_port):
    launch.ports.side_effect = lambda port: port != busy_port
    callback = MagicMock()
    assert ui.start_ui(callback, start_backend=True, start_mcp=True) is None
    launch.popen.assert_not_called()
    launch.run.assert_not_called()
    launch.docker.assert_not_called()
    callback.assert_not_called()


def test_free_mcp_port_preserves_all_three_services(launch):
    callbacks = []
    assert (
        ui.start_ui(callbacks.append, start_backend=True, start_mcp=True, open_browser=False)
        is launch.frontend
    )
    assert len(callbacks) == 3
    assert callbacks[0][0] == 103
    assert callbacks[0][1].startswith("cognee-mcp-")
    assert callbacks[1:] == [101, 102]
    launch.docker.assert_called_once()
    assert launch.popen.call_args_list[0].args[0][:2] == ["docker", "run"]
    assert "8001:8000" in launch.popen.call_args_list[0].args[0]


def test_docker_unavailable_preserves_ui_and_backend(launch):
    launch.docker.return_value = (False, "Docker unavailable")
    callbacks = []
    assert (
        ui.start_ui(callbacks.append, start_backend=True, start_mcp=True, open_browser=False)
        is launch.frontend
    )
    assert callbacks == [101, 102]
    launch.run.assert_not_called()
    assert launch.popen.call_count == 2


def test_disabled_mcp_never_checks_its_port_or_docker(launch):
    assert (
        ui.start_ui(lambda _: None, start_backend=True, start_mcp=False, open_browser=False)
        is launch.frontend
    )
    assert [call.args[0] for call in launch.ports.call_args_list] == [3000, 8000]
    launch.docker.assert_not_called()
    launch.run.assert_not_called()
