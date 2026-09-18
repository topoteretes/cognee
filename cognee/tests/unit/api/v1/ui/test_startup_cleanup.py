"""Startup owns its processes until all UI setup steps have succeeded."""

import importlib
import signal
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

ui = importlib.import_module("cognee.api.v1.ui.ui")


@pytest.fixture
def launch(monkeypatch, tmp_path):
    processes = [MagicMock(pid=pid) for pid in (101, 102, 103)]
    for process in processes:
        process.poll.return_value = None
        process.wait.return_value = 0
    mocks = {
        "_check_required_ports": MagicMock(return_value=(True, [])),
        "_check_docker_available": MagicMock(return_value=(True, "ready")),
        "_stream_process_output": MagicMock(),
        "find_frontend_path": MagicMock(return_value=tmp_path),
        "check_node_npm": MagicMock(return_value=(True, "ready")),
        "install_frontend_dependencies": MagicMock(return_value=True),
        "get_nvm_sh_path": MagicMock(return_value=tmp_path / "missing-nvm.sh"),
        "prompt_user_for_download": MagicMock(return_value=False),
        "download_frontend_assets": MagicMock(return_value=False),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(ui, name, mock)
    popen = MagicMock(side_effect=processes)
    run = MagicMock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    killpg = MagicMock()
    monkeypatch.setattr(ui.subprocess, "Popen", popen)
    monkeypatch.setattr(ui.subprocess, "run", run)
    monkeypatch.setattr(ui.os, "killpg", killpg, raising=False)
    monkeypatch.setattr(ui.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(ui.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ui.time, "sleep", lambda _: None)
    callback = MagicMock()
    return SimpleNamespace(
        processes=processes,
        mocks=mocks,
        popen=popen,
        run=run,
        killpg=killpg,
        callback=callback,
    )


def start(launch, **kwargs):
    return ui.start_ui(
        launch.callback, start_mcp=True, start_backend=True, open_browser=False, **kwargs
    )


def assert_cleaned(launch, count):
    for process in launch.processes[:count]:
        assert call(process.pid, signal.SIGTERM) in launch.killpg.call_args_list
        process.wait.assert_called_with(timeout=5)
    removals = [c.args[0] for c in launch.run.call_args_list if c.args[0][:2] == ["docker", "rm"]]
    docker_command = launch.popen.call_args_list[0].args[0]
    container_name = docker_command[docker_command.index("--name") + 1]
    assert removals == [["docker", "rm", "--force", container_name]]


@pytest.mark.parametrize(
    "failure", ["node", "dependencies", "declined", "download", "missing_after_download"]
)
def test_early_setup_returns_clean_up_owned_services(launch, failure):
    if failure == "node":
        launch.mocks["check_node_npm"].return_value = (False, "missing")
    elif failure == "dependencies":
        launch.mocks["install_frontend_dependencies"].return_value = False
    else:
        launch.mocks["find_frontend_path"].return_value = None
        launch.mocks["prompt_user_for_download"].return_value = failure != "declined"
        launch.mocks["download_frontend_assets"].return_value = failure == "missing_after_download"
    assert start(launch) is None
    assert_cleaned(launch, 2)


@pytest.mark.parametrize(
    "stage", ["find_frontend_path", "check_node_npm", "install_frontend_dependencies"]
)
def test_setup_exceptions_clean_up_owned_services(launch, stage):
    launch.mocks[stage].side_effect = RuntimeError("setup failed")
    assert start(launch) is None
    assert_cleaned(launch, 2)


@pytest.mark.parametrize("index", [1, 2])
def test_early_process_exit_cleans_up_all_started_resources(launch, index):
    launch.processes[index].poll.return_value = 1
    assert start(launch) is None
    assert_cleaned(launch, index + 1)


def test_frontend_spawn_failure_cleans_up_other_services(launch):
    launch.popen.side_effect = [*launch.processes[:2], OSError("cannot spawn npm")]
    assert start(launch) is None
    assert_cleaned(launch, 2)


def test_frontend_callback_failure_also_cleans_up_frontend(launch):
    launch.callback.side_effect = [None, None, RuntimeError("callback failed")]
    assert start(launch) is None
    assert_cleaned(launch, 3)


def test_keyboard_interrupt_cleans_up_and_propagates(launch):
    launch.mocks["install_frontend_dependencies"].side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        start(launch)
    assert_cleaned(launch, 2)


def test_success_hands_resources_to_caller(launch):
    assert start(launch) is launch.processes[2]
    launch.killpg.assert_not_called()
    assert len(launch.run.call_args_list) == 1  # image pull only
    for process in launch.processes:
        process.wait.assert_not_called()
    assert launch.callback.call_count == 3


def test_failed_optional_mcp_setup_is_cleaned_even_when_ui_succeeds(launch):
    launch.callback.side_effect = [RuntimeError("MCP callback failed"), None, None]
    assert start(launch) is launch.processes[2]
    assert_cleaned(launch, 1)
    for process in launch.processes[1:]:
        process.wait.assert_not_called()


def test_unavailable_ports_do_not_touch_existing_services(launch):
    launch.mocks["_check_required_ports"].return_value = (False, ["Frontend"])
    assert start(launch) is None
    launch.popen.assert_not_called()
    launch.killpg.assert_not_called()
    launch.run.assert_not_called()


def test_disabled_services_are_not_cleaned_up(launch):
    launch.mocks["check_node_npm"].return_value = (False, "missing")
    assert ui.start_ui(launch.callback, start_mcp=False, start_backend=False) is None
    launch.popen.assert_not_called()
    launch.killpg.assert_not_called()
    launch.run.assert_not_called()


def test_process_shutdown_escalates_and_bounds_both_waits(launch):
    process = launch.processes[0]
    process.wait.side_effect = subprocess.TimeoutExpired("process", 5)
    ui.stop_ui_process(process)
    assert launch.killpg.call_args_list == [call(101, signal.SIGTERM), call(101, signal.SIGKILL)]
    assert process.wait.call_args_list == [call(timeout=5), call(timeout=5)]


def test_already_exited_process_is_safe_to_clean_twice(launch):
    launch.killpg.side_effect = ProcessLookupError
    process = launch.processes[0]
    process.poll.return_value = 0
    ui.stop_ui_process(process)
    ui.stop_ui_process(process)
    process.wait.assert_called_with(timeout=5)


@pytest.mark.parametrize("error", [OSError("docker gone"), subprocess.TimeoutExpired("docker", 10)])
def test_container_cleanup_failure_does_not_prevent_other_cleanup(launch, error):
    launch.mocks["check_node_npm"].return_value = (False, "missing")

    def fail_the_removal(cmd, *args, **kwargs):
        # Keyed on the command rather than call order: remove_ui_container asks
        # for a graceful `docker stop` before forcing removal, so the forced call
        # is no longer simply the second subprocess.run of the teardown.
        if cmd[:2] == ["docker", "rm"]:
            raise error
        return subprocess.CompletedProcess([], 0, "", "")

    launch.run.side_effect = fail_the_removal
    assert start(launch) is None
    assert_cleaned(launch, 2)


def test_windows_shutdown_stops_tree_and_bounds_fallback_waits(launch, monkeypatch):
    monkeypatch.setattr(ui.platform, "system", lambda: "Windows")
    process = launch.processes[0]
    process.wait.side_effect = [subprocess.TimeoutExpired("process", 5), 0]
    ui.stop_ui_process(process)
    launch.run.assert_called_once_with(
        ["taskkill", "/PID", "101", "/T", "/F"],
        capture_output=True,
        timeout=5,
        check=True,
    )
    process.kill.assert_called_once_with()
    assert process.wait.call_args_list == [call(timeout=5), call(timeout=5)]
    launch.killpg.assert_not_called()


def test_already_removed_container_is_safe_to_clean_twice(launch):
    launch.run.return_value = subprocess.CompletedProcess([], 1, "", "No such container: owned")
    ui.remove_ui_container("owned")
    ui.remove_ui_container("owned")
    for cleanup_call in launch.run.call_args_list:
        assert cleanup_call.kwargs["timeout"] == 10
    # Each removal asks for a graceful stop first, then forces it. A container
    # that is already gone reports "No such container" on both and is ignored.
    assert [c.args[0] for c in launch.run.call_args_list] == [
        ["docker", "stop", "owned"],
        ["docker", "rm", "--force", "owned"],
        ["docker", "stop", "owned"],
        ["docker", "rm", "--force", "owned"],
    ]
