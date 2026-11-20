"""The private venv local mode starts its cognee server from (runtime_env).

Every subprocess is faked: these check which commands run, in what order, and
that the ready marker is written only once the pinned version is really there.
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognee_integration_hermes import runtime_env  # noqa: E402
from cognee_integration_hermes import server_bootstrap as sb  # noqa: E402
from cognee_integration_hermes.backend import CogneeNotInstalledError  # noqa: E402
from cognee_integration_hermes.config import LOCAL_COGNEE_REQUIREMENT  # noqa: E402

PINNED = LOCAL_COGNEE_REQUIREMENT.split("==", 1)[1]
# Taken at import, before conftest's hermetic_env stubs them out for every test.
_REAL_PLUGIN_SERVERS = runtime_env.plugin_servers
_REAL_STOP = runtime_env.stop_plugin_servers


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    monkeypatch.setattr(runtime_env, "RUNTIME_DIR", root)
    monkeypatch.setattr(runtime_env, "VENV_DIR", root / "venv")
    monkeypatch.setattr(runtime_env, "READY_MARKER", root / "ready.json")
    return root


def _python(root):
    return runtime_env.venv_python(root / "venv")


def _mark_ready(root, requirement=LOCAL_COGNEE_REQUIREMENT):
    python = _python(root)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    (root / "ready.json").write_text(json.dumps({"requirement": requirement}))


class _FakeRun:
    """Records install commands; makes `venv` create the interpreter file."""

    def __init__(self, root, version=PINNED, fail_on=None):
        self.root, self.version, self.fail_on = root, version, fail_on
        self.commands = []
        self.envs = []

    def __call__(self, command, **kwargs):
        command = [str(c) for c in command]
        if command[-2:] == ["-c", "import importlib.metadata as m; print(m.version('cognee'))"]:
            return subprocess.CompletedProcess(command, 0, stdout=f"{self.version}\n", stderr="")
        self.commands.append(command)
        self.envs.append(kwargs.get("env"))
        if self.fail_on and self.fail_on in command:
            raise subprocess.CalledProcessError(2, command)
        if "venv" in command:
            _python(self.root).parent.mkdir(parents=True, exist_ok=True)
            _python(self.root).touch()
        return subprocess.CompletedProcess(command, 0)


def test_not_ready_without_a_marker(runtime):
    assert runtime_env.runtime_python() is None


def test_a_marker_for_another_version_is_not_ready(runtime):
    _mark_ready(runtime, requirement="cognee==0.0.1")
    assert runtime_env.runtime_python() is None


def test_an_outdated_runtime_is_not_used(runtime):
    # A runtime an older plugin installed for its own pin keeps local mode off.
    _mark_ready(runtime, requirement="cognee==0.0.1")
    with mock.patch.object(runtime_env, "has_cognee", return_value=False):
        assert runtime_env.outdated_requirement() == "cognee==0.0.1"
        assert runtime_env.server_python() is None
        assert runtime_env.local_ready() is False


@pytest.mark.parametrize("venv", [True, False])
def test_the_not_ready_message_names_both_pins_only_for_an_outdated_runtime(runtime, venv):
    _mark_ready(runtime, requirement="cognee==0.0.1")
    if not venv:
        # A marker whose venv is gone is no runtime at all.
        _python(runtime).unlink()
    message = runtime_env.local_not_ready_message()
    if venv:
        assert "cognee==0.0.1" in message
        assert LOCAL_COGNEE_REQUIREMENT in message
        assert "hermes cognee install-runtime" in message
    else:
        assert "not installed yet" in message


def test_ready_with_a_matching_marker(runtime):
    _mark_ready(runtime)
    assert runtime_env.runtime_python() == str(_python(runtime))


def test_server_python_prefers_the_runtime(runtime):
    _mark_ready(runtime)
    with mock.patch.object(runtime_env, "has_cognee", return_value=True):
        assert runtime_env.server_python() == str(_python(runtime))


def test_server_python_falls_back_to_a_hand_installed_cognee(runtime):
    with mock.patch.object(runtime_env, "has_cognee", return_value=True):
        assert runtime_env.server_python() == sys.executable
    with mock.patch.object(runtime_env, "has_cognee", return_value=False):
        assert runtime_env.server_python() is None


def test_install_with_uv_never_downloads_an_interpreter(runtime):
    run = _FakeRun(runtime)
    with (
        mock.patch.object(runtime_env, "_find_uv", return_value="/bin/uv"),
        mock.patch.object(runtime_env.subprocess, "run", side_effect=run),
    ):
        python = runtime_env.install_runtime(output=lambda _: None)
    assert python == str(_python(runtime))
    assert run.commands == [
        ["/bin/uv", "venv", str(runtime / "venv"), "--python", sys.executable],
        ["/bin/uv", "pip", "install", "--python", str(_python(runtime)), LOCAL_COGNEE_REQUIREMENT],
    ]
    assert all(env["UV_PYTHON_DOWNLOADS"] == "never" for env in run.envs)
    assert json.loads((runtime / "ready.json").read_text())["requirement"] == (
        LOCAL_COGNEE_REQUIREMENT
    )


def test_install_without_uv_uses_venv_and_pip(runtime):
    run = _FakeRun(runtime)
    with (
        mock.patch.object(runtime_env, "_find_uv", return_value=None),
        mock.patch.object(runtime_env.subprocess, "run", side_effect=run),
    ):
        runtime_env.install_runtime(output=lambda _: None)
    assert run.commands == [
        [sys.executable, "-m", "venv", str(runtime / "venv")],
        [str(_python(runtime)), "-m", "pip", "install", LOCAL_COGNEE_REQUIREMENT],
    ]
    assert runtime_env.runtime_python() == str(_python(runtime))


def test_a_ready_runtime_is_not_reinstalled(runtime):
    _mark_ready(runtime)
    with mock.patch.object(runtime_env.subprocess, "run") as run:
        runtime_env.install_runtime(output=lambda _: None)
    run.assert_not_called()


def test_an_older_runtime_is_upgraded_in_place(runtime):
    # A plugin update moves the pin: the existing venv is reused, cognee reinstalled.
    _mark_ready(runtime, requirement="cognee==0.0.1")
    run = _FakeRun(runtime)
    with (
        mock.patch.object(runtime_env, "_find_uv", return_value="/bin/uv"),
        mock.patch.object(runtime_env.subprocess, "run", side_effect=run),
    ):
        runtime_env.install_runtime(output=lambda _: None)
    assert run.commands == [
        ["/bin/uv", "pip", "install", "--python", str(_python(runtime)), LOCAL_COGNEE_REQUIREMENT],
    ]
    assert json.loads((runtime / "ready.json").read_text())["requirement"] == (
        LOCAL_COGNEE_REQUIREMENT
    )
    assert runtime_env.runtime_python() == str(_python(runtime))
    assert runtime_env.outdated_requirement() is None


@pytest.mark.parametrize("outdated", [False, True])
def test_a_failed_step_leaves_no_marker(runtime, outdated):
    if outdated:
        # The marker goes before the install, so a half-upgraded venv is never used.
        _mark_ready(runtime, requirement="cognee==0.0.1")
    run = _FakeRun(runtime, fail_on="install")
    with (
        mock.patch.object(runtime_env, "_find_uv", return_value="/bin/uv"),
        mock.patch.object(runtime_env.subprocess, "run", side_effect=run),
    ):
        with pytest.raises(runtime_env.RuntimeInstallError, match="Installing cognee with uv"):
            runtime_env.install_runtime(output=lambda _: None)
    assert not (runtime / "ready.json").exists()


def test_a_wrong_installed_version_leaves_no_marker(runtime):
    run = _FakeRun(runtime, version="0.0.1")
    with (
        mock.patch.object(runtime_env, "_find_uv", return_value="/bin/uv"),
        mock.patch.object(runtime_env.subprocess, "run", side_effect=run),
    ):
        with pytest.raises(runtime_env.RuntimeInstallError, match="expected"):
            runtime_env.install_runtime(output=lambda _: None)
    assert runtime_env.runtime_python() is None


def test_remove_runtime(runtime):
    assert runtime_env.remove_runtime() is False
    _mark_ready(runtime)
    assert runtime_env.remove_runtime() is True
    assert runtime_env.runtime_python() is None
    assert not (runtime / "venv").exists()


def test_the_server_is_spawned_with_the_runtime_interpreter():
    with (
        mock.patch.object(sb, "health_ok", side_effect=[False, True]),
        mock.patch.object(sb, "server_python", return_value="/runtime/bin/python"),
        mock.patch.object(sb.subprocess, "Popen") as popen,
        mock.patch.object(sb.time, "sleep"),
    ):
        sb.ensure_local_server(8011, log_path="/dev/null")
    argv = popen.call_args.args[0]
    assert argv[:4] == ["/runtime/bin/python", "-m", "uvicorn", "cognee.api.client:app"]


def test_an_outdated_runtime_starts_no_server(runtime):
    _mark_ready(runtime, requirement="cognee==0.0.1")
    with (
        mock.patch.object(sb, "health_ok", return_value=False),
        mock.patch.object(sb.subprocess, "Popen") as popen,
        mock.patch.object(runtime_env.subprocess, "run") as run,
        mock.patch.object(runtime_env, "has_cognee", return_value=False),
    ):
        with pytest.raises(CogneeNotInstalledError, match="install-runtime") as raised:
            sb.ensure_local_server(8011, log_path="/dev/null")
    assert "cognee==0.0.1" in str(raised.value)
    assert LOCAL_COGNEE_REQUIREMENT in str(raised.value)
    popen.assert_not_called()
    run.assert_not_called()  # nothing is installed behind the user's back


def test_status_says_an_outdated_runtime_keeps_local_mode_off(runtime):
    from cognee_integration_hermes import cli

    _mark_ready(runtime, requirement="cognee==0.0.1")
    with mock.patch.object(runtime_env, "has_cognee", return_value=False):
        status = cli._runtime_status()
    assert status.startswith("outdated")
    assert "cognee==0.0.1" in status
    assert "install-runtime" in status
    # A cognee in Hermes's own environment at the pin (or newer) still runs local mode.
    with (
        mock.patch.object(runtime_env, "has_cognee", return_value=True),
        mock.patch.object(runtime_env.importlib.metadata, "version", return_value=PINNED),
    ):
        assert cli._runtime_status() == f"cognee {PINNED} installed in Hermes's environment"


# -- the pinned version or newer, and servers on an older cognee ---------------


@pytest.mark.parametrize(
    ("version", "ok"),
    [
        (PINNED, True),
        ("99.0.0", True),
        (f"{PINNED}.dev1", True),
        (f"{PINNED}-local", True),
        ("1.6.5", False),
        ("0.0.1", False),
        ("", False),
        ("unknown", False),
    ],
)
def test_meets_pin(version, ok):
    assert runtime_env.meets_pin(version) is ok


def _hermes_env_cognee(version):
    return (
        mock.patch.object(runtime_env, "has_cognee", return_value=True),
        mock.patch.object(runtime_env.importlib.metadata, "version", return_value=version),
    )


def test_an_older_cognee_in_hermess_environment_is_never_started(runtime):
    # Releases up to 1.3.x depended on cognee 1.6.5, and `pip install -U` leaves it.
    has, version = _hermes_env_cognee("1.6.5")
    with (
        has,
        version,
        mock.patch.object(sb, "health_ok", return_value=False),
        mock.patch.object(sb.subprocess, "Popen") as popen,
    ):
        assert runtime_env.server_python() is None
        assert runtime_env.local_ready() is False
        with pytest.raises(CogneeNotInstalledError, match="install-runtime") as raised:
            sb.ensure_local_server(8011, log_path="/dev/null")
    assert "Hermes's environment holds cognee==1.6.5" in str(raised.value)
    popen.assert_not_called()


def test_status_says_hermess_older_cognee_keeps_local_mode_off(runtime):
    from cognee_integration_hermes import cli

    has, version = _hermes_env_cognee("1.6.5")
    with has, version:
        status = cli._runtime_status()
    assert status.startswith("outdated (cognee==1.6.5, Hermes's environment)")
    assert "install-runtime" in status


@pytest.mark.parametrize("version", [PINNED, "99.0.0"])
def test_the_pinned_or_a_newer_cognee_in_hermess_environment_runs(runtime, version):
    has, installed = _hermes_env_cognee(version)
    with has, installed:
        assert runtime_env.server_python() == sys.executable


def test_a_cognee_of_unknown_version_in_hermess_environment_is_not_started(runtime):
    with (
        mock.patch.object(runtime_env, "has_cognee", return_value=True),
        mock.patch.object(
            runtime_env.importlib.metadata,
            "version",
            side_effect=runtime_env.importlib.metadata.PackageNotFoundError("cognee"),
        ),
    ):
        assert runtime_env.server_python() is None


@pytest.mark.parametrize(("version", "accepted"), [("1.6.5", False), (PINNED, True), ("", True)])
def test_a_running_server_on_an_older_cognee_is_not_used(runtime, monkeypatch, version, accepted):
    # The server is shared by every cognee plugin: one another plugin started on
    # an older cognee answers /health, but recall against it is never routed.
    monkeypatch.setattr(sb, "health_version", lambda url, timeout=2.0: version)
    with (
        mock.patch.object(sb, "health_ok", return_value=True),
        mock.patch.object(sb.subprocess, "Popen") as popen,
    ):
        if accepted:
            assert sb.ensure_local_server(8011, log_path="/dev/null") == "http://127.0.0.1:8011"
        else:
            with pytest.raises(sb.OutdatedServerError) as raised:
                sb.ensure_local_server(8011, log_path="/dev/null")
            assert "port 8011 runs cognee 1.6.5" in str(raised.value)
            assert "install-runtime" in str(raised.value)
            assert isinstance(raised.value, CogneeNotInstalledError)
    popen.assert_not_called()


@pytest.mark.skipif(runtime_env.fcntl is None, reason="the install lock is POSIX-only")
def test_no_server_spawns_while_the_runtime_is_being_installed(runtime):
    _mark_ready(runtime)
    with (
        runtime_env._install_lock(),
        mock.patch.object(sb, "health_ok", return_value=False),
        mock.patch.object(sb.subprocess, "Popen") as popen,
    ):
        with pytest.raises(CogneeNotInstalledError, match="being installed"):
            sb.ensure_local_server(8011, log_path="/dev/null")
    popen.assert_not_called()


# -- install-runtime stops the servers this plugin started --------------------

_posix_only = pytest.mark.skipif(os.name == "nt", reason="servers are found with ps")


def _child(*, ignore_sigterm=False, app=True):
    """A real process whose command line carries the server app (or not).

    Reaped in a thread, as a detached server would be by init: a zombie still
    answers ``kill(pid, 0)``.
    """
    code = "import signal, sys, time\n"
    if ignore_sigterm:
        code += "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    code += "print('ready', flush=True)\ntime.sleep(60)\n"
    argv = [sys.executable, "-c", code] + (["cognee.api.client:app"] if app else [])
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready"
    threading.Thread(target=proc.wait, daemon=True).start()
    return proc


@pytest.fixture
def real_servers(monkeypatch):
    monkeypatch.setattr(runtime_env, "plugin_servers", _REAL_PLUGIN_SERVERS)


@_posix_only
def test_a_spawned_server_is_found_by_its_pidfile(runtime, real_servers):
    proc = _child()
    try:
        runtime_env.write_server_pidfile(proc.pid, 8011, sys.executable)
        assert runtime_env.plugin_servers() == [(proc.pid, 8011)]
        assert runtime_env.plugin_servers({sys.executable}) == [(proc.pid, 8011)]
        assert runtime_env.plugin_servers({"/other/python"}) == []
    finally:
        proc.kill()


@_posix_only
def test_a_stale_pidfile_naming_another_process_is_not_killed(runtime, real_servers, monkeypatch):
    # The server exited and its pid now belongs to something else.
    other = _child(app=False)
    try:
        runtime_env.write_server_pidfile(other.pid, 8011, sys.executable)
        sent = []
        monkeypatch.setattr(runtime_env.os, "kill", lambda pid, sig: sent.append(sig))
        assert runtime_env.stop_plugin_servers(output=lambda _l: None, timeout=0.1) == []
        assert sent == [0]  # only probed, never signalled
        assert not runtime_env._pidfile(8011).exists()
    finally:
        other.kill()


@_posix_only
def test_a_server_that_exits_on_sigterm_lets_the_install_proceed(runtime, real_servers):
    _mark_ready(runtime, requirement="cognee==0.0.1")
    proc = _child()
    runtime_env.write_server_pidfile(proc.pid, 8011, str(_python(runtime)))
    out = []
    try:
        with (
            mock.patch.object(runtime_env, "_find_uv", return_value=None),
            mock.patch.object(runtime_env.subprocess, "run", side_effect=_FakeRun(runtime)),
            # subprocess.run is faked for the installers, so `ps` is too.
            mock.patch.object(runtime_env, "_runs_server_app", return_value=True),
        ):
            runtime_env.install_runtime(output=out.append, port=8011)
    finally:
        proc.kill()
    assert proc.poll() is not None or not runtime_env._alive(proc.pid)
    assert runtime_env.runtime_python() == str(_python(runtime))
    assert not runtime_env._pidfile(8011).exists()
    assert any(f"pid {proc.pid}" in line for line in out)
    assert any("start a new server on their next request" in line for line in out)


@_posix_only
def test_a_server_that_ignores_sigterm_aborts_the_install_untouched(
    runtime, real_servers, monkeypatch
):
    _mark_ready(runtime, requirement="cognee==0.0.1")
    marker = (runtime / "ready.json").read_text()
    proc = _child(ignore_sigterm=True)
    runtime_env.write_server_pidfile(proc.pid, 8011, str(_python(runtime)))
    monkeypatch.setattr(
        runtime_env,
        "stop_plugin_servers",
        lambda output, interpreters=None: _REAL_STOP(output, timeout=0.5),
    )
    try:
        with mock.patch.object(runtime_env, "_run") as install_step:
            with pytest.raises(runtime_env.ServerStillRunningError, match=f"pid {proc.pid}"):
                runtime_env.install_runtime(output=lambda _l: None, port=8011)
        assert proc.poll() is None  # still running: never SIGKILLed
        install_step.assert_not_called()
    finally:
        proc.kill()
    assert (runtime / "ready.json").read_text() == marker
    assert _python(runtime).exists()
    assert runtime_env._pidfile(8011).exists()


@pytest.mark.parametrize("version", ["1.6.5", ""])
def test_an_outdated_or_versionless_server_answering_aborts_the_install_untouched(
    runtime, monkeypatch, version
):
    # One this plugin did not start (no pidfile): another plugin's, or 1.4.2's,
    # which may report no version.
    _mark_ready(runtime, requirement="cognee==0.0.1")
    marker = (runtime / "ready.json").read_text()
    monkeypatch.setattr(runtime_env, "_server_answering", lambda port: port == 8011)
    monkeypatch.setattr(sb, "health_version", lambda url, timeout=2.0: version)
    monkeypatch.setattr(runtime_env, "_listening_pid", lambda port: 555)
    with mock.patch.object(runtime_env.subprocess, "run") as run:
        with pytest.raises(runtime_env.ServerStillRunningError) as raised:
            runtime_env.install_runtime(output=lambda _l: None, port=8011)
    run.assert_not_called()
    message = str(raised.value)
    assert "127.0.0.1:8011 (pid 555)" in message
    assert (f"cognee {version}" if version else "reports no version") in message
    assert "re-run `hermes cognee install-runtime`" in message
    assert (runtime / "ready.json").read_text() == marker
    assert _python(runtime).exists()


@pytest.mark.parametrize("version", [PINNED, "99.0.0"])
def test_a_current_server_answering_lets_the_install_proceed(runtime, monkeypatch, version):
    # Another plugin's server on the pinned cognee (or newer) runs from its own
    # interpreter: upgrading this plugin's venv does not touch it.
    _mark_ready(runtime, requirement="cognee==0.0.1")
    monkeypatch.setattr(runtime_env, "_server_answering", lambda port: True)
    monkeypatch.setattr(sb, "health_version", lambda url, timeout=2.0: version)
    run = _FakeRun(runtime)
    with (
        mock.patch.object(runtime_env, "_find_uv", return_value=None),
        mock.patch.object(runtime_env.subprocess, "run", side_effect=run),
    ):
        runtime_env.install_runtime(output=lambda _l: None, port=8011)
    assert runtime_env.runtime_python() == str(_python(runtime))


def test_without_lsof_the_message_says_how_to_find_the_server(monkeypatch):
    monkeypatch.setattr(runtime_env, "_listening_pid", lambda port: None)
    message = runtime_env.still_running_message(8011)
    assert ("netstat -ano" if os.name == "nt" else "lsof -t -iTCP:8011 -sTCP:LISTEN") in message


def test_stopping_never_sends_sigkill(runtime, monkeypatch):
    sent = []

    def fake_kill(pid, sig):
        if sig != 0:
            sent.append(sig)

    monkeypatch.setattr(runtime_env, "plugin_servers", lambda interpreters=None: [(7, 8011)])
    monkeypatch.setattr(runtime_env.os, "kill", fake_kill)
    with pytest.raises(runtime_env.ServerStillRunningError, match="pid 7"):
        runtime_env.stop_plugin_servers(output=lambda _l: None, timeout=0.01)
    assert sent == [runtime_env.signal.SIGTERM]


def test_windows_never_signals_a_pidfile_pid(runtime, real_servers, monkeypatch):
    runtime_env.write_server_pidfile(4242, 8011, sys.executable)
    monkeypatch.setattr(runtime_env.os, "name", "nt")
    with mock.patch.object(runtime_env.os, "kill") as kill:
        assert runtime_env.plugin_servers() == []
    kill.assert_not_called()


def test_the_spawn_writes_the_pidfile(runtime, tmp_path):
    with mock.patch.object(sb.subprocess, "Popen") as popen:
        popen.return_value.pid = 4321
        sb._spawn(8011, "", "", str(tmp_path / "server.log"), python="/runtime/bin/python")
    record = json.loads(runtime_env._pidfile(8011).read_text())
    assert record == {"pid": 4321, "port": 8011, "python": "/runtime/bin/python"}


def test_a_ready_runtime_still_stops_a_server_on_hermess_older_cognee(runtime, monkeypatch):
    _mark_ready(runtime)
    asked = []
    monkeypatch.setattr(
        runtime_env,
        "stop_plugin_servers",
        lambda output, interpreters=None: asked.append(interpreters),
    )
    has, version = _hermes_env_cognee("1.6.5")
    with has, version, mock.patch.object(runtime_env.subprocess, "run") as run:
        runtime_env.install_runtime(output=lambda _l: None)
    assert asked == [{sys.executable}]
    run.assert_not_called()


def test_the_outdated_server_message_promises_no_stop():
    message = sb.outdated_server_message(8011, "1.6.5")
    assert "when this plugin started it" in message
    assert "stop the process listening on port 8011" in message


# -- install-runtime says what it stops, and asks -----------------------------


def _cli_install(monkeypatch, *, servers, tty, answer="", yes=False):
    from cognee_integration_hermes import cli

    monkeypatch.setattr(runtime_env, "servers_to_stop", lambda: servers)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: tty, raising=False)
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or answer)
    with mock.patch.object(runtime_env, "install_runtime") as install:
        code = cli._install_runtime(mock.Mock(yes=yes))
    return code, install.called, asked


def test_no_tty_without_yes_stops_nothing(monkeypatch, capsys):
    code, installed, asked = _cli_install(monkeypatch, servers=[(42, 8011)], tty=False)
    assert (code, installed, asked) == (1, False, [])
    out = capsys.readouterr().out
    assert "This stops the local cognee server (pid 42) on port 8011" in out
    assert "--yes" in out


def test_yes_proceeds_without_asking(monkeypatch):
    code, installed, asked = _cli_install(monkeypatch, servers=[(42, 8011)], tty=False, yes=True)
    assert (code, installed, asked) == (0, True, [])


@pytest.mark.parametrize(("answer", "installed"), [("", True), ("y", True), ("n", False)])
def test_a_terminal_asks_before_stopping(monkeypatch, answer, installed):
    code, did, asked = _cli_install(monkeypatch, servers=[(42, 8011)], tty=True, answer=answer)
    assert did is installed and code == (0 if installed else 1)
    assert asked == ["Continue? (y/n) [y]: "]


def test_nothing_running_asks_nothing(monkeypatch):
    code, installed, asked = _cli_install(monkeypatch, servers=[], tty=False)
    assert (code, installed, asked) == (0, True, [])


def test_the_cli_reports_a_server_that_did_not_stop(monkeypatch, capsys):
    from cognee_integration_hermes import cli

    monkeypatch.setattr(runtime_env, "servers_to_stop", lambda: [])
    with mock.patch.object(
        runtime_env,
        "install_runtime",
        side_effect=runtime_env.ServerStillRunningError("the local cognee server (pid 9)"),
    ):
        assert cli._install_runtime(mock.Mock(yes=False)) == 1
    assert "pid 9" in capsys.readouterr().out


def test_install_runtime_help_mentions_the_stop_and_yes():
    import argparse

    from cognee_integration_hermes import cli

    parser = argparse.ArgumentParser()
    cli.register_cli(parser)
    sub = parser._subparsers._group_actions[0].choices["install-runtime"]
    text = sub.format_help()
    assert "--yes" in text and "SIGTERM" in text


@pytest.mark.parametrize("servers", [[], [(42, 8011)]])
def test_the_setup_wizard_names_the_pid_only_when_a_server_runs(monkeypatch, servers):
    from cognee_integration_hermes import provider as provider_mod

    labels = []
    monkeypatch.setattr(runtime_env, "servers_to_stop", lambda: servers)
    monkeypatch.setattr(
        provider_mod, "_prompt_bool", lambda label, default: labels.append(label) or False
    )
    provider_mod._offer_runtime_install()
    (label,) = labels
    assert ("This stops the running local server (pid 42)." in label) is bool(servers)
    assert "about 1 GB from PyPI" in label
