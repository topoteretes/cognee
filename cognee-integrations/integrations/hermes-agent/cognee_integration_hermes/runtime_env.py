"""The private Python environment the local cognee server runs from.

cognee cannot live in Hermes's own environment: Hermes resolves every plugin
into one shared lock, where cognee's pins conflict with Hermes extras and the
whole install is refused. So local mode keeps cognee in a venv of its own and
starts the server from it (``server_bootstrap._spawn``). The plugin itself
never imports cognee in that mode — it talks to the server over HTTP.

The venv is built only on request (``hermes cognee install-runtime``, or the
local branch of ``hermes memory setup``), never as a side effect of a session
starting: it downloads cognee and its dependencies from PyPI. It is built from
the interpreter running Hermes (cognee supports it: Hermes runs 3.14, cognee
3.10-3.14) with uv when one is on PATH, else stdlib ``venv`` + pip, and uv is
never allowed to download an interpreter. One venv serves every Hermes profile,
as the server it runs does (one port, one store under ~/.cognee).

A runtime an earlier plugin version installed for another cognee pin does not
count as installed: local mode stays off, and says to run ``install-runtime``,
which upgrades the venv in place to the current pin. A cognee in Hermes's own
environment (installed by hand, or left behind by a plugin release up to 1.3.x,
which depended on it) is used only when it is the pinned version or newer. The
plugin never starts a server on an older cognee.

``install-runtime`` first stops every local server this plugin started (from the
runtime venv or from Hermes's own interpreter), so the venv is never upgraded
under a running process. Those servers are found by the pidfile ``_spawn``
writes, checked to still run the uvicorn app, and sent SIGTERM only. One that
does not stop aborts the install. If a cognee server older than the pin (or
reporting no version) still answers on the local port afterwards (one another
cognee plugin started, or one an earlier release started without a pidfile), the
install stops before it touches the ready marker or the venv and says which
process to stop; a current server another plugin runs does not block it. The
server is shared: the other cognee plugins on this machine use it too, and start
a new one on their next request.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from .backend import cognee_not_installed_message, has_cognee
from .config import LOCAL_COGNEE_REQUIREMENT, SHARED_PLUGIN_STATE_DIR

try:  # POSIX only; on Windows two concurrent installs are not serialized.
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows only
    fcntl = None  # type: ignore[assignment]

RUNTIME_DIR = SHARED_PLUGIN_STATE_DIR / "hermes" / "runtime"
VENV_DIR = RUNTIME_DIR / "venv"
# Written only after cognee installed and its version was read back, so a build
# cut short leaves no marker and the next install starts over.
READY_MARKER = RUNTIME_DIR / "ready.json"
INSTALL_TIMEOUT_SECS = 1800

Output = Callable[[str], None]


class RuntimeInstallError(RuntimeError):
    """Building the runtime venv failed; the message says which step."""


def pinned_version() -> str:
    return LOCAL_COGNEE_REQUIREMENT.split("==", 1)[1]


def _release(version: str) -> tuple[int, ...]:
    """The numeric release of a version string: "1.6.6.dev1" and "1.6.6-local" -> (1, 6, 6)."""
    match = re.match(r"\s*v?(\d+(?:\.\d+)*)", str(version or ""))
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def meets_pin(version: str) -> bool:
    """True when ``version`` is the pinned cognee or newer; False when unknown."""
    release = _release(version)
    return bool(release) and release >= _release(pinned_version())


def hermes_env_cognee_version() -> str:
    """The cognee version installed in Hermes's own environment, "" when none.

    Reads package metadata only, never imports cognee.
    """
    if not has_cognee():
        return ""
    try:
        return importlib.metadata.version("cognee")
    except importlib.metadata.PackageNotFoundError:
        return ""


def hermes_env_outdated() -> Optional[str]:
    """``cognee==X`` when Hermes's own environment holds a cognee older than the pin.

    Plugin releases up to 1.3.x depended on cognee, and ``pip install -U``
    never removes a dropped dependency, so an upgraded install usually still
    has that version. It is not used (see :func:`server_python`).
    """
    if not has_cognee():
        return None
    version = hermes_env_cognee_version()
    if meets_pin(version):
        return None
    return f"cognee=={version}" if version else "cognee (unknown version)"


def venv_python(venv_dir: Optional[Path] = None) -> Path:
    venv_dir = VENV_DIR if venv_dir is None else venv_dir
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _marker_requirement() -> Optional[str]:
    try:
        marker = json.loads(READY_MARKER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    requirement = marker.get("requirement") if isinstance(marker, dict) else None
    return requirement if isinstance(requirement, str) else None


def runtime_python() -> Optional[str]:
    """The runtime venv's interpreter, when it holds the pinned cognee."""
    if _marker_requirement() != LOCAL_COGNEE_REQUIREMENT:
        return None
    python = venv_python()
    return str(python) if python.exists() else None


def outdated_requirement() -> Optional[str]:
    """The ``cognee==X`` the runtime venv holds when X is not the current pin.

    An earlier plugin version installed it. It is not used: local mode stays
    off until ``install-runtime`` upgrades the venv to the current pin.
    """
    requirement = _marker_requirement()
    if (
        not requirement
        or requirement == LOCAL_COGNEE_REQUIREMENT
        or not requirement.startswith("cognee==")
    ):
        return None
    return requirement if venv_python().exists() else None


def server_python() -> Optional[str]:
    """The interpreter to start the local server with, or None when there is none.

    The runtime venv first: it is the supported path and holds the pinned
    version. A cognee in Hermes's own environment still works when it is the
    pinned version or newer; an older one (or one whose version can't be read)
    is never started.
    """
    python = runtime_python()
    if python:
        return python
    return sys.executable if has_cognee() and not hermes_env_outdated() else None


def local_ready() -> bool:
    return server_python() is not None


def local_not_ready_message() -> str:
    """What to tell a user whose local mode cannot start (no usable runtime)."""
    outdated = outdated_requirement()
    if not outdated:
        stale = hermes_env_outdated()
        if stale:
            return (
                f"cognee local mode is off: Hermes's environment holds {stale}, and this "
                f"plugin version needs cognee {pinned_version()} or newer. Run `hermes cognee "
                "install-runtime` in a terminal: it installs the pinned version into the "
                "plugin's own runtime, after stopping a local server the plugin started on "
                "the old one (a server it cannot stop, it names for you to stop). Then start "
                "a new Hermes session."
            )
        return cognee_not_installed_message("local")
    return (
        f"cognee local mode is off: its runtime holds {outdated}, and this plugin version "
        f"needs {LOCAL_COGNEE_REQUIREMENT}. Run `hermes cognee install-runtime` in a "
        "terminal: it upgrades the runtime in place, after stopping a local server the "
        "plugin started on the old version (a server it cannot stop, it names for you to "
        "stop). Then start a new Hermes session."
    )


def embedded_outdated_message() -> str:
    """Embedded mode with a cognee older than the pin in Hermes's own environment.

    Embedded mode runs cognee in the Hermes process, so the runtime venv (and
    ``install-runtime``) does nothing for it: the fix is in Hermes's environment.
    """
    version = hermes_env_cognee_version()
    found = f"cognee {version}" if version else "a cognee of unknown version"
    return (
        f"cognee embedded mode is off: Hermes's environment holds {found}, and this plugin "
        f"version needs cognee {pinned_version()} or newer. Upgrade it there with "
        f"`{sys.executable} -m pip install {LOCAL_COGNEE_REQUIREMENT}`, or turn embedded mode "
        "off (unset COGNEE_EMBEDDED, and `embedded` in cognee.json) to use local mode, which "
        "runs its own server. Then start a new Hermes session."
    )


def _find_uv() -> Optional[str]:
    return shutil.which("uv")


class RuntimeInstallInProgressError(RuntimeError):
    """A server spawn found ``install-runtime`` holding the install lock."""


@contextlib.contextmanager
def _install_lock(blocking: bool = True):
    """The lock ``install-runtime`` holds while it stops servers and installs.

    A server spawn takes it without blocking (``blocking=False``), so a spawn
    never starts on a venv mid-upgrade: it raises
    :class:`RuntimeInstallInProgressError` instead, and an install never begins
    while a spawn is starting the interpreter.
    """
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if fcntl is None:
        yield
        return
    with open(RUNTIME_DIR / "install.lock", "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeInstallInProgressError(
                "the cognee runtime is being installed (hermes cognee install-runtime); "
                "start a new Hermes session once it finishes"
            ) from exc
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(handle, fcntl.LOCK_UN)


def spawn_lock():
    """Held around starting the local server; see :func:`_install_lock`."""
    return _install_lock(blocking=False)


# The command line every local server this plugin starts carries
# (server_bootstrap._spawn): ``<python> -m uvicorn cognee.api.client:app``.
_SERVER_APP = "cognee.api.client:app"
# How long ``install-runtime`` waits for a server it sent SIGTERM to. Above the
# server's own idle teardown, so an improve step another plugin started can
# drain. There is no SIGKILL: a server still up after this aborts the install.
STOP_TIMEOUT_SECS = 60.0


class ServerStillRunningError(RuntimeInstallError):
    """A local cognee server is still up, so the runtime was left untouched."""


def _pidfile_dir() -> Path:
    return RUNTIME_DIR / "servers"


def _pidfile(port: int) -> Path:
    return _pidfile_dir() / f"{int(port)}.pid"


def write_server_pidfile(pid: int, port: int, python: str) -> None:
    """Record a server ``server_bootstrap._spawn`` started; best-effort.

    ``install-runtime`` stops servers by these files only (see
    :func:`plugin_servers`), never by searching the process list.
    """
    try:
        _pidfile_dir().mkdir(parents=True, exist_ok=True)
        _pidfile(port).write_text(
            json.dumps({"pid": int(pid), "port": int(port), "python": str(python)}),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError):
        pass


def _read_pidfiles() -> list[tuple[Path, dict]]:
    try:
        paths = sorted(_pidfile_dir().glob("*.pid"))
    except OSError:
        return []
    records = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(record, dict) and int(record.get("pid") or 0) > 0:
                records.append((path, record))
                continue
        except (OSError, TypeError, ValueError):
            pass
        with contextlib.suppress(OSError):
            path.unlink()
    return records


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _runs_server_app(pid: int) -> bool:
    """True when ``pid`` is alive and still runs the cognee uvicorn app.

    The pid-reuse guard: a pidfile left behind by a server that exited may name
    an unrelated process by now. Never True on Windows, which has no ``ps``
    (and where ``os.kill(pid, 0)`` is not a probe): a server there is not
    stopped for the user, and the install's port check says what to stop.
    """
    if os.name == "nt" or pid == os.getpid() or not _alive(pid):
        return False
    try:
        # -ww: unlimited width. Linux ps otherwise cuts the line at the width it
        # takes from the environment (80 under pytest), dropping the app name
        # that ends a long interpreter path's command line.
        command = subprocess.run(
            ["ps", "-ww", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return _SERVER_APP in command


def plugin_servers(interpreters: Optional[set[str]] = None) -> list[tuple[int, int]]:
    """``(pid, port)`` of the running local servers this plugin started.

    Read from the pidfiles ``_spawn`` writes, each checked to be alive and still
    running the uvicorn app; a stale pidfile is removed. ``interpreters``
    narrows it to servers started from those (default: any). A server another
    cognee plugin started, or one an earlier plugin release started without a
    pidfile, is not found: the install's port check covers those.
    """
    servers = []
    for path, record in _read_pidfiles():
        pid = int(record["pid"])
        if not _runs_server_app(pid):
            if os.name != "nt":
                with contextlib.suppress(OSError):
                    path.unlink()
            continue
        if interpreters is not None and str(record.get("python") or "") not in interpreters:
            continue
        try:
            port = int(record.get("port") or 0)
        except (TypeError, ValueError):
            port = 0
        servers.append((pid, port))
    return servers


def servers_to_stop() -> list[tuple[int, int]]:
    """The servers :func:`install_runtime` would stop now, for announcing it."""
    if runtime_python():
        return plugin_servers({sys.executable}) if hermes_env_outdated() else []
    return plugin_servers()


def _pids_phrase(servers: list[tuple[int, int]]) -> str:
    pids = ", ".join(str(pid) for pid, _port in servers)
    return f"pid {pids}" if len(servers) == 1 else f"pids {pids}"


def stop_notice(servers: list[tuple[int, int]]) -> str:
    """What ``install-runtime`` says before it stops ``servers``."""
    ports = sorted({port for _pid, port in servers if port})
    where = f" on port {', '.join(str(p) for p in ports)}" if ports else ""
    noun = "server" if len(servers) == 1 else "servers"
    return (
        f"This stops the local cognee {noun} ({_pids_phrase(servers)}){where}; other cognee "
        "plugins using it will restart it on their next request."
    )


def stop_plugin_servers(
    output: Output = print,
    timeout: float = STOP_TIMEOUT_SECS,
    interpreters: Optional[set[str]] = None,
) -> list[int]:
    """Stop the local servers this plugin started; return their pids.

    ``interpreters`` narrows it to servers run by those (default: any this
    plugin started). SIGTERM only, so uvicorn shuts down cleanly and finishes
    the requests in flight; a server still running after ``timeout`` raises
    :class:`ServerStillRunningError` and is left running. The other cognee
    plugins sharing the server start a new one on their next request.
    """
    servers = plugin_servers(interpreters)
    for pid, port in servers:
        where = f" on port {port}" if port else ""
        output(f"Stopping the local cognee server this plugin started (pid {pid}{where}).")
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGTERM)
    pids = [pid for pid, _port in servers]
    deadline = time.monotonic() + timeout
    while any(_alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.2)
    running = [(pid, port) for pid, port in servers if _alive(pid)]
    for pid, port in servers:
        if (pid, port) not in running:
            with contextlib.suppress(OSError):
                _pidfile(port).unlink()
    if running:
        raise ServerStillRunningError(
            f"the local cognee server ({_pids_phrase(running)}) didn't stop; it may be "
            "finishing work for another cognee plugin. Try again in a minute, or stop it "
            "yourself. The runtime was left as it was."
        )
    if pids:
        output(
            "Other cognee plugins on this machine that used it start a new server on "
            "their next request."
        )
    return pids


def _configured_port() -> int:
    from .config import DEFAULT_LOCAL_PORT, load_config

    try:
        return int(load_config().get("local_port") or DEFAULT_LOCAL_PORT)
    except Exception:
        return DEFAULT_LOCAL_PORT


def _server_answering(port: int) -> bool:
    from .server_bootstrap import health_ok

    return health_ok(f"http://127.0.0.1:{int(port)}")


def _listening_pid(port: int) -> Optional[int]:
    """The pid listening on ``port``, from ``lsof`` when it is installed."""
    lsof = shutil.which("lsof")
    if os.name == "nt" or not lsof:
        return None
    try:
        out = subprocess.run(
            [lsof, "-t", f"-iTCP:{int(port)}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    first = out.split()[0] if out.split() else ""
    return int(first) if first.isdigit() else None


def find_listener_hint(port: int) -> str:
    """How to find the process on ``port``, for this platform."""
    if os.name == "nt":
        return f"`netstat -ano | findstr :{int(port)}` shows its pid"
    return f"`lsof -t -iTCP:{int(port)} -sTCP:LISTEN` shows its pid"


def _server_version(port: int) -> str:
    from .server_bootstrap import health_version

    return health_version(f"http://127.0.0.1:{int(port)}")


def still_running_message(port: int, version: str = "") -> str:
    """Why the install stopped: a cognee server older than the pin (or of
    unknown version) still answers on ``port``."""
    pid = _listening_pid(port)
    who = f" (pid {pid})" if pid else f" ({find_listener_hint(port)})"
    runs = f"cognee {version}" if version else "a cognee that reports no version"
    return (
        f"a cognee server is still running on 127.0.0.1:{int(port)}{who} on {runs}, older "
        f"than the pinned {pinned_version()} or unknown, and this plugin could not stop it "
        "(another cognee plugin, or an earlier version of this one, started it). The runtime "
        "was left as it was. Stop that server, then re-run `hermes cognee install-runtime`."
    )


def _require_port_free(port: int) -> None:
    """Abort the install while an outdated cognee server still answers on ``port``.

    A server on the pinned cognee or newer (another plugin's current one) does
    not block it: the venv is upgraded under no process of this plugin's. One
    that reports no version may be a pidfile-less 1.4.2 server, so it blocks.
    """
    if not _server_answering(port):
        return
    version = _server_version(port)
    if version and meets_pin(version):
        return
    raise ServerStillRunningError(still_running_message(port, version))


def _run(command: list[str], *, step: str, env: Optional[dict] = None, timeout: float) -> None:
    try:
        subprocess.run(command, env=env, check=True, timeout=timeout)
    except subprocess.CalledProcessError as exc:
        raise RuntimeInstallError(f"{step} failed (exit code {exc.returncode})") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeInstallError(f"{step} did not finish within {int(timeout)}s") from exc
    except OSError as exc:
        raise RuntimeInstallError(f"{step} could not start: {exc}") from exc


def _installed_version(python: Path, timeout: float = 60) -> str:
    # Reads package metadata only: importing cognee would take seconds and
    # touch its data directories.
    probe = "import importlib.metadata as m; print(m.version('cognee'))"
    try:
        result = subprocess.run(
            [str(python), "-c", probe],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def install_runtime(
    output: Output = print,
    timeout: float = INSTALL_TIMEOUT_SECS,
    port: Optional[int] = None,
) -> str:
    """Build (or update) the runtime venv; return its interpreter.

    Installer output streams straight to the terminal, so the user sees the
    download progress of a step that takes minutes. Concurrent calls wait for
    the first to finish and then find it ready.

    Before anything is touched, the servers this plugin started are stopped
    (:func:`stop_plugin_servers`), and the install aborts with
    :class:`ServerStillRunningError` while a cognee server older than the pin,
    or reporting no version, still answers on ``port`` (default: the configured
    local port). The ready marker and the venv are then left as they were.
    """
    with _install_lock():
        ready = runtime_python()
        if ready:
            # A server started earlier on an outdated cognee in Hermes's own
            # environment would keep answering on the port; stop it too.
            if hermes_env_outdated():
                stop_plugin_servers(output, interpreters={sys.executable})
            output(f"cognee runtime is ready ({LOCAL_COGNEE_REQUIREMENT}): {VENV_DIR}")
            return ready
        # Never upgrade the venv under a running server: a live process would
        # lazy-import modules of the new version into the old one.
        stop_plugin_servers(output)
        _require_port_free(_configured_port() if port is None else port)
        with contextlib.suppress(OSError):
            READY_MARKER.unlink()
        python = venv_python()
        uv = _find_uv()
        output(f"Installing {LOCAL_COGNEE_REQUIREMENT} into {VENV_DIR}")
        output("This downloads cognee and its dependencies from PyPI and takes a few minutes.")
        if uv:
            env = {**os.environ, "UV_PYTHON_DOWNLOADS": "never"}
            if not python.exists():
                _run(
                    [uv, "venv", str(VENV_DIR), "--python", sys.executable],
                    step="Creating the venv with uv",
                    env=env,
                    timeout=timeout,
                )
            _run(
                [uv, "pip", "install", "--python", str(python), LOCAL_COGNEE_REQUIREMENT],
                step="Installing cognee with uv",
                env=env,
                timeout=timeout,
            )
        else:
            if not python.exists():
                _run(
                    [sys.executable, "-m", "venv", str(VENV_DIR)],
                    step="Creating the venv",
                    timeout=timeout,
                )
            _run(
                [str(python), "-m", "pip", "install", LOCAL_COGNEE_REQUIREMENT],
                step="Installing cognee with pip",
                timeout=timeout,
            )
        version = _installed_version(python)
        if version != pinned_version():
            raise RuntimeInstallError(
                f"the venv reports cognee {version or 'missing'}, expected {pinned_version()}"
            )
        READY_MARKER.write_text(
            json.dumps({"requirement": LOCAL_COGNEE_REQUIREMENT, "python": str(python)}),
            encoding="utf-8",
        )
        output(f"cognee runtime is ready ({LOCAL_COGNEE_REQUIREMENT}): {VENV_DIR}")
        return str(python)


def remove_runtime() -> bool:
    """Delete the runtime venv; True when there was one."""
    if not RUNTIME_DIR.exists():
        return False
    with _install_lock():
        with contextlib.suppress(OSError):
            READY_MARKER.unlink()
        shutil.rmtree(VENV_DIR, ignore_errors=True)
    return True
