"""Ensure a local cognee HTTP server is running, so Hermes can be a thin client.

Why a server instead of the in-process SDK: cognee's local stores (SQLite
relational, Kuzu/Ladybug graph, LanceDB vector) are **single-writer**. Driving
them in-process from Hermes's background threads — or from two Hermes processes
sharing one data dir — risks "database is locked"/corruption. cognee uses a
``DatasetQueue`` + subprocess DB workers precisely because the HTTP server is the
intended *single owner* that serializes access. So local mode points the SDK at a
local server (``cognee.serve(url)``) and lets the server own the databases.

This mirrors the proven Claude/Codex bootstrap: health-check, spawn uvicorn if
needed, poll ``/health``. No explicit lock is needed — only one process can bind
the port, so concurrent spawns simply lose the bind and then observe health.
"""

import json
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .backend import CogneeNotInstalledError
from .config import (
    DEFAULT_SERVER_BOOT_TIMEOUT,
    DEFAULT_USER_EMAIL,
    DEFAULT_USER_PASSWORD,
    SHARED_COGNEE_HOME,
    SHARED_PLUGIN_STATE_DIR,
    ollama_embedding_pins,
    secret,
)
from .runtime_env import (
    RuntimeInstallInProgressError,
    find_listener_hint,
    local_not_ready_message,
    meets_pin,
    pinned_version,
    server_python,
    spawn_lock,
    write_server_pidfile,
)

logger = logging.getLogger(__name__)

# LLM providers cognee can reach without an API key (a local model server).
_KEYLESS_LLM_PROVIDERS = frozenset({"ollama"})
_warned_no_llm_key = False
# Every API key cognee's LLM and embedding clients read. The server gets each one
# from the profile's secret scope only, so a profile never bills its calls to a
# key the gateway's environment holds for another profile. Non-secret settings
# (provider, model, endpoint) are inherited as before: calls bill to the key.
_PROFILE_SECRETS = (
    "LLM_API_KEY",
    "EMBEDDING_API_KEY",
    "LLM_EXTRACTION_API_KEY",
    "LLM_QUERY_API_KEY",
    "LLM_SUMMARIZATION_API_KEY",
)


def health_ok(url, timeout=2.0):
    """True when GET {url}/health returns a 2xx."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            return 200 <= int(status) < 300
    except Exception:
        return False


def health_version(url, timeout=2.0):
    """The cognee version a server's /health reports, "" when it reports none."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8") or "{}")
    except Exception:
        return ""
    version = body.get("version") if isinstance(body, dict) else ""
    return str(version or "").strip()


class OutdatedServerError(CogneeNotInstalledError):
    """The healthy server on the local port runs a cognee older than the pin."""


def outdated_server_message(port, version):
    return (
        f"Cognee memory is off: the local cognee server on port {port} runs cognee "
        f"{version}, and this plugin needs cognee {pinned_version()} or newer. Run "
        "`hermes cognee install-runtime`: it installs the pinned version, and stops that "
        "server first when this plugin started it. It cannot stop one another cognee plugin "
        f"or an earlier release started: stop the process listening on port {port} "
        f"({find_listener_hint(port)}), then start a new Hermes session."
    )


def _require_current(url, port):
    """Refuse a server on an older cognee; one reporting no version is accepted.

    The local server is shared by every cognee plugin on this machine, so the
    one answering may have been started by another plugin on an older cognee.
    Recall against it would never be routed, so Hermes reports Cognee as
    unavailable until it is replaced.
    """
    version = health_version(url)
    if version and not meets_pin(version):
        raise OutdatedServerError(outdated_server_message(port, version))
    return url


def port_bound(port, timeout=1.0):
    """True when something accepts TCP connections on the local port.

    Weaker than :func:`health_ok` on purpose: uvicorn binds the port well before
    the app is ready, so this distinguishes "a server is coming up" from "nothing
    is there at all".
    """
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=timeout):
            return True
    except OSError:
        return False


#: Rotate the server log when it is this big at the next spawn. It holds the
#: server's console output for the process lifetime and is read back by the
#: overflow scan (http_backend), so it cannot go to /dev/null like the other
#: plugins' server output — but nothing needs more than the recent past, and an
#: unbounded file is how one looping traceback turned into gigabytes.
SERVER_LOG_MAX_BYTES = 20 * 1024 * 1024
SERVER_LOG_MAX_BYTES_ENV = "COGNEE_SERVER_LOG_MAX_BYTES"


def _server_log_cap():
    raw = os.environ.get(SERVER_LOG_MAX_BYTES_ENV, "").strip()
    if not raw:
        return SERVER_LOG_MAX_BYTES
    try:
        return max(0, int(float(raw)))
    except ValueError:
        return SERVER_LOG_MAX_BYTES


def rotate_oversized_log(log_path, max_bytes=None):
    """Move ``log_path`` to ``<log_path>.1`` when it exceeds ``max_bytes``.

    One generation is kept: the interesting part of a failure is usually the
    stretch right before the cap hit, and that survives in ``.1``. Best-effort —
    a rotation that cannot happen must never stop the server from booting.
    Returns True when a rotation happened. The overflow scan tolerates this: it
    treats a shrunk log as rotated and rescans from the top.
    """
    cap = _server_log_cap() if max_bytes is None else max_bytes
    if cap <= 0:
        return False
    try:
        if os.path.getsize(log_path) <= cap:
            return False
        os.replace(log_path, str(log_path) + ".1")
        return True
    except OSError:
        return False


def server_workdir():
    """Where the spawned server runs: the plugin's own state directory.

    cognee >= 1.6.2 loads the first ``.env`` found walking up from its working
    directory and lets it override the environment, so a server started in the
    directory Hermes was launched from could pick up an unrelated project's
    ``.env``. Its settings come from Hermes instead (``_spawn``).
    """
    workdir = SHARED_PLUGIN_STATE_DIR / "hermes"
    try:
        workdir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return str(workdir)


def _warn_no_llm_key(env):
    """Say once, at boot, that the server runs without an LLM key.

    cognee >= 1.6 then builds memory with local models (GLiNER extraction,
    fastembed embeddings), downloaded on the first build, and recall returns
    stored text rather than generated answers; worth knowing, and the embedder
    choice sticks to the store. A keyless provider (Ollama) needs no key, so it
    gets no message.
    """
    global _warned_no_llm_key
    provider = str(env.get("LLM_PROVIDER") or "").strip().lower()
    if _warned_no_llm_key or provider in _KEYLESS_LLM_PROVIDERS:
        return
    _warned_no_llm_key = True
    logger.warning(
        "Starting the local cognee server without an LLM key: cognee builds memory with "
        "local models instead (GLiNER extraction and fastembed embeddings, about 1.5 GB "
        "downloaded on the first build), and recall returns stored text rather than "
        "generated answers. For LLM extraction, set an LLM key with `hermes memory setup "
        "cognee`; keys kept only for the other cognee plugins are not read by this one."
    )


def _isolated_env_file():
    """An empty ``.env`` the server is told to load instead of searching for one.

    cognee >= 1.6.2 loads the first ``.env`` found walking up from its working
    directory and lets it override the environment. From the plugin's state
    directory that walk reaches ``~/.cognee-plugin/.env``, the file the *other*
    cognee plugins keep their keys in, so the server would run on their LLM key
    and settings instead of the ones Hermes passes in. ``COGNEE_ENV_FILE`` names
    the one file to load and stops the search; this one is empty on purpose.
    """
    path = os.path.join(server_workdir(), "cognee-server.env")
    try:
        if not os.path.isfile(path):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "# Left empty on purpose: the Hermes cognee plugin starts its local\n"
                    "# server with COGNEE_ENV_FILE pointing here, so the server reads its\n"
                    "# settings from Hermes rather than from a .env found above this folder.\n"
                )
    except OSError:
        return None
    return path


def _spawn(port, data_root, system_root, log_path, python=None):
    env = dict(os.environ)
    # An explicit COGNEE_ENV_FILE from the user wins; otherwise pin an empty one.
    if not env.get("COGNEE_ENV_FILE"):
        isolated = _isolated_env_file()
        if isolated:
            env["COGNEE_ENV_FILE"] = isolated
    # The server's API keys come from Hermes (the profile's secret scope), never
    # from a file the plugin reads itself: the plugin-catalog rules forbid that,
    # and under a multi-profile gateway os.environ may hold another profile's.
    # Each is dropped first, so a profile without one never boots the server on
    # the inherited value.
    for name in _PROFILE_SECRETS:
        env.pop(name, None)
        value = secret(name)
        if value:
            env[name] = value
    if "LLM_API_KEY" not in env:
        _warn_no_llm_key(env)
    env["COGNEE_AGENT_MODE"] = "true"  # server tears itself down once idle / no clients
    env["HTTP_API_PORT"] = str(port)
    # The server must behave identically no matter which cognee plugin booted it,
    # so this mirrors the env the claude-code/codex/openclaw bootstraps set
    # (claude-code's apply_cognee_env). setdefault: an explicit user value wins.
    #
    # CACHING gates the session cache: without it cognee's session manager reports
    # ``is_available = False`` and a session write is *silently dropped* while the
    # API still answers ``status: "session_stored"`` — so every turn vanished and
    # improve() had nothing to promote into the graph (live-diagnosed).
    # LLM_INSTRUCTOR_MODE=json_schema_mode uses grammar-constrained decoding for
    # structured output — small local models fail schema-in-prompt mode so often
    # that every improve() timed out (diagnosed in the claude-code plugin).
    # DEFAULT_USER_EMAIL/PASSWORD: cognee >= 1.6.0 creates the default user at
    # startup only when DEFAULT_USER_PASSWORD is set, and a password-less user
    # rejects the login http_backend._mint_api_key relies on. Always the plugin
    # *literals* here, never COGNEE_USER_*: every cognee plugin shares this one
    # server and database, and the server sets the default user's password once
    # and never rewrites it — so a per-plugin login user passed through as
    # DEFAULT_USER_* would let whichever plugin boots first define the default
    # user for all of them. claude-code/codex/antigravity pass the same literals.
    # COGNEE_USER_* only selects which (already existing) user the login uses.
    # setdefault keeps an operator's own DEFAULT_USER_* export authoritative.
    for key, value in (
        ("CACHING", "true"),
        ("AUTO_FEEDBACK", "true"),
        ("CACHE_ROOT_DIRECTORY", str(SHARED_COGNEE_HOME / "cache")),
        ("LLM_INSTRUCTOR_MODE", "json_schema_mode"),
        ("COGNEE_IMPROVE_SUBMIT_TIMEOUT", "420"),
        ("DEFAULT_USER_EMAIL", DEFAULT_USER_EMAIL),
        ("DEFAULT_USER_PASSWORD", DEFAULT_USER_PASSWORD),
    ):
        env.setdefault(key, value)
    # A local Ollama embedder inherits a token ceiling far above its real
    # context and silently mean-pools every overflowing text into a lossy
    # vector — see ollama_embedding_pins for the incident this pins away.
    for key, value in ollama_embedding_pins(env).items():
        env.setdefault(key, value)
    if data_root:
        env["DATA_ROOT_DIRECTORY"] = data_root
    if system_root:
        env["SYSTEM_ROOT_DIRECTORY"] = system_root
    rotate_oversized_log(log_path)
    try:
        log = open(log_path, "ab", buffering=0)  # noqa: SIM115 — handed to the child
    except Exception:
        log = subprocess.DEVNULL
    python = python or sys.executable
    try:
        proc = subprocess.Popen(
            [
                python,
                "-m",
                "uvicorn",
                "cognee.api.client:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=env,
            cwd=server_workdir(),
            stdout=log,
            stderr=log,
            start_new_session=True,  # detach: outlive the spawning call
        )
        # What `install-runtime` stops this server by (runtime_env.plugin_servers).
        write_server_pidfile(proc.pid, port, python)
        return proc
    finally:
        # The child inherited its own dup of the fd; close the parent's copy so we
        # don't leak a descriptor on every initialize().
        if log is not subprocess.DEVNULL:
            log.close()


def default_server_log_path() -> str:
    """Where the spawned server's output lands when no log path is given.

    Same state root as the other cognee plugins (they keep per-host logs in
    ~/.cognee-plugin/<host>/), so diagnostics live in one predictable place.
    Also read back by the HTTP transport's overflow scan.
    """
    log_dir = SHARED_PLUGIN_STATE_DIR / "hermes"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # _spawn falls back to DEVNULL if the file cannot be opened
    return str(log_dir / "server.log")


class ServerStillStartingError(RuntimeError):
    """The server we started is alive but not healthy when the caller stopped waiting.

    A first start on an existing store runs migrations before the server answers,
    which can take minutes. The caller can keep waiting off the hot path with
    :func:`wait_for_server` (never by calling :func:`ensure_local_server` again,
    which would start a second server on the same store).
    """

    def __init__(self, message, *, url, port, proc, log_path, remaining):
        super().__init__(message)
        self.url = url
        self.port = port
        self.proc = proc
        self.log_path = log_path
        self.remaining = remaining


def ensure_local_server(
    port,
    *,
    data_root="",
    system_root="",
    log_path=None,
    boot_timeout=float(DEFAULT_SERVER_BOOT_TIMEOUT),
    wait=None,
):
    """Return the URL of a healthy local cognee server, starting one if needed.

    The deadline is generous (matching the other plugins' 600s) because a *first*
    boot runs migrations and can take minutes — but it is only ever waited out
    for a server that is actually coming up. A dead spawn with nothing listening
    on the port raises immediately: waiting cannot fix a missing dependency or a
    crash, and a 10-minute hang would stall every Hermes session start.

    ``wait`` caps how long this call blocks (default: all of ``boot_timeout``).
    When it runs out while the server we started is still alive, it raises
    :class:`ServerStillStartingError` with what :func:`wait_for_server` needs to
    wait out the rest of ``boot_timeout``.

    Raises RuntimeError when the spawn is dead and the port unowned, or when the
    server does not become healthy within boot_timeout.
    """
    url = "http://127.0.0.1:%d" % int(port)
    if health_ok(url):
        return _require_current(url, port)
    if log_path is None:
        log_path = default_server_log_path()
    proc = None
    spawn_error = None
    try:
        # Held while the interpreter starts, so a spawn never runs on a venv
        # that `install-runtime` is upgrading (it raises instead).
        with spawn_lock():
            # A server another cognee plugin started is used above without cognee
            # here; spawning one runs `python -m uvicorn cognee.api.client:app`,
            # which needs an interpreter that has the pinned cognee or newer: the
            # plugin's runtime venv (runtime_env), or Hermes's own environment.
            python = server_python()
            if python is None:
                raise CogneeNotInstalledError(local_not_ready_message())
            try:
                proc = _spawn(port, data_root, system_root, log_path, python=python)
            except Exception as exc:
                # A spawn failure may just be a port-bind race with another starter,
                # in which case health polling below will still succeed. But it may
                # also be a real problem (missing uvicorn, permission denied) — log
                # it for diagnostics.
                spawn_error = exc
                logger.warning(
                    "cognee server spawn attempt failed (will still poll /health): %s", exc
                )
    except RuntimeInstallInProgressError as exc:
        raise CogneeNotInstalledError(f"Cognee memory is off for now: {exc}.") from exc
    foreground = float(boot_timeout) if wait is None else min(float(wait), float(boot_timeout))
    deadline = time.monotonic() + foreground
    while time.monotonic() < deadline:
        if health_ok(url):
            # Usually ours; but a concurrent starter may have won the port.
            return _require_current(url, port)
        # Our spawn being gone is not itself fatal — losing a port-bind race to a
        # concurrent starter looks exactly like this, and then the port has an
        # owner worth waiting for. Dead spawn AND a silent port is fatal.
        spawn_is_gone = spawn_error is not None or (proc is not None and proc.poll() is not None)
        if spawn_is_gone and not port_bound(port):
            cause = (
                "failed to launch (%s)" % spawn_error
                if spawn_error is not None
                else "exited with code %s" % proc.returncode
            )
            raise RuntimeError(
                "cognee local server %s and nothing is listening on port %s. "
                "See %s for the server output." % (cause, port, log_path)
            )
        time.sleep(1.0)
    remaining = float(boot_timeout) - foreground
    if remaining > 0 and proc is not None and proc.poll() is None:
        raise ServerStillStartingError(
            "cognee local server at %s is still starting after %ss (a first start on an "
            "existing store migrates it, which can take minutes). See %s."
            % (url, int(foreground), log_path),
            url=url,
            port=port,
            proc=proc,
            log_path=log_path,
            remaining=remaining,
        )
    raise RuntimeError(
        "cognee local server did not become healthy at %s within %ss. "
        "See %s for the server output." % (url, boot_timeout, log_path)
    )


def wait_for_server(starting: ServerStillStartingError) -> bool:
    """Keep waiting for a server :func:`ensure_local_server` left starting.

    True once it answers /health; False when it exits with nothing listening, or
    when the rest of the boot deadline runs out. Never starts another server.
    """
    deadline = time.monotonic() + starting.remaining
    while time.monotonic() < deadline:
        if health_ok(starting.url):
            return True
        if starting.proc.poll() is not None and not port_bound(starting.port):
            return False
        time.sleep(2.0)
    return False
