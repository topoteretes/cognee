"""Closing a Hermes session: improve, then unregister — always out of process.

Closing a session means two calls in a fixed order: ``improve(session_ids=[...])``
promotes the session's turns into the permanent graph, and only then may the agent
connection be unregistered. The order is load-bearing. The local server runs with
``COGNEE_AGENT_MODE=true``, so unregistering drops the agent count to zero and its
watchdog SIGTERMs the server within 60s — with no regard for pipelines still
running. Unregister first and the promotion is killed halfway.

Doing that in Hermes itself forces a choice between two things the user should not
have to trade: a fast exit, or a session that actually reaches the graph. Out of
process there is no trade — the worker blocks on the improve and its blocking
costs nobody anything. So both paths out of a session route here:

* **clean** — ``finalize()`` hands the close to a detached worker (``--final``)
  the moment ``on_session_end`` fires, and Hermes exits without waiting.
* **unclean** — ``arm()`` leaves a watcher polling the Hermes PID from session
  start; if Hermes dies without ever handing off, the watcher closes the session
  itself.

Both converge on :func:`fire`, which claims a once-marker beside the state file
so that a session is closed exactly once no matter how many workers reach it. The
poller is deliberately *not* stood down when a finalizer is spawned: if that
spawn dies before claiming, the poller is still there to catch it.

This module runs as a plain script (``python exit_watcher.py --state <path>
[--final]``), outside the package, so nothing here may import from the package —
everything it needs travels in the state file, plus the API key, which stays out
of world-readable places: the provider hands it over in the child's environment,
and the watcher never reads a key from a file of its own accord.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_POLL_SECONDS = 2.0
_UNREGISTER_TIMEOUT = 15.0
_WINDOWS = os.name == "nt"
# How long a finalizer will hold the agent registration open waiting for Hermes to
# exit, once its improve is done. Only ever waited out if Hermes outlives its own
# session end; normally the parent is long gone and this returns immediately.
_DEFAULT_UNREGISTER_GRACE = 60.0


# --- state file ---------------------------------------------------------------


def read_state(state_path):
    """The parsed state dict, or None when missing/unreadable (i.e. disarmed)."""
    try:
        data = json.loads(Path(state_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def write_state(state_path, state):
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    os.replace(tmp, path)


def pid_alive(pid):
    """True when *pid* is a live process.

    POSIX probes with signal 0, which touches nothing. Windows MUST NOT go
    through ``os.kill``: CPython implements non-console signals there as
    ``OpenProcess`` + ``TerminateProcess`` — "probing" would kill the very
    Hermes this watcher is guarding — so it queries the process handle instead.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        # POSIX gives pid<=0 group/broadcast semantics; never treat those as
        # "a process worth waiting on".
        return False
    if _WINDOWS:
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def _pid_alive_windows(pid):
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    ERROR_ACCESS_DENIED = 5

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            # Access denied means the pid exists but belongs to someone else.
            return ctypes.get_last_error() == ERROR_ACCESS_DENIED
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            # Documented Windows ambiguity: a process that exited with the
            # literal code 259 reads as alive. Acceptable for a 2s poll —
            # the watcher just fires one poll later than it could have.
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        # An identity we cannot determine is treated as alive: firing improve/
        # unregister against a session that is still running is worse than
        # firing late.
        return True


def marker_path(state_path):
    """The once-marker guarding the session this state file describes."""
    return Path(str(state_path) + ".done")


def claim_once(state_path):
    """True when this process is the one that gets to close the session.

    A finalizer and the poller can both reach :func:`fire` for the same session,
    by design — the poller stays armed as a backstop in case the finalizer never
    starts. The filesystem arbitrates: whoever creates the marker acts, the other
    bows out. A failure for any *other* reason returns True, because a duplicate
    improve costs some wasted work while a skipped one costs the whole session.
    """
    try:
        os.close(os.open(str(marker_path(state_path)), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        return True
    except FileExistsError:
        return False
    except Exception:
        return True


def _spawn(state_path, extra_args, *, api_key, log_path):
    """Start this file as a detached process that outlives its parent."""
    env = dict(os.environ)
    # Exactly the key the provider resolved, never one inherited from Hermes's
    # environment (under a multi-profile gateway that may be another profile's).
    env.pop("COGNEE_API_KEY", None)
    if api_key:
        env["COGNEE_API_KEY"] = str(api_key)
    log = subprocess.DEVNULL
    if log_path:
        try:
            log = open(log_path, "ab", buffering=0)  # noqa: SIM115 — handed to the child
        except Exception:
            log = subprocess.DEVNULL
    try:
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--state",
                str(state_path),
                *extra_args,
            ],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,  # detach: must outlive Hermes to be of any use
        )
    finally:
        if log is not subprocess.DEVNULL:
            log.close()


# --- the provider-facing API ---------------------------------------------------


def arm(
    *,
    state_path,
    log_path,
    parent_pid,
    url,
    api_key,
    agent_session_name,
    dataset,
    session_id,
    improve,
    improve_timeout,
    poll_interval=_DEFAULT_POLL_SECONDS,
):
    """Write the watcher's state file and spawn the detached watcher process.

    The state file deliberately carries no API key — the key goes through the
    child's environment instead. The provider arms the watcher after connecting,
    so it hands over the key the session resolved. That key is empty when the
    session runs without one (a server with auth off, or a failed mint); the
    watcher then logs that its calls may be rejected.
    """
    # State files are keyed on the Hermes pid, and pids recycle. A marker left by
    # whoever held this pid last would silently no-op this session's close.
    try:
        marker_path(state_path).unlink(missing_ok=True)
    except Exception as exc:
        logger.debug("could not clear a stale cognee session-close marker: %s", exc)
    write_state(
        state_path,
        {
            "parent_pid": int(parent_pid),
            "url": str(url),
            "agent_session_name": str(agent_session_name),
            "dataset": str(dataset),
            "session_id": str(session_id),
            "improve": bool(improve),
            "improve_timeout": float(improve_timeout),
            "poll_interval": float(poll_interval),
            # Recorded so finalize() can reuse the same log without re-deriving it.
            "log_path": str(log_path) if log_path else "",
        },
    )
    _spawn(state_path, [], api_key=api_key, log_path=log_path)


def update(state_path, **fields):
    """Merge *fields* into the state file (e.g. a new session id on switch).

    Best-effort and never raises: the watcher is insurance, and no memory
    operation should fail because insurance paperwork did.
    """
    try:
        state = read_state(state_path)
        if state is None:
            return
        state.update(fields)
        write_state(state_path, state)
    except Exception as exc:
        logger.debug("could not update the cognee exit watcher state: %s", exc)


def finalize(
    state_path,
    *,
    api_key,
    session_id,
    dataset,
    improve,
    improve_timeout,
    unregister_grace=_DEFAULT_UNREGISTER_GRACE,
):
    """Hand improve-then-unregister to a detached worker. True when it took them.

    False means nothing was handed off and the caller still owes this session a
    close of its own — there is no armed watcher to take it (embedded mode), or
    the spawn failed.

    The armed poller is deliberately left in place either way. If this worker dies
    before claiming the once-marker, the poller still catches Hermes exiting; if
    it claims, the poller finds the marker taken and bows out.
    """
    state = read_state(state_path)
    if state is None:
        return False
    state.update(
        {
            "session_id": str(session_id),
            "dataset": str(dataset),
            "improve": bool(improve),
            "improve_timeout": float(improve_timeout),
            "unregister_grace": float(unregister_grace),
        }
    )
    try:
        write_state(state_path, state)
        _spawn(state_path, ["--final"], api_key=api_key, log_path=state.get("log_path"))
        return True
    except Exception as exc:
        # State left intact: the poller is still valid insurance.
        logger.debug("could not spawn the cognee session-close worker: %s", exc)
        return False


def disarm(state_path):
    """Delete the state file; the watcher notices and exits without acting."""
    for path in (Path(state_path), marker_path(state_path)):
        try:
            path.unlink(missing_ok=True)
        except Exception as exc:
            logger.debug("could not disarm the cognee exit watcher: %s", exc)


# --- the watcher process --------------------------------------------------------


def _log(message):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + message, flush=True)


def _post(url, path, payload, *, api_key, timeout):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Api-Key"] = api_key
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return getattr(response, "status", None) or response.getcode()


def _await_parent_exit(state, grace):
    """Wait, bounded, for Hermes to be gone."""
    deadline = time.monotonic() + float(grace)
    interval = float(state.get("poll_interval") or _DEFAULT_POLL_SECONDS)
    while time.monotonic() < deadline and pid_alive(state.get("parent_pid", -1)):
        time.sleep(interval)


def fire(state, *, state_path=None, wait_for_parent=0.0):
    """Close the session: improve first, then unregister. True when we did it.

    That order is load-bearing (and matches openclaw's exit watcher): the improve
    must finish while this connection still counts as a registered agent, or the
    server's idle watchdog could tear the server down mid-promotion.

    ``state_path`` enables the once-marker, so whichever worker gets here first
    is the only one that acts. ``wait_for_parent`` holds the registration open,
    bounded, until Hermes is actually gone — the improve is already done by then,
    so this only avoids pulling the server out from under a Hermes still using it.
    """
    if state_path is not None and not claim_once(state_path):
        _log("session already closed by another worker; exiting")
        return False
    url = str(state.get("url") or "")
    # Only the key the provider put in this process's environment at spawn time.
    api_key = os.environ.get("COGNEE_API_KEY", "").strip()
    if not api_key:
        _log("no API key was handed over; a server that requires one will reject this close")
    session_id = str(state.get("session_id") or "")
    if state.get("improve") and session_id:
        try:
            _post(
                url,
                "/api/v1/improve",
                {
                    "dataset_name": str(state.get("dataset") or ""),
                    "session_ids": [session_id],
                    "run_in_background": False,
                },
                api_key=api_key,
                timeout=float(state.get("improve_timeout") or 300.0),
            )
            _log("improve submitted for session %s" % session_id)
        except urllib.error.HTTPError as exc:
            if exc.code == 402:
                # cognee >= 1.6.3: the server's LLM budget is exhausted.
                _log(
                    "improve skipped for session %s: the Cognee server's LLM budget "
                    "is exhausted (402); not retrying" % session_id
                )
            else:
                _log("improve failed for session %s: %s" % (session_id, exc))
        except Exception as exc:
            _log("improve failed for session %s: %s" % (session_id, exc))
    if wait_for_parent:
        _await_parent_exit(state, wait_for_parent)
    try:
        _post(
            url,
            "/api/v1/agents/unregister",
            {"agent_session_name": str(state.get("agent_session_name") or "hermes")},
            api_key=api_key,
            timeout=_UNREGISTER_TIMEOUT,
        )
        _log("agent connection unregistered")
    except Exception as exc:
        _log("unregister failed: %s" % exc)
    return True


def watch(state_path):
    """Poll until disarmed, superseded, or the parent dies (then act)."""
    state = read_state(state_path)
    if state is None:
        return 0
    # Claim the state file. A later arm() for the same path overwrites this, and
    # the superseded watcher sees a foreign pid below and bows out.
    state["watcher_pid"] = os.getpid()
    write_state(state_path, state)
    _log(
        "watching hermes pid %s for session %s" % (state.get("parent_pid"), state.get("session_id"))
    )
    while True:
        state = read_state(state_path)
        if state is None:
            _log("disarmed (clean shutdown); exiting")
            return 0
        if state.get("watcher_pid") != os.getpid():
            _log("superseded by a newer watcher; exiting")
            return 0
        if not pid_alive(state.get("parent_pid", -1)):
            _log("hermes pid %s is gone; closing the session" % state.get("parent_pid"))
            if fire(state, state_path=state_path):
                disarm(state_path)
            return 0
        time.sleep(float(state.get("poll_interval") or _DEFAULT_POLL_SECONDS))


def run_final(state_path):
    """The clean path: Hermes asked us to close the session, so close it now."""
    state = read_state(state_path)
    if state is None:
        return 0
    _log("closing session %s on request" % state.get("session_id"))
    if fire(
        state,
        state_path=state_path,
        wait_for_parent=float(state.get("unregister_grace") or _DEFAULT_UNREGISTER_GRACE),
    ):
        disarm(state_path)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="cognee-hermes session-close worker")
    parser.add_argument("--state", required=True)
    parser.add_argument(
        "--final",
        action="store_true",
        help="close the session now instead of polling for the parent to die",
    )
    args = parser.parse_args(argv)
    return run_final(args.state) if args.final else watch(args.state)


if __name__ == "__main__":
    sys.exit(main())
