"""Generic subprocess harness: request/response dataclasses, queue event loop,
and a main-process Session that owns a Process + queues.

Stdlib only. Never import cognee from this module.
"""

from __future__ import annotations

import asyncio
import ctypes
import itertools
import os
import pickle
import queue as std_queue
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


SHUTDOWN = "__SUBPROCESS_HARNESS_SHUTDOWN__"
_DEFAULT_SHUTDOWN_TIMEOUT = 10.0
_DEFAULT_INIT_TIMEOUT = 60.0


def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    """Parse a float-or-disabled env var. Empty / unset → ``default``; a
    value <= 0 disables the timeout (``None``); any other float is used.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        val = float(raw)
    except ValueError:
        return default
    return None if val <= 0 else val


# Per-RPC deadline for subprocess calls. Guards against hung native libraries.
# Override with ``SUBPROCESS_CALL_TIMEOUT`` env var (seconds, or <=0 to
# disable entirely — not recommended outside benchmarks).
_DEFAULT_CALL_TIMEOUT: Optional[float] = _env_float("SUBPROCESS_CALL_TIMEOUT", 300.0)
_DEFAULT_MAX_RETRIES = int(os.environ.get("SUBPROCESS_MAX_RETRIES", "2"))
_PROCESS_CHECK_INTERVAL = 1.0
_READY_SENTINEL = "__SUBPROCESS_HARNESS_READY__"

# Universal op code — every worker's DISPATCH includes an entry for it so
# callers can force a ``gc.collect()`` inside the worker (useful before
# reading the subprocess's RSS so the number reflects reachable objects
# only, not uncollected cycles).
OP_GC_COLLECT = 99


class SubprocessTransportError(RuntimeError):
    """Raised when the subprocess transport itself is broken — the worker
    exited unexpectedly or the RPC deadline was missed. Distinct from
    application errors raised *inside* the worker (those come back via
    ``Response.error`` / ``Response.exception``) so the session can decide
    which kind to retry.
    """


@dataclass
class Request:
    op: int
    handle_id: Optional[int] = None
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)


@dataclass
class Response:
    result: Any = None
    new_handle_id: Optional[int] = None
    error: Optional[str] = None
    exception: Optional[BaseException] = None


@dataclass
class HandleResult:
    """Sentinel returned by a dispatcher that allocated a new worker-side handle."""

    value: Any
    handle_id: int


@dataclass
class ReplayStep:
    """A setup RPC replayed when the worker is respawned after a crash.

    ``make_request`` is invoked at replay time (not registration time), so
    lambdas that read ``self._handle_id`` naturally pick up values freshly
    assigned by earlier steps.

    ``apply_new_handle`` receives the newly allocated worker-side handle and
    returns the *old* handle id it replaces. The session records that
    ``old → new`` mapping and rewrites pending requests whose ``handle_id``
    matches before the retry fires. Return ``None`` for steps that don't
    allocate a handle (e.g. ``OP_DB_INIT``, ``OP_LOAD_EXTENSION``).
    """

    make_request: "Callable[[], Request]"
    apply_new_handle: Optional["Callable[[int], Optional[int]]"] = None


class HandleRegistry:
    """Integer-keyed registry of worker-side native objects."""

    def __init__(self) -> None:
        self._counter = itertools.count(1)
        self._handles: Dict[int, Any] = {}

    def register(self, obj: Any) -> int:
        hid = next(self._counter)
        self._handles[hid] = obj
        return hid

    def get(self, hid: int) -> Any:
        return self._handles[hid]

    def pop(self, hid: int) -> Any:
        return self._handles.pop(hid, None)


def set_pdeathsig() -> None:
    """Linux only: ask the kernel to SIGTERM this process when the parent dies.

    No-op on other platforms. Complements ``daemon=True`` which is bypassed on
    abnormal parent termination.
    """
    if sys.platform != "linux":
        return
    try:
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:
        pass


def start_parent_liveness_watchdog(poll_interval: float = 1.0) -> None:
    """Portable fallback for platforms without ``pdeathsig`` (macOS, Windows).

    Starts a daemon thread that polls ``os.getppid()``. When the original
    parent dies the child is reparented (to init/launchd, typically pid 1),
    so a ppid change is a reliable signal. The watchdog then terminates the
    worker with ``os._exit`` to avoid orphaned DB processes holding file locks.

    Redundant on Linux (``set_pdeathsig`` covers it) but cheap enough to run
    everywhere as defense-in-depth.
    """
    import threading

    try:
        original_ppid = os.getppid()
    except Exception:
        return

    def _watch() -> None:
        while True:
            try:
                current_ppid = os.getppid()
            except Exception:
                return
            if current_ppid != original_ppid or current_ppid == 1:
                # Parent is gone; exit fast without running atexit handlers
                # (those may try to use resources owned by the dead parent).
                os._exit(0)
            try:
                time.sleep(poll_interval)
            except Exception:
                return

    t = threading.Thread(target=_watch, name="parent-liveness-watchdog", daemon=True)
    t.start()


class spawn_without_main:
    """Temporarily hide ``__main__.__spec__`` / ``__main__.__file__`` so that
    ``multiprocessing``'s spawn bootstrap does not re-execute the parent's main
    script in the child. Without this, the child re-imports every top-level
    import performed by the main script — which for cognee means a ~200 MB
    import tax every time a subprocess starts.
    """

    def __enter__(self):
        main_mod = sys.modules.get("__main__")
        self._main = main_mod
        self._saved_spec = getattr(main_mod, "__spec__", None) if main_mod else None
        self._saved_file = getattr(main_mod, "__file__", None) if main_mod else None
        if main_mod is not None:
            try:
                main_mod.__spec__ = None
            except Exception:
                pass
            try:
                main_mod.__file__ = None
            except Exception:
                pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._main is None:
            return False
        try:
            self._main.__spec__ = self._saved_spec
        except Exception:
            pass
        try:
            self._main.__file__ = self._saved_file
        except Exception:
            pass
        return False


Dispatcher = Callable[[HandleRegistry, Request], Any]


def run_worker_loop(
    dispatch: Dict[int, Dispatcher],
    req_q,
    resp_q,
    init: Optional[Callable[[HandleRegistry], None]] = None,
) -> None:
    """Serialize all requests on a single event loop. Handlers may be sync or
    return a coroutine; coroutines are awaited on the worker's asyncio loop.
    """
    set_pdeathsig()
    start_parent_liveness_watchdog()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    registry = HandleRegistry()

    try:
        if init is not None:
            init(registry)
        resp_q.put(Response(result=_READY_SENTINEL))
    except Exception as e:
        resp_q.put(
            Response(error=traceback.format_exc(), exception=_safe_pickle_exception(e))
        )
        return

    try:
        while True:
            msg = req_q.get()
            if msg == SHUTDOWN:
                resp_q.put(Response())
                break

            handler = dispatch.get(msg.op)
            if handler is None:
                resp_q.put(Response(error=f"Unknown op {msg.op!r}"))
                continue

            try:
                result = handler(registry, msg)
                if asyncio.iscoroutine(result):
                    result = loop.run_until_complete(result)
                if isinstance(result, HandleResult):
                    resp_q.put(
                        Response(result=result.value, new_handle_id=result.handle_id)
                    )
                else:
                    resp_q.put(Response(result=result))
            except Exception as e:
                resp_q.put(
                    Response(
                        error=traceback.format_exc(),
                        exception=_safe_pickle_exception(e),
                    )
                )
    finally:
        try:
            loop.close()
        except Exception:
            pass


def _safe_pickle_exception(e: BaseException) -> Optional[BaseException]:
    try:
        pickle.dumps(e)
        return e
    except Exception:
        return None


def _op_gc_collect(_registry: HandleRegistry, _req: Request):
    """Worker-side handler for ``OP_GC_COLLECT`` — runs Python's garbage
    collector inside the worker process. Included in every adapter worker's
    DISPATCH via ``DEFAULT_DISPATCH``.
    """
    import gc as _gc

    return int(_gc.collect())


DEFAULT_DISPATCH: Dict[int, Dispatcher] = {
    OP_GC_COLLECT: _op_gc_collect,
}


import weakref

# Weak registry of all live sessions. Used by ``collect_garbage_in_all_workers``
# so callers (e.g. benchmark scripts that want to read accurate per-child
# RSS) can trigger ``gc.collect()`` in every worker without threading a
# session reference through the call site.
_all_sessions: "weakref.WeakSet[SubprocessSession]" = weakref.WeakSet()


def collect_garbage_in_all_workers(timeout: float = 5.0) -> int:
    """Send ``OP_GC_COLLECT`` to every live session's worker. Returns the
    count of sessions that responded successfully. Best-effort: a session
    that's mid-shutdown, crashed, or busy is skipped silently.

    Intended as a preamble to RSS measurement so subprocess memory numbers
    reflect reachable objects only.
    """
    collected = 0
    # Materialize a snapshot before iterating: ``WeakSet`` entries can
    # disappear at any time when the last strong reference is dropped, which
    # would raise ``RuntimeError`` mid-iteration. The snapshot holds temporary
    # strong refs, but the broad try/except below still covers the narrower
    # race where a session is mid-shutdown by the time we call into it.
    for session in list(_all_sessions):
        try:
            session.call(Request(op=OP_GC_COLLECT), timeout=timeout)
            collected += 1
        except Exception:
            continue
    return collected


class SubprocessSession:
    """Main-process owner of a spawned worker process + its queues.

    Concrete subclasses / callers are expected to create the Process and
    queues, pass them in, then call ``wait_for_ready()`` before issuing calls.
    """

    def __init__(
        self,
        proc,
        req_q,
        resp_q,
        *,
        shutdown_timeout: float = _DEFAULT_SHUTDOWN_TIMEOUT,
        init_timeout: float = _DEFAULT_INIT_TIMEOUT,
        call_timeout: Optional[float] = _DEFAULT_CALL_TIMEOUT,
        respawn_factory: Optional["Callable[[], tuple]"] = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self._proc = proc
        self._req_q = req_q
        self._resp_q = resp_q
        self._closed = False
        self._last_accessed_at = time.time()
        self._shutdown_timeout = shutdown_timeout
        self._init_timeout = init_timeout
        # None disables the per-call deadline (not recommended in production:
        # a hung native library inside the worker will block the main process
        # forever). Override per-session if needed.
        self._call_timeout = call_timeout
        # Retry / respawn wiring. ``respawn_factory`` is a no-arg callable
        # returning a fresh ``(proc, req_q, resp_q)`` triple; typically this
        # is supplied by the subclass' ``start()`` classmethod. When
        # ``max_retries > 0`` the session will tear down a dead worker,
        # spawn a new one, replay any registered ``ReplayStep``s, remap
        # handle ids, and retry the failed RPC.
        self._respawn_factory = respawn_factory
        self._max_retries = max(0, int(max_retries)) if respawn_factory else 0
        self._replay_steps: "list[ReplayStep]" = []
        self._handle_remap: "dict[int, int]" = {}
        # Eager construction avoids a lazy "is None then assign" race that
        # could give two concurrent callers different Lock instances.
        # asyncio.Lock in Python 3.10+ constructs without a running loop.
        import threading

        self._rpc_lock = asyncio.Lock()
        self._sync_rpc_lock = threading.Lock()
        self._terminate_lock = threading.Lock()
        self._respawn_lock = threading.Lock()

        # Register in the global weak set so ``collect_garbage_in_all_workers``
        # can reach this session without an explicit reference.
        _all_sessions.add(self)

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid

    @property
    def last_accessed_at(self) -> float:
        return self._last_accessed_at

    def touch(self) -> None:
        self._last_accessed_at = time.time()

    def wait_for_ready(self) -> None:
        try:
            resp = self._resp_q.get(timeout=self._init_timeout)
        except std_queue.Empty:
            self._terminate()
            self._closed = True
            raise SubprocessTransportError(
                f"Subprocess init timed out after {self._init_timeout}s"
            )
        if resp.error:
            self._terminate()
            self._closed = True
            raise SubprocessTransportError(
                f"Subprocess init failed:\n{resp.error}"
            )
        if resp.result != _READY_SENTINEL:
            self._terminate()
            self._closed = True
            raise SubprocessTransportError(
                f"Unexpected subprocess startup response: {resp.result!r}"
            )

    def add_replay_step(self, step: ReplayStep) -> None:
        """Register a setup RPC to replay after a respawn. Order is preserved.

        Typically called by a proxy in its ``__init__`` right after the initial
        setup RPC succeeded — so the step captures the same kwargs used the
        first time and propagates the new handle id back via
        ``apply_new_handle``.
        """
        self._replay_steps.append(step)

    def remove_replay_step(self, step: ReplayStep) -> None:
        try:
            self._replay_steps.remove(step)
        except ValueError:
            pass

    def call(self, req: Request, timeout: Optional[float] = ...) -> Response:
        """Blocking synchronous call. Retries transport failures up to
        ``max_retries`` times (respawning + replaying setup steps between
        attempts). Application errors raised inside the worker are NOT
        retried — they would fail the same way on any new subprocess.
        """
        attempts_left = self._max_retries
        while True:
            deadline = self._resolve_deadline(timeout)
            req_to_send = self._apply_remap(req)
            try:
                with self._sync_rpc_lock:
                    self._check_alive()
                    self.touch()
                    self._req_q.put(req_to_send)
                    resp = self._wait_response(deadline)
                    return self._handle_response(resp)
            except (SubprocessTransportError, TimeoutError):
                if attempts_left <= 0 or not self._can_respawn():
                    raise
                attempts_left -= 1
                self._respawn()

    async def call_async(self, req: Request, timeout: Optional[float] = ...) -> Response:
        """Async counterpart of ``call`` with identical retry semantics."""
        attempts_left = self._max_retries
        while True:
            deadline = self._resolve_deadline(timeout)
            req_to_send = self._apply_remap(req)
            try:
                async with self._rpc_lock:
                    self._check_alive()
                    self.touch()
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self._req_q.put, req_to_send)
                    resp = await loop.run_in_executor(None, self._wait_response, deadline)
                    return self._handle_response(resp)
            except (SubprocessTransportError, TimeoutError):
                if attempts_left <= 0 or not self._can_respawn():
                    raise
                attempts_left -= 1
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._respawn)

    def _can_respawn(self) -> bool:
        return self._respawn_factory is not None

    def _apply_remap(self, req: Request) -> Request:
        """After a successful respawn, previously-allocated handle ids are
        dead. Any in-flight Request still referring to an old id gets
        rewritten to the new id recorded in ``_handle_remap``.
        """
        if not self._handle_remap or req.handle_id is None:
            return req
        new_id = self._handle_remap.get(req.handle_id)
        if new_id is None:
            return req
        from dataclasses import replace as _replace

        return _replace(req, handle_id=new_id)

    def _resolve_deadline(self, timeout) -> Optional[float]:
        """Collapse (``...``, ``None``, float) into an absolute deadline."""
        if timeout is ...:
            timeout = self._call_timeout
        if timeout is None:
            return None
        return time.time() + float(timeout)

    def _handle_response(self, resp: Response) -> Response:
        if resp.exception is not None:
            raise resp.exception
        if resp.error:
            raise RuntimeError(resp.error)
        return resp

    def _wait_response(self, deadline: Optional[float] = None) -> Response:
        while True:
            remaining = _PROCESS_CHECK_INTERVAL
            if deadline is not None:
                remaining = max(0.0, min(remaining, deadline - time.time()))
                if remaining <= 0.0:
                    # Past deadline — mark session dead so follow-up calls
                    # fail fast and the caller can rebuild an adapter.
                    self._closed = True
                    raise TimeoutError(
                        f"Subprocess call exceeded {self._call_timeout}s deadline; "
                        f"session marked closed"
                    )
            try:
                return self._resp_q.get(timeout=remaining)
            except std_queue.Empty:
                if not self._proc.is_alive():
                    # Detected worker death — flip to closed so subsequent
                    # callers see a consistent state instead of racing on
                    # proc.is_alive() each time.
                    self._closed = True
                    raise SubprocessTransportError(
                        f"Subprocess exited unexpectedly "
                        f"(exit code {self._proc.exitcode})"
                    )

    def _check_alive(self) -> None:
        if self._closed:
            raise SubprocessTransportError("Subprocess session is closed")
        if not self._proc.is_alive():
            self._closed = True
            raise SubprocessTransportError(
                f"Subprocess exited unexpectedly "
                f"(exit code {self._proc.exitcode})"
            )

    def _respawn(self) -> None:
        """Tear down the dead worker, spawn a fresh one, and replay the
        registered setup steps. Remaps any allocated handle ids so in-flight
        requests targeting the old handles are rewritten to the new ones.

        Serialized by ``_respawn_lock`` so concurrent retries coalesce onto
        one new process rather than spawning N new workers.
        """
        if self._respawn_factory is None:
            raise SubprocessTransportError(
                "Subprocess session has no respawn factory; retry is disabled"
            )

        with self._respawn_lock:
            self._terminate()

            new_proc, new_req_q, new_resp_q = self._respawn_factory()
            self._proc = new_proc
            self._req_q = new_req_q
            self._resp_q = new_resp_q
            self._closed = False

            # Wait for the new worker's ready sentinel before replay.
            self.wait_for_ready()

            # Replay setup in registration order. Each step runs against the
            # already-open session; handle remaps accumulate so later steps
            # that reference earlier handles (e.g. OPEN_CONNECTION uses the
            # just-reopened Database handle via its proxy's ``self._handle_id``
            # which apply_new_handle updated) work correctly.
            new_remap: "dict[int, int]" = {}
            for step in list(self._replay_steps):
                req = step.make_request()
                # Rewrite any handle_id that was already remapped in this
                # replay pass (e.g. OP_DB_INIT after OP_OPEN_DATABASE).
                if new_remap and req.handle_id in new_remap:
                    from dataclasses import replace as _replace

                    req = _replace(req, handle_id=new_remap[req.handle_id])
                # Bypass the retry loop and the per-call timeout for replay
                # itself; we don't want recursive retries here.
                resp = self._raw_call_locked(req)
                if resp.new_handle_id is not None and step.apply_new_handle is not None:
                    old_id = step.apply_new_handle(resp.new_handle_id)
                    if old_id is not None:
                        new_remap[old_id] = resp.new_handle_id
            # Merge replay's remap with any prior one (handles can be remapped
            # across multiple successive respawns).
            self._handle_remap = new_remap

    def _raw_call_locked(self, req: Request) -> Response:
        """Single-shot RPC without retry or rpc_lock (caller holds
        ``_respawn_lock``). Used during replay.
        """
        self._check_alive()
        self.touch()
        self._req_q.put(req)
        resp = self._wait_response(self._resolve_deadline(...))
        return self._handle_response(resp)

    def shutdown(self, timeout: Optional[float] = None) -> None:
        if self._closed:
            return
        self._closed = True
        t = timeout if timeout is not None else self._shutdown_timeout

        try:
            if self._proc.is_alive():
                self._req_q.put(SHUTDOWN)
                try:
                    self._resp_q.get(timeout=t)
                except std_queue.Empty:
                    pass
        except Exception:
            pass
        self._terminate(timeout=t)

        # Break the ref cycle: each ReplayStep's ``make_request`` is a
        # closure that captures a proxy object, and the proxy holds
        # ``self._session`` = this session. Without clearing the list here,
        # evicted sessions stay alive via that cycle until the gc gets
        # around to it (and ``__del__`` used to block cycle collection
        # pre-PEP-442). Clearing also lets the child's worker-side state —
        # held indirectly through the proxies — become collectable sooner.
        self._replay_steps.clear()
        self._handle_remap.clear()

    def _terminate(self, timeout: float = 2.0) -> None:
        """Force-terminate the worker process. Idempotent and serialized by
        ``self._terminate_lock`` so concurrent ``shutdown`` / ``__del__`` /
        ``clean`` paths can't race each other.
        """
        with self._terminate_lock:
            try:
                self._proc.join(timeout=timeout)
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=timeout)
                if self._proc.is_alive():
                    self._proc.kill()
                    self._proc.join(timeout=timeout)
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.shutdown(timeout=2.0)
        except Exception:
            pass


def get_process_rss_bytes(pid: int) -> int:
    """Return RSS of a process in bytes. Reads /proc on Linux, falls back to
    ``ps`` elsewhere. Returns 0 if the process is gone or unreachable."""
    if pid is None:
        return 0
    try:
        with open(f"/proc/{pid}/statm") as f:
            rss_pages = int(f.read().split()[1])
            return rss_pages * os.sysconf("SC_PAGESIZE")
    except OSError:
        pass
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)], text=True, stderr=subprocess.DEVNULL
        )
        return int(out.strip()) * 1024
    except Exception:
        return 0
