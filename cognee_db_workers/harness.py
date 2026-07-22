"""Generic subprocess harness: request/response dataclasses, queue event loop,
and a main-process Session that owns a Process + queues.

Stdlib only. Never import cognee from this module.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import ctypes
import ctypes.util
import itertools
import os
import pickle
import queue as std_queue
import signal
import subprocess
import sys
import threading
import time
import traceback
import weakref
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional


SHUTDOWN = "__SUBPROCESS_HARNESS_SHUTDOWN__"
_DEFAULT_SHUTDOWN_TIMEOUT = 10.0
_DEFAULT_INIT_TIMEOUT = 300.0  # was 60.0 — raised for Cognee MCP stdio bootstrap to finish Neo4j + LanceDB init


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


def _env_int(name: str, default: int) -> int:
    """Graceful ``int`` parse. Returns the default if the env var is missing,
    empty, or not a valid integer — we don't want a typo in a deployment's
    env vars to crash module import.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Per-RPC deadline for subprocess calls. Guards against hung native libraries.
# Override with ``SUBPROCESS_CALL_TIMEOUT`` env var (seconds, or <=0 to
# disable entirely — not recommended outside benchmarks).
_DEFAULT_CALL_TIMEOUT: Optional[float] = _env_float("SUBPROCESS_CALL_TIMEOUT", 300.0)
_DEFAULT_MAX_RETRIES = _env_int("SUBPROCESS_MAX_RETRIES", 2)
# Reader-thread poll interval for the response queue and for the
# ``_closed_event`` flag. NOTE: any caller setting an explicit
# ``shutdown_timeout`` should keep it >= this interval — otherwise the
# reader-thread ``join()`` in ``shutdown()`` can time out before the
# reader observes the closed flag, producing spurious "join timed out"
# warnings even on a clean shutdown.
_PROCESS_CHECK_INTERVAL = 1.0
# Per-call timeouts in a row before the retry loop forces a respawn.
# A single per-call timeout under concurrent RPC is just a slow op (the
# session stays open so sibling calls keep working); two in a row almost
# always means the worker is wedged.
_TIMEOUT_BEFORE_RESPAWN = 2

# Number of times the worker retries opening a database that is currently
# lock-held by another (still-shutting-down) worker for the same file, and the
# initial backoff between attempts (seconds, exponential). These exist as a
# backstop for the brief window where one worker is releasing a file lock while
# another opens the same path; see ``kuzu_worker._open_database``. Override with
# ``SUBPROCESS_OPEN_LOCK_RETRIES`` / ``SUBPROCESS_OPEN_LOCK_BACKOFF``.
OPEN_LOCK_RETRIES = _env_int("SUBPROCESS_OPEN_LOCK_RETRIES", 10)
# ``_env_float`` returns the default (0.1) when unset/invalid and ``None`` for an
# explicit ``<= 0``. Treat that explicit ``<= 0`` as ``0.0`` (immediate retries,
# no delay) rather than resetting to the default, so ``SUBPROCESS_OPEN_LOCK_BACKOFF=0``
# is honored.
_OPEN_LOCK_BACKOFF_RAW = _env_float("SUBPROCESS_OPEN_LOCK_BACKOFF", 0.1)
OPEN_LOCK_BACKOFF = 0.0 if _OPEN_LOCK_BACKOFF_RAW is None else _OPEN_LOCK_BACKOFF_RAW
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
    # Per-request id used to correlate responses with waiters in the main
    # process. ``0`` is reserved for protocol sentinels (the SHUTDOWN ack
    # and the ``_READY_SENTINEL`` response). The session assigns ids from
    # ``itertools.count(1)`` in ``_issue`` / ``_issue_async``.
    request_id: int = 0
    handle_id: Optional[int] = None
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)


@dataclass
class Response:
    # Echoed from the originating Request so the main-process reader thread
    # can route this response to the correct pending future. ``0`` for
    # protocol sentinels (SHUTDOWN ack, READY).
    request_id: int = 0
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

    def register_at(self, hid: int, obj: Any) -> None:
        """Register ``obj`` at a fixed pre-known handle id.

        Used for singleton resources whose id the protocol has reserved
        out-of-band (e.g. the LanceDB worker's per-process connection
        slot). Keeps callers from reaching into ``self._handles``
        directly so any future locking / invariants the registry
        adopts apply uniformly.
        """
        self._handles[hid] = obj

    def get(self, hid: int) -> Any:
        return self._handles[hid]

    def pop(self, hid: int) -> Any:
        return self._handles.pop(hid, None)


# Annotations for the most common SIGKILL/SIGTERM/SIGSEGV/SIGABRT scenarios.
# Keeps the diagnostic message immediately actionable instead of requiring
# the reader to know what each signal usually means in production. Kept
# small on purpose — only the signals we actually want to flag with extra
# context get an entry; everything else falls back to just the signal name.
_SIGNAL_HINTS: Dict[str, str] = {
    "SIGKILL": "likely OOM kill or `docker kill`",
    "SIGSEGV": "native crash — check faulthandler dump in worker stderr",
    "SIGABRT": "abort/assert in native code",
    "SIGTERM": "external termination (parent shutdown or platform stop)",
    "SIGBUS": "bad memory access in native code",
    "SIGFPE": "arithmetic error in native code",
}


def _describe_exitcode(exitcode: Optional[int]) -> str:
    """Render an exitcode for human consumption.

    ``None`` and non-negative codes pass through unchanged; negative codes
    are decoded into the signal that killed the worker (``-9`` → ``SIGKILL``)
    with a short hint about the typical cause in production. POSIX exit
    semantics: ``multiprocessing.Process.exitcode`` is the negated signal
    number when the child died from an uncaught signal.

    Note: positive exitcodes are NOT decoded as signals. On Windows a process
    "killed" via ``os.kill(pid, sig)`` (``TerminateProcess``) exits with a
    positive code, but positive codes are overwhelmingly ordinary exit codes
    (e.g. ``1`` = generic error), so decoding them as signals would mislabel
    normal exits.
    """
    if exitcode is None or exitcode >= 0:
        return str(exitcode)
    try:
        sig = signal.Signals(-exitcode)
    except (ValueError, AttributeError):
        return str(exitcode)
    hint = _SIGNAL_HINTS.get(sig.name)
    if hint:
        return f"{exitcode} ({sig.name} — {hint})"
    return f"{exitcode} ({sig.name})"


def set_pdeathsig() -> bool:
    """Linux only: ask the kernel to SIGTERM this process when the parent dies.

    Returns ``True`` when the signal was successfully armed, ``False`` on any
    other platform or if the prctl call failed. Callers use the return value
    to decide whether the portable polling watchdog is still needed.

    Resolves libc via ``ctypes.util.find_library`` rather than hard-coding
    ``libc.so.6`` so musl-based distros (Alpine) and other non-glibc Linuxes
    still arm pdeathsig instead of silently falling through to the polling
    watchdog — which would re-expose the Docker-PID-1 hang this module was
    built to prevent.

    No-op on other platforms. Complements ``daemon=True`` which is bypassed on
    abnormal parent termination.
    """
    if sys.platform != "linux":
        return False
    try:
        PR_SET_PDEATHSIG = 1
        # ``find_library`` returns the SONAME of the system libc (e.g.
        # ``libc.so.6`` on glibc, ``libc.musl-x86_64.so.1`` on Alpine).
        # Passing ``None`` to ``CDLL`` opens the main program's symbol table
        # which on Linux includes the dynamic linker's libc symbols — used as
        # a last-ditch fallback if ``find_library`` returns nothing (rare).
        libc_name = ctypes.util.find_library("c")
        libc = ctypes.CDLL(libc_name, use_errno=True) if libc_name else ctypes.CDLL(None)
        rc = libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
        return rc == 0
    except Exception:
        return False


def start_parent_liveness_watchdog(poll_interval: float = 1.0) -> None:
    """Portable fallback when ``set_pdeathsig`` is unavailable or failed.

    Used on macOS, Windows, and on Linux configurations where the prctl call
    in ``set_pdeathsig`` did not succeed (no libc found, sandboxed prctl,
    etc.). The call site in ``run_worker_loop`` gates this watchdog behind
    ``set_pdeathsig()`` returning ``False`` so a successfully-armed kernel
    signal isn't shadowed by a redundant polling thread.

    Starts a daemon thread that polls ``os.getppid()``. When the original
    parent dies the child is reparented (typically to launchd/init), so a
    ppid change is a reliable signal. The watchdog then terminates the worker
    with ``os._exit`` to avoid orphaned DB processes holding file locks.

    Only checks for ``current_ppid != original_ppid``. An earlier version
    also exited when ``current_ppid == 1`` as a heuristic for "reparented to
    init" — but in container deployments the legitimate parent is frequently
    pid 1 (cognee running as the container's entrypoint), which made the
    watchdog kill workers on their very first poll. The ppid-change check
    alone covers reparenting correctly in all cases.
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
            if current_ppid != original_ppid:
                # Parent is gone; exit fast without running atexit handlers
                # (those may try to use resources owned by the dead parent).
                os._exit(0)
            try:
                time.sleep(poll_interval)
            except Exception:
                return

    t = threading.Thread(target=_watch, name="parent-liveness-watchdog", daemon=True)
    t.start()


# Serializes all ``spawn_without_main`` enter/exit transitions. The
# mutation is on a single, process-global object (``sys.modules["__main__"]``),
# so two threads entering concurrently would race: thread B would capture
# thread A's already-cleared ``None`` as "saved", then on exit restore it,
# permanently breaking the main module for the rest of the process. A
# threading lock is sufficient because ``Process.start()`` blocks on the
# child reading its pickled state, so holding this lock for the duration of
# the spawn is fine.
_MAIN_MODULE_MUTATION_LOCK = threading.Lock()


class spawn_without_main:
    """Temporarily hide ``__main__.__spec__`` / ``__main__.__file__`` so that
    ``multiprocessing``'s spawn bootstrap does not re-execute the parent's main
    script in the child. Without this, the child re-imports every top-level
    import performed by the main script — which for cognee means a ~200 MB
    import tax every time a subprocess starts.
    """

    def __enter__(self):
        _MAIN_MODULE_MUTATION_LOCK.acquire()
        try:
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
        except BaseException:
            _MAIN_MODULE_MUTATION_LOCK.release()
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
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
        finally:
            _MAIN_MODULE_MUTATION_LOCK.release()


Dispatcher = Callable[[HandleRegistry, Request], Any]


def _enable_faulthandler() -> None:
    """Install Python's ``faulthandler`` so a native crash inside the worker
    (Kuzu / LanceDB C++) dumps the Python traceback that triggered it to
    stderr before the process dies. Without this, segfaults surface to the
    parent as a bare ``exitcode=-11`` with no context about which query or
    handler was on the stack.

    Set ``SUBPROCESS_FAULTHANDLER_DISABLED=1`` to opt out (e.g. when running
    under a debugger that wants to handle SIGSEGV itself).
    """
    if os.environ.get("SUBPROCESS_FAULTHANDLER_DISABLED") == "1":
        return
    try:
        import faulthandler

        # ``sys.__stderr__`` is the real fd 2 the worker inherited from the
        # parent. ``sys.stderr`` may have been wrapped by an embedding host;
        # the underlying file descriptor is what container log collectors
        # capture, so prefer it.
        faulthandler.enable(file=sys.__stderr__ or sys.stderr, all_threads=True)
    except Exception:
        # Best-effort: if faulthandler can't be enabled (no usable stderr,
        # etc.) we just lose this diagnostic. Don't crash the worker over a
        # debugging aid.
        pass


def run_worker_loop(
    dispatch: Dict[int, Dispatcher],
    req_q,
    resp_q,
    init: Optional[Callable[[HandleRegistry], None]] = None,
) -> None:
    """Concurrent worker dispatch.

    Async handlers (any handler that returns a coroutine) are launched as
    ``asyncio.create_task`` so multiple can be in flight at once — bounded
    by a semaphore (``SUBPROCESS_WORKER_MAX_INFLIGHT``, default 16) to keep
    the worker's memory footprint predictable. Sync handlers continue to
    run inline on the event loop thread, preserving today's serialization
    for adapters whose underlying library is not thread-safe (e.g. Kuzu).

    Responses carry the originating ``request_id`` so the main-process
    reader thread can route them to the correct waiter; the protocol
    sentinels (READY, SHUTDOWN ack) use ``request_id=0``.
    """
    _enable_faulthandler()
    # pdeathsig is the authoritative parent-death signal on Linux. Only fall
    # back to the portable polling watchdog when the kernel hook is
    # unavailable (macOS, Windows) or failed to arm — that watchdog has no
    # way to distinguish "legitimate parent happens to be pid 1" from
    # "reparented to init", so we avoid running it whenever pdeathsig has
    # us covered.
    if not set_pdeathsig():
        start_parent_liveness_watchdog()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    registry = HandleRegistry()
    # ``SUBPROCESS_WORKER_MAX_INFLIGHT`` must be > 0. A zero or negative
    # value would create a semaphore that deadlocks on every ``async with sem``.
    # This is a load-bearing validation: silent degradation (clamping to 1) would
    # hide configuration errors. Callers wanting "no cap" should pass a
    # sufficiently large value explicitly (e.g., 10000).
    max_inflight = _env_int("SUBPROCESS_WORKER_MAX_INFLIGHT", 16)
    if max_inflight <= 0:
        raise ValueError(
            f"SUBPROCESS_WORKER_MAX_INFLIGHT must be > 0, got {max_inflight}. "
            "This controls the worker's concurrent task semaphore; zero or negative "
            "values would deadlock all async handlers."
        )
    sem = asyncio.Semaphore(max_inflight)

    try:
        if init is not None:
            init(registry)
        resp_q.put(Response(result=_READY_SENTINEL))
    except Exception as e:
        resp_q.put(Response(error=traceback.format_exc(), exception=_safe_pickle_exception(e)))
        return

    pending: "set[asyncio.Task]" = set()

    def _emit(rid: int, result: Any) -> None:
        if isinstance(result, HandleResult):
            resp_q.put(
                Response(request_id=rid, result=result.value, new_handle_id=result.handle_id)
            )
        else:
            resp_q.put(Response(request_id=rid, result=result))

    def _emit_error(rid: int, e: BaseException) -> None:
        resp_q.put(
            Response(
                request_id=rid,
                error=traceback.format_exc(),
                exception=_safe_pickle_exception(e),
            )
        )

    async def _run_async(coro, rid: int) -> None:
        # The semaphore bounds in-flight async ops on the worker. The cap
        # is the whole point of running these in a subprocess — we don't
        # want to trade main-process memory pressure for unbounded worker
        # memory pressure.
        async with sem:
            try:
                _emit(rid, await coro)
            except Exception as e:
                _emit_error(rid, e)

    async def serve() -> None:
        while True:
            # Block the request fetch on a thread so the event loop stays
            # free to drive in-flight async handlers.
            msg = await loop.run_in_executor(None, req_q.get)
            if msg == SHUTDOWN:
                # Drain in-flight tasks before acking so callers that
                # already submitted a request before SHUTDOWN see a real
                # response rather than a closed-session error.
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                resp_q.put(Response())  # request_id=0 ack
                return

            rid = getattr(msg, "request_id", 0)
            handler = dispatch.get(msg.op)
            if handler is None:
                resp_q.put(Response(request_id=rid, error=f"Unknown op {msg.op!r}"))
                continue

            try:
                result = handler(registry, msg)
            except Exception as e:
                _emit_error(rid, e)
                continue

            if asyncio.iscoroutine(result):
                t = asyncio.create_task(_run_async(result, rid))
                pending.add(t)
                t.add_done_callback(pending.discard)
            else:
                _emit(rid, result)

    try:
        loop.run_until_complete(serve())
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


# Weak registry of all live sessions. Used by ``collect_garbage_in_all_workers``
# so callers (e.g. benchmark scripts that want to read accurate per-child
# RSS) can trigger ``gc.collect()`` in every worker without threading a
# session reference through the call site.
_all_sessions: "weakref.WeakSet[SubprocessSession]" = weakref.WeakSet()


def collect_garbage_in_all_workers(timeout: float = 5.0) -> int:
    """Send ``OP_GC_COLLECT`` to every live session's worker. Returns the
    count of sessions that responded successfully. Best-effort: a session
    that's mid-shutdown or crashed is skipped silently.

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
            if session._closed_event.is_set() or not session._proc.is_alive():
                continue
            # Goes through the normal id-routed call path. With concurrent
            # RPC, this no longer blocks other in-flight calls — it just
            # runs alongside them. The worker's GC handler is sync and
            # quick, so the practical impact on other ops is negligible.
            session.call(Request(op=OP_GC_COLLECT), timeout=timeout)
            collected += 1
        except Exception:
            continue
    return collected


def _reap_all_sessions_atexit() -> None:
    """Force-reap any still-live subprocess workers at interpreter exit.

    Workers are normally reaped via ``SubprocessSession.__del__ -> shutdown
    -> _terminate`` during garbage collection. At interpreter shutdown,
    however, GC and ``__del__`` ordering is not guaranteed to run — notably
    on Windows with ``spawn`` daemon processes — which can leave a worker
    alive and block the parent from exiting until an external timeout (e.g.
    the 60-minute CI job timeout on ``Custom Graph Delete``). Registering an
    ``atexit`` reaper makes teardown deterministic: every live session's
    worker is force-terminated (bounded ``_terminate`` kill chain) before the
    interpreter exits, regardless of GC timing. Best-effort and idempotent —
    ``_terminate`` is serialized by ``_terminate_lock`` and safe to call on an
    already-closed session.
    """
    for session in list(_all_sessions):
        try:
            session._terminate(timeout=2.0)
        except Exception:
            pass


atexit.register(_reap_all_sessions_atexit)


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
        # Per-request pending registry: request_id -> concurrent.futures.Future.
        # The reader thread sets ``set_result(resp)`` on the matching future;
        # both sync and async callers wait on the same primitive (sync via
        # ``Future.result(timeout)``, async via ``asyncio.wrap_future``).
        # ``Future.set_result`` / ``set_exception`` are thread-safe so the
        # reader can resolve them without a ``call_soon_threadsafe`` dance.
        self._id_counter = itertools.count(1)
        self._pending: "dict[int, concurrent.futures.Future]" = {}
        self._pending_lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        # "No more RPCs accepted" flag. An Event (vs the previous plain
        # bool) lets the reader poll it cheaply at the top of its loop and
        # avoids a separate mutex for the closed-state read/write.
        self._closed_event = threading.Event()
        # Per-call retry state. The session-level deadline (per-call
        # timeout) does NOT mark the session closed — sibling calls may
        # still succeed — but consecutive timeouts on the same caller
        # almost certainly mean the worker is wedged. After
        # ``_TIMEOUT_BEFORE_RESPAWN`` timeouts in a row, the retry loop
        # forces a respawn instead of issuing the same RPC against the
        # same hung worker.
        self._consecutive_timeouts = 0
        self._consecutive_timeouts_lock = threading.Lock()
        self._terminate_lock = threading.Lock()
        self._respawn_lock = threading.Lock()
        # Set by ``wait_for_ready`` after the worker emits its READY
        # sentinel. Used to gate registration in the global ``_all_sessions``
        # set: a session that's still waiting for its sentinel must not be
        # visible to ``collect_garbage_in_all_workers`` — a concurrent GC
        # sweep would consume the sentinel from ``_resp_q`` and cause
        # ``wait_for_ready`` to time out and kill a healthy worker.
        self._ready = False

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid

    @property
    def last_accessed_at(self) -> float:
        return self._last_accessed_at

    def touch(self) -> None:
        self._last_accessed_at = time.time()

    # Backward-compat shim: a handful of external call sites (and the old
    # internal logic the reader thread now subsumes) read/write ``_closed``
    # as a bool. Route both through the underlying ``Event``.
    @property
    def _closed(self) -> bool:
        return self._closed_event.is_set()

    @_closed.setter
    def _closed(self, value: bool) -> None:
        if value:
            self._closed_event.set()
        else:
            self._closed_event.clear()

    def _init_diagnostics(self) -> str:
        """Compact ``pid=… exitcode=… alive=…`` summary of the worker process,
        included in init-failure messages so callers can tell a silent
        ``os._exit(0)`` apart from a timeout where the child is still
        running.

        Negative exit codes are decoded into the killing signal's name
        (``exitcode=-9 (SIGKILL — likely OOM/docker kill)``) so the most
        common production failure shapes don't require POSIX trivia to
        read the log line.
        """
        try:
            pid = self._proc.pid
        except Exception:
            pid = None
        try:
            exitcode = self._proc.exitcode
        except Exception:
            exitcode = None
        try:
            alive = self._proc.is_alive()
        except Exception:
            alive = None
        return f"pid={pid} exitcode={_describe_exitcode(exitcode)} alive={alive}"

    def _init_failure_message(self, reason: str) -> str:
        return f"Subprocess init {reason} after {self._init_timeout}s ({self._init_diagnostics()})"

    # Total time we'll wait for the producer-side feeder thread's bytes to
    # cross the pipe after the worker exits. Long enough to absorb scheduler
    # jitter under load; short enough that an empty queue resolves to the
    # "exited before signalling ready" message quickly.
    _POST_DEATH_DRAIN_TIMEOUT = 0.5
    _POST_DEATH_DRAIN_POLL = 0.02

    def _drain_response_after_death(self) -> Optional[Response]:
        """Try to read a Response that the worker queued just before exiting.

        See the caller's comment for the race this addresses: the producer's
        feeder thread may not have pushed pickled bytes onto the pipe before
        the worker exited, so ``get_nowait()`` immediately after observing
        ``is_alive() == False`` can return ``Empty`` even when data is in
        flight. Poll for a short bounded window so legitimate worker-side
        errors aren't lost.

        ``multiprocessing.Queue`` can also raise ``EOFError`` / ``OSError``
        when the underlying pipe is closed or corrupted (e.g. the worker
        died mid-``put``). We treat those the same as "no message recovered"
        and fall through to the caller's diagnostic path — surfacing a raw
        pipe error here would mask the much more useful "exited before
        signalling ready" message with ``pid=… exitcode=…`` context.
        """
        deadline = time.monotonic() + self._POST_DEATH_DRAIN_TIMEOUT
        while True:
            try:
                return self._resp_q.get_nowait()
            except std_queue.Empty:
                if time.monotonic() >= deadline:
                    return None
                time.sleep(self._POST_DEATH_DRAIN_POLL)
            except (EOFError, OSError):
                # Pipe closed/corrupted — no salvageable response.
                return None

    def wait_for_ready(self) -> None:
        # Poll the response queue in short slices so we notice if the child
        # dies before producing a READY sentinel. A single blocking ``get``
        # with the full init timeout hides that case entirely: the caller
        # learns nothing beyond "60s elapsed" even when the child exited 50ms
        # in. By interleaving ``is_alive()`` checks we can fail fast and
        # surface the worker's pid / exitcode in the error message.
        deadline = time.monotonic() + self._init_timeout
        poll_interval = min(0.5, self._init_timeout) if self._init_timeout > 0 else 0.5
        resp = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate()
                self._closed_event.set()
                raise SubprocessTransportError(self._init_failure_message("timed out"))
            try:
                resp = self._resp_q.get(timeout=min(poll_interval, remaining))
                break
            except std_queue.Empty:
                if not self._proc.is_alive():
                    # Drain anything the child managed to queue before dying
                    # (a Response with .error is common when init raised).
                    #
                    # ``multiprocessing.Queue`` uses a background feeder
                    # thread in the producer to push pickled bytes from an
                    # in-memory buffer onto the pipe — those bytes may not
                    # have crossed yet when the worker exits, so a single
                    # ``get_nowait()`` races the flush and would drop the
                    # error the child intended to send. Poll for a short
                    # bounded window (sub-second) before giving up. We
                    # cannot use ``proc.join()`` to force a flush: the
                    # feeder thread lives in the *worker* process and is
                    # already gone once the worker has exited.
                    drained = self._drain_response_after_death()
                    if drained is not None:
                        resp = drained
                        break
                    self._terminate()
                    self._closed_event.set()
                    raise SubprocessTransportError(
                        self._init_failure_message("exited before signalling ready")
                    )
        if resp.error:
            self._terminate()
            self._closed_event.set()
            raise SubprocessTransportError(
                f"Subprocess init failed ({self._init_diagnostics()}):\n{resp.error}"
            )
        if resp.result != _READY_SENTINEL:
            self._terminate()
            self._closed_event.set()
            raise SubprocessTransportError(
                f"Unexpected subprocess startup response "
                f"({self._init_diagnostics()}): {resp.result!r}"
            )
        # Only register in ``_all_sessions`` after the worker is actually
        # ready. Doing this in ``__init__`` exposed the session to
        # ``collect_garbage_in_all_workers`` before the READY sentinel was
        # consumed; a concurrent GC sweep could steal the sentinel from
        # ``_resp_q``, causing ``wait_for_ready`` (this method) to time
        # out and kill a healthy worker.
        self._ready = True
        _all_sessions.add(self)
        # The reader is the single consumer of ``_resp_q`` from this point
        # on. Starting it BEFORE returning is important: if a caller
        # invokes ``call_async`` immediately after ``wait_for_ready``, the
        # reader must already be draining responses or the future never
        # resolves.
        self._start_reader_thread()

    def _start_reader_thread(self) -> None:
        t = threading.Thread(
            target=self._reader_loop,
            name=f"subproc-session-reader-{self._proc.pid}",
            daemon=True,
        )
        self._reader_thread = t
        t.start()

    def _reader_loop(self) -> None:
        """Single consumer of ``self._resp_q``. Routes responses to pending
        futures by ``request_id`` and fails every waiter on any path that
        ends the session.

        **Invariant (load-bearing):** the ``finally`` block always runs
        ``_fail_all_pending``, so worker crashes, transport-level errors,
        and explicit shutdown all propagate ``SubprocessTransportError`` to
        every pending future. No path that stops the reader can leak
        futures.
        """
        transport_err: Optional[BaseException] = None
        try:
            while not self._closed_event.is_set():
                try:
                    resp = self._resp_q.get(timeout=_PROCESS_CHECK_INTERVAL)
                except std_queue.Empty:
                    if not self._proc.is_alive():
                        transport_err = SubprocessTransportError(
                            f"Subprocess exited unexpectedly (exit code {self._proc.exitcode})"
                        )
                        break
                    continue
                except (EOFError, OSError, BrokenPipeError) as e:
                    # Queue's underlying pipe died — treat as transport
                    # failure rather than letting the reader die silently.
                    transport_err = SubprocessTransportError(
                        f"Subprocess response queue broken: {e!r}"
                    )
                    break
                except (pickle.UnpicklingError, EOFError) as e:
                    # The worker emitted a value we couldn't deserialize
                    # (corrupted queue payload, partial write, etc.).
                    # Treat as a transport failure too — every pending
                    # caller will be failed via ``_fail_all_pending`` in
                    # the outer ``finally``, and the session is no longer
                    # trustworthy for further RPCs.
                    transport_err = SubprocessTransportError(
                        f"Subprocess response decode failed: {e!r}"
                    )
                    break
                except Exception as e:
                    # Defensive: any unexpected exception inside the
                    # reader must propagate as a transport error so the
                    # ``finally`` invariant (drain every pending future)
                    # still applies. Without this, a fresh exception
                    # type that slipped past the targeted clauses above
                    # would bubble out and leave callers waiting on
                    # futures forever.
                    transport_err = SubprocessTransportError(
                        f"Subprocess reader internal error: {e!r}"
                    )
                    break
                rid = getattr(resp, "request_id", 0)
                if rid == 0:
                    # Protocol sentinel (SHUTDOWN ack, an unsolicited
                    # response, or a response from the legacy
                    # un-id'd protocol). Drop it.
                    continue
                with self._pending_lock:
                    fut = self._pending.pop(rid, None)
                if fut is None or fut.done():
                    # Caller already gave up (timeout / cancel) and
                    # popped its own entry, or a racing failure already
                    # resolved the future. Dropping the response here
                    # is correct.
                    continue
                try:
                    fut.set_result(resp)
                except concurrent.futures.InvalidStateError:
                    # A near-simultaneous ``_fail_all_pending`` may have
                    # raced our ``pop`` + ``set_result``. Either resolution
                    # is acceptable — the caller wakes up either way.
                    pass
        finally:
            self._closed_event.set()
            self._fail_all_pending(
                transport_err or SubprocessTransportError("Subprocess session is closed")
            )

    def _fail_all_pending(self, err: BaseException) -> None:
        """Atomically drain the pending registry and propagate ``err`` to
        every future. Runs from the reader's ``finally`` (the load-bearing
        path) and also from explicit lifecycle transitions.

        The dict swap-with-``{}`` under ``_pending_lock`` is what gives the
        "no leaked futures" guarantee — every entry that existed at the
        moment the session closed is failed exactly once.
        """
        with self._pending_lock:
            pending, self._pending = self._pending, {}
        for fut in pending.values():
            if fut.done():
                continue
            try:
                fut.set_exception(err)
            except concurrent.futures.InvalidStateError:
                # Raced with a ``set_result`` from the reader for the same
                # id. Harmless — the caller already woke.
                pass

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

    def _register_pending(self, rid: int, fut: "concurrent.futures.Future") -> None:
        """Atomically insert ``fut`` into the pending registry, refusing
        the insert if the session has been closed.

        Called by ``_issue`` / ``_issue_async`` immediately after a
        lock-free ``_check_alive`` probe; the in-lock re-check below is
        the one that closes the TOCTOU window with a racing close.
        """
        with self._pending_lock:
            if self._closed_event.is_set():
                raise SubprocessTransportError("Subprocess session is closed")
            self._pending[rid] = fut

    def _issue(self, req: Request, timeout) -> Response:
        """Single-shot sync RPC: register a future, send the request, block
        on ``future.result(timeout)``. ``try/finally`` pops the registry
        entry on every exit path (success, timeout, exception) so the
        reader never has to clean up after sync callers.
        """
        deadline, eff = self._resolve_deadline(timeout)
        # ``_check_alive`` does the ``proc.is_alive()`` syscall and may set
        # ``_closed_event`` if the worker exited; running it outside the
        # registry lock keeps that syscall off the hot path. The in-lock
        # re-check in ``_register_pending`` catches a close racing this
        # caller (TOCTOU-safe under ``_pending_lock``).
        self._check_alive()
        rid = next(self._id_counter)
        fut: concurrent.futures.Future = concurrent.futures.Future()
        req_to_send = self._apply_remap(replace(req, request_id=rid))
        self._register_pending(rid, fut)
        try:
            self.touch()
            self._req_q.put(req_to_send)
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                resp = fut.result(timeout=remaining)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Subprocess call exceeded {eff}s deadline")
            # The worker responded — even an application error proves it
            # is still serving. Reset the consecutive-timeout counter
            # BEFORE ``_handle_response`` so a worker-side raise doesn't
            # leave the counter pointing at a wedged-worker state and
            # trigger a spurious respawn on the next per-call timeout.
            self._reset_timeout_counter()
            return self._handle_response(resp)
        finally:
            with self._pending_lock:
                self._pending.pop(rid, None)

    async def _issue_async(self, req: Request, timeout) -> Response:
        """Async counterpart of ``_issue``. Wraps the ``concurrent.futures.Future``
        for the current event loop and awaits with ``asyncio.wait_for`` so
        per-call deadlines work uniformly across sync / async callers.
        """
        deadline, eff = self._resolve_deadline(timeout)
        # See ``_issue`` for why ``_check_alive`` runs outside the lock.
        self._check_alive()
        rid = next(self._id_counter)
        fut: concurrent.futures.Future = concurrent.futures.Future()
        req_to_send = self._apply_remap(replace(req, request_id=rid))
        self._register_pending(rid, fut)
        try:
            self.touch()
            self._req_q.put(req_to_send)
            # ``wrap_future`` auto-detects the running loop. Passing
            # ``loop=`` explicitly is deprecated (and removed in newer
            # Pythons) so we let asyncio infer it from the running
            # context here.
            aio_fut = asyncio.wrap_future(fut)
            timeout_s = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                resp = await asyncio.wait_for(aio_fut, timeout=timeout_s)
            except asyncio.TimeoutError:
                # Cancel the wrapped concurrent future so a late response
                # from the worker for this id is dropped by the reader.
                if not fut.done():
                    fut.cancel()
                raise TimeoutError(f"Subprocess call_async exceeded {eff}s deadline")
            # Worker responded — reset the streak (see ``_issue`` for why
            # this is unconditional, not gated on success vs app error).
            self._reset_timeout_counter()
            return self._handle_response(resp)
        finally:
            with self._pending_lock:
                self._pending.pop(rid, None)

    def _record_timeout(self) -> int:
        """Bump the session-wide consecutive-timeout counter and return the
        new value. Reset to zero on any successful call.
        """
        with self._consecutive_timeouts_lock:
            self._consecutive_timeouts += 1
            return self._consecutive_timeouts

    def _reset_timeout_counter(self) -> None:
        with self._consecutive_timeouts_lock:
            self._consecutive_timeouts = 0

    def _force_respawn_after_timeouts(self) -> None:
        """Mark the session closed so the next ``_respawn`` actually
        respawns instead of early-returning. Called after
        ``_TIMEOUT_BEFORE_RESPAWN`` consecutive per-call timeouts on the
        assumption the worker is wedged.

        Without this hook, the retry loop would keep issuing RPCs against
        the same hung worker because ``_respawn`` short-circuits when the
        process is still ``is_alive()``.
        """
        self._closed_event.set()
        self._reset_timeout_counter()

    def call(self, req: Request, timeout: Optional[float] = ...) -> Response:
        """Blocking synchronous call. Retries transport failures up to
        ``max_retries`` times (respawning + replaying setup steps between
        attempts). Application errors raised inside the worker are NOT
        retried — they would fail the same way on any new subprocess.

        Per-call timeouts no longer mark the session closed (sibling
        in-flight calls keep working), but ``_TIMEOUT_BEFORE_RESPAWN``
        consecutive timeouts force a respawn — the worker is almost
        certainly wedged at that point and issuing the same RPC against
        it would just time out again.
        """
        attempts_left = self._max_retries
        while True:
            try:
                return self._issue(req, timeout)
            except TimeoutError:
                if attempts_left <= 0 or not self._can_respawn():
                    raise
                if self._record_timeout() >= _TIMEOUT_BEFORE_RESPAWN:
                    self._force_respawn_after_timeouts()
                attempts_left -= 1
                self._respawn()
            except SubprocessTransportError:
                if attempts_left <= 0 or not self._can_respawn():
                    raise
                attempts_left -= 1
                self._respawn()

    async def call_async(self, req: Request, timeout: Optional[float] = ...) -> Response:
        """Async counterpart of ``call`` with identical retry semantics.

        Concurrent ``call_async`` invocations no longer queue at a session
        lock: per-request ids let the reader thread route responses, and
        the per-call ``concurrent.futures.Future`` carries the response or
        the close-error to the awaiting coroutine.
        """
        loop = asyncio.get_running_loop()
        attempts_left = self._max_retries
        while True:
            try:
                return await self._issue_async(req, timeout)
            except TimeoutError:
                if attempts_left <= 0 or not self._can_respawn():
                    raise
                if self._record_timeout() >= _TIMEOUT_BEFORE_RESPAWN:
                    self._force_respawn_after_timeouts()
                attempts_left -= 1
                await loop.run_in_executor(None, self._respawn)
            except SubprocessTransportError:
                if attempts_left <= 0 or not self._can_respawn():
                    raise
                attempts_left -= 1
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
        return replace(req, handle_id=new_id)

    def _resolve_deadline(self, timeout) -> tuple:
        """Collapse (``...``, ``None``, float) into ``(deadline, effective_timeout)``.

        ``deadline`` is an absolute monotonic-ish timestamp (``None`` = no
        deadline). ``effective_timeout`` is the numeric timeout actually in
        effect for this call after collapsing the ``...`` sentinel to the
        session default; it's used for diagnostic error messages so timeout
        errors report the value the caller actually got, not the session
        default.
        """
        if timeout is ...:
            timeout = self._call_timeout
        if timeout is None:
            return (None, None)
        # ``time.monotonic()`` (not ``time.time()``) for deadline math: the
        # wall clock can jump backward or forward via NTP corrections, VM
        # suspend/resume, or clock-skew fixes, which would either fire
        # spurious ``TimeoutError`` (clock jumped forward) or hang past the
        # intended deadline (clock jumped backward). Monotonic time only
        # ever increases.
        return (time.monotonic() + float(timeout), float(timeout))

    def _handle_response(self, resp: Response) -> Response:
        if resp.exception is not None:
            # The worker also formatted the remote traceback into
            # ``resp.error``; without preserving it, the local traceback
            # only shows the ``call()`` site and the worker-side stack
            # frames are lost. ``add_note`` (PEP 678, Python 3.11+) is
            # the lightest way to attach the remote stack without
            # rewriting the exception type or chaining. Older Pythons
            # silently skip this — the caller still gets the original
            # exception, just without the remote traceback.
            if resp.error and hasattr(resp.exception, "add_note"):
                resp.exception.add_note(f"Remote subprocess traceback:\n{resp.error}")
            raise resp.exception
        if resp.error:
            raise RuntimeError(resp.error)
        return resp

    def _check_alive(self) -> None:
        if self._closed_event.is_set():
            raise SubprocessTransportError("Subprocess session is closed")
        if not self._proc.is_alive():
            self._closed_event.set()
            raise SubprocessTransportError(
                f"Subprocess exited unexpectedly (exit code {self._proc.exitcode})"
            )

    def _respawn(self) -> None:
        """Tear down the dead worker, spawn a fresh one, and replay the
        registered setup steps. Remaps any allocated handle ids so in-flight
        requests targeting the old handles are rewritten to the new ones.

        Serialized by ``_respawn_lock`` so concurrent retries coalesce onto
        one new process rather than spawning N new workers. The old reader
        thread is stopped (its ``finally`` block fails any still-pending
        futures); the new one is started after replay completes.
        """
        if self._respawn_factory is None:
            raise SubprocessTransportError(
                "Subprocess session has no respawn factory; retry is disabled"
            )

        with self._respawn_lock:
            # Another concurrent retry may have already respawned successfully
            # between the moment the current caller's RPC failed and us
            # acquiring the lock. If the process is alive again, bail out —
            # the caller's outer retry loop will just try the RPC again.
            if not self._closed_event.is_set() and self._proc.is_alive():
                return

            # Stop the old reader if still running; its ``finally`` block
            # fails any futures that hadn't been resolved yet. After
            # ``_fail_all_pending`` runs from inside the reader, no new
            # callers can add to ``_pending`` until we clear the closed
            # flag below (``_check_alive`` rejects under ``_pending_lock``).
            self._closed_event.set()
            if self._reader_thread is not None and self._reader_thread.is_alive():
                self._reader_thread.join(timeout=self._shutdown_timeout)
                if self._reader_thread.is_alive():
                    # The old reader is still running. If we proceeded
                    # and reassigned ``self._resp_q`` below, the old
                    # reader's next iteration would pull from the NEW
                    # queue (it dereferences ``self._resp_q`` per loop),
                    # producing two consumers on the same queue and
                    # nondeterministic response routing. Bail out
                    # instead — the session is now bricked and callers
                    # will see this as a transport error.
                    raise SubprocessTransportError(
                        f"Reader thread did not stop within "
                        f"{self._shutdown_timeout}s; refusing to respawn"
                    )
            self._reader_thread = None

            self._terminate()

            new_proc, new_req_q, new_resp_q = self._respawn_factory()
            self._proc = new_proc
            self._req_q = new_req_q
            self._resp_q = new_resp_q

            # ``_closed_event`` stays set through the entire ready +
            # replay sequence. New callers (including non-retry ones)
            # see ``_closed_event.is_set()`` in ``_register_pending`` and
            # bail with ``SubprocessTransportError`` — they retry via the
            # outer loop and end up parked on ``_respawn_lock``. Without
            # this, a caller could insert into ``_pending`` after the old
            # reader exited but before the new reader started, then
            # block on ``fut.result()`` forever.
            #
            # Replay calls use ``_raw_call_locked`` which bypasses
            # ``_pending`` and reads directly from ``_resp_q`` — so
            # keeping the close flag set does not block replay itself.
            self._ready = False
            try:
                resp = self._resp_q.get(timeout=self._init_timeout)
            except std_queue.Empty:
                self._terminate()
                raise SubprocessTransportError(
                    f"Respawn init timed out after {self._init_timeout}s"
                )
            if resp.error:
                self._terminate()
                raise SubprocessTransportError(f"Respawn init failed:\n{resp.error}")
            if resp.result != _READY_SENTINEL:
                self._terminate()
                raise SubprocessTransportError(
                    f"Unexpected respawn startup response: {resp.result!r}"
                )
            self._ready = True

            # Replay setup in registration order BEFORE starting the reader.
            # Replay calls read responses directly from ``_resp_q``; running
            # them with the reader live would race the reader for those
            # messages.
            new_remap: "dict[int, int]" = {}
            for step in list(self._replay_steps):
                req = step.make_request()
                # Rewrite any handle_id that was already remapped in this
                # replay pass (e.g. OP_DB_INIT after OP_OPEN_DATABASE).
                if new_remap and req.handle_id in new_remap:
                    req = replace(req, handle_id=new_remap[req.handle_id])
                resp = self._raw_call_locked(req)
                if resp.new_handle_id is not None and step.apply_new_handle is not None:
                    old_id = step.apply_new_handle(resp.new_handle_id)
                    if old_id is not None:
                        new_remap[old_id] = resp.new_handle_id

            # Compose this respawn's remap with the accumulated one so that
            # in-flight Requests carrying a handle id from *any* previous
            # incarnation (not just the immediately preceding one) get
            # rewritten to the current generation's id.
            #
            # Illustration across two respawns:
            #   prior self._handle_remap = {original -> id_A}   (after respawn #1)
            #   this respawn's new_remap = {id_A -> id_B}       (replay promoted id_A to id_B)
            # We want the composed remap to route both ``original`` and
            # ``id_A`` to ``id_B``. Overwriting with ``new_remap`` alone would
            # strand ``original`` — it would keep pointing at the now-dead
            # ``id_A``.
            composed: "dict[int, int]" = {}
            for orig, intermediate in self._handle_remap.items():
                composed[orig] = new_remap.get(intermediate, intermediate)
            for old, new in new_remap.items():
                composed.setdefault(old, new)
            self._handle_remap = composed

            # Replay finished, remap is final, reader can take over.
            # Clear the close flag and start the reader in that order so
            # any caller that races past the in-lock check finds a
            # running reader.
            self._closed_event.clear()
            self._start_reader_thread()

    def _raw_call_locked(self, req: Request) -> Response:
        """Single-shot RPC for use during ``_respawn`` replay only.

        Replay runs while the reader thread is NOT live (we stopped the old
        one before reseting state, and we haven't started the new one yet),
        so we can read responses directly from ``_resp_q``. The
        ``request_id`` check below catches a stale response lingering
        from the previous worker — that response would never line up with
        the freshly-spawned worker's reply and silently corrupt replay.
        It's an ``if/raise`` (not ``assert``) so it stays in place under
        ``python -O`` / ``PYTHONOPTIMIZE``, where ``assert`` is stripped.
        """
        rid = next(self._id_counter)
        req_to_send = replace(req, request_id=rid)
        self.touch()
        self._req_q.put(req_to_send)
        _deadline, effective_timeout = self._resolve_deadline(...)
        # ``effective_timeout is None`` means the caller explicitly
        # disabled per-call deadlines (``SUBPROCESS_CALL_TIMEOUT<=0``).
        # Honor that here by issuing a blocking ``get()`` rather than
        # inventing an arbitrary fallback; otherwise benchmark runs that
        # disable the timeout would still see a 60s ceiling on replay.
        if effective_timeout is None:
            resp = self._resp_q.get()
        else:
            try:
                resp = self._resp_q.get(timeout=effective_timeout)
            except std_queue.Empty:
                raise TimeoutError(f"Subprocess replay call exceeded {effective_timeout}s deadline")
        if resp.request_id != rid:
            raise SubprocessTransportError(
                f"Replay response id mismatch: expected {rid}, got {resp.request_id!r}"
            )
        return self._handle_response(resp)

    def shutdown(self, timeout: Optional[float] = None) -> None:
        """Tear down the worker process. Always reaches ``_terminate``, even
        if the session is already marked closed — a timeout or crash flips
        the closed flag without reaping the child, and a second call would
        orphan a still-alive process holding file locks.

        ``timeout`` (or ``self._shutdown_timeout`` when ``None``) is the
        *shutdown* budget. The reader thread is signaled to exit, joined,
        and its ``finally`` block fails any still-pending futures. Then we
        send the graceful SHUTDOWN sentinel (if the worker is still alive)
        and force-terminate.
        """
        t = timeout if timeout is not None else self._shutdown_timeout

        already_closed = self._closed_event.is_set()
        # Set closed under the registry lock so any caller about to add a
        # new entry sees the closed state and bails before inserting.
        with self._pending_lock:
            self._closed_event.set()

        # Try to stop the reader cleanly. If it joins within ``t`` we
        # own ``_resp_q`` exclusively and can drain the SHUTDOWN ack
        # ourselves. If it doesn't, the reader is still alive — we let
        # ``_terminate`` break the queue pipes below, which forces the
        # reader's ``get()`` to raise (EOFError / OSError caught by the
        # reader loop's exception clauses) and then we do a second
        # join. Importantly: we do NOT reassign ``_resp_q``, so an
        # in-flight reader can't end up as a second consumer of a
        # fresh queue (contrast ``_respawn``, which DOES reassign and
        # therefore raises if the join times out).
        reader_owned = False
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=t)
            reader_owned = not self._reader_thread.is_alive()
        else:
            reader_owned = True

        if not already_closed and self._proc.is_alive():
            try:
                self._req_q.put(SHUTDOWN)
                # Only drain the ack here if the reader is confirmed
                # gone. With a still-running reader we'd race it for
                # the message; it'll drop the ack (request_id=0) on
                # its own.
                if reader_owned:
                    try:
                        self._resp_q.get(timeout=t)
                    except std_queue.Empty:
                        pass
            except Exception:
                pass

        # ``_terminate`` reaps the worker process, which closes the
        # multiprocessing queue's underlying pipe; that unblocks any
        # reader still parked in ``get()``.
        self._terminate(timeout=t)

        # Second join opportunity for a reader that was stuck inside
        # ``get()`` — the queue pipe just died, so its exception path
        # should run and the thread should exit promptly. If it
        # doesn't, the thread is a daemon and the process will reap
        # it at exit; nothing more we can do without unsafe forcible
        # cancellation.
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=t)
        self._reader_thread = None

        # Safety net: if the reader didn't drain pending (e.g. it was
        # never started because ``wait_for_ready`` failed, or it was
        # still wedged above), do it now.
        self._fail_all_pending(SubprocessTransportError("Subprocess session is closed"))

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
        """Force-terminate the worker process and wait until it has actually
        exited. Idempotent and serialized by ``self._terminate_lock`` so
        concurrent ``shutdown`` / ``__del__`` / ``clean`` paths can't race each
        other.

        The escalating join/terminate/kill chain is followed by a bounded poll
        that does not return while ``is_alive()`` is still True. This matters
        for the on-disk file lock held by DB workers (Ladybug/LanceDB): the OS
        releases a process's file locks only once the process is truly gone, so
        a caller that re-opens the same DB path right after ``shutdown()``
        returns would hit "Could not set lock on file" if we returned while the
        child were still alive. ``join(timeout)`` alone can return early (e.g.
        right after ``kill()`` the SIGKILL may not have been delivered yet), so
        we poll ``is_alive()`` until the process is reaped, capped so a wedged
        kernel state can't hang teardown forever.
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
                # Final wait for true exit: after SIGKILL the process should die
                # promptly, but ``join`` above may have timed out before the
                # kernel reaped it. Poll up to a generous cap so we don't return
                # while the child still holds its file lock.
                deadline = time.monotonic() + max(timeout, 5.0)
                while self._proc.is_alive() and time.monotonic() < deadline:
                    self._proc.join(timeout=0.05)
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
