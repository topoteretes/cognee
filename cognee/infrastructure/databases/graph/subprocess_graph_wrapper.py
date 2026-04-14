"""Subprocess wrapper that runs any GraphDBInterface adapter in an isolated process.

All method calls are serialized via multiprocessing queues. The subprocess
owns the real adapter instance and its resources (DB connections, memory).
"""

import asyncio
import importlib
import inspect
import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass
from typing import Any, Optional, Type

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.memory_cleanup import get_process_rss
from cognee.shared.logging_utils import get_logger

logger = get_logger()

_SHUTDOWN = "__SUBPROCESS_GRAPH_SHUTDOWN__"
_DEFAULT_SHUTDOWN_TIMEOUT = 10
_DEFAULT_INIT_TIMEOUT = 60
_PROCESS_CHECK_INTERVAL = 1.0


@dataclass
class _Request:
    method: str
    args: tuple
    kwargs: dict


_ATTRIBUTE_ERROR_PREFIX = "__ATTRIBUTE_ERROR__:"


class _SerializedModel:
    """Thin wrapper around a dict that mimics a Pydantic model for the adapter.

    KuzuAdapter (and others) call ``node.model_dump()`` or ``vars(node)`` to
    extract properties.  Dynamic Pydantic models created by ``create_model()``
    cannot be pickled, so we convert them to dicts *before* sending through the
    queue and wrap them here so the adapter's ``model_dump()`` calls still work.
    """

    def __init__(self, data: dict):
        self.__dict__.update(data)

    def model_dump(self, **kwargs):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def dict(self, **kwargs):
        return self.model_dump(**kwargs)


def _prepare_for_pickle(obj):
    """Recursively convert Pydantic BaseModel instances to _SerializedModel."""
    try:
        from pydantic import BaseModel
    except ImportError:
        return obj

    if isinstance(obj, BaseModel):
        return _SerializedModel(obj.model_dump())
    if isinstance(obj, (list, tuple)):
        converted = [_prepare_for_pickle(item) for item in obj]
        return type(obj)(converted)
    if isinstance(obj, dict):
        return {k: _prepare_for_pickle(v) for k, v in obj.items()}
    return obj


@dataclass
class _Response:
    result: Any = None
    error: Optional[str] = None


def _set_pdeathsig() -> None:
    """Ask the kernel to send SIGTERM to this process when its parent exits.

    This is Linux-specific (prctl PR_SET_PDEATHSIG).  It complements
    ``daemon=True``: daemon mode relies on Python's atexit machinery, which is
    bypassed on SIGKILL or abnormal parent termination.  With pdeathsig the
    kernel itself sends SIGTERM to this child regardless of *how* the parent
    dies, preventing orphaned DB subprocesses that hold file locks.

    The call is a no-op on non-Linux platforms so it is always safe to call.
    """
    import sys

    if sys.platform != "linux":
        return
    try:
        import ctypes
        import signal

        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        ret = libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
        if ret != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"prctl(PR_SET_PDEATHSIG) failed: errno={errno}")
    except Exception:
        pass  # best-effort; daemon=True is still the fallback


def _worker(
    adapter_module: str,
    adapter_name: str,
    constructor_args: tuple,
    constructor_kwargs: dict,
    req_q: mp.Queue,
    resp_q: mp.Queue,
):
    """Subprocess entry point. Creates the adapter and processes requests."""
    # Ensure this process dies when the parent dies, even on SIGKILL.
    _set_pdeathsig()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        mod = importlib.import_module(adapter_module)
        adapter_cls = getattr(mod, adapter_name)
        adapter = adapter_cls(*constructor_args, **constructor_kwargs)
    except Exception:
        resp_q.put(_Response(error=traceback.format_exc()))
        return

    # Signal successful initialization; send the adapter's public attribute
    # names so the proxy can answer hasattr() truthfully.
    adapter_attrs = {name for name in dir(adapter) if not name.startswith("_")}
    resp_q.put(_Response(result=adapter_attrs))

    try:
        while True:
            msg = req_q.get()

            if msg == _SHUTDOWN:
                if hasattr(adapter, "close"):
                    try:
                        result = adapter.close()
                        if asyncio.iscoroutine(result):
                            loop.run_until_complete(result)
                    except Exception:
                        pass
                resp_q.put(_Response())
                break

            try:
                method = getattr(adapter, msg.method)
                result = method(*msg.args, **msg.kwargs)
                if asyncio.iscoroutine(result):
                    result = loop.run_until_complete(result)
                resp_q.put(_Response(result=result))
            except AttributeError:
                resp_q.put(_Response(error=_ATTRIBUTE_ERROR_PREFIX + traceback.format_exc()))
            except Exception:
                resp_q.put(_Response(error=traceback.format_exc()))
    finally:
        loop.close()


def _make_proxy(method_name: str):
    """Create an async proxy method that delegates to the subprocess."""

    async def proxy(self, *args, **kwargs):
        return await self._call(method_name, *args, **kwargs)

    proxy.__name__ = method_name
    proxy.__qualname__ = f"SubprocessGraphDBWrapper.{method_name}"
    return proxy


class SubprocessGraphDBWrapper(GraphDBInterface):
    """Runs any GraphDBInterface implementation in a dedicated subprocess.

    Usage::

        wrapper = SubprocessGraphDBWrapper(KuzuAdapter, db_path="/path/to/db")
        result = await wrapper.query("MATCH (n) RETURN n LIMIT 1")
        await wrapper.close()
    """

    def __init__(
        self,
        adapter_cls: Type[GraphDBInterface],
        *args: Any,
        shutdown_timeout: float = _DEFAULT_SHUTDOWN_TIMEOUT,
        init_timeout: float = _DEFAULT_INIT_TIMEOUT,
        **kwargs: Any,
    ):
        ctx = mp.get_context("spawn")
        self._req_q = ctx.Queue()
        self._resp_q = ctx.Queue()
        self._closed = False
        self._last_accessed_at = time.time()
        self._shutdown_timeout = shutdown_timeout

        self._proc = ctx.Process(
            target=_worker,
            args=(
                adapter_cls.__module__,
                adapter_cls.__name__,
                args,
                kwargs,
                self._req_q,
                self._resp_q,
            ),
            daemon=True,
        )
        self._proc.start()

        try:
            resp = self._resp_q.get(timeout=init_timeout)
        except queue.Empty:
            self._proc.terminate()
            self._proc.join(timeout=5)
            raise RuntimeError(
                f"Adapter initialization in subprocess timed out after {init_timeout}s"
            )

        if resp.error:
            self._proc.join(timeout=5)
            raise RuntimeError(f"Adapter initialization failed in subprocess:\n{resp.error}")

        # The worker sends the adapter's public attribute names on success.
        self._adapter_attrs: set = resp.result or set()

        logger.info(
            "SubprocessGraphDBWrapper: subprocess started",
            adapter=adapter_cls.__name__,
            pid=self._proc.pid,
        )

        # Stamp proxy methods for adapter-specific methods not already on the
        # class.  This makes class-level checks like
        #   getattr(adapter.__class__, "some_method", None)
        # work correctly — code in the wild uses this pattern to detect
        # optional adapter capabilities.
        for attr_name in self._adapter_attrs:
            if attr_name not in type(self).__dict__:
                setattr(type(self), attr_name, _make_proxy(attr_name))

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Send a method call to the subprocess and return the result."""
        self._touch()

        if self._closed:
            raise RuntimeError("Subprocess wrapper is closed")
        if not self._proc.is_alive():
            raise RuntimeError(f"Subprocess exited unexpectedly (exit code {self._proc.exitcode})")

        args = _prepare_for_pickle(args)
        kwargs = _prepare_for_pickle(kwargs)
        req = _Request(method=method, args=args, kwargs=kwargs)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._req_q.put, req)
        resp = await loop.run_in_executor(None, self._wait_response)

        if resp.error:
            if resp.error.startswith(_ATTRIBUTE_ERROR_PREFIX):
                raise AttributeError(f"Subprocess adapter has no attribute '{method}'")
            raise RuntimeError(f"Subprocess call to '{method}' failed:\n{resp.error}")
        return resp.result

    def _wait_response(self) -> _Response:
        """Block until a response arrives, checking process liveness periodically."""
        while True:
            try:
                return self._resp_q.get(timeout=_PROCESS_CHECK_INTERVAL)
            except queue.Empty:
                if not self._proc.is_alive():
                    raise RuntimeError(
                        f"Subprocess exited unexpectedly (exit code {self._proc.exitcode})"
                    )

    def _touch(self) -> None:
        self._last_accessed_at = time.time()

    def _close_sync(self) -> None:
        if self._closed:
            return
        self._closed = True

        if not hasattr(self, "_proc") or not self._proc.is_alive():
            return

        pid = self._proc.pid
        logger.info("SubprocessGraphDBWrapper: stopping subprocess", pid=pid)

        try:
            self._req_q.put(_SHUTDOWN)
            self._proc.join(timeout=self._shutdown_timeout)
        except Exception:
            logger.warning("Error during graceful subprocess shutdown", exc_info=True)

        if self._proc.is_alive():
            logger.warning("SubprocessGraphDBWrapper: subprocess did not exit gracefully, terminating forcibly", pid=pid)
            self._proc.terminate()
            self._proc.join(timeout=5)
            logger.info("SubprocessGraphDBWrapper: subprocess forcibly terminated", pid=pid)
        else:
            logger.info("SubprocessGraphDBWrapper: subprocess stopped gracefully", pid=pid)

    def memory_used(self) -> int:
        if self._closed or not hasattr(self, "_proc") or not self._proc.is_alive():
            return 0
        return get_process_rss(self._proc.pid)

    def last_accessed_ts(self) -> float:
        return self._last_accessed_at

    def clean(self) -> None:
        self._close_sync()

    async def close(self) -> None:
        """Gracefully shut down the subprocess, force-kill if it doesn't exit in time."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._close_sync)

    def __del__(self):
        self._close_sync()

    def __getattr__(self, name: str):
        """Proxy for methods not on GraphDBInterface (adapter-specific methods)."""
        if name.startswith("_"):
            raise AttributeError(name)
        if hasattr(self, "_adapter_attrs") and name not in self._adapter_attrs:
            raise AttributeError(f"Subprocess adapter has no attribute '{name}'")

        async def dynamic_proxy(*args, **kwargs):
            return await self._call(name, *args, **kwargs)

        dynamic_proxy.__name__ = name
        return dynamic_proxy


# Stamp proxy methods for all public async methods on GraphDBInterface.
# This satisfies the ABC requirement that all abstract methods are implemented.
_stamped = set()
for _name, _method in inspect.getmembers(GraphDBInterface, predicate=inspect.iscoroutinefunction):
    if not _name.startswith("_") and _name not in SubprocessGraphDBWrapper.__dict__:
        setattr(SubprocessGraphDBWrapper, _name, _make_proxy(_name))
        _stamped.add(_name)

SubprocessGraphDBWrapper.__abstractmethods__ = frozenset(
    SubprocessGraphDBWrapper.__abstractmethods__ - _stamped
)
