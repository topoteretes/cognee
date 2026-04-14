"""Subprocess wrapper that runs any VectorDBInterface adapter in an isolated process.

All method calls are serialized via multiprocessing queues. The subprocess
owns the real adapter instance and its resources (DB connections, caches,
embedding engine).
"""

import asyncio
import importlib
import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass
from typing import Any, Optional, Type

from cognee.infrastructure.memory_cleanup import get_process_rss
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface
from cognee.shared.logging_utils import get_logger

logger = get_logger()

_SHUTDOWN = "__SUBPROCESS_VECTOR_SHUTDOWN__"
_DEFAULT_SHUTDOWN_TIMEOUT = 10
_DEFAULT_INIT_TIMEOUT = 60
_PROCESS_CHECK_INTERVAL = 1.0
_ATTRIBUTE_ERROR_PREFIX = "__ATTRIBUTE_ERROR__:"


@dataclass
class _Request:
    method: str
    args: Any
    kwargs: Any


class _SerializedModel:
    """Thin wrapper around a dict that mimics a Pydantic model for the adapter."""

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
        return {key: _prepare_for_pickle(value) for key, value in obj.items()}
    return obj


@dataclass
class _Response:
    result: Any = None
    error: Any = None
    traceback_text: Optional[str] = None


def _resolve_target(adapter: Any, method_path: str):
    target = adapter
    for part in method_path.split("."):
        target = getattr(target, part)
    return target


class _EmbeddingEngineProxy:
    """Proxy nested embedding engine access through the parent wrapper."""

    def __init__(self, wrapper: "SubprocessVectorDBWrapper"):
        self._wrapper = wrapper

    async def embed_text(self, text):
        return await self._wrapper._call("embedding_engine.embed_text", text)

    def get_vector_size(self) -> int:
        return self._wrapper._call_sync("embedding_engine.get_vector_size")

    def get_batch_size(self) -> int:
        return self._wrapper._call_sync("embedding_engine.get_batch_size")

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._wrapper._call_sync(f"embedding_engine.{name}")


def _build_adapter(
    adapter_module: str,
    adapter_name: str,
    constructor_args: tuple,
    constructor_kwargs: dict,
    initialize_embedding_engine: bool,
):
    mod = importlib.import_module(adapter_module)
    adapter_cls = getattr(mod, adapter_name)

    if initialize_embedding_engine and "embedding_engine" not in constructor_kwargs:
        from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine

        constructor_kwargs = {
            **constructor_kwargs,
            "embedding_engine": get_embedding_engine(),
        }

    return adapter_cls(*constructor_args, **constructor_kwargs)


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
    initialize_embedding_engine: bool,
    req_q: mp.Queue,
    resp_q: mp.Queue,
):
    """Subprocess entry point. Creates the adapter and processes requests."""
    # Ensure this process dies when the parent dies, even on SIGKILL.
    _set_pdeathsig()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        adapter = _build_adapter(
            adapter_module,
            adapter_name,
            constructor_args,
            constructor_kwargs,
            initialize_embedding_engine,
        )
    except Exception:
        resp_q.put(
            _Response(
                error=RuntimeError("adapter initialization failed"),
                traceback_text=traceback.format_exc(),
            )
        )
        return

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
                method = _resolve_target(adapter, msg.method)
                result = method(*msg.args, **msg.kwargs) if callable(method) else method
                if asyncio.iscoroutine(result):
                    result = loop.run_until_complete(result)
                resp_q.put(_Response(result=result))
            except AttributeError:
                resp_q.put(
                    _Response(
                        error=AttributeError(msg.method),
                        traceback_text=_ATTRIBUTE_ERROR_PREFIX + traceback.format_exc(),
                    )
                )
            except Exception as error:
                resp_q.put(_Response(error=error, traceback_text=traceback.format_exc()))
    finally:
        loop.close()


def _make_proxy(method_name: str):
    """Create an async proxy method that delegates to the subprocess."""

    async def proxy(self, *args, **kwargs):
        return await self._call(method_name, *args, **kwargs)

    proxy.__name__ = method_name
    proxy.__qualname__ = f"SubprocessVectorDBWrapper.{method_name}"
    return proxy


class SubprocessVectorDBWrapper(VectorDBInterface):
    """Runs any VectorDBInterface implementation in a dedicated subprocess."""

    def __init__(
        self,
        adapter_cls: Type[VectorDBInterface],
        *args: Any,
        shutdown_timeout: float = _DEFAULT_SHUTDOWN_TIMEOUT,
        init_timeout: float = _DEFAULT_INIT_TIMEOUT,
        initialize_embedding_engine: bool = True,
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
                initialize_embedding_engine,
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
            raise RuntimeError(
                f"Adapter initialization failed in subprocess:\n{resp.traceback_text or resp.error}"
            )

        self._adapter_attrs: set = resp.result or set()
        self.embedding_engine = None

        logger.info(
            "SubprocessVectorDBWrapper: subprocess started",
            adapter=adapter_cls.__name__,
            pid=self._proc.pid,
        )

        if "embedding_engine" in self._adapter_attrs:
            self.embedding_engine = _EmbeddingEngineProxy(self)

        for attr_name in self._adapter_attrs:
            if attr_name not in type(self).__dict__:
                setattr(type(self), attr_name, _make_proxy(attr_name))

    async def has_collection(self, collection_name: str) -> bool:
        return await self._call("has_collection", collection_name)

    async def create_collection(self, collection_name: str, payload_schema: Optional[Any] = None):
        return await self._call("create_collection", collection_name, payload_schema)

    async def create_data_points(self, collection_name: str, data_points: list[Any]):
        return await self._call("create_data_points", collection_name, data_points)

    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
        return await self._call("retrieve", collection_name, data_point_ids)

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[list[float]] = None,
        limit: Optional[int] = None,
        with_vector: bool = False,
        include_payload: bool = False,
        node_name: Optional[list[str]] = None,
    ):
        return await self._call(
            "search",
            collection_name,
            query_text,
            query_vector,
            limit,
            with_vector,
            include_payload,
            node_name,
        )

    async def batch_search(
        self,
        collection_name: str,
        query_texts: list[str],
        limit: Optional[int],
        with_vectors: bool = False,
        include_payload: bool = False,
        node_name: Optional[list[str]] = None,
    ):
        return await self._call(
            "batch_search",
            collection_name,
            query_texts,
            limit,
            with_vectors,
            include_payload,
            node_name,
        )

    async def delete_data_points(self, collection_name: str, data_point_ids: list[Any]):
        return await self._call("delete_data_points", collection_name, data_point_ids)

    async def prune(self):
        return await self._call("prune")

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        return await self._call("embed_data", data)

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Send a method call to the subprocess and return the result."""
        self._touch()

        if self._closed:
            raise RuntimeError("Subprocess wrapper is closed")
        if not self._proc.is_alive():
            raise RuntimeError(f"Subprocess exited unexpectedly (exit code {self._proc.exitcode})")

        req = _Request(
            method=method,
            args=_prepare_for_pickle(args),
            kwargs=_prepare_for_pickle(kwargs),
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._req_q.put, req)
        resp = await loop.run_in_executor(None, self._wait_response)

        if resp.error:
            if isinstance(resp.error, AttributeError):
                raise AttributeError(f"Subprocess adapter has no attribute '{method}'")
            if isinstance(resp.error, BaseException):
                raise resp.error
            raise RuntimeError(
                f"Subprocess call to '{method}' failed:\n{resp.traceback_text or resp.error}"
            )
        return resp.result

    def _call_sync(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Send a synchronous method call to the subprocess and return the result."""
        self._touch()

        if self._closed:
            raise RuntimeError("Subprocess wrapper is closed")
        if not self._proc.is_alive():
            raise RuntimeError(f"Subprocess exited unexpectedly (exit code {self._proc.exitcode})")

        req = _Request(
            method=method,
            args=_prepare_for_pickle(args),
            kwargs=_prepare_for_pickle(kwargs),
        )
        self._req_q.put(req)
        resp = self._wait_response()

        if resp.error:
            if isinstance(resp.error, AttributeError):
                raise AttributeError(f"Subprocess adapter has no attribute '{method}'")
            if isinstance(resp.error, BaseException):
                raise resp.error
            raise RuntimeError(
                f"Subprocess call to '{method}' failed:\n{resp.traceback_text or resp.error}"
            )
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
        logger.info("SubprocessVectorDBWrapper: stopping subprocess", pid=pid)

        try:
            self._req_q.put(_SHUTDOWN)
            self._proc.join(timeout=self._shutdown_timeout)
        except Exception:
            logger.warning("Error during graceful subprocess shutdown", exc_info=True)

        if self._proc.is_alive():
            logger.warning("SubprocessVectorDBWrapper: subprocess did not exit gracefully, terminating forcibly", pid=pid)
            self._proc.terminate()
            self._proc.join(timeout=5)
            logger.info("SubprocessVectorDBWrapper: subprocess forcibly terminated", pid=pid)
        else:
            logger.info("SubprocessVectorDBWrapper: subprocess stopped gracefully", pid=pid)

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
        """Proxy for methods not on VectorDBInterface (adapter-specific methods)."""
        if name.startswith("_"):
            raise AttributeError(name)
        if hasattr(self, "_adapter_attrs") and name not in self._adapter_attrs:
            raise AttributeError(f"Subprocess adapter has no attribute '{name}'")

        async def dynamic_proxy(*args, **kwargs):
            return await self._call(name, *args, **kwargs)

        dynamic_proxy.__name__ = name
        return dynamic_proxy
