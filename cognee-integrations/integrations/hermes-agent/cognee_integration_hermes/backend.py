"""The transport seam between the Hermes provider and cognee.

The provider owns everything Hermes-shaped — scope routing, session-id
derivation, tool envelopes, the circuit breaker, the write gating. A backend owns
only *how the bytes reach cognee*. Keeping the split at exactly that line is what
lets the transport change without touching a single Hermes semantic.

The protocol is **synchronous** on purpose. cognee's SDK is async, so
:class:`SdkBackend` owns the event-loop thread; an HTTP backend needs no loop at
all. The provider therefore never learns that one of its transports happens to be
async, and the bridge disappears with the backend that needs it.

``SdkBackend`` is the in-process/served-SDK transport: it drives ``cognee.*``
directly, and in served mode routes through the SDK's own ``CloudClient``. Note
that this path silently drops ``session_ids`` on ``improve()`` (a CloudClient
limitation) — the reason a direct-HTTP backend exists.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import threading
from typing import Any, Optional

from .config import LOCAL_COGNEE_REQUIREMENT, ollama_embedding_pins, str_to_bool

logger = logging.getLogger(__name__)


def has_cognee() -> bool:
    """True when the cognee package is importable. Never imports it."""
    return importlib.util.find_spec("cognee") is not None


class CogneeNotInstalledError(RuntimeError):
    """Local or embedded mode needs the cognee package, which is not installed."""


def cognee_not_installed_message(mode: str) -> str:
    remote = (
        "Or use Cognee Cloud or another cognee server: run `hermes memory setup cognee`, "
        "choose remote, and enter the service URL and API key."
    )
    if mode == "local":
        return (
            "cognee local mode needs its runtime, which is not installed yet. Run "
            "`hermes cognee install-runtime` in a terminal: it installs "
            f"{LOCAL_COGNEE_REQUIREMENT} into a private environment (about 1 GB from PyPI), "
            f"kept apart from Hermes's own because their dependencies conflict. {remote}"
        )
    return (
        f"cognee {mode} mode needs the cognee package ({LOCAL_COGNEE_REQUIREMENT}) in "
        "Hermes's own Python environment, which this plugin does not install: its "
        "dependencies conflict with Hermes's. Local mode runs cognee in a private "
        f"environment instead (unset COGNEE_EMBEDDED). {remote}"
    )


class MemoryBackend:
    """How the provider reaches cognee.

    Operations are required. The configuration hooks default to no-ops because
    they only mean something to an in-process SDK: an HTTP transport has no local
    model config, no local data roots, and no local user table.
    """

    # -- connection / lifecycle -------------------------------------------

    def configure_models(self, *, llm_api_key: str, llm_model: str) -> None:
        """Point the transport at an LLM (in-process transports only)."""

    def configure_local_roots(self, *, data_root: str, system_root: str) -> None:
        """Point the transport at local storage (in-process transports only)."""

    def connect(self, *, url: str, api_key: str, timeout: float) -> None:
        """Attach to a cognee instance at *url*."""
        raise NotImplementedError

    def resolve_identity(self, *, email: str, password: str, timeout: float) -> Any:
        """Resolve the principal this transport writes as, if it owns one."""
        return None

    def close(self, *, timeout: float = 5.0, unregister: bool = True) -> None:
        """Release connections and background resources.

        ``unregister=False`` keeps the agent registration alive on the way out,
        because a detached worker owns it and is still running this session's
        improve. Dropping it here would let the server retire mid-promotion.
        """

    def connection_info(self) -> Optional[dict[str, Any]]:
        """How to reach the server this transport is attached to, if that server
        outlives this process (``url``, ``api_key``, ``agent_session_name``).

        None for in-process transports — there is nothing to close out after the
        process dies. The provider uses this to arm the crash-safe exit watcher.
        """
        return None

    def ensure_dataset(self, *, dataset: str, timeout: float) -> None:
        """Make sure *dataset* exists (server transports only).

        In-process transports need nothing: the SDK creates datasets on write,
        and grants live in the same process. On a server, a recall-first session
        against a fresh instance would otherwise hit a missing dataset.
        """

    def probe_health(self, *, timeout: float) -> bool:
        """Whether the server answers ``/health`` within ``timeout``.

        In-process transports have no server to probe and always answer.
        """
        return True

    # -- operations --------------------------------------------------------

    def recall(
        self,
        *,
        query: str,
        session_id: Optional[str],
        datasets: Optional[list[str]],
        top_k: int,
        auto_route: bool,
        query_type: Optional[str],
        scope: Any = None,
        code_query: Optional[dict[str, Any]] = None,
        only_context: bool = False,
        timeout: float,
    ) -> list[Any]:
        """Search memory. ``scope`` is the list of server scopes to read —
        ``["graph"]`` for memory, ``["code"]`` for the deterministic code graph
        — alongside the targets it implies. Memory is read from the graph only:
        the session-cache scopes and the server's ``auto`` scope (which folds
        them in) are never requested, so a transport handed no scope states
        ``["graph"]`` rather than leaving the server to infer one.

        A transport must state the scope outright rather than leave the server
        to infer it from which targets happen to be set: that inference is
        conditional on other fields, so it silently changes meaning when one of
        them moves.
        """
        raise NotImplementedError

    def remember_session(self, *, text: str, session_id: str, dataset: str, timeout: float) -> Any:
        """Store a turn in the session cache (cheap, no graph extraction)."""
        raise NotImplementedError

    def remember_permanent(
        self,
        *,
        text: str,
        dataset: str,
        session_ids: list[str],
        timeout: float,
        chunk_size: int = 0,
    ) -> Any:
        """Store content in the permanent graph. ``chunk_size`` > 0 caps the
        tokens per chunk; 0 leaves it to the server."""
        raise NotImplementedError

    def remember_entry(
        self, *, entry: dict[str, Any], session_id: str, dataset: str, timeout: float
    ) -> Any:
        """Store one typed session entry (a ``trace`` of a tool call, or a ``qa``)."""
        raise self._http_only("typed session entries (tool-call traces)")

    def forget(
        self,
        *,
        dataset: Optional[str],
        everything: bool,
        memory_only: bool,
        timeout: float,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def improve(
        self, *, dataset: str, session_ids: list[str], background: bool, timeout: float
    ) -> Any:
        """Bridge session-cache content into the permanent graph."""
        raise NotImplementedError

    # -- inspection / targeted deletion / code graph -------------------------
    #
    # These operations exist only on the server's REST API — the SDK's client
    # objects have no equivalents — so only the HTTP transport implements them.
    # The shared default raises a uniform, actionable error instead of
    # NotImplementedError so a tool envelope can carry it verbatim.

    def _http_only(self, operation: str) -> RuntimeError:
        return RuntimeError(
            f"{operation} requires the HTTP transport to a cognee server. "
            "Unset COGNEE_TRANSPORT/COGNEE_EMBEDDED (local-server mode is the "
            "default) or set COGNEE_TRANSPORT=http."
        )

    def list_datasets(self, *, timeout: float) -> list[dict[str, Any]]:
        raise self._http_only("listing datasets")

    def list_dataset_data(
        self,
        *,
        dataset_id: str,
        timeout: float,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        raise self._http_only("listing dataset documents")

    def list_sessions(self, *, range: str, limit: int, offset: int, timeout: float) -> Any:
        raise self._http_only("listing sessions")

    def get_session(self, *, session_id: str, timeout: float) -> Optional[dict[str, Any]]:
        raise self._http_only("reading a session")

    def read_raw_data(self, *, dataset_id: str, data_id: str, timeout: float) -> str:
        raise self._http_only("reading a stored document")

    def forget_document(self, *, dataset_id: str, data_id: str, timeout: float) -> dict[str, Any]:
        raise self._http_only("per-document deletion")

    def index_repository(
        self,
        *,
        repo: str,
        dataset: str,
        index_vectors: bool = False,
        run_in_background: bool = True,
        timeout: float,
    ) -> dict[str, Any]:
        raise self._http_only("code-graph indexing")

    def dataset_pipeline_status(
        self, *, dataset_id: str, pipeline: str = "code_graph_pipeline", timeout: float
    ) -> str:
        raise self._http_only("pipeline status polling")

    # -- diagnostics -------------------------------------------------------

    def empty_recall_hint(self, *, timeout: float = 5.0) -> Optional[str]:
        """A reason recall *structurally* could not match, or None.

        Called only when a recall comes back empty, to distinguish a genuine miss
        from a misconfiguration. Best-effort: never raises.
        """
        return None

    def overflow_hint(self) -> Optional[str]:
        """Evidence the server's embedder is truncating/pooling oversized inputs,
        or None. Best-effort: never raises.

        Only the HTTP transport against a *local* server can answer — it reads
        the server log this plugin spawned. In-process transports keep the
        default: an embedding failure there raises in this process and needs no
        after-the-fact detection.
        """
        return None


class _AsyncBridge:
    """Run cognee's async SDK from the provider's synchronous interface."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _ensure_loop(self) -> None:
        with self._lock:
            if self._loop is not None and self._loop.is_running():
                return
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._loop.run_forever,
                daemon=True,
                name="cognee-hermes-event-loop",
            )
            self._thread.start()

    def run(self, coro, timeout: float):
        self._ensure_loop()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def shutdown(self) -> None:
        with self._lock:
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5.0)
            self._loop = None
            self._thread = None


class SdkBackend(MemoryBackend):
    """Drives the cognee Python SDK, either in-process or via ``cognee.serve()``.

    ``served`` records whether :meth:`connect` ran. It gates two things that only
    make sense in-process: the local user identity (a served instance owns
    identity via the api-key principal) and the embedding-dimension probe (a
    served instance owns the vector store).
    """

    def __init__(self) -> None:
        self._bridge = _AsyncBridge()
        self._user: Any = None
        self.served = False

    @property
    def user(self) -> Any:
        return self._user

    # -- connection / lifecycle -------------------------------------------

    def configure_models(self, *, llm_api_key: str, llm_model: str) -> None:
        try:
            import cognee

            if llm_api_key:
                cognee.config.set_llm_api_key(str(llm_api_key))
            if llm_model:
                cognee.config.set_llm_model(str(llm_model))
        except Exception as exc:
            logger.debug("Cognee model configuration failed: %s", exc)

    def configure_local_roots(self, *, data_root: str, system_root: str) -> None:
        # Only embedded mode gets here, and it runs cognee in this process; fail
        # with the fix rather than with an ImportError on the first memory call.
        if not has_cognee():
            raise CogneeNotInstalledError(cognee_not_installed_message("embedded"))
        # Embedded mode runs cognee's session manager in *this* process, and it is
        # gated on CACHING: without it a session write is silently dropped while
        # still reporting success, so turns never reach the graph. The spawned
        # server gets the same flags in server_bootstrap._spawn. setdefault so an
        # explicit user value wins.
        for key, value in (("CACHING", "true"), ("AUTO_FEEDBACK", "true")):
            os.environ.setdefault(key, value)
        # Embedded mode builds cognee's (lru-cached) embedding config in this
        # process, so the Ollama overflow protection has to land before the
        # first cognee call — same pins the spawned server gets in
        # server_bootstrap._spawn, see ollama_embedding_pins.
        for key, value in ollama_embedding_pins(os.environ).items():
            os.environ.setdefault(key, value)
        try:
            import cognee

            cognee.config.data_root_directory(str(data_root))
            cognee.config.system_root_directory(str(system_root))
        except Exception as exc:
            logger.debug("Cognee root configuration failed: %s", exc)

    def connect(self, *, url: str, api_key: str, timeout: float) -> None:
        self._bridge.run(self._do_serve(url, api_key), timeout=timeout)
        self.served = True

    def resolve_identity(self, *, email: str, password: str, timeout: float) -> Any:
        self._user = self._bridge.run(self._ensure_identity(email, password), timeout=timeout)
        return self._user

    def close(self, *, timeout: float = 5.0, unregister: bool = True) -> None:
        # ``unregister`` is accepted and ignored: this transport never registers
        # an agent connection, and its connection_info() is None, so it is never
        # on the handoff path. Disconnecting and stopping the bridge are pure
        # local teardown with nothing for a detached worker to inherit.
        del unregister
        if self.served:
            try:
                self._bridge.run(self._do_disconnect(), timeout=timeout)
            except Exception:
                pass
        self._bridge.shutdown()

    # -- operations --------------------------------------------------------

    def recall(
        self,
        *,
        query,
        session_id,
        datasets,
        top_k,
        auto_route,
        query_type,
        scope=None,
        code_query=None,
        only_context=False,
        timeout,
    ) -> list[Any]:
        # ``scope`` and ``only_context`` are forwarded: both are ``cognee.recall``
        # keywords on every version this plugin pins (>= 1.4), and the per-prompt
        # memory lane depends on them — ``scope=["graph"]`` with ``only_context``
        # and a ``session_id`` is what makes the server hand back the full
        # prompt-shaped memory item instead of running an LLM completion.
        # ``code_query`` is dropped: the SDK entry point has no such keyword, and
        # the code lane is HTTP-only anyway.
        del code_query
        return self._bridge.run(
            self._do_recall(
                query,
                session_id,
                datasets,
                top_k,
                auto_route,
                query_type,
                scope=scope,
                only_context=only_context,
            ),
            timeout=timeout,
        )

    def remember_session(self, *, text, session_id, dataset, timeout) -> Any:
        return self._bridge.run(
            self._do_remember_session(text, session_id, dataset), timeout=timeout
        )

    def remember_permanent(self, *, text, dataset, session_ids, timeout, chunk_size=0) -> Any:
        return self._bridge.run(
            self._do_remember_permanent(text, dataset, session_ids, chunk_size=chunk_size),
            timeout=timeout,
        )

    def forget(self, *, dataset, everything, memory_only, timeout) -> dict[str, Any]:
        return self._bridge.run(
            self._do_forget(dataset, everything=everything, memory_only=memory_only),
            timeout=timeout,
        )

    def improve(self, *, dataset, session_ids, background, timeout) -> Any:
        return self._bridge.run(
            self._do_improve(dataset, session_ids, run_in_background=background),
            timeout=timeout,
        )

    # -- diagnostics -------------------------------------------------------

    def empty_recall_hint(self, *, timeout: float = 5.0) -> Optional[str]:
        """Confirmed embedding-dimension mismatch, or None. In-process only.

        A served instance owns the vector store, so introspecting the local
        in-process engine would be meaningless — skip.
        """
        if self.served:
            return None
        try:
            return self._bridge.run(dimension_mismatch_hint(), timeout=timeout)
        except Exception:
            return None

    # -- the async SDK surface ---------------------------------------------

    def _add_user_kwarg(self, kwargs: dict[str, Any]) -> None:
        """Inject the local user only when running in-process.

        In served mode the instance owns identity (api-key principal) and
        ``self._user`` is None. Passing ``user=None`` is not the same as omitting
        the key — the SDK may treat an explicit None differently (overriding a
        default, affecting tenant scoping) — so we omit it entirely instead.
        """
        if not self.served and self._user is not None:
            kwargs["user"] = self._user

    async def _ensure_identity(self, email: str, password: str):
        try:
            from cognee.modules.users.methods import (
                create_user,
                get_default_user,
                get_user_by_email,
            )
        except Exception:
            return None

        user = await get_user_by_email(email)
        if user:
            return user
        try:
            return await create_user(
                email=email,
                password=password,
                is_verified=True,
                is_active=True,
            )
        except Exception:
            user = await get_user_by_email(email)
            if user:
                return user
            return await get_default_user()

    async def _do_serve(self, url: str, api_key: str):
        import cognee

        kwargs = {"url": url}
        if api_key:
            kwargs["api_key"] = api_key
        return await cognee.serve(**kwargs)

    async def _do_disconnect(self):
        import cognee

        return await cognee.disconnect()

    async def _do_recall(
        self,
        query: str,
        session_id: Optional[str],
        datasets: Optional[list[str]],
        top_k: int,
        auto_route: bool,
        query_type: Optional[str],
        scope: Any = None,
        only_context: bool = False,
    ) -> list[Any]:
        import cognee

        kwargs: dict[str, Any] = {
            "top_k": top_k,
            "auto_route": auto_route,
        }
        self._add_user_kwarg(kwargs)
        if session_id is not None:
            kwargs["session_id"] = session_id
        if datasets is not None:
            kwargs["datasets"] = datasets
        if query_type:
            kwargs["query_type"] = resolve_search_type(query_type)
        # Always stated: left out, cognee resolves the scope to ``auto`` and
        # folds the session cache in whenever a session id travels.
        kwargs["scope"] = scope or ["graph"]
        if only_context:
            kwargs["only_context"] = True

        return await cognee.recall(query_text=query, **kwargs)

    async def _do_remember_session(self, content: str, session_id: str, dataset: str):
        import cognee

        kwargs: dict[str, Any] = {
            "data": content,
            "dataset_name": dataset,
            "session_id": session_id,
            "self_improvement": False,
        }
        self._add_user_kwarg(kwargs)
        return await cognee.remember(**kwargs)

    async def _do_remember_permanent(
        self, content: str, dataset: str, session_ids: list[str], chunk_size: int = 0
    ):
        import cognee

        kwargs: dict[str, Any] = {
            "data": content,
            "dataset_name": dataset,
            "self_improvement": True,
            "session_ids": session_ids,
        }
        if chunk_size > 0:
            kwargs["chunk_size"] = chunk_size
        self._add_user_kwarg(kwargs)
        return await cognee.remember(**kwargs)

    async def _do_forget(
        self,
        dataset: Optional[str],
        *,
        everything: bool = False,
        memory_only: bool = False,
    ) -> dict[str, Any]:
        import cognee

        kwargs: dict[str, Any] = {
            "everything": everything,
            "memory_only": memory_only,
        }
        if dataset and not everything:
            kwargs["dataset"] = dataset
        self._add_user_kwarg(kwargs)
        return await cognee.forget(**kwargs)

    async def _do_improve(
        self, dataset: str, session_ids: list[str], run_in_background: bool = False
    ):
        import cognee

        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "session_ids": session_ids,
            "run_in_background": run_in_background,
        }
        self._add_user_kwarg(kwargs)
        return await cognee.improve(**kwargs)


# --- Embedding-dimension mismatch detection ---------------------------------
# When the embedding model changes between writing and reading, stored vectors
# and query vectors differ in size, so recall silently matches nothing. These
# helpers turn that silent miss into a one-line actionable error naming both
# dimensions and the active embedder. Best-effort and fail-safe (any uncertainty
# returns None). Only valid in-process, where this process owns the vector store;
# a served instance owns it and the in-process engine here would not reflect it.


async def sample_stored_vector_dim(engine) -> Optional[int]:
    """Sample the dimension of a stored vector from any populated collection, or None.

    Enumerates the store's actual collections via the vector interface's
    ``get_connection().table_names()`` (the same path cognee's own ``has_collection``
    uses) rather than assuming fixed collection names, so it also covers custom
    pipelines. Never raises: each collection is probed independently and any
    unreadable one is skipped. Covers cognee's default local backend (LanceDB); other
    backends whose connection can't enumerate return None and fall back to the normal
    empty-recall path.
    """
    try:
        connection = await engine.get_connection()
        names = await connection.table_names()
    except Exception:
        return None
    for name in names:
        try:
            collection = await engine.get_collection(name)
            rows = await collection.query().limit(1).to_list()
            if rows:
                vector = rows[0].get("vector")
                if vector is not None:
                    return len(vector)
        except Exception:
            continue
    return None


async def dimension_mismatch_hint(engine=None) -> Optional[str]:
    """One-line diagnostic when the stored vectors differ in size from the active
    embedder's query vectors (so recall can never match), else None.

    ``engine`` is injectable for testing. Best-effort and fail-safe: any error, or
    an indeterminate/matching dimension, returns None so the caller keeps the
    normal empty-recall behavior.
    """
    try:
        if engine is None:
            from cognee.infrastructure.databases.vector import get_vector_engine

            engine = get_vector_engine()
        embed = getattr(engine, "embedding_engine", None)
        if embed is None:
            return None
        query_dim = int(embed.get_vector_size())
        stored_dim = await sample_stored_vector_dim(engine)
        if not stored_dim or not query_dim or stored_dim == query_dim:
            return None
        model = getattr(embed, "model", None) or "unknown-model"
        provider = getattr(embed, "provider", None) or "unknown-provider"
        return (
            "Cognee recall found nothing because the embedder changed: stored vectors are "
            f"{stored_dim}-d but the active embedder '{model}' (provider '{provider}') produces "
            f"{query_dim}-d queries. Re-index this data with the current embedder, or set "
            f"EMBEDDING_MODEL/EMBEDDING_DIMENSIONS back to the {stored_dim}-d model that wrote it."
        )
    except Exception:
        return None


def resolve_search_type(search_type: str):
    """Map a search-type name onto cognee's ``SearchType``, defaulting sanely.

    An unrecognized name falls back to ``GRAPH_COMPLETION`` rather than failing the
    recall. Kept in the backend so the provider needs no cognee import — and so a
    future cognee search type keeps working without a provider change.
    """
    from cognee.modules.search.types import SearchType

    key = str(search_type).upper().strip()
    return getattr(SearchType, key, SearchType.GRAPH_COMPLETION)


def build_backend(config: Optional[dict[str, Any]] = None):
    """Pick a transport from config.

    Direct HTTP is the default: it is what the other cognee plugins use, and the
    only transport that sends ``session_ids`` on ``improve()`` — the SDK's
    ``CloudClient`` drops them, so the session-to-graph bridge silently becomes a
    dataset-wide improve.

    The SDK transport is selected by ``COGNEE_EMBEDDED=true``, which it is the only
    one able to serve (it runs cognee in this process, with no server involved), or
    explicitly by ``COGNEE_TRANSPORT=sdk`` for a like-for-like comparison.
    """
    config = config or {}
    transport = str(config.get("transport") or "").strip().lower()
    if transport in {"sdk", "cognee"}:
        return SdkBackend()
    # Embedded means "no server" — only the in-process SDK can do that, so it wins
    # over the default regardless of transport.
    if str_to_bool(config.get("embedded"), False):
        return SdkBackend()

    from .http_backend import HttpBackend

    return HttpBackend()


def default_backend() -> MemoryBackend:
    """The transport used before config is loaded, or when none is injected.

    Deliberately the SDK: it needs no URL, no key and no reachable server, so
    constructing a provider is side-effect free. ``initialize()`` replaces it with
    the configured transport once it knows what to build.
    """
    return SdkBackend()
