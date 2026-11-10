"""Shared stubbing for the characterization suite.

The characterization tests pin the behaviour the HTTP refactor must preserve:
the JSON envelopes Hermes hands to the model, the request payloads the provider
builds, the session-id derivation, and the lifecycle-hook semantics that exist
only in Hermes (``agent_context`` write gating, the prefetch two-phase protocol,
``on_memory_write``, ``on_delegation``).

**Two fakes, one per layer.**

* :func:`fake_backend` records the provider's calls to the transport protocol.
  Provider-level tests use this and never mention cognee — which is what lets the
  transport change underneath them.
* :func:`fake_cognee` installs a fake ``cognee`` package in ``sys.modules``. Only
  the transport's own tests (``test_sdk_backend.py``) use it, to assert wire
  format.

None of these tests need the real cognee installed, and none of them touch the
network or the filesystem outside a temporary directory.
"""

import contextlib
import sys
import threading
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognee_integration_hermes import provider as provider_mod  # noqa: E402
from cognee_integration_hermes.backend import MemoryBackend  # noqa: E402

# Module paths the fake has to occupy for ``from cognee.x.y import z`` to
# resolve. Keep in sync with the call-time imports in provider.py.
_FAKE_MODULE_PATHS = (
    "cognee",
    "cognee.modules",
    "cognee.modules.search",
    "cognee.modules.search.types",
)


class RememberResultStub:
    """Stands in for cognee's ``RememberResult`` (an object with ``.status``)."""

    def __init__(self, status="completed"):
        self.status = status


class _SearchType:
    """Enough of cognee's ``SearchType`` for ``_resolve_search_type`` to work.

    Attribute lookup returns the name as a plain string so recorded
    ``query_type`` values stay readable in assertions.
    """

    GRAPH_COMPLETION = "GRAPH_COMPLETION"
    HYBRID_COMPLETION = "HYBRID_COMPLETION"
    RAG_COMPLETION = "RAG_COMPLETION"
    CHUNKS = "CHUNKS"
    CHUNKS_LEXICAL = "CHUNKS_LEXICAL"
    TEMPORAL = "TEMPORAL"
    FEELING_LUCKY = "FEELING_LUCKY"


class _FakeCogneeConfig:
    def __init__(self, recorder):
        self._recorder = recorder

    def set_llm_api_key(self, value):
        self._recorder.record("config.set_llm_api_key", {"value": value})

    def set_llm_model(self, value):
        self._recorder.record("config.set_llm_model", {"value": value})

    def data_root_directory(self, value):
        self._recorder.record("config.data_root_directory", {"value": value})

    def system_root_directory(self, value):
        self._recorder.record("config.system_root_directory", {"value": value})


class FakeCognee:
    """Records what the provider asks cognee to do, and what it hands back.

    ``results[name]`` sets the return value for an operation; ``errors[name]``
    makes it raise. Threaded provider paths (``sync_turn``, ``on_memory_write``,
    ``queue_prefetch``) are awaited with :meth:`wait`.
    """

    def __init__(self):
        self.calls = []
        self.results = {
            "recall": [],
            "remember": RememberResultStub(),
            "improve": {},
            "forget": {"deleted": True},
            "serve": None,
            "disconnect": None,
        }
        self.errors = {}
        # ``gates[name] = threading.Event()`` holds that operation inside the call
        # until the test sets the event — the only way to deterministically
        # observe a provider worker that is mid-flight.
        self.gates = {}
        self.config = _FakeCogneeConfig(self)
        self._lock = threading.Lock()
        self._events = {}

    # -- recording ---------------------------------------------------------

    def record(self, name, kwargs):
        with self._lock:
            self.calls.append((name, kwargs))
            event = self._events.setdefault(name, threading.Event())
        event.set()
        gate = self.gates.get(name)
        if gate is not None:
            gate.wait(timeout=5.0)
        error = self.errors.get(name)
        if error is not None:
            raise error
        return self.results.get(name)

    def _event(self, name):
        with self._lock:
            return self._events.setdefault(name, threading.Event())

    def wait(self, name, timeout=2.0):
        """Block until *name* has been called at least once. Returns True on success."""
        return self._event(name).wait(timeout)

    # -- inspection --------------------------------------------------------

    def names(self):
        return [name for name, _ in self.calls]

    def kwargs_for(self, name):
        return [kwargs for called, kwargs in self.calls if called == name]

    def only_call(self, name):
        """The kwargs of the single call to *name*; fails loudly if not exactly one."""
        matches = self.kwargs_for(name)
        if len(matches) != 1:
            raise AssertionError(f"expected exactly 1 {name!r} call, got {len(matches)}")
        return matches[0]

    # -- the async surface provider.py awaits ------------------------------

    async def recall(self, **kwargs):
        return self.record("recall", kwargs)

    async def remember(self, **kwargs):
        return self.record("remember", kwargs)

    async def improve(self, **kwargs):
        return self.record("improve", kwargs)

    async def forget(self, **kwargs):
        return self.record("forget", kwargs)

    async def serve(self, **kwargs):
        return self.record("serve", kwargs)

    async def disconnect(self, **kwargs):
        return self.record("disconnect", kwargs)


@contextlib.contextmanager
def fake_cognee():
    """Install a :class:`FakeCognee` as the ``cognee`` package for the duration.

    Used by ``test_sdk_backend.py`` to assert the SDK transport's wire format.
    Provider-level tests use :func:`fake_backend` instead — they should not know
    that cognee exists.
    """
    recorder = FakeCognee()

    modules = {}
    root = types.ModuleType("cognee")
    root.recall = recorder.recall
    root.remember = recorder.remember
    root.improve = recorder.improve
    root.forget = recorder.forget
    root.serve = recorder.serve
    root.disconnect = recorder.disconnect
    root.config = recorder.config
    modules["cognee"] = root

    for path in _FAKE_MODULE_PATHS[1:]:
        modules[path] = types.ModuleType(path)
    modules["cognee.modules.search.types"].SearchType = _SearchType
    # Parent-attribute wiring, so both ``import a.b`` and ``from a.b import c`` work.
    modules["cognee"].modules = modules["cognee.modules"]
    modules["cognee.modules"].search = modules["cognee.modules.search"]
    modules["cognee.modules.search"].types = modules["cognee.modules.search.types"]

    saved = {path: sys.modules.get(path) for path in _FAKE_MODULE_PATHS}
    sys.modules.update(modules)
    try:
        yield recorder
    finally:
        for path, original in saved.items():
            if original is None:
                sys.modules.pop(path, None)
            else:
                sys.modules[path] = original


class FakeBackend(MemoryBackend):
    """Records the protocol calls the provider makes, and what it hands back.

    This is the seam the contract tests assert against: provider behaviour on one
    side, transport on the other. Each transport then has its own small test that
    it faithfully turns these protocol args into wire format
    (``test_sdk_backend.py``, and later an HTTP equivalent).

    Shares ``FakeCognee``'s recorder API — ``results``, ``errors``, ``gates``,
    ``wait``, ``kwargs_for``, ``only_call`` — so retargeting a test is a one-line
    change of which recorder it reads.
    """

    def __init__(self):
        self._recorder = FakeCognee()
        self._recorder.results = {
            "recall": [],
            "remember_session": None,
            "remember_entry": None,
            "remember_permanent": RememberResultStub(),
            "forget": {"deleted": True},
            "forget_document": {"deleted": True},
            "improve": {},
            "connect": None,
            "resolve_identity": "USER",
            "list_datasets": [],
            "list_dataset_data": [],
            "list_sessions": {"sessions": [], "has_more": False},
            "get_session": None,
            "read_raw_data": "",
            "index_repository": {},
            "dataset_pipeline_status": "",
        }
        self.empty_recall_hint_value = None
        self.overflow_hint_value = None

    # Recorder passthrough ------------------------------------------------

    @property
    def results(self):
        return self._recorder.results

    @property
    def errors(self):
        return self._recorder.errors

    @property
    def gates(self):
        return self._recorder.gates

    @property
    def calls(self):
        return self._recorder.calls

    def wait(self, name, timeout=2.0):
        return self._recorder.wait(name, timeout=timeout)

    def names(self):
        return self._recorder.names()

    def kwargs_for(self, name):
        return self._recorder.kwargs_for(name)

    def only_call(self, name):
        return self._recorder.only_call(name)

    # MemoryBackend -------------------------------------------------------

    def configure_models(self, **kwargs):
        self._recorder.record("configure_models", kwargs)

    def configure_local_roots(self, **kwargs):
        self._recorder.record("configure_local_roots", kwargs)

    def connect(self, **kwargs):
        return self._recorder.record("connect", kwargs)

    def resolve_identity(self, **kwargs):
        return self._recorder.record("resolve_identity", kwargs)

    def ensure_dataset(self, **kwargs):
        return self._recorder.record("ensure_dataset", kwargs)

    def close(self, **kwargs):
        self._recorder.record("close", kwargs)

    def recall(self, **kwargs):
        result = self._recorder.record("recall", kwargs)
        # The guidance lane (``scope=["session_context"]``) answers from its own
        # slot, so a test's ``results["recall"]`` stays the graph's answer.
        # ``errors["recall:session_context"]`` makes only that lane fail.
        if list(kwargs.get("scope") or []) == ["session_context"]:
            error = self._recorder.errors.get("recall:session_context")
            if error is not None:
                raise error
            return self._recorder.results.get("recall:session_context", [])
        return result

    def remember_session(self, **kwargs):
        return self._recorder.record("remember_session", kwargs)

    def remember_entry(self, **kwargs):
        return self._recorder.record("remember_entry", kwargs)

    def remember_permanent(self, **kwargs):
        return self._recorder.record("remember_permanent", kwargs)

    def forget(self, **kwargs):
        return self._recorder.record("forget", kwargs)

    def improve(self, **kwargs):
        return self._recorder.record("improve", kwargs)

    def list_datasets(self, **kwargs):
        return self._recorder.record("list_datasets", kwargs)

    def list_sessions(self, **kwargs):
        return self._recorder.record("list_sessions", kwargs)

    def get_session(self, **kwargs):
        return self._recorder.record("get_session", kwargs)

    def list_dataset_data(self, **kwargs):
        return self._recorder.record("list_dataset_data", kwargs)

    def read_raw_data(self, **kwargs):
        return self._recorder.record("read_raw_data", kwargs)

    def forget_document(self, **kwargs):
        return self._recorder.record("forget_document", kwargs)

    def index_repository(self, **kwargs):
        return self._recorder.record("index_repository", kwargs)

    def dataset_pipeline_status(self, **kwargs):
        return self._recorder.record("dataset_pipeline_status", kwargs)

    def empty_recall_hint(self, **kwargs):
        self._recorder.record("empty_recall_hint", kwargs)
        return self.empty_recall_hint_value

    def overflow_hint(self, **kwargs):
        self._recorder.record("overflow_hint", kwargs)
        return self.overflow_hint_value


_CURRENT_FAKE_BACKEND = None


@contextlib.contextmanager
def fake_backend():
    """Yield a :class:`FakeBackend` and make it the default for ``make_provider``.

    Ambient on purpose, mirroring how ``fake_cognee`` installs itself into
    ``sys.modules``: a test says ``with fake_backend() as fake`` and every
    provider built inside the block records to it, with no plumbing at the call
    sites.
    """
    global _CURRENT_FAKE_BACKEND
    previous = _CURRENT_FAKE_BACKEND
    backend = FakeBackend()
    _CURRENT_FAKE_BACKEND = backend
    try:
        yield backend
    finally:
        _CURRENT_FAKE_BACKEND = previous


def make_provider(
    *,
    backend=None,
    dataset="hermes",
    top_k=5,
    session_id="s-1",
    session_cognee_id=None,
    remote_mode=True,
    watcher_state_path=None,
    writes_enabled=True,
    auto_route=True,
    improve_on_end=True,
    stored_since_improve=1,
    config=None,
):
    """A provider with post-``initialize()`` state set and a fake transport.

    Uses the ambient backend from an enclosing ``fake_backend()`` block, or a
    throwaway one when there is none.
    """
    provider = provider_mod.CogneeMemoryProvider(
        backend=backend or _CURRENT_FAKE_BACKEND or FakeBackend()
    )
    provider._initialized = True
    provider._remote_mode = remote_mode
    # Non-None makes the provider look armed, so on_session_end will try to hand
    # the close to a detached worker instead of improving in-process.
    provider._watcher_state_path = watcher_state_path
    provider._writes_enabled = writes_enabled
    provider._dataset = dataset
    provider._top_k = top_k
    provider._auto_route = auto_route
    provider._improve_on_end = improve_on_end
    provider._session_id = session_id
    provider._session_cognee_id = session_cognee_id or f"hermes_{session_id}"
    provider._default_dataset = dataset
    # Entries stored since the session's last improve. A session that stored
    # nothing skips its pre-compress and session-end improves, so the default is
    # a session that has written something; pass 0 to test the skip.
    provider._stored_since_improve = stored_since_improve
    provider._config = {
        "recall_timeout": 5,
        "write_timeout": 5,
        "improve_timeout": 5,
        "improve_background": "",
        # The hit header has its own tests; everything else runs without it.
        "memory_hits": False,
        # So does the memory steer (test_memory_steer.py).
        "memory_steer": False,
        **(config or {}),
    }
    return provider
