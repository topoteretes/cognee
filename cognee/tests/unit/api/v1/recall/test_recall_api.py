import importlib
import types
from uuid import uuid4

import pytest

from cognee.modules.search.types import SearchType


def _make_user(user_id=None, tenant_id=None):
    return types.SimpleNamespace(id=user_id or uuid4(), tenant_id=tenant_id)


@pytest.fixture
def api_recall_mod():
    return importlib.import_module("cognee.api.v1.recall.recall")


@pytest.mark.asyncio
async def test_recall_dataset_ids_takes_precedence_over_datasets(monkeypatch, api_recall_mod):
    user = _make_user()
    explicit_id = uuid4()

    captured = {}

    async def dummy_set_session_user_context_variable(_user):
        return None

    async def dummy_authorized_search(**kwargs):
        captured["dataset_ids"] = kwargs.get("dataset_ids")
        return []

    async def dummy_get_authorized_existing_datasets(*args, **kwargs):
        captured["resolved_from_datasets"] = True
        return []

    def dummy_get_remote_client():
        return None

    monkeypatch.setattr(
        api_recall_mod,
        "set_session_user_context_variable",
        dummy_set_session_user_context_variable,
    )
    monkeypatch.setattr(
        api_recall_mod,
        "get_authorized_existing_datasets",
        dummy_get_authorized_existing_datasets,
    )

    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    search_methods = importlib.import_module("cognee.modules.search.methods.search")

    monkeypatch.setattr(serve_state, "get_remote_client", dummy_get_remote_client)
    monkeypatch.setattr(search_methods, "authorized_search", dummy_authorized_search)

    out = await api_recall_mod.recall(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=["some_dataset_name"],
        dataset_ids=[explicit_id],
        auto_route=False,
        user=user,
    )

    assert out == []
    assert captured["dataset_ids"] == [explicit_id]
    assert "resolved_from_datasets" not in captured


@pytest.mark.asyncio
async def test_recall_remote_client_forwards_dataset_ids(monkeypatch, api_recall_mod):
    user = _make_user()
    explicit_id = uuid4()

    captured = {}

    async def dummy_remote_recall(query_text, query_type, **kwargs):
        captured["dataset_ids"] = kwargs.get("dataset_ids")
        return []

    dummy_remote_client = types.SimpleNamespace(recall=dummy_remote_recall)

    def dummy_get_remote_client():
        return dummy_remote_client

    async def dummy_get_authorized_existing_datasets(*args, **kwargs):
        captured["resolved_from_datasets"] = True
        return []

    monkeypatch.setattr(
        api_recall_mod,
        "get_authorized_existing_datasets",
        dummy_get_authorized_existing_datasets,
    )

    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(serve_state, "get_remote_client", dummy_get_remote_client)

    out = await api_recall_mod.recall(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=["some_dataset_name"],
        dataset_ids=[explicit_id],
        auto_route=False,
        user=user,
    )

    assert out == []
    assert captured["dataset_ids"] == [explicit_id]
    assert "resolved_from_datasets" not in captured


@pytest.mark.asyncio
async def test_recall_session_with_dataset_ids_does_not_short_circuit_graph(
    monkeypatch, api_recall_mod
):
    user = _make_user()
    explicit_id = uuid4()

    captured = {}

    async def dummy_set_session_user_context_variable(_user):
        return None

    async def dummy_authorized_search(**kwargs):
        captured["dataset_ids"] = kwargs.get("dataset_ids")
        return []

    async def dummy_get_authorized_existing_datasets(*args, **kwargs):
        captured["resolved_from_datasets"] = True
        return []

    def dummy_get_remote_client():
        return None

    async def dummy_search_session(**kwargs):
        return []

    async def dummy_search_trace(**kwargs):
        return []

    monkeypatch.setattr(
        api_recall_mod,
        "set_session_user_context_variable",
        dummy_set_session_user_context_variable,
    )
    monkeypatch.setattr(
        api_recall_mod,
        "get_authorized_existing_datasets",
        dummy_get_authorized_existing_datasets,
    )
    monkeypatch.setattr(api_recall_mod, "_search_session", dummy_search_session)
    monkeypatch.setattr(api_recall_mod, "_search_trace", dummy_search_trace)

    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    search_methods = importlib.import_module("cognee.modules.search.methods.search")

    monkeypatch.setattr(serve_state, "get_remote_client", dummy_get_remote_client)
    monkeypatch.setattr(search_methods, "authorized_search", dummy_authorized_search)

    out = await api_recall_mod.recall(
        query_text="q",
        query_type=None,
        dataset_ids=[explicit_id],
        session_id="session-1",
        user=user,
    )

    assert out == []
    assert captured["dataset_ids"] == [explicit_id]
    assert "resolved_from_datasets" not in captured


# ----------------------------------------------------------------- session_context scope


def test_normalize_scope_accepts_session_context():
    from cognee.memory.entries import normalize_scope

    assert normalize_scope("session_context") == ["session_context"]
    assert normalize_scope("graph_context") == ["graph"]
    assert normalize_scope(["graph_context", "session_context"]) == ["graph", "session_context"]
    assert "session_context" in normalize_scope("all")
    assert "graph_context" not in normalize_scope("all")


@pytest.mark.asyncio
async def test_recall_session_context_scope_threads_profile(monkeypatch, api_recall_mod):
    from cognee.modules.recall.types.RecallResponse import ResponseSessionContextEntry

    user = _make_user()
    captured = {}

    async def dummy_fetch_session_context(query_text, session_id, context_profile, user=None):
        captured["context_profile"] = context_profile
        captured["session_id"] = session_id
        return [
            ResponseSessionContextEntry(
                content="block", context_profile=context_profile, source="session_context"
            )
        ]

    monkeypatch.setattr(api_recall_mod, "_fetch_session_context", dummy_fetch_session_context)
    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)

    out = await api_recall_mod.recall(
        query_text="q",
        scope=["session_context"],
        context_profile="agent",
        session_id="s",
        user=user,
    )

    assert captured == {"context_profile": "agent", "session_id": "s"}
    assert len(out) == 1
    assert out[0].source == "session_context"
    assert out[0].context_profile == "agent"


@pytest.mark.asyncio
async def test_recall_remote_client_forwards_context_profile(monkeypatch, api_recall_mod):
    user = _make_user()
    captured = {}

    async def dummy_remote_recall(query_text, query_type, **kwargs):
        captured["context_profile"] = kwargs.get("context_profile")
        return []

    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(
        serve_state, "get_remote_client", lambda: types.SimpleNamespace(recall=dummy_remote_recall)
    )

    out = await api_recall_mod.recall(
        query_text="q",
        scope=["session_context"],
        context_profile="agent",
        session_id="s",
        user=user,
    )

    assert out == []
    assert captured["context_profile"] == "agent"


@pytest.mark.asyncio
async def test_recall_remote_client_receives_resolved_all_scope(monkeypatch, api_recall_mod):
    user = _make_user()
    captured = {}

    async def dummy_remote_recall(query_text, query_type, **kwargs):
        captured["scope"] = kwargs.get("scope")
        return []

    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(
        serve_state, "get_remote_client", lambda: types.SimpleNamespace(recall=dummy_remote_recall)
    )

    out = await api_recall_mod.recall(
        query_text="q",
        scope="all",
        session_id="s",
        user=user,
    )

    assert out == []
    assert captured["scope"] == ["graph", "session", "trace", "session_context"]


@pytest.mark.asyncio
async def test_recall_remote_client_receives_resolved_legacy_graph_context_scope(
    monkeypatch, api_recall_mod
):
    user = _make_user()
    captured = {}

    async def dummy_remote_recall(query_text, query_type, **kwargs):
        captured["scope"] = kwargs.get("scope")
        return []

    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(
        serve_state, "get_remote_client", lambda: types.SimpleNamespace(recall=dummy_remote_recall)
    )

    out = await api_recall_mod.recall(
        query_text="q",
        scope="graph_context",
        session_id="s",
        user=user,
    )

    assert out == []
    assert captured["scope"] == ["graph"]


@pytest.mark.asyncio
async def test_recall_remote_client_receives_resolved_legacy_graph_context_scope_list(
    monkeypatch, api_recall_mod
):
    user = _make_user()
    captured = {}

    async def dummy_remote_recall(query_text, query_type, **kwargs):
        captured["scope"] = kwargs.get("scope")
        return []

    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(
        serve_state, "get_remote_client", lambda: types.SimpleNamespace(recall=dummy_remote_recall)
    )

    out = await api_recall_mod.recall(
        query_text="q",
        scope=["graph_context", "session_context"],
        session_id="s",
        user=user,
    )

    assert out == []
    assert captured["scope"] == ["graph", "session_context"]


@pytest.mark.asyncio
async def test_recall_remote_client_preserves_default_scope(monkeypatch, api_recall_mod):
    user = _make_user()
    captured = {}

    async def dummy_remote_recall(query_text, query_type, **kwargs):
        captured["scope"] = kwargs.get("scope")
        return []

    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(
        serve_state, "get_remote_client", lambda: types.SimpleNamespace(recall=dummy_remote_recall)
    )

    out = await api_recall_mod.recall(
        query_text="q",
        session_id="s",
        user=user,
    )

    assert out == []
    assert captured["scope"] is None


class _FakeRecallResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return []

    async def text(self):
        return ""


class _FakeRecallSession:
    """Captures the JSON body of the recall POST."""

    def __init__(self):
        self.last_json = None

    def post(self, _url, json=None, **kwargs):
        self.last_json = json
        return _FakeRecallResponse()

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_fetch_session_context_renders_profile_read_only(monkeypatch, api_recall_mod):
    from cognee.infrastructure.session.session_context_models import SessionContextEntry

    agent_row = SessionContextEntry(
        id="a1",
        section="tool_rules",
        context_profile="agent",
        content="Use uv run",
        created_at="2026-06-11T10:00:00",
    ).model_dump()

    class _FakeSM:
        is_available = True

        def __init__(self):
            self.updates = 0

        async def get_session_context_entries(self, user_id, session_id):
            return [agent_row]

        async def update_session_context_entry(self, **kwargs):
            self.updates += 1
            return True

    fake = _FakeSM()

    async def fake_resolve_cache_user(session_id, caller_user_id):
        return caller_user_id

    gsm_mod = importlib.import_module("cognee.infrastructure.session.get_session_manager")
    monkeypatch.setattr(gsm_mod, "get_session_manager", lambda: fake)
    monkeypatch.setattr(api_recall_mod, "_resolve_session_cache_user_id", fake_resolve_cache_user)

    out = await api_recall_mod._fetch_session_context(
        query_text="q",
        session_id="s",
        context_profile="agent",
        user=_make_user(),
    )

    assert len(out) == 1
    assert out[0].source == "session_context"
    assert out[0].context_profile == "agent"
    assert "Use uv run" in out[0].content
    assert fake.updates == 0  # read-only: stamp_served=False


@pytest.mark.asyncio
async def test_fetch_session_context_empty_when_no_lessons(monkeypatch, api_recall_mod):
    class _EmptySM:
        is_available = True

        async def get_session_context_entries(self, user_id, session_id):
            return []

    async def fake_resolve_cache_user(session_id, caller_user_id):
        return caller_user_id

    gsm_mod = importlib.import_module("cognee.infrastructure.session.get_session_manager")
    monkeypatch.setattr(gsm_mod, "get_session_manager", lambda: _EmptySM())
    monkeypatch.setattr(api_recall_mod, "_resolve_session_cache_user_id", fake_resolve_cache_user)

    out = await api_recall_mod._fetch_session_context(
        query_text="q", session_id="s", context_profile="agent", user=_make_user()
    )
    assert out == []


@pytest.mark.asyncio
async def test_cloud_client_recall_payload_includes_context_profile(monkeypatch):
    # cloud_client.recall takes **kwargs and only forwards keys it picks, so a missed forward
    # is a silent drop — assert context_profile actually lands in the POST body.
    from cognee.api.v1.serve.cloud_client import CloudClient

    client = CloudClient("http://example", "key")
    fake = _FakeRecallSession()

    async def fake_get_session():
        return fake

    monkeypatch.setattr(client, "_get_session", fake_get_session)

    await client.recall(
        "q", None, scope=["session_context"], context_profile="agent", session_id="s"
    )

    assert fake.last_json.get("context_profile") == "agent"


@pytest.mark.asyncio
async def test_record_recall_context_usage_accumulates(monkeypatch, api_recall_mod):
    """Context-only recall attributes query + served context to the session."""

    class _Entry:
        def __init__(self, text):
            self._text = text

        def model_dump_json(self):
            return self._text

    recorded = {}

    async def fake_record(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(
        "cognee.modules.session_lifecycle.usage_tracking.record_transcript_usage", fake_record
    )

    await api_recall_mod._record_recall_context_usage(
        session_id="s1",
        user=_make_user(),
        query_text="what did I do",
        entries=[_Entry("ctx-a"), _Entry("ctx-b")],
    )

    assert recorded["session_id"] == "s1"
    assert recorded["input_text"] == "what did I do"
    assert "ctx-a" in recorded["output_text"] and "ctx-b" in recorded["output_text"]


@pytest.mark.asyncio
async def test_record_recall_context_usage_noop_without_user_id(monkeypatch, api_recall_mod):
    """No attribution when the user id can't be resolved."""
    called = False

    async def fake_record(**kwargs):
        nonlocal called
        called = True

    async def no_user(_user):
        return None

    monkeypatch.setattr(
        "cognee.modules.session_lifecycle.usage_tracking.record_transcript_usage", fake_record
    )
    monkeypatch.setattr(api_recall_mod, "_resolve_user_id", no_user)

    await api_recall_mod._record_recall_context_usage(
        session_id="s1", user=None, query_text="q", entries=[]
    )

    assert called is False
