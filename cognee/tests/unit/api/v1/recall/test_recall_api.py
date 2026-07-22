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


def _session_row(user_id, dataset_id):
    return types.SimpleNamespace(user_id=user_id, dataset_id=dataset_id)


def test_session_cache_owner_prefers_callers_scoped_row(api_recall_mod):
    caller_id = uuid4()
    other_id = uuid4()
    caller_dataset_id = uuid4()
    other_dataset_id = uuid4()
    rows = [
        _session_row(other_id, other_dataset_id),
        _session_row(caller_id, caller_dataset_id),
    ]

    owner = api_recall_mod._select_session_cache_owner(
        rows,
        caller_id,
        {caller_dataset_id, other_dataset_id},
        {caller_dataset_id, other_dataset_id},
    )

    assert owner == caller_id


def test_session_cache_owner_honors_requested_dataset_scope(api_recall_mod):
    caller_id = uuid4()
    agent_id = uuid4()
    caller_dataset_id = uuid4()
    agent_dataset_id = uuid4()
    rows = [
        _session_row(caller_id, caller_dataset_id),
        _session_row(agent_id, agent_dataset_id),
    ]

    owner = api_recall_mod._select_session_cache_owner(
        rows,
        caller_id,
        {caller_dataset_id, agent_dataset_id},
        {agent_dataset_id},
    )

    assert owner == agent_id
    assert (
        api_recall_mod._select_session_cache_owner(
            rows,
            caller_id,
            {caller_dataset_id, agent_dataset_id},
            {uuid4()},
        )
        is None
    )


def test_session_cache_owner_rejects_ambiguous_non_owner_rows(api_recall_mod):
    caller_id = uuid4()
    first_agent_id = uuid4()
    second_agent_id = uuid4()
    dataset_id = uuid4()
    rows = [
        _session_row(first_agent_id, dataset_id),
        _session_row(second_agent_id, dataset_id),
    ]

    owner = api_recall_mod._select_session_cache_owner(
        rows,
        caller_id,
        {dataset_id},
        {dataset_id},
    )

    assert owner is None


def test_session_cache_owner_preserves_single_agent_session_fallback(api_recall_mod):
    caller_id = uuid4()
    agent_id = uuid4()
    dataset_id = uuid4()
    rows = [
        _session_row(caller_id, None),
        _session_row(agent_id, dataset_id),
    ]

    owner = api_recall_mod._select_session_cache_owner(
        rows,
        caller_id,
        {dataset_id},
        None,
    )

    assert owner == agent_id


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
        captured["session_dataset_ids"] = kwargs.get("dataset_ids")
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
    assert captured["session_dataset_ids"] == [explicit_id]
    assert "resolved_from_datasets" not in captured


@pytest.mark.asyncio
async def test_recall_threads_dataset_scope_to_all_session_sources(monkeypatch, api_recall_mod):
    user = _make_user()
    explicit_id = uuid4()
    captured = {}

    async def dummy_search_session(**kwargs):
        captured["session"] = kwargs.get("dataset_ids")
        return []

    async def dummy_search_trace(**kwargs):
        captured["trace"] = kwargs.get("dataset_ids")
        return []

    async def dummy_fetch_session_context(**kwargs):
        captured["session_context"] = kwargs.get("dataset_ids")
        return []

    monkeypatch.setattr(api_recall_mod, "_search_session", dummy_search_session)
    monkeypatch.setattr(api_recall_mod, "_search_trace", dummy_search_trace)
    monkeypatch.setattr(api_recall_mod, "_fetch_session_context", dummy_fetch_session_context)

    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)

    out = await api_recall_mod.recall(
        query_text="q",
        dataset_ids=[explicit_id],
        session_id="session-1",
        scope=["session", "trace", "session_context"],
        user=user,
    )

    assert out == []
    assert captured == {
        "session": [explicit_id],
        "trace": [explicit_id],
        "session_context": [explicit_id],
    }


@pytest.mark.asyncio
async def test_recall_resolves_dataset_names_for_session_sources(monkeypatch, api_recall_mod):
    user = _make_user()
    resolved_id = uuid4()
    captured = {}

    async def dummy_get_authorized_existing_datasets(datasets, permission, resolved_user):
        captured["authorization"] = (datasets, permission, resolved_user)
        return [types.SimpleNamespace(id=resolved_id)]

    async def dummy_search_session(**kwargs):
        captured["session_dataset_ids"] = kwargs.get("dataset_ids")
        return []

    monkeypatch.setattr(
        api_recall_mod,
        "get_authorized_existing_datasets",
        dummy_get_authorized_existing_datasets,
    )
    monkeypatch.setattr(api_recall_mod, "_search_session", dummy_search_session)

    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)

    out = await api_recall_mod.recall(
        query_text="q",
        datasets=["shared-dataset"],
        session_id="session-1",
        scope=["session"],
        user=user,
    )

    assert out == []
    assert captured["authorization"] == (["shared-dataset"], "read", user)
    assert captured["session_dataset_ids"] == [resolved_id]


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

    async def dummy_fetch_session_context(
        query_text, session_id, context_profile, user=None, dataset_ids=None
    ):
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

    async def fake_resolve_cache_user(session_id, caller_user_id, dataset_ids=None):
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

    async def fake_resolve_cache_user(session_id, caller_user_id, dataset_ids=None):
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
