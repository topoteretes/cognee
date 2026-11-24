"""recall() bills the whole call to an explicit session_id (SDK-1092).

Retrieval-stage LLM and embedding calls run before the session turn opens its
own usage scope; binding in the SDK function covers them on every surface.
"""

import importlib
import types
from uuid import uuid4

import pytest

from cognee.modules.recall.config import RecallConfig
from cognee.modules.search.types import SearchType
from cognee.modules.session_lifecycle import usage_tracking

pytestmark = pytest.mark.asyncio


@pytest.fixture
def api_recall_mod():
    return importlib.import_module("cognee.api.v1.recall.recall")


@pytest.fixture
def graph_lane_targets(monkeypatch, api_recall_mod):
    """Stub the graph lane; returns the usage target seen inside each search."""
    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    search_methods = importlib.import_module("cognee.modules.search.methods.search")
    search_operations = importlib.import_module("cognee.modules.search.operations")
    recall_config_mod = importlib.import_module("cognee.modules.recall.config")

    seen = []

    async def spy_authorized_search(**_kwargs):
        seen.append(usage_tracking._active_session.get())
        return []

    async def noop(*_args, **_kwargs):
        return None

    config = RecallConfig(_env_file=None, recall_warmup_shortcircuit=False)
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr(search_methods, "authorized_search", spy_authorized_search)
    monkeypatch.setattr(search_operations, "log_search_history", noop)
    monkeypatch.setattr(api_recall_mod, "set_session_user_context_variable", noop)
    monkeypatch.setattr(recall_config_mod, "get_recall_config", lambda: config)
    return seen


async def _recall(api_recall_mod, user, **kwargs):
    return await api_recall_mod.recall(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4()],
        scope="graph",
        auto_route=False,
        user=user,
        **kwargs,
    )


async def test_explicit_session_binds_the_graph_lane(api_recall_mod, graph_lane_targets):
    user = types.SimpleNamespace(id=uuid4(), tenant_id=None)

    await _recall(api_recall_mod, user, session_id="trial-1")

    assert graph_lane_targets == [("trial-1", user.id)]
    assert usage_tracking._active_session.get() is None


async def test_no_session_id_binds_nothing(api_recall_mod, graph_lane_targets):
    user = types.SimpleNamespace(id=uuid4(), tenant_id=None)

    await _recall(api_recall_mod, user)

    assert graph_lane_targets == [None]


async def test_default_user_is_resolved_for_the_binding(
    monkeypatch, api_recall_mod, graph_lane_targets
):
    default_user = types.SimpleNamespace(id=uuid4(), tenant_id=None)

    async def fake_default_user():
        return default_user

    monkeypatch.setattr(api_recall_mod, "get_default_user", fake_default_user)

    await _recall(api_recall_mod, None, session_id="trial-1")

    assert graph_lane_targets == [("trial-1", default_user.id)]
