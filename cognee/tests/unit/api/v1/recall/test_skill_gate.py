import importlib
import types
from uuid import uuid4

import pytest

from cognee.api.v1.recall.skill_gate import (
    should_search_skills,
    skill_gate_enabled,
)
from cognee.modules.search.types import SearchType


# ── gate classification: pure regex, no LLM ──────────────────────────────────


@pytest.mark.parametrize(
    "query",
    [
        "how do I deploy to staging",
        "how to rotate the API keys",
        "steps to onboard a new tenant",
        "what is the process for a release",
        "walk me through the database migration",
        "is there a runbook for incident response",
        "setup guide for the staging cluster",
        "which skills are available",
    ],
)
def test_gate_fires_on_procedural_queries(query):
    assert should_search_skills(query).fired


@pytest.mark.parametrize(
    "query",
    [
        "what is our churn rate",
        "who owns the billing service",
        "deploy notes from yesterday",
        "summary of last week's incidents",
        "",
    ],
)
def test_gate_stays_closed_on_non_procedural_queries(query):
    assert not should_search_skills(query).fired


def test_gate_negation_suppresses_match():
    result = should_search_skills("do not walk me through it")
    assert not result.fired


def test_gate_result_carries_score_and_matches():
    result = should_search_skills("how do I deploy to staging")
    assert result.score >= 3.0
    assert any("how do" in fragment.lower() for fragment in result.matched)


def test_gate_enabled_flag(monkeypatch):
    monkeypatch.delenv("SKILL_GATE_ENABLED", raising=False)
    assert skill_gate_enabled() is True
    monkeypatch.setenv("SKILL_GATE_ENABLED", "false")
    assert skill_gate_enabled() is False
    monkeypatch.setenv("SKILL_GATE_ENABLED", "0")
    assert skill_gate_enabled() is False
    monkeypatch.setenv("SKILL_GATE_ENABLED", "true")
    assert skill_gate_enabled() is True


# ── recall() wiring: the gate appends source="skills" entries ─────────────────


def _make_user():
    return types.SimpleNamespace(id=uuid4(), tenant_id=None)


def _skill_item(**overrides):
    item = {
        "id": str(uuid4()),
        "name": "deploy-checklist",
        "description": "Steps to deploy to staging",
        "score": 0.3,
    }
    item.update(overrides)
    return item


@pytest.fixture
def api_recall_mod():
    return importlib.import_module("cognee.api.v1.recall.recall")


def _patch_recall_plumbing(monkeypatch, api_recall_mod, search_calls, skill_items):
    """Patch recall's external dependencies; record authorized_search calls."""

    async def dummy_set_session_user_context_variable(_user):
        return None

    async def dummy_authorized_search(**kwargs):
        from cognee.modules.search.models.SearchResultPayload import SearchResultPayload

        search_calls.append(kwargs)
        if kwargs.get("query_type") is SearchType.SKILLS:
            # A real payload: the graph lane normalizes it when SKILLS is the
            # explicit query_type; the gate lane only reads .completion.
            return [
                SearchResultPayload(
                    result_object=None,
                    context=None,
                    completion=list(skill_items),
                    search_type=SearchType.SKILLS,
                    only_context=False,
                )
            ]
        return []

    def dummy_get_remote_client():
        return None

    async def dummy_log_search_history(*args, **kwargs):
        return None

    monkeypatch.setattr(
        api_recall_mod,
        "set_session_user_context_variable",
        dummy_set_session_user_context_variable,
    )
    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    search_methods = importlib.import_module("cognee.modules.search.methods.search")
    search_operations = importlib.import_module("cognee.modules.search.operations")
    monkeypatch.setattr(serve_state, "get_remote_client", dummy_get_remote_client)
    monkeypatch.setattr(search_methods, "authorized_search", dummy_authorized_search)
    monkeypatch.setattr(search_operations, "log_search_history", dummy_log_search_history)


@pytest.mark.asyncio
async def test_gate_appends_skill_entries_for_procedural_query(monkeypatch, api_recall_mod):
    user = _make_user()
    dataset_id = uuid4()
    search_calls = []
    _patch_recall_plumbing(monkeypatch, api_recall_mod, search_calls, [_skill_item()])

    out = await api_recall_mod.recall(
        query_text="how do I deploy to staging",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[dataset_id],
        auto_route=False,
        user=user,
    )

    called_types = [call.get("query_type") for call in search_calls]
    assert SearchType.SKILLS in called_types
    skills_call = next(call for call in search_calls if call.get("query_type") is SearchType.SKILLS)
    assert skills_call["dataset_ids"] == [dataset_id]

    skill_entries = [entry for entry in out if getattr(entry, "source", None) == "skills"]
    assert len(skill_entries) == 1
    entry = skill_entries[0]
    assert entry.text == "deploy-checklist: Steps to deploy to staging"
    assert entry.score == 0.3
    assert entry.skill["name"] == "deploy-checklist"
    assert "score" not in entry.skill


@pytest.mark.asyncio
async def test_gate_skipped_for_non_procedural_query(monkeypatch, api_recall_mod):
    user = _make_user()
    search_calls = []
    _patch_recall_plumbing(monkeypatch, api_recall_mod, search_calls, [_skill_item()])

    out = await api_recall_mod.recall(
        query_text="who owns the billing service",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4()],
        auto_route=False,
        user=user,
    )

    assert SearchType.SKILLS not in [call.get("query_type") for call in search_calls]
    assert out == []


@pytest.mark.asyncio
async def test_gate_skipped_when_disabled(monkeypatch, api_recall_mod):
    monkeypatch.setenv("SKILL_GATE_ENABLED", "false")
    user = _make_user()
    search_calls = []
    _patch_recall_plumbing(monkeypatch, api_recall_mod, search_calls, [_skill_item()])

    await api_recall_mod.recall(
        query_text="how do I deploy to staging",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4()],
        auto_route=False,
        user=user,
    )

    assert SearchType.SKILLS not in [call.get("query_type") for call in search_calls]


@pytest.mark.asyncio
async def test_gate_skipped_without_exactly_one_dataset(monkeypatch, api_recall_mod):
    """The skill invariant: lookup requires exactly one dataset — else skip silently."""
    user = _make_user()
    search_calls = []
    _patch_recall_plumbing(monkeypatch, api_recall_mod, search_calls, [_skill_item()])

    await api_recall_mod.recall(
        query_text="how do I deploy to staging",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4(), uuid4()],
        auto_route=False,
        user=user,
    )

    assert SearchType.SKILLS not in [call.get("query_type") for call in search_calls]


@pytest.mark.asyncio
async def test_gate_bypassed_for_explicit_skills_query_type(monkeypatch, api_recall_mod):
    user = _make_user()
    search_calls = []
    _patch_recall_plumbing(monkeypatch, api_recall_mod, search_calls, [_skill_item()])

    await api_recall_mod.recall(
        query_text="how do I deploy to staging",
        query_type=SearchType.SKILLS,
        dataset_ids=[uuid4()],
        auto_route=False,
        user=user,
    )

    # Exactly one SKILLS call — the graph lane's own — not a second gate call.
    assert [call.get("query_type") for call in search_calls].count(SearchType.SKILLS) == 1


@pytest.mark.asyncio
async def test_gate_failure_never_fails_recall(monkeypatch, api_recall_mod):
    """A gate-lane exception is swallowed; the main lanes still answer."""
    user = _make_user()
    search_calls = []

    async def dummy_set_session_user_context_variable(_user):
        return None

    async def dummy_authorized_search(**kwargs):
        search_calls.append(kwargs)
        if kwargs.get("query_type") is SearchType.SKILLS:
            raise RuntimeError("skill lane exploded")
        return []

    async def dummy_log_search_history(*args, **kwargs):
        return None

    monkeypatch.setattr(
        api_recall_mod,
        "set_session_user_context_variable",
        dummy_set_session_user_context_variable,
    )
    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    search_methods = importlib.import_module("cognee.modules.search.methods.search")
    search_operations = importlib.import_module("cognee.modules.search.operations")
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr(search_methods, "authorized_search", dummy_authorized_search)
    monkeypatch.setattr(search_operations, "log_search_history", dummy_log_search_history)

    out = await api_recall_mod.recall(
        query_text="how do I deploy to staging",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4()],
        auto_route=False,
        user=user,
    )

    assert out == []
    assert SearchType.SKILLS in [call.get("query_type") for call in search_calls]
