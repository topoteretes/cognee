"""Tests for recall's graph-lane warm-up short-circuit."""

import importlib
import types
from uuid import uuid4

import pytest
from pydantic import TypeAdapter

from cognee.modules.recall.config import RecallConfig
from cognee.modules.recall.types.RecallResponse import RecallResponse, ResponseMarkerEntry
from cognee.modules.search.types import SearchType


def _make_user(user_id=None, tenant_id=None):
    return types.SimpleNamespace(id=user_id or uuid4(), tenant_id=tenant_id)


@pytest.fixture
def api_recall_mod():
    return importlib.import_module("cognee.api.v1.recall.recall")


@pytest.fixture
def graph_warmup_mod():
    return importlib.import_module("cognee.modules.recall.methods.graph_warmup")


@pytest.fixture
def recall_config_mod():
    return importlib.import_module("cognee.modules.recall.config")


@pytest.fixture
def no_remote_client(monkeypatch):
    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)


@pytest.fixture
def stub_dataset_auth(monkeypatch):
    """Stub the pre-probe dataset authorization; returns the spy call list."""
    permission_methods = importlib.import_module("cognee.modules.users.permissions.methods")
    calls = []

    async def fake_get_specific_user_permission_datasets(user_id, permission_type, dataset_ids):
        calls.append((user_id, permission_type, dataset_ids))
        return [types.SimpleNamespace(id=dataset_id) for dataset_id in dataset_ids]

    monkeypatch.setattr(
        permission_methods,
        "get_specific_user_permission_datasets",
        fake_get_specific_user_permission_datasets,
    )
    return calls


@pytest.fixture
def stubbed_graph_lane(monkeypatch, api_recall_mod):
    """Stub everything downstream of the guard; returns the spy call list."""
    search_methods = importlib.import_module("cognee.modules.search.methods.search")
    search_operations = importlib.import_module("cognee.modules.search.operations")

    calls = []

    async def spy_authorized_search(**kwargs):
        calls.append(kwargs)
        return []

    async def dummy_log_search_history(*args, **kwargs):
        return None

    async def dummy_set_session_user_context_variable(_user):
        return None

    monkeypatch.setattr(search_methods, "authorized_search", spy_authorized_search)
    monkeypatch.setattr(search_operations, "log_search_history", dummy_log_search_history)
    monkeypatch.setattr(
        api_recall_mod,
        "set_session_user_context_variable",
        dummy_set_session_user_context_variable,
    )
    return calls


def _use_config(monkeypatch, recall_config_mod, **overrides):
    config = RecallConfig(_env_file=None, **overrides)
    monkeypatch.setattr(recall_config_mod, "get_recall_config", lambda: config)


def _stub_count(monkeypatch, graph_warmup_mod, value):
    """Stub the build-status probe with count semantics: 0 = cold, >0 = warm."""
    calls = []

    async def counting_stub(user, dataset_ids):
        calls.append((user, dataset_ids))
        if isinstance(value, Exception):
            raise value
        if value and value > 0:
            return graph_warmup_mod.WarmupProbe(graph_warmup_mod.STATE_WARM, value)
        return graph_warmup_mod.WarmupProbe(graph_warmup_mod.STATE_NEVER_BUILT, 0)

    monkeypatch.setattr(graph_warmup_mod, "get_graph_build_status", counting_stub)
    return calls


def _stub_probe(monkeypatch, graph_warmup_mod, probe):
    """Stub the build-status probe with an explicit WarmupProbe."""
    calls = []

    async def probe_stub(user, dataset_ids):
        calls.append((user, dataset_ids))
        return probe

    monkeypatch.setattr(graph_warmup_mod, "get_graph_build_status", probe_stub)
    return calls


def test_shortcircuit_is_enabled_by_default():
    # The guard must be on for fresh installs with no env vars set; every
    # other test injects an explicit config, so only this pins the default.
    config = RecallConfig(_env_file=None)
    assert config.recall_warmup_shortcircuit is True
    assert config.recall_warmup_threshold == 1
    assert config.recall_warmup_cache_ttl == 60.0


class TestResponseUnion:
    def test_marker_entry_round_trips_through_union(self):
        adapter = TypeAdapter(list[RecallResponse])
        entries = adapter.validate_python(
            [
                {
                    "source": "system",
                    "status": "memory_warming_up",
                    "text": "memory still warming up, no graph data yet",
                    "datapoint_count": 0,
                    "threshold": 1,
                }
            ]
        )
        assert isinstance(entries[0], ResponseMarkerEntry)
        assert entries[0].status == "memory_warming_up"
        assert entries[0].datapoint_count == 0


@pytest.mark.asyncio
async def test_empty_graph_returns_marker_and_skips_search(
    monkeypatch,
    api_recall_mod,
    graph_warmup_mod,
    recall_config_mod,
    no_remote_client,
    stub_dataset_auth,
):
    _use_config(monkeypatch, recall_config_mod)
    _stub_count(monkeypatch, graph_warmup_mod, 0)
    search_methods = importlib.import_module("cognee.modules.search.methods.search")
    search_operations = importlib.import_module("cognee.modules.search.operations")

    authorized_calls = []
    history_calls = []

    async def spy_authorized_search(**kwargs):
        authorized_calls.append(kwargs)
        return []

    async def spy_log_search_history(*args, **kwargs):
        history_calls.append(args)

    async def dummy_set_session_user_context_variable(_user):
        return None

    monkeypatch.setattr(search_methods, "authorized_search", spy_authorized_search)
    monkeypatch.setattr(search_operations, "log_search_history", spy_log_search_history)
    monkeypatch.setattr(
        api_recall_mod,
        "set_session_user_context_variable",
        dummy_set_session_user_context_variable,
    )

    out = await api_recall_mod.recall(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4()],
        auto_route=False,
        user=_make_user(),
    )

    assert len(out) == 1
    marker = out[0]
    assert isinstance(marker, ResponseMarkerEntry)
    assert marker.source == "system"
    assert marker.status == "memory_warming_up"
    assert marker.datapoint_count == 0
    assert marker.threshold == 1
    assert authorized_calls == []
    assert history_calls == []
    # The guard authorized the caller-supplied ids before probing.
    assert len(stub_dataset_auth) == 1


@pytest.mark.asyncio
async def test_warm_graph_runs_normal_search(
    monkeypatch,
    api_recall_mod,
    graph_warmup_mod,
    recall_config_mod,
    no_remote_client,
    stubbed_graph_lane,
    stub_dataset_auth,
):
    _use_config(monkeypatch, recall_config_mod)
    _stub_count(monkeypatch, graph_warmup_mod, 5)

    out = await api_recall_mod.recall(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4()],
        auto_route=False,
        user=_make_user(),
    )

    assert out == []
    assert len(stubbed_graph_lane) == 1


@pytest.mark.asyncio
async def test_probe_failure_fails_open(
    monkeypatch,
    api_recall_mod,
    graph_warmup_mod,
    recall_config_mod,
    no_remote_client,
    stubbed_graph_lane,
    stub_dataset_auth,
):
    _use_config(monkeypatch, recall_config_mod)
    _stub_count(monkeypatch, graph_warmup_mod, RuntimeError("probe broke"))

    out = await api_recall_mod.recall(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4()],
        auto_route=False,
        user=_make_user(),
    )

    assert out == []
    assert len(stubbed_graph_lane) == 1


@pytest.mark.asyncio
async def test_kill_switch_off_runs_search_even_when_cold(
    monkeypatch,
    api_recall_mod,
    graph_warmup_mod,
    recall_config_mod,
    no_remote_client,
    stubbed_graph_lane,
):
    _use_config(monkeypatch, recall_config_mod, recall_warmup_shortcircuit=False)
    probe_calls = _stub_count(monkeypatch, graph_warmup_mod, 0)

    out = await api_recall_mod.recall(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4()],
        auto_route=False,
        user=_make_user(),
    )

    assert out == []
    assert len(stubbed_graph_lane) == 1
    assert probe_calls == []


@pytest.mark.asyncio
async def test_cold_verdicts_are_never_cached(monkeypatch, graph_warmup_mod, recall_config_mod):
    # A cold verdict must not stick: a graph populated right after a cold
    # probe has to be searchable on the very next recall.
    _use_config(monkeypatch, recall_config_mod)
    probes = iter(
        [
            graph_warmup_mod.WarmupProbe(graph_warmup_mod.STATE_NEVER_BUILT, 0),
            graph_warmup_mod.WarmupProbe(graph_warmup_mod.STATE_WARM, 5),
        ]
    )
    probe_calls = []

    async def counting_stub(user, dataset_ids):
        probe_calls.append((user, dataset_ids))
        return next(probes)

    monkeypatch.setattr(graph_warmup_mod, "get_graph_build_status", counting_stub)
    graph_warmup_mod.clear_warmup_cache()
    user = _make_user()
    dataset_ids = [uuid4()]

    assert await graph_warmup_mod.is_memory_warm(user, dataset_ids) == (False, 0)
    # Second call re-probes (no cached cold verdict) and sees the new data.
    assert await graph_warmup_mod.is_memory_warm(user, dataset_ids) == (True, 5)
    assert len(probe_calls) == 2


@pytest.mark.asyncio
async def test_ttl_cache_caches_warm_verdicts_too(monkeypatch, graph_warmup_mod, recall_config_mod):
    _use_config(monkeypatch, recall_config_mod)
    probe_calls = _stub_count(monkeypatch, graph_warmup_mod, 7)
    graph_warmup_mod.clear_warmup_cache()
    user = _make_user()

    assert await graph_warmup_mod.is_memory_warm(user, None) == (True, 7)
    assert await graph_warmup_mod.is_memory_warm(user, None) == (True, 7)
    assert len(probe_calls) == 1


@pytest.mark.asyncio
async def test_expired_ttl_re_queries(monkeypatch, graph_warmup_mod, recall_config_mod):
    # TTL of 0 makes every cached (warm) verdict already expired.
    _use_config(monkeypatch, recall_config_mod, recall_warmup_cache_ttl=0.0)
    probe_calls = _stub_count(monkeypatch, graph_warmup_mod, 7)
    graph_warmup_mod.clear_warmup_cache()
    user = _make_user()

    await graph_warmup_mod.is_memory_warm(user, None)
    await graph_warmup_mod.is_memory_warm(user, None)
    assert len(probe_calls) == 2


@pytest.mark.asyncio
async def test_empty_dataset_list_does_not_share_all_datasets_cache_key(
    monkeypatch, graph_warmup_mod, recall_config_mod
):
    # dataset_ids=[] and dataset_ids=None must map to distinct cache keys.
    _use_config(monkeypatch, recall_config_mod)
    probe_calls = _stub_count(monkeypatch, graph_warmup_mod, 7)
    graph_warmup_mod.clear_warmup_cache()
    user = _make_user()

    assert await graph_warmup_mod.is_memory_warm(user, None) == (True, 7)
    # [] must not hit the cached "__all__" verdict.
    assert await graph_warmup_mod.is_memory_warm(user, []) == (True, 7)
    assert len(probe_calls) == 2


@pytest.mark.asyncio
async def test_threshold_above_probe_cap_is_clamped(
    monkeypatch, graph_warmup_mod, recall_config_mod
):
    # The probe is binary and reports at most _WARM_COUNT; a larger
    # threshold must not read every populated graph as cold.
    _use_config(monkeypatch, recall_config_mod, recall_warmup_threshold=2**40)
    _stub_count(monkeypatch, graph_warmup_mod, graph_warmup_mod._WARM_COUNT)
    graph_warmup_mod.clear_warmup_cache()

    warm, count = await graph_warmup_mod.is_memory_warm(_make_user(), None)
    assert warm is True
    assert count == graph_warmup_mod._WARM_COUNT


@pytest.mark.asyncio
async def test_config_failure_fails_open_and_skips_guard(
    monkeypatch,
    api_recall_mod,
    graph_warmup_mod,
    recall_config_mod,
    no_remote_client,
    stubbed_graph_lane,
):
    def broken_config():
        raise ValueError("malformed RECALL_WARMUP_* env value")

    monkeypatch.setattr(recall_config_mod, "get_recall_config", broken_config)
    probe_calls = _stub_count(monkeypatch, graph_warmup_mod, 0)

    out = await api_recall_mod.recall(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4()],
        auto_route=False,
        user=_make_user(),
    )

    assert out == []
    assert len(stubbed_graph_lane) == 1
    assert probe_calls == []


@pytest.mark.asyncio
async def test_unauthorized_dataset_ids_raise_before_probe(
    monkeypatch,
    api_recall_mod,
    graph_warmup_mod,
    recall_config_mod,
    no_remote_client,
    stubbed_graph_lane,
):
    from cognee.modules.users.exceptions import PermissionDeniedError

    _use_config(monkeypatch, recall_config_mod)
    probe_calls = _stub_count(monkeypatch, graph_warmup_mod, 0)
    permission_methods = importlib.import_module("cognee.modules.users.permissions.methods")

    async def deny(user_id, permission_type, dataset_ids):
        raise PermissionDeniedError("no read permission")

    monkeypatch.setattr(permission_methods, "get_specific_user_permission_datasets", deny)

    with pytest.raises(PermissionDeniedError):
        await api_recall_mod.recall(
            query_text="q",
            query_type=SearchType.GRAPH_COMPLETION,
            dataset_ids=[uuid4()],
            auto_route=False,
            user=_make_user(),
        )

    assert probe_calls == []
    assert stubbed_graph_lane == []


@pytest.mark.asyncio
async def test_only_context_skips_guard(
    monkeypatch,
    api_recall_mod,
    graph_warmup_mod,
    recall_config_mod,
    no_remote_client,
    stubbed_graph_lane,
    stub_dataset_auth,
):
    _use_config(monkeypatch, recall_config_mod)
    probe_calls = _stub_count(monkeypatch, graph_warmup_mod, 0)

    out = await api_recall_mod.recall(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4()],
        auto_route=False,
        only_context=True,
        user=_make_user(),
    )

    assert out == []
    assert len(stubbed_graph_lane) == 1
    assert probe_calls == []


@pytest.mark.asyncio
async def test_multi_source_cold_graph_returns_empty_not_marker(
    monkeypatch,
    api_recall_mod,
    graph_warmup_mod,
    recall_config_mod,
    no_remote_client,
    stubbed_graph_lane,
    stub_dataset_auth,
):
    _use_config(monkeypatch, recall_config_mod)
    _stub_count(monkeypatch, graph_warmup_mod, 0)

    out = await api_recall_mod.recall(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4()],
        auto_route=False,
        scope=["session", "graph"],
        user=_make_user(),
    )

    # A cold graph in a multi-source recall contributes nothing instead of
    # injecting a marker next to other sources' results (and instead of
    # suppressing the tools "on_empty" fallback).
    assert out == []
    assert stubbed_graph_lane == []


@pytest.mark.asyncio
async def test_probe_failure_verdict_is_not_cached(
    monkeypatch, graph_warmup_mod, recall_config_mod
):
    _use_config(monkeypatch, recall_config_mod)
    probe_calls = _stub_count(monkeypatch, graph_warmup_mod, RuntimeError("boom"))
    graph_warmup_mod.clear_warmup_cache()
    user = _make_user()

    warm, count = await graph_warmup_mod.is_memory_warm(user, [uuid4()])
    assert warm is True
    assert count >= 1
    assert len(probe_calls) == 1


class TestGlobalConfigWiring:
    """cognee.config exposes the warm-up settings as runtime setters that
    mutate the same cached RecallConfig instance the guard reads."""

    def test_set_recall_config_updates_cached_instance(self):
        from cognee.api.v1.config.config import config as global_config
        from cognee.modules.recall.config import get_recall_config

        cfg = get_recall_config()
        original = cfg.to_dict()
        try:
            # String payloads (the CLI contract) must coerce to real types.
            global_config.set_recall_config(
                {
                    "recall_warmup_shortcircuit": "false",
                    "recall_warmup_threshold": "5",
                    "recall_warmup_cache_ttl": "1.5",
                }
            )
            assert cfg.recall_warmup_shortcircuit is False
            assert cfg.recall_warmup_threshold == 5
            assert cfg.recall_warmup_cache_ttl == 1.5
            # The guard reads the same lru_cached instance the setter mutated.
            assert get_recall_config() is cfg
        finally:
            global_config.set_recall_config(original)

    def test_generic_set_and_get_cover_warmup_keys(self):
        from cognee.api.v1.config.config import config as global_config
        from cognee.modules.recall.config import get_recall_config

        original = get_recall_config().to_dict()
        try:
            global_config.set("recall_warmup_shortcircuit", "true")
            global_config.set("recall_warmup_threshold", 3)
            global_config.set("recall_warmup_cache_ttl", "30")
            assert global_config.get("recall_warmup_shortcircuit") is True
            assert global_config.get("recall_warmup_threshold") == 3
            assert global_config.get("recall_warmup_cache_ttl") == 30.0
            assert global_config.get_all()["recall_warmup_shortcircuit"] is True
        finally:
            global_config.set_recall_config(original)

    def test_invalid_values_are_rejected(self):
        from cognee.api.v1.config.config import config as global_config
        from cognee.modules.recall.config import get_recall_config

        original = get_recall_config().to_dict()
        try:
            with pytest.raises(ValueError):
                global_config.set_recall_warmup_shortcircuit("not-a-bool")
            with pytest.raises(ValueError):
                global_config.set_recall_warmup_threshold(0)
            with pytest.raises(ValueError):
                global_config.set_recall_warmup_cache_ttl(-1)
            assert get_recall_config().to_dict() == original
        finally:
            global_config.set_recall_config(original)


@pytest.mark.asyncio
async def test_build_failed_returns_diagnostic_marker(
    monkeypatch,
    api_recall_mod,
    graph_warmup_mod,
    recall_config_mod,
    no_remote_client,
    stub_dataset_auth,
    stubbed_graph_lane,
):
    """A failed build yields a build_failed marker carrying the classified cause.

    Previously an ERRORED run read as 'warm' and recall returned empty results
    with no explanation — the exact dead end that turns a failed first cognify
    into churn.
    """
    _use_config(monkeypatch, recall_config_mod)
    _stub_probe(
        monkeypatch,
        graph_warmup_mod,
        graph_warmup_mod.WarmupProbe(
            graph_warmup_mod.STATE_BUILD_FAILED,
            0,
            error_class="AuthenticationError",
            error_message="invalid api key",
        ),
    )
    graph_warmup_mod.clear_warmup_cache()

    out = await api_recall_mod.recall(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        dataset_ids=[uuid4()],
        auto_route=False,
        user=_make_user(),
    )

    assert len(out) == 1
    marker = out[0]
    assert marker.status == "build_failed"
    assert marker.error_class == "AuthenticationError"
    assert "AuthenticationError: invalid api key" in marker.text
    assert stubbed_graph_lane == []


@pytest.mark.asyncio
async def test_cold_states_are_not_cached_but_warm_is(
    monkeypatch, graph_warmup_mod, recall_config_mod
):
    _use_config(monkeypatch, recall_config_mod)
    probes = iter(
        [
            graph_warmup_mod.WarmupProbe(graph_warmup_mod.STATE_BUILD_FAILED, 0, error_class="X"),
            graph_warmup_mod.WarmupProbe(graph_warmup_mod.STATE_WARM, 9),
            # Would be returned on a third probe call — must never be reached,
            # because the warm verdict above is cached.
            graph_warmup_mod.WarmupProbe(graph_warmup_mod.STATE_NEVER_BUILT, 0),
        ]
    )
    calls = []

    async def probe_stub(user, dataset_ids):
        calls.append(1)
        return next(probes)

    monkeypatch.setattr(graph_warmup_mod, "get_graph_build_status", probe_stub)
    graph_warmup_mod.clear_warmup_cache()
    user = _make_user()

    first = await graph_warmup_mod.assess_memory_readiness(user, None)
    assert first.state == graph_warmup_mod.STATE_BUILD_FAILED
    second = await graph_warmup_mod.assess_memory_readiness(user, None)
    assert second.state == graph_warmup_mod.STATE_WARM
    third = await graph_warmup_mod.assess_memory_readiness(user, None)
    assert third.state == graph_warmup_mod.STATE_WARM
    assert len(calls) == 2
