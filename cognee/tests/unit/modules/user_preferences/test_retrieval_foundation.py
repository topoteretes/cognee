"""Unit tests for Phase 3 of user preferences: the retrieval foundation.

Covers the deterministic pieces only — no graph database, no LLM, no cache:
- ``personal_factor``: exact no-op at neutral weight and at zero influence in
  both spaces, and opposite movement across the two spaces.
- ``load_active_preferences``: flag-off short-circuit without touching the
  graph, fail-open on errors, render-header handling, decay applied on read
  (callers never see the stored weight), and ContextVar memoization.
- ``compose_session_prompt``: layer ordering with preference text, and
  byte-identical output when the preference text is empty.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

import cognee.modules.user_preferences.lookup as lookup_module
from cognee.context_global_variables import current_dataset_id, session_user
from cognee.infrastructure.session.session_turn import compose_session_prompt
from cognee.modules.user_preferences.constants import PREFERENCE_RENDER_HEADER
from cognee.modules.user_preferences.lookup import load_active_preferences
from cognee.modules.user_preferences.weights import effective_weight, personal_factor

BETA = 0.02


class TestPersonalFactor:
    @pytest.mark.parametrize("distance_space", [True, False])
    def test_exactly_one_at_neutral_weight(self, distance_space):
        assert personal_factor(0.5, 0.3, distance_space=distance_space) == 1.0

    @pytest.mark.parametrize("distance_space", [True, False])
    @pytest.mark.parametrize("weight", [0.0, 0.05, 0.5, 0.95, 1.0])
    def test_exactly_one_at_zero_influence(self, distance_space, weight):
        assert personal_factor(weight, 0.0, distance_space=distance_space) == 1.0

    def test_opposite_directions_across_spaces(self):
        # A preferred item: smaller distance, larger score.
        assert personal_factor(0.9, 0.3, distance_space=True) < 1.0
        assert personal_factor(0.9, 0.3, distance_space=False) > 1.0
        # A disliked item: larger distance, smaller score.
        assert personal_factor(0.1, 0.3, distance_space=True) > 1.0
        assert personal_factor(0.1, 0.3, distance_space=False) < 1.0

    def test_influence_bounds_the_move(self):
        # influence reads as "the most personalization may move a score":
        # 0.3 means at most 30%, reached only at the weight extremes.
        assert personal_factor(1.0, 0.3, distance_space=True) == pytest.approx(0.7)
        assert personal_factor(0.0, 0.3, distance_space=True) == pytest.approx(1.3)
        assert personal_factor(1.0, 0.3, distance_space=False) == pytest.approx(1.3)
        assert personal_factor(0.0, 0.3, distance_space=False) == pytest.approx(0.7)


@pytest.fixture
def clear_lookup_cache():
    lookup_module._active_preferences_cache.set(None)
    yield
    lookup_module._active_preferences_cache.set(None)


@pytest.fixture
def identity_in_context():
    user_token = session_user.set(SimpleNamespace(id=uuid4()))
    dataset_token = current_dataset_id.set(uuid4())
    yield
    session_user.reset(user_token)
    current_dataset_id.reset(dataset_token)


def _patch_config(monkeypatch, *, enabled):
    config = SimpleNamespace(
        personalization_enabled=enabled,
        personalization_influence=0.3,
        preference_beta=BETA,
    )
    monkeypatch.setattr(lookup_module, "get_base_config", lambda: config)
    return config


def _patch_state(monkeypatch, node, stored):
    calls = []

    async def fake_load_preference_state(user_id, dataset_id):
        calls.append((user_id, dataset_id))
        return node, stored

    monkeypatch.setattr(lookup_module, "load_preference_state", fake_load_preference_state)
    return calls


@pytest.mark.asyncio
@pytest.mark.usefixtures("clear_lookup_cache")
class TestLoadActivePreferences:
    async def test_flag_off_returns_empty_without_touching_the_graph(
        self, monkeypatch, identity_in_context
    ):
        _patch_config(monkeypatch, enabled=False)

        async def must_not_be_called(user_id, dataset_id):
            raise AssertionError("graph read must not happen with the flag off")

        monkeypatch.setattr(lookup_module, "load_preference_state", must_not_be_called)

        assert await load_active_preferences() == ("", {})

    async def test_missing_identity_returns_empty(self, monkeypatch):
        _patch_config(monkeypatch, enabled=True)
        calls = _patch_state(monkeypatch, {"text": "x", "turn_counter": 1}, {})
        user_token = session_user.set(None)
        dataset_token = current_dataset_id.set(None)
        try:
            assert await load_active_preferences() == ("", {})
            assert calls == []
        finally:
            session_user.reset(user_token)
            current_dataset_id.reset(dataset_token)

    async def test_no_node_returns_empty(self, monkeypatch, identity_in_context):
        _patch_config(monkeypatch, enabled=True)
        _patch_state(monkeypatch, None, {})
        assert await load_active_preferences() == ("", {})

    async def test_fails_open_when_the_read_raises(self, monkeypatch, identity_in_context):
        _patch_config(monkeypatch, enabled=True)

        async def broken(user_id, dataset_id):
            raise RuntimeError("graph is down")

        monkeypatch.setattr(lookup_module, "load_preference_state", broken)
        assert await load_active_preferences() == ("", {})

    async def test_header_prepended_only_when_text_is_non_empty(
        self, monkeypatch, identity_in_context
    ):
        _patch_config(monkeypatch, enabled=True)
        _patch_state(
            monkeypatch,
            {"text": "prefers concise answers", "turn_counter": 0},
            {},
        )
        text, weights = await load_active_preferences()
        assert text == PREFERENCE_RENDER_HEADER + "\nprefers concise answers"
        assert weights == {}

    async def test_empty_text_returns_empty_string_with_no_stray_header(
        self, monkeypatch, identity_in_context
    ):
        _patch_config(monkeypatch, enabled=True)
        _patch_state(monkeypatch, {"text": "", "turn_counter": 3}, {"n1": {"weight": 0.9}})
        text, _weights = await load_active_preferences()
        assert text == ""

    async def test_weights_are_decayed_never_stored(self, monkeypatch, identity_in_context):
        # Demo 7: a stored weight of 0.9 with updated_at_turn well behind the
        # node's counter reads back decayed — callers have no way to see 0.9.
        _patch_config(monkeypatch, enabled=True)
        _patch_state(
            monkeypatch,
            {"text": "", "turn_counter": 40},
            {"node-1": {"weight": 0.9, "updated_at_turn": 6}},
        )
        _text, weights = await load_active_preferences()
        expected = effective_weight(0.9, 6, 40, BETA)
        assert weights == {"node-1": expected}
        assert weights["node-1"] != 0.9
        assert 0.5 < weights["node-1"] < 0.9

    async def test_untouched_edge_is_not_decayed(self, monkeypatch, identity_in_context):
        _patch_config(monkeypatch, enabled=True)
        _patch_state(
            monkeypatch,
            {"text": "", "turn_counter": 7},
            {"node-1": {"weight": 0.8, "updated_at_turn": 7}},
        )
        _text, weights = await load_active_preferences()
        assert weights == {"node-1": 0.8}

    async def test_memoized_per_user_and_dataset(self, monkeypatch, identity_in_context):
        _patch_config(monkeypatch, enabled=True)
        calls = _patch_state(
            monkeypatch,
            {"text": "prefers X", "turn_counter": 1},
            {"node-1": {"weight": 0.7, "updated_at_turn": 1}},
        )
        first = await load_active_preferences()
        second = await load_active_preferences()
        assert first == second
        assert len(calls) == 1

    async def test_cache_misses_on_a_different_dataset(self, monkeypatch, identity_in_context):
        _patch_config(monkeypatch, enabled=True)
        calls = _patch_state(monkeypatch, {"text": "", "turn_counter": 0}, {})
        await load_active_preferences()
        dataset_token = current_dataset_id.set(uuid4())
        try:
            await load_active_preferences()
        finally:
            current_dataset_id.reset(dataset_token)
        assert len(calls) == 2


class TestComposeSessionPromptPreferenceLayer:
    def test_preference_text_layers_ahead_of_active_context(self):
        result = compose_session_prompt("BLOCK", "HISTORY", "PREFS")
        assert result == "PREFS\n\nBLOCK\n\nHISTORY"

    def test_preference_text_without_active_context(self):
        assert compose_session_prompt("", "HISTORY", "PREFS") == "PREFS\n\nHISTORY"

    def test_empty_preference_text_is_byte_identical_to_two_arg_call(self):
        for block, history in [("BLOCK", "HISTORY"), ("", "HISTORY"), ("BLOCK", ""), ("", "")]:
            assert compose_session_prompt(block, history, "") == compose_session_prompt(
                block, history
            )

    def test_default_is_empty(self):
        assert compose_session_prompt("BLOCK", "HISTORY") == "BLOCK\n\nHISTORY"
