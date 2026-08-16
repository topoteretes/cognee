"""Unit tests for Phase 2 of user preferences: rating plumbing and the update.

Covers the deterministic pieces only — no graph database, no LLM, no cache:
- The 1-5 rating validators on ``SessionTurnAnalysis`` and
  ``SessionFeedbackEntry`` (out-of-range coerces to None, never raises).
- ``effective_weight`` against the plan's decay table (idle 34 halves the
  distance from neutral; idle 173 falls under the delete threshold).
- ``_run_preference_update`` with a fake session manager and mocked store:
  rating-3 skip, memify_metadata merge-not-replace, explicit-feedback
  precedence, clock/prune behavior, and the empty no-op case.
- ``refresh_preference_text`` watermark, cap, and section-filter logic.
"""

from types import SimpleNamespace

import pytest

import cognee.modules.user_preferences.update as update_module
from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.infrastructure.session.session_context_models import (
    SessionContextEntry,
    SessionFeedbackEntry,
)
from cognee.modules.user_preferences.constants import (
    MAX_PREFERENCE_TEXT_CHARS,
    PREFERENCE_DELETE_THRESHOLD,
    PREFERENCE_TURN_COUNTED_KEY,
    PREFERENCE_WEIGHTS_APPLIED_KEY,
)
from cognee.modules.user_preferences.update import (
    PreferenceUpdateScope,
    _run_preference_update,
    build_rating_map,
    refresh_preference_text,
    resolve_turn_rating,
)
from cognee.modules.user_preferences.weights import effective_weight
from cognee.tasks.memify.apply_feedback_weights import stream_update_weight

ALPHA = 0.3
BETA = 0.02


class TestPreviousAnswerRatingValidator:
    @pytest.mark.parametrize("value", [1, 2, 3, 4, 5])
    def test_in_range_kept(self, value):
        assert SessionTurnAnalysis(previous_answer_rating=value).previous_answer_rating == value

    @pytest.mark.parametrize("value", [0, 6, -1, 100, "junk", 3.7, True, [4], {"rating": 4}])
    def test_out_of_range_or_malformed_coerces_to_none(self, value):
        assert SessionTurnAnalysis(previous_answer_rating=value).previous_answer_rating is None

    def test_none_stays_none_and_is_the_default(self):
        assert SessionTurnAnalysis().previous_answer_rating is None
        assert SessionTurnAnalysis(previous_answer_rating=None).previous_answer_rating is None

    def test_integral_string_is_accepted(self):
        assert SessionTurnAnalysis(previous_answer_rating="4").previous_answer_rating == 4


class TestReferencedQaRatingValidator:
    def _entry(self, rating):
        return SessionFeedbackEntry(
            id="fb-1",
            created_at="2026-01-01T00:00:00+00:00",
            raw_text="feedback",
            referenced_qa_rating=rating,
        )

    @pytest.mark.parametrize("value", [1, 5])
    def test_in_range_kept(self, value):
        assert self._entry(value).referenced_qa_rating == value

    @pytest.mark.parametrize("value", [0, 6, "junk", 2.5, True])
    def test_out_of_range_or_malformed_coerces_to_none(self, value):
        assert self._entry(value).referenced_qa_rating is None

    def test_default_is_none(self):
        assert self._entry(None).referenced_qa_rating is None


class TestEffectiveWeightDecay:
    """The plan's decay table: a three-fives edge (0.8285) under BETA=0.02."""

    def _three_fives_weight(self):
        weight = 0.5
        for _ in range(3):
            weight = stream_update_weight(weight, 1.0, ALPHA)
        return weight

    def test_three_fives_reach_083(self):
        assert self._three_fives_weight() == pytest.approx(0.8285)

    def test_idle_zero_reads_stored_value(self):
        assert effective_weight(0.8285, 10, 10, BETA) == pytest.approx(0.8285)

    def test_idle_34_halves_the_distance_from_neutral(self):
        weight = self._three_fives_weight()
        decayed = effective_weight(weight, 0, 34, BETA)
        assert decayed == pytest.approx(0.665, abs=0.005)

    def test_idle_173_falls_under_the_delete_threshold(self):
        weight = self._three_fives_weight()
        decayed = effective_weight(weight, 0, 173, BETA)
        assert abs(decayed - 0.5) <= PREFERENCE_DELETE_THRESHOLD

    def test_idle_172_still_survives(self):
        weight = self._three_fives_weight()
        decayed = effective_weight(weight, 0, 172, BETA)
        assert abs(decayed - 0.5) > PREFERENCE_DELETE_THRESHOLD

    def test_clock_never_amplifies(self):
        # updated_at_turn ahead of the counter clamps to zero idle turns.
        assert effective_weight(0.9, 10, 5, BETA) == pytest.approx(0.9)


class TestRatingResolution:
    def test_explicit_feedback_score_wins(self):
        rating_map = {"qa-1": 5}
        row = {"qa_id": "qa-1", "feedback_score": 1}
        assert resolve_turn_rating(row, rating_map) == 1

    def test_inferred_rating_used_when_no_explicit_score(self):
        rating_map = {"qa-1": 4}
        assert resolve_turn_rating({"qa_id": "qa-1"}, rating_map) == 4

    def test_no_signal_is_none(self):
        assert resolve_turn_rating({"qa_id": "qa-1"}, {}) is None

    def test_rating_map_latest_entry_wins(self):
        rows = [
            {"referenced_qa_rating": 5, "referenced_qa_ids": ["qa-1"]},
            {"referenced_qa_rating": 2, "referenced_qa_ids": ["qa-1"]},
        ]
        assert build_rating_map(rows) == {"qa-1": 2}

    def test_rating_map_ignores_invalid_ratings(self):
        rows = [
            {"referenced_qa_rating": 9, "referenced_qa_ids": ["qa-1"]},
            {"referenced_qa_rating": None, "referenced_qa_ids": ["qa-2"]},
        ]
        assert build_rating_map(rows) == {}


class FakeSessionManager:
    def __init__(self, qas_by_session=None, context_by_session=None):
        self.qas_by_session = qas_by_session or {}
        self.context_by_session = context_by_session or {}
        self.update_qa_calls = []

    async def get_session(self, *, user_id, session_id, formatted=False, **kwargs):
        return list(self.qas_by_session.get(session_id, []))

    async def get_session_context_entries(self, *, user_id, session_id):
        return list(self.context_by_session.get(session_id, []))

    async def update_qa(self, *, user_id, session_id, qa_id, memify_metadata=None, **kwargs):
        self.update_qa_calls.append(
            {"session_id": session_id, "qa_id": qa_id, "memify_metadata": memify_metadata}
        )
        return True


class StoreRecorder:
    def __init__(self, node=None, stored=None):
        self.node = node
        self.stored = stored or {}
        self.upserts = []
        self.edge_writes = []
        self.deletes = []

    async def load_preference_state(self, user_id, dataset_id):
        return self.node, self.stored

    async def upsert_preference_node(
        self, user_id, dataset_id, *, text, turn_counter, text_watermark
    ):
        self.upserts.append(
            {"text": text, "turn_counter": turn_counter, "text_watermark": text_watermark}
        )
        return "pref-id"

    async def write_prefers_edges(self, user_id, dataset_id, weights_by_target, updated_at_turn):
        self.edge_writes.append(
            {"weights": dict(weights_by_target), "updated_at_turn": updated_at_turn}
        )

    async def delete_prefers_edges(self, user_id, dataset_id, target_ids):
        self.deletes.append(sorted(target_ids))


def _scope():
    return PreferenceUpdateScope(
        user=SimpleNamespace(id="user-1"),
        dataset=SimpleNamespace(id="dataset-1", owner_id="user-1"),
    )


def _wire(monkeypatch, session_manager, store):
    monkeypatch.setattr(update_module, "get_session_manager", lambda: session_manager)
    monkeypatch.setattr(update_module, "load_preference_state", store.load_preference_state)
    monkeypatch.setattr(update_module, "upsert_preference_node", store.upsert_preference_node)
    monkeypatch.setattr(update_module, "write_prefers_edges", store.write_prefers_edges)
    monkeypatch.setattr(update_module, "delete_prefers_edges", store.delete_prefers_edges)
    monkeypatch.setattr(
        update_module,
        "get_base_config",
        lambda: SimpleNamespace(preference_alpha=ALPHA, preference_beta=BETA),
    )


def _qa(qa_id, time, *, feedback_score=None, node_ids=None, memify_metadata=None):
    row = {
        "time": time,
        "qa_id": qa_id,
        "question": "q",
        "context": "",
        "answer": "a",
        "feedback_score": feedback_score,
        "memify_metadata": memify_metadata,
    }
    if node_ids is not None:
        row["used_graph_element_ids"] = {"node_ids": node_ids, "edge_ids": ["edge-x"]}
    return row


class TestRunPreferenceUpdate:
    @pytest.mark.asyncio
    async def test_rated_turn_writes_edges_and_marks_applied(self, monkeypatch):
        session_manager = FakeSessionManager(
            qas_by_session={
                "s1": [_qa("qa-1", "t1", feedback_score=5, node_ids=["n1", "n2"])],
            }
        )
        store = StoreRecorder()
        _wire(monkeypatch, session_manager, store)

        result = await _run_preference_update(_scope(), ["s1"])

        assert result.status == "completed"
        assert result.turns_applied == 1
        assert result.edges_written == 2
        assert store.edge_writes == [{"weights": {"n1": 0.65, "n2": 0.65}, "updated_at_turn": 1}]
        assert store.upserts and store.upserts[0]["turn_counter"] == 1
        [call] = session_manager.update_qa_calls
        assert call["memify_metadata"] == {
            PREFERENCE_TURN_COUNTED_KEY: True,
            PREFERENCE_WEIGHTS_APPLIED_KEY: True,
        }

    @pytest.mark.asyncio
    async def test_rating_three_is_counted_but_never_applied(self, monkeypatch):
        session_manager = FakeSessionManager(
            qas_by_session={
                "s1": [_qa("qa-1", "t1", feedback_score=3, node_ids=["n1"])],
            }
        )
        store = StoreRecorder()
        _wire(monkeypatch, session_manager, store)

        result = await _run_preference_update(_scope(), ["s1"])

        assert store.edge_writes == []
        assert result.turns_applied == 0
        # Counted so the clock advanced, but left without the applied key so a
        # later real rating still gets spent on this turn.
        [call] = session_manager.update_qa_calls
        assert call["memify_metadata"] == {PREFERENCE_TURN_COUNTED_KEY: True}
        assert PREFERENCE_WEIGHTS_APPLIED_KEY not in call["memify_metadata"]
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_memify_metadata_merges_and_keeps_real_bools(self, monkeypatch):
        session_manager = FakeSessionManager(
            qas_by_session={
                "s1": [
                    _qa(
                        "qa-1",
                        "t1",
                        feedback_score=5,
                        node_ids=["n1"],
                        memify_metadata={"feedback_weights_applied": True},
                    )
                ],
            }
        )
        store = StoreRecorder()
        _wire(monkeypatch, session_manager, store)

        await _run_preference_update(_scope(), ["s1"])

        [call] = session_manager.update_qa_calls
        metadata = call["memify_metadata"]
        # Merge, not replace: the global feedback marker survives.
        assert metadata["feedback_weights_applied"] is True
        assert metadata[PREFERENCE_TURN_COUNTED_KEY] is True
        assert metadata[PREFERENCE_WEIGHTS_APPLIED_KEY] is True
        assert all(isinstance(value, bool) for value in metadata.values())

    @pytest.mark.asyncio
    async def test_inferred_rating_from_feedback_entry(self, monkeypatch):
        feedback_row = {
            "kind": "feedback",
            "id": "fb-1",
            "created_at": "t2",
            "referenced_qa_ids": ["qa-1"],
            "referenced_qa_rating": 1,
        }
        session_manager = FakeSessionManager(
            qas_by_session={"s1": [_qa("qa-1", "t1", node_ids=["n1"])]},
            context_by_session={"s1": [feedback_row]},
        )
        store = StoreRecorder()
        _wire(monkeypatch, session_manager, store)

        result = await _run_preference_update(_scope(), ["s1"])

        # target = normalize_feedback_score(1) = 0.0 -> 0.5 + 0.3*(0 - 0.5) = 0.35
        assert store.edge_writes == [{"weights": {"n1": 0.35}, "updated_at_turn": 1}]
        assert result.turns_applied == 1

    @pytest.mark.asyncio
    async def test_already_processed_turn_is_a_no_op(self, monkeypatch):
        session_manager = FakeSessionManager(
            qas_by_session={
                "s1": [
                    _qa(
                        "qa-1",
                        "t1",
                        feedback_score=5,
                        node_ids=["n1"],
                        memify_metadata={
                            PREFERENCE_TURN_COUNTED_KEY: True,
                            PREFERENCE_WEIGHTS_APPLIED_KEY: True,
                        },
                    )
                ],
            }
        )
        store = StoreRecorder(node={"turn_counter": 1, "text": "", "text_watermark": ""})
        _wire(monkeypatch, session_manager, store)

        result = await _run_preference_update(_scope(), ["s1"])

        assert result.status == "no_changes"
        assert store.edge_writes == []
        assert store.upserts == []
        assert session_manager.update_qa_calls == []

    @pytest.mark.asyncio
    async def test_unrated_turns_advance_clock_and_prune_neutral_edges(self, monkeypatch):
        session_manager = FakeSessionManager(
            qas_by_session={"s1": [_qa(f"qa-{i}", f"t{i}") for i in range(10)]}
        )
        store = StoreRecorder(
            node={"turn_counter": 5, "text": "", "text_watermark": ""},
            stored={"n-old": {"weight": 0.505, "updated_at_turn": 0}},
        )
        _wire(monkeypatch, session_manager, store)

        result = await _run_preference_update(_scope(), ["s1"])

        # Counter advanced by 10 unrated turns; the near-neutral edge decays
        # within the threshold and is collected.
        assert store.upserts and store.upserts[0]["turn_counter"] == 15
        assert store.deletes == [["n-old"]]
        assert result.edges_pruned == 1
        assert result.edges_written == 0
        assert len(session_manager.update_qa_calls) == 10

    @pytest.mark.asyncio
    async def test_strong_edge_survives_the_prune_pass(self, monkeypatch):
        session_manager = FakeSessionManager(qas_by_session={"s1": [_qa("qa-1", "t1")]})
        store = StoreRecorder(
            node={"turn_counter": 0, "text": "", "text_watermark": ""},
            stored={"n-strong": {"weight": 0.9, "updated_at_turn": 0}},
        )
        _wire(monkeypatch, session_manager, store)

        result = await _run_preference_update(_scope(), ["s1"])

        assert store.deletes == []
        assert result.edges_pruned == 0

    @pytest.mark.asyncio
    async def test_empty_sessions_create_nothing(self, monkeypatch):
        session_manager = FakeSessionManager()
        store = StoreRecorder()
        _wire(monkeypatch, session_manager, store)

        result = await _run_preference_update(_scope(), ["s1"])

        assert result.status == "no_changes"
        assert store.upserts == []
        assert store.edge_writes == []
        assert store.deletes == []

    @pytest.mark.asyncio
    async def test_unrated_turns_without_existing_node_do_not_create_one(self, monkeypatch):
        session_manager = FakeSessionManager(qas_by_session={"s1": [_qa("qa-1", "t1")]})
        store = StoreRecorder()
        _wire(monkeypatch, session_manager, store)

        result = await _run_preference_update(_scope(), ["s1"])

        assert store.upserts == []
        # The turn is still marked counted so re-runs never re-count it.
        [call] = session_manager.update_qa_calls
        assert call["memify_metadata"] == {PREFERENCE_TURN_COUNTED_KEY: True}
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_preference_text_folds_without_any_ratings(self, monkeypatch):
        context_row = {
            "kind": "context",
            "id": "ctx-1",
            "section": "preferences",
            "content": "Keep answers concise.",
            "confidence": 0.9,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        session_manager = FakeSessionManager(context_by_session={"s1": [context_row]})
        store = StoreRecorder()
        _wire(monkeypatch, session_manager, store)

        result = await _run_preference_update(_scope(), ["s1"])

        assert result.text_lines_added == 1
        assert result.status == "completed"
        [upsert] = store.upserts
        assert upsert["text"] == "Keep answers concise."
        assert upsert["text_watermark"] == "2026-01-01T00:00:00+00:00"


def _context_entry(
    entry_id, content, created_at, *, section="preferences", confidence=0.9, harmful=0
):
    return SessionContextEntry(
        id=entry_id,
        section=section,
        content=content,
        confidence=confidence,
        created_at=created_at,
        harmful_count=harmful,
    )


class TestRefreshPreferenceText:
    def test_sections_other_than_preferences_are_ignored(self):
        entries = [
            _context_entry("e1", "a goal", "t1", section="goals"),
            _context_entry("e2", "a lesson", "t2", section="lessons_learned"),
            _context_entry("e3", "a rule", "t3", section="rules"),
        ]
        text, watermark, added = refresh_preference_text("", "", entries)
        assert (text, watermark, added) == ("", "", 0)

    def test_gates_reuse_distillations_thresholds(self):
        entries = [
            _context_entry("e1", "harmful", "t1", harmful=1),
            _context_entry("e2", "low confidence", "t2", confidence=0.5),
        ]
        text, watermark, added = refresh_preference_text("", "", entries)
        assert added == 0

    def test_newest_first_and_repeats_kept(self):
        entries = [
            _context_entry("e1", "prefers tables", "2026-01-01T00:00:00"),
            _context_entry("e2", "prefers tables", "2026-01-03T00:00:00"),
            _context_entry("e3", "short answers", "2026-01-02T00:00:00"),
        ]
        text, watermark, added = refresh_preference_text("", "", entries)
        assert text.splitlines() == ["prefers tables", "short answers", "prefers tables"]
        assert watermark == "2026-01-03T00:00:00"
        assert added == 3

    def test_watermark_drops_already_folded_rows(self):
        entries = [
            _context_entry("e1", "old line", "2026-01-01T00:00:00"),
            _context_entry("e2", "boundary line", "2026-01-02T00:00:00"),
            _context_entry("e3", "new line", "2026-01-03T00:00:00"),
        ]
        text, watermark, added = refresh_preference_text("existing", "2026-01-02T00:00:00", entries)
        # Strictly newer only: the boundary row was folded on a previous run.
        assert text == "new line\nexisting"
        assert watermark == "2026-01-03T00:00:00"
        assert added == 1

    def test_rerun_with_no_new_rows_is_byte_identical(self):
        entries = [_context_entry("e1", "line", "2026-01-01T00:00:00")]
        text, watermark, added = refresh_preference_text("", "", entries)
        text2, watermark2, added2 = refresh_preference_text(text, watermark, entries)
        assert (text2, watermark2, added2) == (text, watermark, 0)

    def test_cap_drops_the_oldest_lines(self):
        old_text = "z" * MAX_PREFERENCE_TEXT_CHARS
        entries = [_context_entry("e1", "newest line", "2026-01-05T00:00:00")]
        text, watermark, added = refresh_preference_text(old_text, "", entries)
        assert len(text) == MAX_PREFERENCE_TEXT_CHARS
        assert text.startswith("newest line\n")
        assert added == 1
