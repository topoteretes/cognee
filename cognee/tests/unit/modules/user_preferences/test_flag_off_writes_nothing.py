"""With PERSONALIZATION_ENABLED off (the default), the feature writes nothing.

The flag must gate the writing half of personalization, not just the reading
half in ``lookup.py``: ``update_user_preferences`` returns early — no preference
node, no ``prefers`` edges, no new ``memify_metadata`` keys on session rows.

The implicit 1-5 answer rating is *not* personalization's to gate (plan item
C6). It is part of automatic feedback analysis: the per-turn LLM call asks the
rating question whenever ``AUTO_FEEDBACK`` is on, and a rating-only turn saves
its feedback row regardless of the personalization flag. Personalization is one
consumer of that row; with the flag off it simply never reads it.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import cognee.infrastructure.session.feedback_detection as feedback_detection_module
import cognee.infrastructure.session.session_turn as session_turn_module
import cognee.modules.user_preferences.update as update_module
from cognee.infrastructure.session.feedback_detection import analyze_turn_for_session_context
from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.infrastructure.session.session_turn import apply_session_turn_analysis
from cognee.modules.user_preferences.update import update_user_preferences


def _config(enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        personalization_enabled=enabled,
        preference_alpha=0.3,
        preference_beta=0.02,
    )


class FlagOffSessionManager:
    """Records every write; get calls return empty."""

    def __init__(self):
        self.created_entries = []
        self.update_qa_calls = []

    async def get_session(self, **kwargs):
        return []

    async def get_session_context_entries(self, **kwargs):
        return []

    async def create_session_context_entry(self, *, user_id, entry_dump, session_id):
        self.created_entries.append(entry_dump)
        return entry_dump.get("id")

    async def update_qa(self, **kwargs):
        self.update_qa_calls.append(kwargs)
        return True

    async def update_session_context_entry(self, **kwargs):
        raise AssertionError("update_session_context_entry must not be called")


class TestUpdateUserPreferencesFlagOff:
    @pytest.mark.asyncio
    async def test_flag_off_returns_early_and_writes_nothing(self, monkeypatch):
        session_manager = FlagOffSessionManager()
        monkeypatch.setattr(update_module, "get_base_config", lambda: _config(False))
        monkeypatch.setattr(update_module, "get_session_manager", lambda: session_manager)

        async def _fail(*args, **kwargs):
            raise AssertionError("flag off must return before touching scope or store")

        monkeypatch.setattr(update_module, "resolve_preference_scope", _fail)
        monkeypatch.setattr(update_module, "load_preference_state", _fail)
        monkeypatch.setattr(update_module, "upsert_preference_node", _fail)
        monkeypatch.setattr(update_module, "write_prefers_edges", _fail)
        monkeypatch.setattr(update_module, "delete_prefers_edges", _fail)

        result = await update_user_preferences(session_ids=["s1"], dataset="d1")

        assert result.status == "personalization_disabled"
        assert result.turns_applied == 0
        assert result.edges_written == 0
        assert result.edges_pruned == 0
        assert result.text_lines_added == 0
        assert session_manager.update_qa_calls == []
        assert session_manager.created_entries == []


class TestTurnAnalysisRatingQuestionIgnoresPersonalizationFlag:
    """The rating question follows AUTO_FEEDBACK, not PERSONALIZATION_ENABLED (C6)."""

    async def _captured_system_prompt(self, *, auto_feedback: bool) -> str:
        llm_mock = AsyncMock(return_value=SessionTurnAnalysis())
        cache_config = SimpleNamespace(caching=True, auto_feedback=auto_feedback)
        with (
            patch.object(feedback_detection_module, "get_cache_config", lambda: cache_config),
            patch.object(
                feedback_detection_module.LLMGateway,
                "acreate_structured_output",
                llm_mock,
            ),
        ):
            await analyze_turn_for_session_context(
                "That answer was perfect, thanks!",
                previous_question="q",
                previous_answer="a",
            )
        return llm_mock.call_args.kwargs["system_prompt"]

    @pytest.mark.asyncio
    async def test_personalization_off_still_asks_the_rating_question(self, monkeypatch):
        # PERSONALIZATION_ENABLED is off by default; the module no longer reads it.
        assert not hasattr(feedback_detection_module, "get_base_config")
        system_prompt = await self._captured_system_prompt(auto_feedback=True)
        assert "Previous answer rating" in system_prompt

    @pytest.mark.asyncio
    async def test_auto_feedback_off_omits_the_rating_question(self):
        system_prompt = await self._captured_system_prompt(auto_feedback=False)
        assert "Previous answer rating" not in system_prompt
        assert "previous_answer_rating" not in system_prompt


class TestFeedbackRowSaveIgnoresPersonalizationFlag:
    """A produced rating is persisted whether or not personalization consumes it (C6)."""

    def _rating_only_analysis(self) -> SessionTurnAnalysis:
        return SessionTurnAnalysis(previous_answer_rating=5)

    @pytest.mark.asyncio
    async def test_rating_only_turn_saves_the_feedback_row_without_personalization(self):
        # session_turn no longer consults base config at all.
        assert not hasattr(session_turn_module, "get_base_config")
        session_manager = FlagOffSessionManager()

        touched = await apply_session_turn_analysis(
            session_manager,
            user_id="user-1",
            session_id="s1",
            query="That answer was perfect, thanks!",
            analysis=self._rating_only_analysis(),
            previous_qa_id="qa-1",
            served_ids=[],
        )

        assert touched == []
        [entry] = session_manager.created_entries
        assert entry["referenced_qa_rating"] == 5
        assert entry["referenced_qa_ids"] == ["qa-1"]

    @pytest.mark.asyncio
    async def test_rating_without_a_previous_turn_saves_nothing(self):
        """No previous QA means nothing the rating can refer to, so no row."""
        session_manager = FlagOffSessionManager()

        touched = await apply_session_turn_analysis(
            session_manager,
            user_id="user-1",
            session_id="s1",
            query="That answer was perfect, thanks!",
            analysis=self._rating_only_analysis(),
            previous_qa_id=None,
            served_ids=[],
        )

        assert touched == []
        assert session_manager.created_entries == []
