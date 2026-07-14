"""Integration-style tests for the session-context turn flow.

These exercise SessionManager._generate_completion_with_session_inner against a REAL FS-backed
cache (so the session-context CRUD round-trips) while the LLM calls are mocked. They cover
the four branches:

  1. first-turn       -> no served context, empty block, served_ids recorded as None
  2. ordinary question-> active block built + prepended above history, served_ids on new QA
  3. feedback-only    -> response_to_user returned, QA recorded, counters bumped, candidate applied
  4. feedback+request -> answer request, new QA with served_ids, candidate applied

The whole session-guidance layer is gated on caching and auto_feedback.
"""

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.infrastructure.session.feedback_models import FeedbackDetectionResult
from cognee.infrastructure.session.session_context_models import (
    CandidateContextUpdate,
    ServedContextRating,
    normalize_content,
)
from cognee.infrastructure.session.session_manager import SessionManager
from cognee.infrastructure.session.session_turn import apply_served_context_ratings


@pytest.fixture
def fs_adapter():
    """FSCacheAdapter backed by a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch(
            "cognee.infrastructure.databases.cache.fscache.FsCacheAdapter.get_storage_config",
            return_value={"data_root_directory": tmpdir},
        ):
            from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import (
                FSCacheAdapter,
            )

            inst = FSCacheAdapter()
            yield inst
            inst.cache.close()


@pytest.fixture
def session_manager(fs_adapter) -> SessionManager:
    return SessionManager(cache_engine=fs_adapter)


@pytest.fixture(autouse=True)
def session_vector_mocks():
    with (
        patch(
            "cognee.infrastructure.session.session_manager.index_session_qa", new_callable=AsyncMock
        ),
        patch(
            "cognee.infrastructure.session.session_turn.search_session_qa_ids",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        yield


def _config(*, auto_feedback: bool = True):
    cfg = MagicMock()
    cfg.caching = True
    cfg.auto_feedback = auto_feedback
    cfg.max_session_context_chars = None
    return cfg


def _patches(completion_return, analysis_return=None):
    """Patch session_user, CacheConfig, turn analysis, and completion."""
    user = MagicMock()
    user.id = "owner-1"  # non-UUID -> skips track_session_usage + session_records side effects

    mock_user = patch("cognee.infrastructure.session.session_manager.session_user")
    mock_cfg = patch("cognee.infrastructure.session.session_manager.CacheConfig")
    mock_analyze = patch(
        "cognee.infrastructure.session.session_turn.analyze_turn_for_session_context",
        new_callable=AsyncMock,
        return_value=analysis_return or FeedbackDetectionResult(),
    )
    mock_gen = patch(
        "cognee.infrastructure.session.session_turn.generate_session_completion_with_optional_summary",
        new_callable=AsyncMock,
        return_value=completion_return,
    )
    return user, mock_user, mock_cfg, mock_analyze, mock_gen


async def _seed_context_entry(sm, entry_id, section, content):
    await sm.create_session_context_entry(
        user_id="owner-1",
        entry_dump={
            "id": entry_id,
            "kind": "context",
            "section": section,
            "content": content,
            "normalized_content": normalize_content(content),
            "confidence": 0.9,
            "created_at": "2026-01-01T00:00:00",
        },
        session_id="s1",
    )


@pytest.mark.asyncio
async def test_first_turn_no_block_empty_served_ids(session_manager):
    """First turn: no stored context -> empty block, QA stored with no served context ids."""
    analysis = FeedbackDetectionResult(query_to_answer="What is X?")
    user, mock_user, mock_cfg, mock_analyze, mock_gen = _patches(
        ("Answer one", "", None), analysis_return=analysis
    )
    with mock_user as mu, mock_cfg as mc, mock_analyze as ma, mock_gen as mg:
        mu.get.return_value = user
        mc.return_value = _config()

        result = await session_manager.generate_completion_with_session(
            session_id="s1",
            query="What is X?",
            context="ctx",
            user_prompt_path="user.txt",
            system_prompt_path="sys.txt",
        )

    assert result == "Answer one"
    assert ma.call_args.kwargs["served_context"] == []
    history = mg.call_args.kwargs["conversation_history"]
    assert "## Active session guidance" not in history

    entries = await session_manager.get_session(user_id="owner-1", session_id="s1")
    assert len(entries) == 1
    assert entries[0].used_session_context_ids is None


@pytest.mark.asyncio
async def test_non_feedback_block_prepended_and_served_ids_recorded(session_manager):
    """Ordinary question: active block built and served_ids recorded on new QA."""
    await _seed_context_entry(session_manager, "c-rule", "rules", "Always answer in metric units.")

    analysis = FeedbackDetectionResult(query_to_answer="Give me the distance.")
    user, mock_user, mock_cfg, mock_analyze, mock_gen = _patches(
        ("Answer two", "", None), analysis_return=analysis
    )
    with mock_user as mu, mock_cfg as mc, mock_analyze, mock_gen as mg:
        mu.get.return_value = user
        mc.return_value = _config()

        result = await session_manager.generate_completion_with_session(
            session_id="s1",
            query="Give me the distance.",
            context="ctx",
            user_prompt_path="user.txt",
            system_prompt_path="sys.txt",
        )

    assert result == "Answer two"
    history = mg.call_args.kwargs["conversation_history"]
    assert "## Active session guidance" in history
    assert "Background knowledge from the knowledge graph" not in history
    assert "Always answer in metric units." in history

    entries = await session_manager.get_session(user_id="owner-1", session_id="s1")
    new_qa = entries[-1]
    assert new_qa.question == "Give me the distance."
    assert new_qa.used_session_context_ids == ["c-rule"]


@pytest.mark.asyncio
async def test_feedback_only_returns_thanks_records_qa_and_applies_candidate(session_manager):
    """Feedback-only turn: response_to_user returned, QA recorded, counter bumped, candidate stored."""
    await _seed_context_entry(session_manager, "c-served", "rules", "Be concise.")
    # A previous QA that served c-served, so it can be rated this turn.
    await session_manager.add_qa(
        user_id="owner-1",
        question="prev?",
        context="",
        answer="prev answer",
        session_id="s1",
        used_session_context_ids=["c-served"],
    )

    fb = FeedbackDetectionResult(
        response_to_user="Glad it helped!",
        served_context_ratings=[ServedContextRating(entry_id="c-served", rating="helpful")],
        candidate_context_updates=[
            CandidateContextUpdate(
                section="preferences", content="Prefer bullet points.", confidence=0.9
            )
        ],
    )

    user, mock_user, mock_cfg, mock_analyze, mock_gen = _patches(
        ("ignored answer", "", None),
        analysis_return=fb,
    )
    with (
        mock_user as mu,
        mock_cfg as mc,
        mock_analyze,
        mock_gen,
        patch.object(session_manager, "add_feedback", new_callable=AsyncMock) as mock_add_feedback,
    ):
        mu.get.return_value = user
        mc.return_value = _config()

        result = await session_manager.generate_completion_with_session(
            session_id="s1",
            query="that was great",
            context="ctx",
            user_prompt_path="user.txt",
            system_prompt_path="sys.txt",
        )

    assert result == "Glad it helped!"
    mock_add_feedback.assert_not_called()

    entries = await session_manager.get_session(user_id="owner-1", session_id="s1")
    assert len(entries) == 2
    assert entries[-1].question == "that was great"
    assert entries[-1].answer == "Glad it helped!"
    assert entries[-1].used_session_context_ids is None

    ctx_entries = await session_manager.get_session_context_entries(
        user_id="owner-1", session_id="s1"
    )
    served = next(e for e in ctx_entries if e.get("id") == "c-served")
    assert served["helpful_count"] == 1

    # The candidate was applied -> a new preferences context entry exists.
    prefs = [
        e
        for e in ctx_entries
        if e.get("kind", "context") == "context" and e.get("section") == "preferences"
    ]
    assert any(p["content"] == "Prefer bullet points." for p in prefs)

    # A feedback record was persisted in the same list.
    feedback_records = [e for e in ctx_entries if e.get("kind") == "feedback"]
    assert len(feedback_records) == 1
    assert feedback_records[0]["referenced_qa_ids"]


@pytest.mark.asyncio
async def test_duplicate_served_context_ratings_accumulate(session_manager):
    """Duplicate ratings for the same context entry increment from the latest local count."""
    await _seed_context_entry(session_manager, "c-served", "rules", "Be concise.")

    await apply_served_context_ratings(
        session_manager,
        user_id="owner-1",
        session_id="s1",
        ratings=[
            ServedContextRating(entry_id="c-served", rating="helpful"),
            ServedContextRating(entry_id="c-served", rating="helpful"),
            ServedContextRating(entry_id="c-served", rating="harmful"),
        ],
    )

    ctx_entries = await session_manager.get_session_context_entries(
        user_id="owner-1", session_id="s1"
    )
    served = next(e for e in ctx_entries if e.get("id") == "c-served")
    assert served["helpful_count"] == 2
    assert served["harmful_count"] == 1


@pytest.mark.asyncio
async def test_preference_only_turn_applies_candidate_without_answering(session_manager):
    """Instruction-only turn: preference is stored, acknowledgement is recorded."""
    await session_manager.add_qa(
        user_id="owner-1",
        question="prev?",
        context="",
        answer="prev answer",
        session_id="s1",
    )

    analysis = FeedbackDetectionResult(
        response_to_user="Got it.",
        candidate_context_updates=[
            CandidateContextUpdate(
                section="preferences",
                content="Prefer 2 informative bullet points for answers in this session.",
                confidence=0.9,
            )
        ],
    )

    user, mock_user, mock_cfg, mock_analyze, mock_gen = _patches(
        ("ignored answer", "", None),
        analysis_return=analysis,
    )
    with mock_user as mu, mock_cfg as mc, mock_analyze, mock_gen as mg:
        mu.get.return_value = user
        mc.return_value = _config()

        result = await session_manager.generate_completion_with_session(
            session_id="s1",
            query="For now, answer with 2 informative bullet points.",
            context="ctx",
            user_prompt_path="user.txt",
            system_prompt_path="sys.txt",
        )

    assert result == "Got it."
    mg.assert_not_called()

    entries = await session_manager.get_session(user_id="owner-1", session_id="s1")
    assert len(entries) == 2
    assert entries[-1].question == "For now, answer with 2 informative bullet points."
    assert entries[-1].answer == "Got it."
    assert entries[-1].used_session_context_ids is None

    ctx_entries = await session_manager.get_session_context_entries(
        user_id="owner-1", session_id="s1"
    )
    preferences = [
        e
        for e in ctx_entries
        if e.get("kind", "context") == "context" and e.get("section") == "preferences"
    ]
    assert any(
        p["content"] == "Prefer 2 informative bullet points for answers in this session."
        for p in preferences
    )


@pytest.mark.asyncio
async def test_feedback_followup_prepends_thanks_and_stores_qa(session_manager):
    """Feedback+follow-up: thanks prepended to answer, new QA stored with served_ids."""
    await _seed_context_entry(session_manager, "c-goal", "goals", "Help the user ship faster.")
    await session_manager.add_qa(
        user_id="owner-1",
        question="prev?",
        context="",
        answer="prev answer",
        session_id="s1",
        used_session_context_ids=["c-goal"],
    )

    fb = FeedbackDetectionResult(
        response_to_user="Thanks, noted.",
        query_to_answer="Now what about Y?",
        served_context_ratings=[ServedContextRating(entry_id="c-goal", rating="harmful")],
        candidate_context_updates=[
            CandidateContextUpdate(
                section="lessons_learned", content="Double-check unit conversions.", confidence=0.8
            )
        ],
    )

    user, mock_user, mock_cfg, mock_analyze, mock_gen = _patches(
        ("Here is Y.", "", None),
        analysis_return=fb,
    )
    with mock_user as mu, mock_cfg as mc, mock_analyze as ma, mock_gen:
        mu.get.return_value = user
        mc.return_value = _config()

        result = await session_manager.generate_completion_with_session(
            session_id="s1",
            query="that was wrong, now what about Y?",
            context="ctx",
            user_prompt_path="user.txt",
            system_prompt_path="sys.txt",
        )

    assert result == "Here is Y."

    # served_context (the previously-served entry) was passed to the feedback detector.
    served_arg = ma.call_args.kwargs["served_context"]
    assert served_arg == [{"id": "c-goal", "content": "Help the user ship faster."}]

    # A new QA was added with this turn's served_ids. The previous goal and the newly accepted
    # lesson are both available to the follow-up answer.
    entries = await session_manager.get_session(user_id="owner-1", session_id="s1")
    assert len(entries) == 2
    new_qa = entries[-1]
    assert new_qa.question == "that was wrong, now what about Y?"
    assert "c-goal" in new_qa.used_session_context_ids

    ctx_entries = await session_manager.get_session_context_entries(
        user_id="owner-1", session_id="s1"
    )
    served = next(e for e in ctx_entries if e.get("id") == "c-goal")
    assert served["harmful_count"] == 1
    lessons = [
        e
        for e in ctx_entries
        if e.get("kind", "context") == "context" and e.get("section") == "lessons_learned"
    ]
    matching_lessons = [p for p in lessons if p["content"] == "Double-check unit conversions."]
    assert matching_lessons
    assert matching_lessons[0]["id"] in new_qa.used_session_context_ids


@pytest.mark.asyncio
async def test_layer_disabled_when_auto_feedback_off(session_manager):
    """With auto_feedback False, no block, no feedback analysis, no served_ids on QA."""
    await _seed_context_entry(session_manager, "c-rule", "rules", "Always answer in metric units.")

    user, mock_user, mock_cfg, mock_analyze, mock_gen = _patches(("Answer", "", None))
    with mock_user as mu, mock_cfg as mc, mock_analyze as ma, mock_gen as mg:
        mu.get.return_value = user
        mc.return_value = _config(auto_feedback=False)

        await session_manager.generate_completion_with_session(
            session_id="s1",
            query="Q?",
            context="ctx",
            user_prompt_path="user.txt",
            system_prompt_path="sys.txt",
        )

    history = mg.call_args.kwargs["conversation_history"]
    assert "## Active session guidance" not in history
    ma.assert_not_called()

    entries = await session_manager.get_session(user_id="owner-1", session_id="s1")
    assert entries[-1].used_session_context_ids is None
