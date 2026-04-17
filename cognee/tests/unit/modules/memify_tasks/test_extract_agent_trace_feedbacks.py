import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.exceptions import CogneeSystemError
from cognee.modules.users.models import User
from cognee.tasks.memify.extract_agent_trace_feedbacks import extract_agent_trace_feedbacks

extract_agent_trace_feedbacks_module = sys.modules[
    "cognee.tasks.memify.extract_agent_trace_feedbacks"
]


@pytest.fixture
def mock_user():
    user = MagicMock(spec=User)
    user.id = "test-user-123"
    return user


def _make_mock_session_manager(feedback_entries, is_available: bool = True):
    mock_session_manager = MagicMock()
    mock_session_manager.is_available = is_available

    async def _get_agent_trace_feedback(*, user_id, session_id, last_n=None):
        del user_id, session_id
        if last_n is None:
            return feedback_entries
        return feedback_entries[-last_n:]

    mock_session_manager.get_agent_trace_feedback = AsyncMock(side_effect=_get_agent_trace_feedback)
    mock_session_manager.get_agent_trace_session = AsyncMock(return_value=[])
    return mock_session_manager


@pytest.mark.asyncio
async def test_extract_agent_trace_feedbacks_success(mock_user):
    mock_session_manager = _make_mock_session_manager(
        ["draft plan succeeded.", "   ", "write_summary failed."]
    )

    with (
        patch.object(extract_agent_trace_feedbacks_module, "session_user") as mock_session_user,
        patch.object(
            extract_agent_trace_feedbacks_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        feedback_sessions = []
        async for feedback in extract_agent_trace_feedbacks([{}], session_ids=["trace_session"]):
            feedback_sessions.append(feedback)

    assert feedback_sessions == [
        "Session ID: trace_session\n\ndraft plan succeeded.\nwrite_summary failed."
    ]
    mock_session_manager.get_agent_trace_feedback.assert_called_once_with(
        user_id="test-user-123",
        session_id="trace_session",
        last_n=None,
    )


@pytest.mark.asyncio
async def test_extract_agent_trace_feedbacks_multiple_sessions(mock_user):
    mock_session_manager = _make_mock_session_manager(["step completed."])

    with (
        patch.object(extract_agent_trace_feedbacks_module, "session_user") as mock_session_user,
        patch.object(
            extract_agent_trace_feedbacks_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        feedback_sessions = []
        async for feedback in extract_agent_trace_feedbacks(
            [{}], session_ids=["session1", "session2"]
        ):
            feedback_sessions.append(feedback)

    assert len(feedback_sessions) == 2
    assert mock_session_manager.get_agent_trace_feedback.await_count == 2
    mock_session_manager.get_agent_trace_session.assert_not_called()


@pytest.mark.asyncio
async def test_extract_agent_trace_feedbacks_skips_empty_feedback(mock_user):
    mock_session_manager = _make_mock_session_manager(["   ", ""])

    with (
        patch.object(extract_agent_trace_feedbacks_module, "session_user") as mock_session_user,
        patch.object(
            extract_agent_trace_feedbacks_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        feedback_sessions = []
        async for feedback in extract_agent_trace_feedbacks([{}], session_ids=["empty_session"]):
            feedback_sessions.append(feedback)

    assert feedback_sessions == []


@pytest.mark.asyncio
async def test_extract_agent_trace_feedbacks_no_session_ids(mock_user):
    mock_session_manager = _make_mock_session_manager(["step completed."])

    with (
        patch.object(extract_agent_trace_feedbacks_module, "session_user") as mock_session_user,
        patch.object(
            extract_agent_trace_feedbacks_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        feedback_sessions = []
        async for feedback in extract_agent_trace_feedbacks([{}], session_ids=None):
            feedback_sessions.append(feedback)

    assert feedback_sessions == []
    mock_session_manager.get_agent_trace_feedback.assert_not_called()


@pytest.mark.asyncio
async def test_extract_agent_trace_feedbacks_session_manager_unavailable(mock_user):
    mock_session_manager = _make_mock_session_manager([], is_available=False)

    with (
        patch.object(extract_agent_trace_feedbacks_module, "session_user") as mock_session_user,
        patch.object(
            extract_agent_trace_feedbacks_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        with pytest.raises(CogneeSystemError) as exc_info:
            async for _ in extract_agent_trace_feedbacks([{}], session_ids=["trace_session"]):
                pass

    assert "SessionManager not available" in str(exc_info.value)


@pytest.mark.asyncio
async def test_extract_agent_trace_feedbacks_continues_when_one_session_fails(mock_user):
    mock_session_manager = _make_mock_session_manager(["ignored"])
    mock_session_manager.get_agent_trace_feedback.side_effect = [
        ["first feedback"],
        Exception("SessionManager error"),
        ["third feedback"],
    ]

    with (
        patch.object(extract_agent_trace_feedbacks_module, "session_user") as mock_session_user,
        patch.object(
            extract_agent_trace_feedbacks_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        feedback_sessions = []
        async for feedback in extract_agent_trace_feedbacks(
            [{}], session_ids=["session1", "session2", "session3"]
        ):
            feedback_sessions.append(feedback)

    assert feedback_sessions == [
        "Session ID: session1\n\nfirst feedback",
        "Session ID: session3\n\nthird feedback",
    ]


@pytest.mark.asyncio
async def test_extract_agent_trace_feedbacks_can_extract_raw_return_values(mock_user):
    mock_session_manager = _make_mock_session_manager([])
    mock_session_manager.get_agent_trace_session.return_value = [
        {"method_return_value": "draft ready"},
        {"method_return_value": {"summary": "done", "steps": 2}},
        {"method_return_value": "   "},
        {"method_return_value": None},
    ]

    with (
        patch.object(extract_agent_trace_feedbacks_module, "session_user") as mock_session_user,
        patch.object(
            extract_agent_trace_feedbacks_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        extracted_values = []
        async for value in extract_agent_trace_feedbacks(
            [{}],
            session_ids=["trace_session"],
            raw_trace_content=True,
        ):
            extracted_values.append(value)

    assert extracted_values == [
        'Session ID: trace_session\n\ndraft ready\n{"steps": 2, "summary": "done"}'
    ]
    mock_session_manager.get_agent_trace_session.assert_awaited_once_with(
        user_id="test-user-123",
        session_id="trace_session",
        last_n=None,
    )
    mock_session_manager.get_agent_trace_feedback.assert_not_called()


@pytest.mark.asyncio
async def test_extract_agent_trace_feedbacks_skips_empty_raw_return_values(mock_user):
    mock_session_manager = _make_mock_session_manager([])
    mock_session_manager.get_agent_trace_session.return_value = [
        {"method_return_value": "   "},
        {"method_return_value": None},
    ]

    with (
        patch.object(extract_agent_trace_feedbacks_module, "session_user") as mock_session_user,
        patch.object(
            extract_agent_trace_feedbacks_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        extracted_values = []
        async for value in extract_agent_trace_feedbacks(
            [{}],
            session_ids=["trace_session"],
            raw_trace_content=True,
        ):
            extracted_values.append(value)

    assert extracted_values == []
    mock_session_manager.get_agent_trace_session.assert_awaited_once_with(
        user_id="test-user-123",
        session_id="trace_session",
        last_n=None,
    )
    mock_session_manager.get_agent_trace_feedback.assert_not_called()


@pytest.mark.asyncio
async def test_extract_agent_trace_feedbacks_rejects_non_boolean_raw_trace_content(mock_user):
    mock_session_manager = _make_mock_session_manager(["draft plan succeeded."])

    with (
        patch.object(extract_agent_trace_feedbacks_module, "session_user") as mock_session_user,
        patch.object(
            extract_agent_trace_feedbacks_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        with pytest.raises(CogneeSystemError, match="raw_trace_content must be a boolean"):
            async for _ in extract_agent_trace_feedbacks(
                [{}],
                session_ids=["trace_session"],
                raw_trace_content="yes",
            ):
                pass


@pytest.mark.asyncio
async def test_extract_agent_trace_feedbacks_limits_to_last_n_steps(mock_user):
    mock_session_manager = _make_mock_session_manager(
        ["first step", "second step", "third step", "fourth step"]
    )

    with (
        patch.object(extract_agent_trace_feedbacks_module, "session_user") as mock_session_user,
        patch.object(
            extract_agent_trace_feedbacks_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        feedback_sessions = []
        async for feedback in extract_agent_trace_feedbacks(
            [{}],
            session_ids=["trace_session"],
            last_n_steps=2,
        ):
            feedback_sessions.append(feedback)

    assert feedback_sessions == ["Session ID: trace_session\n\nthird step\nfourth step"]
    mock_session_manager.get_agent_trace_feedback.assert_called_once_with(
        user_id="test-user-123",
        session_id="trace_session",
        last_n=2,
    )


@pytest.mark.asyncio
async def test_extract_agent_trace_feedbacks_limits_raw_return_values_to_last_n_steps(mock_user):
    mock_session_manager = _make_mock_session_manager([])
    trace_entries = [
        {"method_return_value": "first return"},
        {"method_return_value": "second return"},
        {"method_return_value": "third return"},
    ]

    async def _get_agent_trace_session(*, user_id, session_id, last_n=None):
        del user_id, session_id
        if last_n is None:
            return trace_entries
        return trace_entries[-last_n:]

    mock_session_manager.get_agent_trace_session = AsyncMock(side_effect=_get_agent_trace_session)

    with (
        patch.object(extract_agent_trace_feedbacks_module, "session_user") as mock_session_user,
        patch.object(
            extract_agent_trace_feedbacks_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        extracted_values = []
        async for value in extract_agent_trace_feedbacks(
            [{}],
            session_ids=["trace_session"],
            raw_trace_content=True,
            last_n_steps=2,
        ):
            extracted_values.append(value)

    assert extracted_values == ["Session ID: trace_session\n\nsecond return\nthird return"]


@pytest.mark.asyncio
async def test_extract_agent_trace_feedbacks_passes_last_n_to_raw_trace_lookup(mock_user):
    mock_session_manager = _make_mock_session_manager([])
    mock_session_manager.get_agent_trace_session.return_value = [
        {"method_return_value": "second return"},
        {"method_return_value": "third return"},
    ]

    with (
        patch.object(extract_agent_trace_feedbacks_module, "session_user") as mock_session_user,
        patch.object(
            extract_agent_trace_feedbacks_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        extracted_values = []
        async for value in extract_agent_trace_feedbacks(
            [{}],
            session_ids=["trace_session"],
            raw_trace_content=True,
            last_n_steps=2,
        ):
            extracted_values.append(value)

    assert extracted_values == ["Session ID: trace_session\n\nsecond return\nthird return"]
    mock_session_manager.get_agent_trace_session.assert_awaited_once_with(
        user_id="test-user-123",
        session_id="trace_session",
        last_n=2,
    )
