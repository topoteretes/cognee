import sys
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.infrastructure.session.session_persist_watermark import (
    TRACE_PERSIST_WATERMARK,
    TracePersistWindow,
)
from cognee.modules.pipelines.exceptions import CognifyFailedError
from cognee.tasks.memify.cognify_agent_trace_feedback import cognify_agent_trace_feedback

# The module object (not the re-exported function), for patching get_session_manager.
cognify_agent_trace_feedback_module = sys.modules[
    "cognee.tasks.memify.cognify_agent_trace_feedback"
]


@pytest.mark.asyncio
async def test_cognify_agent_trace_feedback_success():
    trace_content = "Session ID: trace_session\n\ndraft plan succeeded.\nwrite_summary failed."

    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
    ):
        await cognify_agent_trace_feedback(trace_content, dataset_id="123")

    mock_add.assert_called_once_with(
        trace_content,
        dataset_id="123",
        node_set=["agent_trace_feedbacks"],
        user=None,
    )
    mock_cognify.assert_called_once_with(datasets=["123"], user=None)


@pytest.mark.asyncio
async def test_cognify_agent_trace_feedback_forwards_user():
    """Test user is forwarded to cognee.add/cognee.cognify."""
    trace_content = "Session ID: trace_session\n\ndraft plan succeeded."
    user = object()

    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
    ):
        await cognify_agent_trace_feedback(trace_content, dataset_id="123", user=user)

    mock_add.assert_called_once_with(
        trace_content,
        dataset_id="123",
        node_set=["agent_trace_feedbacks"],
        user=user,
    )
    mock_cognify.assert_called_once_with(datasets=["123"], user=user)


@pytest.mark.asyncio
async def test_cognify_agent_trace_feedback_custom_node_set_name():
    trace_content = "Session ID: trace_session\n\ndraft plan succeeded."

    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock),
    ):
        await cognify_agent_trace_feedback(
            trace_content,
            dataset_id="123",
            node_set_name="custom_trace_feedbacks",
        )

    mock_add.assert_called_once_with(
        trace_content,
        dataset_id="123",
        node_set=["custom_trace_feedbacks"],
        user=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["", "   \n\t  ", None])
async def test_cognify_agent_trace_feedback_rejects_empty_input(value):
    with pytest.raises(CogneeValidationError, match="Agent trace content cannot be empty"):
        await cognify_agent_trace_feedback(value)


@pytest.mark.asyncio
async def test_cognify_agent_trace_feedback_add_failure():
    trace_content = "Session ID: trace_session\n\nfeedback"

    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock),
    ):
        mock_add.side_effect = Exception("Add operation failed")

        with pytest.raises(CogneeSystemError, match="Failed to cognify agent trace content"):
            await cognify_agent_trace_feedback(trace_content)


@pytest.mark.asyncio
async def test_cognify_agent_trace_feedback_cognify_failure():
    trace_content = "Session ID: trace_session\n\nfeedback"

    with (
        patch("cognee.add", new_callable=AsyncMock),
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
    ):
        mock_cognify.side_effect = Exception("Cognify operation failed")

        with pytest.raises(CogneeSystemError, match="Failed to cognify agent trace content"):
            await cognify_agent_trace_feedback(trace_content)


# ----------------------------------------------------------------- watermark windows


class _FakeSessionManager:
    def __init__(self):
        self.rows: dict[str, list[dict]] = {}

    @property
    def store(self) -> list[dict]:
        return [row for rows in self.rows.values() for row in rows]

    async def get_session_context_entries(self, *, user_id, session_id=None):
        return list(self.rows.get(session_id, []))

    async def update_session_context_entry(self, *, user_id, entry_id, merge, session_id=None):
        for row in self.rows.get(session_id, []):
            if row.get("id") == entry_id:
                row.update(merge)
                return True
        return False

    async def create_session_context_entry(self, *, user_id, entry_dump, session_id=None):
        self.rows.setdefault(session_id, []).append(dict(entry_dump))
        return True


def _window(text="Session ID: s\n\nstep", persisted_trace_count=3, session_id="s"):
    return TracePersistWindow(
        user_id="u", session_id=session_id, text=text, persisted_trace_count=persisted_trace_count
    )


@pytest.mark.asyncio
async def test_a_batch_of_windows_is_one_cognify_run_and_every_watermark_advances():
    manager = _FakeSessionManager()
    windows = [
        _window(text=f"Session ID: {sid}\n\nstep", persisted_trace_count=count, session_id=sid)
        for sid, count in (("a", 4), ("b", 7), ("c", 9))
    ]
    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock, return_value={}) as mock_cognify,
        patch.object(
            cognify_agent_trace_feedback_module, "get_session_manager", return_value=manager
        ),
    ):
        await cognify_agent_trace_feedback(windows, dataset_id="123")

    assert mock_add.await_count == 3
    mock_cognify.assert_awaited_once()
    for sid, count in (("a", 4), ("b", 7), ("c", 9)):
        assert await TRACE_PERSIST_WATERMARK.read_count(manager, "u", sid) == count


@pytest.mark.asyncio
async def test_a_failed_batch_run_advances_no_watermark():
    manager = _FakeSessionManager()
    windows = [
        _window(text=f"Session ID: {sid}\n\nstep", persisted_trace_count=count, session_id=sid)
        for sid, count in (("a", 4), ("b", 7))
    ]
    failed = CognifyFailedError(
        dataset_name="docs", error_class="RuntimeError", error_message="one window could not build"
    )
    with (
        patch("cognee.add", new_callable=AsyncMock),
        patch("cognee.cognify", new_callable=AsyncMock, side_effect=failed),
        patch.object(
            cognify_agent_trace_feedback_module, "get_session_manager", return_value=manager
        ),
        pytest.raises(CognifyFailedError, match="one window could not build"),
    ):
        await cognify_agent_trace_feedback(windows, dataset_id="123")

    assert await TRACE_PERSIST_WATERMARK.read_count(manager, "u", "a") == 0
    assert await TRACE_PERSIST_WATERMARK.read_count(manager, "u", "b") == 0


@pytest.mark.asyncio
async def test_window_advances_watermark_after_successful_cognify():
    manager = _FakeSessionManager()
    window = _window(persisted_trace_count=5)

    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock),
        patch.object(
            cognify_agent_trace_feedback_module, "get_session_manager", return_value=manager
        ),
    ):
        await cognify_agent_trace_feedback(window, dataset_id="123", user="user")

    mock_add.assert_called_once_with(
        window.text, dataset_id="123", node_set=["agent_trace_feedbacks"], user="user"
    )
    assert await TRACE_PERSIST_WATERMARK.read_count(manager, "u", "s") == 5


@pytest.mark.asyncio
async def test_batched_windows_each_advance_their_own_session():
    manager = _FakeSessionManager()
    windows = [
        _window(session_id="s1", persisted_trace_count=2),
        _window(session_id="s2", persisted_trace_count=7),
    ]

    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
        patch.object(
            cognify_agent_trace_feedback_module, "get_session_manager", return_value=manager
        ),
    ):
        await cognify_agent_trace_feedback(windows, dataset_id="123")

    assert mock_add.await_count == 2
    assert mock_cognify.await_count == 1  # one run for the whole batch
    assert await TRACE_PERSIST_WATERMARK.read_count(manager, "u", "s1") == 2
    assert await TRACE_PERSIST_WATERMARK.read_count(manager, "u", "s2") == 7


@pytest.mark.asyncio
async def test_cognify_exception_keeps_watermark_put():
    manager = _FakeSessionManager()

    with (
        patch("cognee.add", new_callable=AsyncMock),
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
        patch.object(
            cognify_agent_trace_feedback_module, "get_session_manager", return_value=manager
        ),
    ):
        mock_cognify.side_effect = Exception("Cognify operation failed")
        with pytest.raises(CogneeSystemError, match="Failed to cognify agent trace content"):
            await cognify_agent_trace_feedback(_window(persisted_trace_count=5), dataset_id="123")

    assert await TRACE_PERSIST_WATERMARK.read_count(manager, "u", "s") == 0


@pytest.mark.asyncio
async def test_blank_window_is_rejected_like_empty_text():
    with pytest.raises(CogneeValidationError, match="Agent trace content cannot be empty"):
        await cognify_agent_trace_feedback([_window(text="   \n")])


@pytest.mark.asyncio
async def test_plain_text_never_touches_the_session_manager():
    def boom():
        raise AssertionError("plain text carries no watermark")

    with (
        patch("cognee.add", new_callable=AsyncMock),
        patch("cognee.cognify", new_callable=AsyncMock),
        patch.object(cognify_agent_trace_feedback_module, "get_session_manager", boom),
    ):
        await cognify_agent_trace_feedback("Session ID: s\n\nfeedback", dataset_id="123")


@pytest.mark.asyncio
async def test_session_node_set_is_appended_to_the_stage_node_set():
    """The stage node set stays first; the session's pinned set follows (SDK-336)."""
    window = TracePersistWindow(
        user_id="u",
        session_id="s",
        text="Session ID: s\n\nedit succeeded.",
        persisted_trace_count=1,
        node_set=("project-a",),
    )
    manager = MagicMock()
    manager.get_session_context_entries = AsyncMock(return_value=[])
    manager.update_session_context_entry = AsyncMock(return_value=True)
    manager.create_session_context_entry = AsyncMock(return_value=True)

    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock),
        patch.object(
            cognify_agent_trace_feedback_module, "get_session_manager", return_value=manager
        ),
    ):
        await cognify_agent_trace_feedback(window, dataset_id="123")

    assert mock_add.await_args.kwargs["node_set"] == ["agent_trace_feedbacks", "project-a"]


@pytest.mark.asyncio
async def test_a_failed_step_after_the_run_is_raised_after_the_watermarks_move():
    """The traces are cognified; only the step after the run failed. The watermarks
    still move, so the windows are not rebuilt, and then the error is raised."""
    from cognee.modules.pipelines.exceptions import AfterRunHookError

    manager = _FakeSessionManager()
    windows = [
        _window(text=f"Session ID: {sid}\n\nstep", persisted_trace_count=count, session_id=sid)
        for sid, count in (("a", 4), ("b", 7))
    ]
    failure = AfterRunHookError([("123", "ds", RuntimeError("compaction"))])
    with (
        patch("cognee.add", new_callable=AsyncMock),
        patch("cognee.cognify", new_callable=AsyncMock, side_effect=failure),
        patch.object(
            cognify_agent_trace_feedback_module, "get_session_manager", return_value=manager
        ),
        pytest.raises(AfterRunHookError) as raised,
    ):
        await cognify_agent_trace_feedback(windows, dataset_id="123")

    assert raised.value is failure
    assert await TRACE_PERSIST_WATERMARK.read_count(manager, "u", "a") == 4
    assert await TRACE_PERSIST_WATERMARK.read_count(manager, "u", "b") == 7
