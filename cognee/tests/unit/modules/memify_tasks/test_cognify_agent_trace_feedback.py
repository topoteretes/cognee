from unittest.mock import AsyncMock, patch

import pytest

from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.tasks.memify.cognify_agent_trace_feedback import cognify_agent_trace_feedback


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


@pytest.mark.asyncio
async def test_watermark_advances_only_after_successful_cognify():
    """A TracePersistWindow advances the trace persist watermark after cognify;
    a failed cognify leaves it untouched so the window retries next run."""
    import sys

    task_module = sys.modules["cognee.tasks.memify.cognify_agent_trace_feedback"]
    from unittest.mock import AsyncMock, patch

    from cognee.infrastructure.session.session_persist_watermark import TracePersistWindow
    from cognee.tasks.memify.cognify_agent_trace_feedback import cognify_agent_trace_feedback

    window = TracePersistWindow(
        user_id="u1", session_id="s1", text="Session ID: s1\n\nstep", persisted_trace_count=7
    )

    save_spy = AsyncMock()
    session_manager = object()
    with (
        patch.object(task_module.cognee, "add", new=AsyncMock()),
        patch.object(task_module.cognee, "cognify", new=AsyncMock()),
        patch.object(task_module, "save_persisted_trace_count", new=save_spy),
        patch(
            "cognee.infrastructure.session.get_session_manager.get_session_manager",
            return_value=session_manager,
        ),
    ):
        await cognify_agent_trace_feedback(window, dataset_id="d1", user=None)

    save_spy.assert_awaited_once_with(session_manager, "u1", "s1", persisted_trace_count=7)

    save_spy.reset_mock()
    with (
        patch.object(task_module.cognee, "add", new=AsyncMock(side_effect=RuntimeError("down"))),
        patch.object(task_module, "save_persisted_trace_count", new=save_spy),
    ):
        with pytest.raises(Exception):
            await cognify_agent_trace_feedback(window, dataset_id="d1", user=None)

    save_spy.assert_not_awaited()
