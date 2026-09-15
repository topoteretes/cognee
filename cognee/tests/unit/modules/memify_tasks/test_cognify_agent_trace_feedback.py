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
    mock_cognify.assert_called_once_with(datasets=["123"], user=None, raise_on_error=False)


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
    mock_cognify.assert_called_once_with(datasets=["123"], user=user, raise_on_error=False)


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
async def test_cognify_agent_trace_feedback_errored_run_info_does_not_raise():
    """An errored build (raise_on_error=False path) is logged, not raised,
    so one bad trace session can't kill the whole memify run."""
    from uuid import uuid4

    from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunErrored

    errored = PipelineRunErrored(
        pipeline_run_id=uuid4(),
        dataset_id=uuid4(),
        dataset_name="ds",
        error_class="AuthenticationError",
        error_message="invalid api key",
    )

    with (
        patch("cognee.add", new_callable=AsyncMock),
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
    ):
        mock_cognify.return_value = {"ds": errored}

        await cognify_agent_trace_feedback("Session ID: trace_session\n\nfeedback")
