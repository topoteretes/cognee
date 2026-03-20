from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.memify_pipelines.apply_feedback_weights import apply_feedback_weights_pipeline


@pytest.mark.asyncio
async def test_apply_feedback_weights_pipeline_validates_session_ids():
    user = MagicMock()
    user.id = "u1"

    with pytest.raises(CogneeValidationError, match="session_ids must be a non-empty list"):
        await apply_feedback_weights_pipeline(user=user, session_ids=[])

    with pytest.raises(CogneeValidationError, match="session_ids must be a non-empty list"):
        await apply_feedback_weights_pipeline(user=user, session_ids="session_1")


@pytest.mark.asyncio
async def test_apply_feedback_weights_pipeline_wires_memify_tasks():
    user = MagicMock()
    user.id = "u1"
    authorized_dataset = SimpleNamespace(id="dataset-1", owner_id="owner-1")

    with (
        patch(
            "cognee.memify_pipelines.apply_feedback_weights.set_session_user_context_variable",
            new=AsyncMock(),
        ) as set_user_ctx,
        patch(
            "cognee.memify_pipelines.apply_feedback_weights.get_authorized_existing_datasets",
            new=AsyncMock(return_value=[authorized_dataset]),
        ) as get_authorized_dataset,
        patch(
            "cognee.memify_pipelines.apply_feedback_weights.set_database_global_context_variables",
            new=AsyncMock(),
        ) as set_db_ctx,
        patch(
            "cognee.memify_pipelines.apply_feedback_weights.memify",
            new=AsyncMock(return_value={"status": "ok"}),
        ) as memify_mock,
    ):
        result = await apply_feedback_weights_pipeline(
            user=user,
            session_ids=["s1", "s2"],
            dataset="main_dataset",
            alpha=0.2,
            batch_size=25,
        )

    assert result == {"status": "ok"}
    set_user_ctx.assert_awaited_once_with(user)
    get_authorized_dataset.assert_awaited_once()
    set_db_ctx.assert_awaited_once_with("dataset-1", "owner-1")

    memify_kwargs = memify_mock.call_args.kwargs
    assert memify_kwargs["dataset"] == "dataset-1"
    assert memify_kwargs["data"] == [{}]
    assert len(memify_kwargs["extraction_tasks"]) == 1
    assert len(memify_kwargs["enrichment_tasks"]) == 1

    extraction_task = memify_kwargs["extraction_tasks"][0]
    enrichment_task = memify_kwargs["enrichment_tasks"][0]
    assert extraction_task.default_params["kwargs"]["session_ids"] == ["s1", "s2"]
    assert enrichment_task.default_params["kwargs"]["alpha"] == 0.2
    assert enrichment_task.task_config["batch_size"] == 25
